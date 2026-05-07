"""
Multi-Node Research Agent — Core (no tracing)
==============================================

6-node LangGraph pipeline with conditional routing and a review loop:

    planner → classifier ──┬──→ researcher → analyst → reviewer ──┬──→ writer → END
                           │                                      │
                           └───→ analyst → reviewer               ├──→ researcher (retry)
                                                                  │
                                                                  └──→ analyst   (retry)

  - planner:    LLM + tools → gathers initial facts for the question
  - classifier: LLM (no tools) → classifies question type, picks route
  - researcher: LLM + tools → deep-dives on missing facts, cross-references
  - analyst:    LLM + tools → computations, unit chains, derived metrics
  - reviewer:   LLM (no tools) → checks completeness/correctness, loops back or passes
  - writer:     LLM (no tools) → writes the final answer
"""

import json
import math
import operator
from typing import List, TypedDict, Literal

from typing_extensions import Annotated
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END


# ============================================================
# KNOWLEDGE BASE — expanded with cross-referencing traps
# ============================================================
# Values deliberately differ from real-world in some cases so the
# agent MUST use tools rather than its training data.

KNOWLEDGE_BASE = {
    # --- Cities ---
    "tokyo":       {"type": "city", "population": 13_960_000, "country": "Japan",
                    "area_km2": 2_194, "elevation_m": 40, "avg_temp_celsius": 16.3,
                    "gdp_billion_usd": 1_920, "timezone": "UTC+9"},
    "new york":    {"type": "city", "population": 8_260_000, "country": "United States",
                    "area_km2": 783, "elevation_m": 10, "avg_temp_celsius": 12.9,
                    "gdp_billion_usd": 1_500, "timezone": "UTC-5"},
    "paris":       {"type": "city", "population": 2_161_000, "country": "France",
                    "area_km2": 105, "elevation_m": 35, "avg_temp_celsius": 12.4,
                    "gdp_billion_usd": 850, "timezone": "UTC+1"},
    "london":      {"type": "city", "population": 8_982_000, "country": "United Kingdom",
                    "area_km2": 1_572, "elevation_m": 11, "avg_temp_celsius": 11.6,
                    "gdp_billion_usd": 1_100, "timezone": "UTC+0"},
    "sydney":      {"type": "city", "population": 5_312_000, "country": "Australia",
                    "area_km2": 12_368, "elevation_m": 58, "avg_temp_celsius": 18.4,
                    "gdp_billion_usd": 440, "timezone": "UTC+11"},
    "mumbai":      {"type": "city", "population": 20_667_000, "country": "India",
                    "area_km2": 603, "elevation_m": 14, "avg_temp_celsius": 27.2,
                    "gdp_billion_usd": 368, "timezone": "UTC+5:30"},
    "sao paulo":   {"type": "city", "population": 12_330_000, "country": "Brazil",
                    "area_km2": 1_521, "elevation_m": 760, "avg_temp_celsius": 19.2,
                    "gdp_billion_usd": 699, "timezone": "UTC-3"},

    # --- Planets ---
    "earth":   {"type": "planet", "radius_km": 6_371, "mass_kg": 5.972e24,
                "distance_from_sun_km": 149_600_000, "gravity_m_s2": 9.81,
                "orbital_period_days": 365.25, "avg_temp_celsius": 15,
                "atmosphere": "nitrogen/oxygen", "moons": 1},
    "mars":    {"type": "planet", "radius_km": 3_390, "mass_kg": 6.39e23,
                "distance_from_sun_km": 227_900_000, "gravity_m_s2": 3.72,
                "orbital_period_days": 687, "avg_temp_celsius": -65,
                "atmosphere": "carbon dioxide", "moons": 2},
    "jupiter": {"type": "planet", "radius_km": 69_911, "mass_kg": 1.898e27,
                "distance_from_sun_km": 778_500_000, "gravity_m_s2": 24.79,
                "orbital_period_days": 4_333, "avg_temp_celsius": -110,
                "atmosphere": "hydrogen/helium", "moons": 95},
    "venus":   {"type": "planet", "radius_km": 6_052, "mass_kg": 4.867e24,
                "distance_from_sun_km": 108_200_000, "gravity_m_s2": 8.87,
                "orbital_period_days": 224.7, "avg_temp_celsius": 464,
                "atmosphere": "carbon dioxide", "moons": 0},
    "saturn":  {"type": "planet", "radius_km": 58_232, "mass_kg": 5.683e26,
                "distance_from_sun_km": 1_434_000_000, "gravity_m_s2": 10.44,
                "orbital_period_days": 10_759, "avg_temp_celsius": -140,
                "atmosphere": "hydrogen/helium", "moons": 146},

    # --- Substances / elements ---
    "water":    {"type": "substance", "chemical_formula": "H2O",
                 "boiling_point_celsius": 100, "freezing_point_celsius": 0,
                 "density_kg_m3": 1000, "specific_heat_j_kg_k": 4186},
    "gold":     {"type": "element", "symbol": "Au", "atomic_number": 79,
                 "density_kg_m3": 19_300, "melting_point_celsius": 1_064,
                 "price_per_oz_usd": 2_350, "thermal_conductivity_w_mk": 317},
    "iron":     {"type": "element", "symbol": "Fe", "atomic_number": 26,
                 "density_kg_m3": 7_874, "melting_point_celsius": 1_538,
                 "price_per_kg_usd": 0.42, "thermal_conductivity_w_mk": 80},
    "copper":   {"type": "element", "symbol": "Cu", "atomic_number": 29,
                 "density_kg_m3": 8_960, "melting_point_celsius": 1_085,
                 "price_per_kg_usd": 8.50, "thermal_conductivity_w_mk": 401},
    "aluminum": {"type": "element", "symbol": "Al", "atomic_number": 13,
                 "density_kg_m3": 2_700, "melting_point_celsius": 660,
                 "price_per_kg_usd": 2.35, "thermal_conductivity_w_mk": 237},

    # --- Physical constants ---
    "speed of light": {"type": "physical_constant", "value_m_s": 299_792_458,
                       "value_km_s": 299_792.458, "symbol": "c"},
    "speed of sound":  {"type": "physical_constant", "value_m_s": 343,
                        "medium": "air at 20C", "symbol": "v_s"},
    "gravitational constant": {"type": "physical_constant",
                               "value_si": 6.674e-11, "symbol": "G",
                               "units": "m3/(kg*s2)"},

    # --- Programming languages ---
    "python":     {"type": "programming_language", "created_year": 1991,
                   "creator": "Guido van Rossum", "typing": "dynamically typed",
                   "paradigm": "multi-paradigm", "latest_major": "3.12"},
    "javascript": {"type": "programming_language", "created_year": 1995,
                   "creator": "Brendan Eich", "typing": "dynamically typed",
                   "paradigm": "multi-paradigm", "latest_major": "ES2024"},
    "rust":       {"type": "programming_language", "created_year": 2010,
                   "creator": "Graydon Hoare", "typing": "statically typed",
                   "paradigm": "multi-paradigm", "latest_major": "1.77"},

    # --- Countries (for GDP per capita chains) ---
    "japan":          {"type": "country", "population": 125_700_000,
                       "gdp_billion_usd": 4_231, "area_km2": 377_975,
                       "capital": "tokyo", "continent": "Asia"},
    "united states":  {"type": "country", "population": 331_900_000,
                       "gdp_billion_usd": 25_460, "area_km2": 9_833_520,
                       "capital": "washington dc", "continent": "North America"},
    "india":          {"type": "country", "population": 1_428_600_000,
                       "gdp_billion_usd": 3_730, "area_km2": 3_287_263,
                       "capital": "new delhi", "continent": "Asia"},
    "brazil":         {"type": "country", "population": 216_400_000,
                       "gdp_billion_usd": 2_170, "area_km2": 8_515_767,
                       "capital": "brasilia", "continent": "South America"},
}

CONVERSIONS = {
    ("km", "miles"): 0.621371,   ("miles", "km"): 1.60934,
    ("kg", "lbs"):   2.20462,    ("lbs", "kg"):   0.453592,
    ("celsius", "fahrenheit"):    lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"):    lambda v: (v - 32) * 5 / 9,
    ("meters", "feet"): 3.28084, ("feet", "meters"): 0.3048,
    ("liters", "gallons"): 0.264172, ("gallons", "liters"): 3.78541,
    ("cm", "inches"): 0.393701,  ("inches", "cm"): 2.54,
    ("km2", "miles2"): 0.386102, ("miles2", "km2"): 2.58999,
    ("m3", "ft3"):  35.3147,     ("ft3", "m3"):  0.0283168,
    ("kg/m3", "lbs/ft3"): 0.062428, ("lbs/ft3", "kg/m3"): 16.0185,
    ("oz", "grams"): 28.3495,   ("grams", "oz"): 0.035274,
    ("troy_oz", "grams"): 31.1035, ("grams", "troy_oz"): 0.032151,
}


# ============================================================
# TOOLS
# ============================================================

@tool
def knowledge_lookup(topic: str) -> str:
    """Look up facts about a topic in the knowledge base.
    You MUST use this tool — do not answer from memory.
    Available: tokyo, new york, paris, london, sydney, mumbai, sao paulo,
    earth, mars, jupiter, venus, saturn, water, gold, iron, copper, aluminum,
    speed of light, speed of sound, gravitational constant,
    python, javascript, rust, japan, united states, india, brazil."""
    key = topic.lower().strip()
    if key in KNOWLEDGE_BASE:
        return json.dumps(KNOWLEDGE_BASE[key], indent=2)
    matches = [k for k in KNOWLEDGE_BASE if key in k or k in key]
    if matches:
        return json.dumps({m: KNOWLEDGE_BASE[m] for m in matches[:3]}, indent=2)
    return f"NOT FOUND: '{topic}'. Available: {', '.join(sorted(KNOWLEDGE_BASE.keys()))}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. You MUST use this for ALL calculations.
    Examples: '13960000 / 2194', 'math.sqrt(144)', '4/3 * math.pi * 6371**3'.
    Available: math.*, abs, round, min, max, sum, pow."""
    allowed = {"__builtins__": {}, "math": math, "abs": abs, "round": round,
               "min": min, "max": max, "sum": sum, "pow": pow}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units. You MUST use this tool — do not convert manually.
    Supported pairs: km/miles, kg/lbs, celsius/fahrenheit, meters/feet,
    liters/gallons, cm/inches, km2/miles2, m3/ft3, kg/m3 to lbs/ft3,
    oz/grams, troy_oz/grams."""
    key = (from_unit.lower().strip(), to_unit.lower().strip())
    if key not in CONVERSIONS:
        avail = [f"{a} -> {b}" for a, b in CONVERSIONS.keys()]
        return f"Unsupported: {from_unit} -> {to_unit}. Available: {', '.join(avail)}"
    factor = CONVERSIONS[key]
    result = factor(value) if callable(factor) else value * factor
    return f"{value} {from_unit} = {round(result, 6)} {to_unit}"


ALL_TOOLS = [knowledge_lookup, calculator, unit_converter]
TOOL_MAP  = {t.name: t for t in ALL_TOOLS}


# ============================================================
# TOOL EXECUTION HELPER
# ============================================================

def execute_tool_calls(ai_message: AIMessage) -> List[ToolMessage]:
    results = []
    for tc in ai_message.tool_calls:
        tool_fn = TOOL_MAP.get(tc["name"])
        output = tool_fn.invoke(tc["args"]) if tool_fn else f"Unknown tool: {tc['name']}"
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return results


def llm_with_tool_loop(
    llm_with_tools, messages: List[BaseMessage], max_rounds: int = 6
) -> tuple[AIMessage, List[BaseMessage]]:
    new_messages: List[BaseMessage] = []
    response = None
    for _ in range(max_rounds):
        response = llm_with_tools.invoke(messages + new_messages)
        new_messages.append(response)
        if not response.tool_calls:
            return response, new_messages
        new_messages.extend(execute_tool_calls(response))
    return response, new_messages


# ============================================================
# GRAPH STATE
# ============================================================

MAX_REVIEW_CYCLES = 2

class AgentState(TypedDict):
    question:        str
    research_facts:  str
    analysis:        str
    final_answer:    str
    messages:        Annotated[List[BaseMessage], operator.add]
    model:           str
    route:           str            # set by classifier
    review_verdict:  str            # set by reviewer
    review_cycles:   int            # guards against infinite loops


def get_init_state(question: str, model: str) -> AgentState:
    return {
        "question": question,
        "research_facts": "",
        "analysis": "",
        "final_answer": "",
        "messages": [],
        "model": model,
        "route": "",
        "review_verdict": "",
        "review_cycles": 0,
    }


# ============================================================
# NODE FUNCTIONS
# ============================================================

def planner_node(state: AgentState) -> dict:
    """Node 1 — gather initial facts via tools."""
    llm = ChatOpenAI(model=state["model"], temperature=0).bind_tools(ALL_TOOLS)
    messages = [
        SystemMessage(content=(
            "You are a research planner. Given the user's question, "
            "use the knowledge_lookup tool to gather ALL raw facts needed. "
            "Call the tool ONCE PER TOPIC — look up every entity mentioned or implied. "
            "For comparison questions, look up BOTH sides. "
            "For country/city questions, look up BOTH the city AND the country if relevant. "
            "Do NOT compute anything. Do NOT answer. ONLY gather data via tools."
        )),
        HumanMessage(content=state["question"]),
    ]
    final_response, new_messages = llm_with_tool_loop(llm, messages)
    facts = [m.content for m in new_messages if isinstance(m, ToolMessage)]
    return {
        "research_facts": "\n\n".join(facts) if facts else final_response.content,
        "messages": [HumanMessage(content=state["question"])] + new_messages,
    }


def classifier_node(state: AgentState) -> dict:
    """Node 2 — classify the question to pick a route."""
    llm = ChatOpenAI(model=state["model"], temperature=0)
    messages = [
        SystemMessage(content=(
            "You are a question classifier. Based on the question and gathered facts, "
            "decide what the question primarily needs:\n\n"
            "- 'research_first' — needs more lookups before any calculations "
            "  (e.g., multi-entity comparisons, cross-referencing country+city data)\n"
            "- 'analysis_first' — facts are sufficient, needs computation/conversion "
            "  (e.g., pure math, simple unit conversion on already-known values)\n"
            "- 'research_and_analysis' — needs both more lookups AND heavy computation "
            "  (e.g., multi-hop chains: look up -> compute -> look up more -> compute)\n\n"
            "Respond with ONLY one of: research_first, analysis_first, research_and_analysis"
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Facts gathered so far:\n{state['research_facts']}"
        )),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower().replace("'", "").replace('"', '')
    if "research_and" in route:
        route = "research_and_analysis"
    elif "research" in route:
        route = "research_first"
    else:
        route = "analysis_first"
    return {"route": route, "messages": [response]}


def researcher_node(state: AgentState) -> dict:
    """Deep research — cross-reference, fill gaps, look up related entities."""
    llm = ChatOpenAI(model=state["model"], temperature=0).bind_tools(ALL_TOOLS)

    review_context = ""
    if state.get("review_verdict") == "needs_research":
        review_context = (
            "\n\nIMPORTANT: The reviewer found gaps in the previous research. "
            "Look up any missing entities or facts that are needed. "
            "Check if all entities referenced in the question have been looked up."
        )

    messages = [
        SystemMessage(content=(
            "You are a thorough researcher. Your job:\n"
            "1. Review the question and all facts gathered so far.\n"
            "2. Identify ANY missing information — entities not yet looked up, "
            "   related data needed for the computation, cross-references.\n"
            "3. Use knowledge_lookup to fill every gap. Look up EACH entity separately.\n"
            "4. Summarize ALL confirmed facts clearly, organized by entity.\n\n"
            "CRITICAL: You MUST use tools. Do NOT fill in values from memory. "
            "If the knowledge base doesn't have something, say so explicitly."
            + review_context
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Facts gathered so far:\n{state['research_facts']}"
        )),
    ]
    final_response, new_messages = llm_with_tool_loop(llm, messages)
    extra = [m.content for m in new_messages if isinstance(m, ToolMessage)]
    all_facts = state["research_facts"]
    if extra:
        all_facts += "\n\nAdditional findings:\n" + "\n\n".join(extra)
    return {
        "research_facts": all_facts + "\n\nResearcher summary: " + final_response.content,
        "messages": new_messages,
    }


def analyst_node(state: AgentState) -> dict:
    """Compute derived values — multi-step math, unit chains, ratios."""
    llm = ChatOpenAI(model=state["model"], temperature=0).bind_tools(ALL_TOOLS)

    review_context = ""
    if state.get("review_verdict") == "needs_analysis":
        review_context = (
            "\n\nIMPORTANT: The reviewer found errors or gaps in the previous analysis. "
            "Redo the calculations carefully. Show each step. Verify intermediate results."
        )

    messages = [
        SystemMessage(content=(
            "You are a data analyst. Your job:\n"
            "1. Read the question and all research facts.\n"
            "2. Plan the calculation steps BEFORE computing.\n"
            "3. Use the calculator tool for EVERY arithmetic operation — no mental math.\n"
            "4. Use unit_converter for EVERY unit conversion — no manual conversion.\n"
            "5. Chain results: use output of one calculation as input to the next.\n"
            "6. State each intermediate result clearly.\n"
            "7. End with a clear final computed answer.\n\n"
            "CRITICAL: Use tools for ALL math. Even simple multiplications. "
            "Double-check that you used the right values from the research facts, "
            "not from your own memory."
            + review_context
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Research facts:\n{state['research_facts']}\n\n"
            f"Previous analysis (if any): {state.get('analysis', 'None yet')}"
        )),
    ]
    final_response, new_messages = llm_with_tool_loop(llm, messages)
    return {"analysis": final_response.content, "messages": new_messages}


def reviewer_node(state: AgentState) -> dict:
    """Check quality and decide: pass to writer, or loop back."""
    llm = ChatOpenAI(model=state["model"], temperature=0)
    cycles = state.get("review_cycles", 0)

    if cycles >= MAX_REVIEW_CYCLES:
        return {
            "review_verdict": "pass",
            "review_cycles": cycles,
            "messages": [AIMessage(content="Max review cycles reached. Passing to writer.")],
        }

    messages = [
        SystemMessage(content=(
            "You are a quality reviewer. Check the research and analysis for:\n\n"
            "1. COMPLETENESS: Were all entities in the question looked up via tools?\n"
            "2. CORRECTNESS: Do the calculations use values from the research (not hallucinated)?\n"
            "3. TOOL USAGE: Were calculator/unit_converter used (not mental math)?\n"
            "4. CHAIN INTEGRITY: In multi-step problems, does each step feed correctly into the next?\n"
            "5. UNITS: Are all unit conversions done via the tool?\n\n"
            "Respond with EXACTLY one of these verdicts as the FIRST WORD:\n"
            "- 'pass' — everything looks correct and complete\n"
            "- 'needs_research' — missing lookups (say which entities)\n"
            "- 'needs_analysis' — calculation errors or missing steps (say which)\n\n"
            "Start with your verdict word, then explain briefly."
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Research facts:\n{state['research_facts']}\n\n"
            f"Analysis:\n{state['analysis']}\n\n"
            f"Review cycle: {cycles + 1}/{MAX_REVIEW_CYCLES + 1}"
        )),
    ]
    response = llm.invoke(messages)
    verdict_text = response.content.strip().lower()
    if verdict_text.startswith("needs_research"):
        verdict = "needs_research"
    elif verdict_text.startswith("needs_analysis"):
        verdict = "needs_analysis"
    else:
        verdict = "pass"

    return {
        "review_verdict": verdict,
        "review_cycles": cycles + 1,
        "messages": [response],
    }


def writer_node(state: AgentState) -> dict:
    """Write the final polished answer."""
    llm = ChatOpenAI(model=state["model"], temperature=0)
    messages = [
        SystemMessage(content=(
            "You are a skilled technical writer. Write the final answer:\n"
            "1. Lead with the DIRECT answer (number, name, or yes/no).\n"
            "2. Show the key calculation steps that led there.\n"
            "3. Include specific numbers with units.\n"
            "4. For comparisons, state both values and the difference/ratio.\n"
            "5. Mention any caveats briefly.\n"
            "6. Keep it under 200 words.\n\n"
            "CRITICAL: Only use values from the research and analysis provided. "
            "Do not add facts from your own knowledge."
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Research facts:\n{state['research_facts']}\n\n"
            f"Analysis:\n{state['analysis']}\n\n"
            f"Write the final answer:"
        )),
    ]
    response = llm.invoke(messages)
    return {"final_answer": response.content, "messages": [response]}


# ============================================================
# ROUTING FUNCTIONS (conditional edges)
# ============================================================

def route_after_classifier(state: AgentState) -> str:
    """Classifier -> one of two entry paths."""
    route = state.get("route", "research_and_analysis")
    if route == "analysis_first":
        return "analyst"
    else:  # research_first or research_and_analysis
        return "researcher"


def route_after_reviewer(state: AgentState) -> str:
    """Reviewer -> writer, or loop back to researcher/analyst."""
    verdict = state.get("review_verdict", "pass")
    if verdict == "needs_research":
        return "researcher"
    elif verdict == "needs_analysis":
        return "analyst"
    else:
        return "writer"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("planner",    planner_node)
    g.add_node("classifier", classifier_node)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst",    analyst_node)
    g.add_node("reviewer",   reviewer_node)
    g.add_node("writer",     writer_node)

    # Entry
    g.set_entry_point("planner")

    # Fixed edges
    g.add_edge("planner",    "classifier")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst",    "reviewer")
    g.add_edge("writer",     END)

    # Conditional edges
    g.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"researcher": "researcher", "analyst": "analyst"},
    )
    g.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer"},
    )

    return g.compile()


# ============================================================
# 20 SAMPLE INPUTS — designed to stress-test the pipeline
# ============================================================

SAMPLE_INPUTS = [
    # ---- Multi-hop chains (lookup -> compute -> lookup -> compute) ----

    # 1. Requires: tokyo city GDP, japan country GDP, then percentage calc
    "What percentage of Japan's total GDP does the city of Tokyo alone account for?",

    # 2. Requires: mumbai population + area -> density, tokyo population + area -> density, compare
    "Which city is more densely populated — Mumbai or Tokyo? By what factor?",

    # 3. Requires: earth distance, mars distance -> difference, speed of light -> time, convert to minutes
    "How many minutes does it take light to travel from Earth to Mars "
    "(at their average distances from the Sun)? Show each step.",

    # 4. Requires: gold density (kg/m3) -> convert to lbs/ft3, compute weight of 1 ft3,
    #    then convert lbs to grams, grams to troy_oz, multiply by price_per_oz_usd
    "How much would a solid gold cube with 1-foot sides weigh in lbs, "
    "and what would it be worth in USD at the current price per troy ounce?",

    # 5. Requires: all 7 cities, compute density for each, rank, identify top 3
    "Rank all available cities by population density (people per km2). "
    "Which three are densest and what are their exact densities?",

    # ---- Cross-referencing (city <-> country data) ----

    # 6. Requires: sao paulo city GDP, brazil country GDP
    "What fraction of Brazil's GDP comes from Sao Paulo?",

    # 7. Requires: mumbai city data to get country -> india, look up india, compute GDP per capita
    "What is the GDP per capita of the country Mumbai is in? "
    "Express the answer in USD and show the calculation.",

    # 8. Requires: japan area (km2) -> convert to miles2, US area -> convert to miles2, ratio
    "How many times larger is the United States than Japan in square miles?",

    # ---- Unit conversion chains (multiple hops) ----

    # 9. Requires: iron density kg/m3, convert to lbs/ft3, then compute 5 ft3 weight
    "How much would a solid block of iron measuring 5 cubic feet weigh in pounds?",

    # 10. Requires: venus avg temp (celsius) -> fahrenheit, earth avg temp -> fahrenheit, difference
    "What is the temperature difference between Venus and Earth "
    "in Fahrenheit? Show both temperatures.",

    # ---- Tricky reasoning (agent must NOT use training data) ----

    # 11. Requires: speed of sound (343 m/s) -> km/h calc, then compute time for 100km
    "How long in minutes would it take sound to travel 100 km "
    "(at standard conditions)? Use the speed of sound from the knowledge base.",

    # 12. Requires: copper, gold, aluminum thermal conductivity -> rank, compute ratio best/worst
    "Rank copper, gold, and aluminum by thermal conductivity. "
    "How much better is the best conductor compared to the worst? Give the ratio.",

    # 13. Requires: saturn moons + jupiter moons -> sum, mars moons -> percentage
    "What percentage of the total moons of Jupiter and Saturn "
    "does Mars account for?",

    # ---- Multi-step with intermediate failures expected ----

    # 14. Requires: gravitational constant, earth mass, earth radius -> g = G*M/R^2
    #     Compare to stored gravity value
    "Using the gravitational constant G and Earth's mass and radius, "
    "compute Earth's surface gravity from scratch. "
    "Does your result match the value stored in the knowledge base?",

    # 15. Requires: earth radius -> volume (4/3 pi r^3) in km3,
    #     mass / volume -> density in kg/km3, convert to kg/m3,
    #     compare to water and iron densities
    "Calculate Earth's average density in kg/m3 using its mass and radius. "
    "Is Earth denser than water? Than iron? Show all steps.",

    # ---- Comparison + conversion combos ----

    # 16. Requires: all 5 planets, orbital periods -> convert to earth years, rank by distance
    "List all available planets ordered by distance from the Sun. "
    "For each, convert the orbital period to Earth years.",

    # 17. Requires: london and new york avg temps -> fahrenheit, GDP / area for both, compare
    "Compare London and New York: which has higher GDP per square kilometer, "
    "and which has a warmer average temperature in Fahrenheit?",

    # ---- Edge cases / trap questions ----

    # 18. KB has Sao Paulo at 760m — agent must use this, not guess
    "Which available city has the highest elevation? "
    "Convert that elevation to feet.",

    # 19. Look up python + javascript, compare years, compute difference — trivial but often skipped
    "How many years older is Python than JavaScript? "
    "Who created each language?",

    # 20. All 4 countries -> GDP per capita, rank, ratio highest/lowest
    "Rank Japan, United States, India, and Brazil by GDP per capita. "
    "What is the ratio between the highest and lowest?",
]