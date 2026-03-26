"""
Extract LLM calls from Langfuse traces.

Supports LangGraph and CrewAI frameworks.
Handles multiple providers: OpenAI, Anthropic, Gemini, etc.

Output fields match the unified CSV schema:
  id, Name, intent, task_id, step_in_trace_general, step_in_trace_node,
  model_input, response, tool_or_agent, meta_data, traj_score
"""

import json
from typing import Any, Dict, List, Callable

from .trace_utils import (
    safe_json,
    normalize_input_messages,
    normalize_response,
    extract_tool_calls,
    extract_api_spec,
    extract_intent_from_input,
    truncate_middle,
    _INTENT_LIMIT,
)


# ----------------- row building -----------------

def _build_llm_rows_for_observation(
    *,
    rows_out: List[Dict[str, Any]],
    trace_id: str,
    session_id: str,
    intent: str,
    obs: Dict[str, Any],
    agent_name: str,
    step_counter_ref: Dict[str, int],
    node_idx: int,
    separate_tools: bool = True,
    system_trunc_limit: int = 50_000,
    traj_score: Any = None,
) -> None:
    """
    Given a GENERATION observation, append 1..N CSV rows.

    Args:
        rows_out: List to append rows to (mutated)
        trace_id: Trace identifier
        session_id: Session identifier
        intent: User intent/query
        obs: Observation dict from Langfuse
        agent_name: Name of the calling agent/node
        step_counter_ref: Mutable counter dict {'v': int}
        node_idx: Index of this observation in the trace
        separate_tools: If True, emit separate rows for tool calls vs text
        system_trunc_limit: Max chars for system messages before truncation
        traj_score: Optional trajectory score
    """
    obs_id = obs.get("id")
    parent_id = obs.get("parentObservationId")
    model = obs.get("model", "")

    # Input - normalize and serialize
    input_data = safe_json(obs.get("input"))
    messages = input_data if isinstance(input_data, list) else input_data.get("messages") or input_data.get("contents")
    if messages:
        model_input_normalized = normalize_input_messages(messages, system_trunc_limit)
    else:
        model_input_normalized = normalize_input_messages(input_data, system_trunc_limit) if input_data else []

    # Serialize: if already string keep as-is, otherwise JSON encode
    if isinstance(model_input_normalized, str):
        model_input_str = model_input_normalized
    else:
        model_input_str = json.dumps(model_input_normalized, ensure_ascii=False)

    # Output
    output_data = safe_json(obs.get("output"))
    response_text = normalize_response(output_data)
    tool_calls = extract_tool_calls(output_data)

    # Extract bound tools (API spec)
    api_spec = extract_api_spec(input_data)
    api_spec_str = json.dumps(api_spec) if api_spec else ""

    # Build metadata with span-level info
    meta_data = {
        # Span-level info
        "span_id": obs_id,
        "span_name": obs.get("name"),
        "span_type": obs.get("type"),
        "parent_span_id": parent_id,
        "duration_ms": obs.get("latency"),
        "status": obs.get("statusMessage"),
        # Model-level info
        "observation_id": obs_id,
        "parent_observation_id": parent_id,
        "Name": agent_name,
        "session_id": session_id,
        "model": model,
        "tokens": {
            "prompt": obs.get("promptTokens", 0),
            "completion": obs.get("completionTokens", 0),
            "total": obs.get("totalTokens", 0),
        },
        "latency": obs.get("latency"),
        "cost": obs.get("calculatedTotalCost"),
    }

    if separate_tools:
        # Emit separate rows for tool calls
        if tool_calls:
            for tc in tool_calls:
                step_counter_ref["v"] += 1
                rows_out.append({
                    "id": f"{trace_id}_{step_counter_ref['v']}",
                    "Name": agent_name,
                    "intent": intent,
                    "task_id": trace_id,
                    "step_in_trace_general": step_counter_ref["v"],
                    "step_in_trace_node": node_idx + 1,
                    "model_input": model_input_str,
                    "response": json.dumps(tc, indent=2),
                    "tool_or_agent": "tool",
                    "api_spec": api_spec_str,
                    "meta_data": json.dumps(meta_data),
                    "traj_score": traj_score,
                })

        # Emit row for text response
        if response_text and response_text.strip() not in ("null", "None", ""):
            step_counter_ref["v"] += 1
            rows_out.append({
                "id": f"{trace_id}_{step_counter_ref['v']}",
                "Name": agent_name,
                "intent": intent,
                "task_id": trace_id,
                "step_in_trace_general": step_counter_ref["v"],
                "step_in_trace_node": node_idx + 1,
                "model_input": model_input_str,
                "response": response_text,
                "tool_or_agent": "agent",
                "api_spec": api_spec_str,
                "meta_data": json.dumps(meta_data),
                "traj_score": traj_score,
            })
    else:
        # Single row mode: combine everything
        step_counter_ref["v"] += 1
        combined_response = response_text
        if tool_calls:
            tool_parts = [json.dumps(tc, indent=2) for tc in tool_calls]
            if tool_parts:
                combined_response = "\n---\n".join(tool_parts)
                if response_text:
                    combined_response += f"\n---\n{response_text}"

        rows_out.append({
            "id": f"{trace_id}_{step_counter_ref['v']}",
            "Name": agent_name,
            "intent": intent,
            "task_id": trace_id,
            "step_in_trace_general": step_counter_ref["v"],
            "step_in_trace_node": node_idx + 1,
            "model_input": model_input_str,
            "response": combined_response,
            "tool_or_agent": "agent",
            "api_spec": api_spec_str,
            "meta_data": json.dumps(meta_data),
            "traj_score": traj_score,
        })


# ----------------- trace parsing -----------------

def _extract_trace_intent(
    json_data: Any,
    observations: List[Dict[str, Any]],
) -> str:
    """
    Extract a single intent for the entire Langfuse trace.

    Resolution order (first non-empty wins):
      1. Explicit trace-level metadata (user_query, intent, task, goal, …).
      2. Trace-level ``input`` field.
      3. Root observation's input (first user message or scalar field).
      4. Earliest observation's input (fallback).
    """
    # --- 1. Trace-level metadata ---
    if isinstance(json_data, dict):
        metadata = safe_json(json_data.get("metadata")) or {}
        for field in ("user_query", "intent", "task", "goal", "query", "question"):
            val = metadata.get(field)
            if val and isinstance(val, str) and val.strip():
                return truncate_middle(val.strip(), _INTENT_LIMIT)

        # --- 2. Trace-level input field ---
        trace_input = safe_json(json_data.get("input"))
        if trace_input:
            intent = extract_intent_from_input(trace_input)
            if intent:
                return intent

    # --- 3/4. Root / earliest observation input ---
    if observations:
        obs_sorted = sorted(observations, key=lambda x: x.get("startTime") or "")
        roots = [o for o in obs_sorted if not o.get("parentObservationId")]
        for obs in (roots or obs_sorted):
            obs_input = safe_json(obs.get("input"))
            if obs_input:
                intent = extract_intent_from_input(obs_input)
                if intent:
                    return intent

    return ""


def _extract_common_from_trace_root(json_data: Any, file_name: str):
    """
    Normalize the top-level trace container into:
      - intent, trace_id, session_id, observations[], traj_score
    Accepts either the full trace dict or a raw list of observations.
    """
    if isinstance(json_data, dict):
        metadata = safe_json(json_data.get("metadata")) or {}
        trace_id = json_data.get("id", file_name)
        session_id = json_data.get("sessionId")
        observations = json_data.get("observations", [])
        traj_score = json_data.get("traj_score") or metadata.get("traj_score")
    elif isinstance(json_data, list):
        observations = json_data
        session_id, traj_score = None, None
        trace_id = file_name
    else:
        raise TypeError("json_data must be a dict (trace) or a list (observations)")

    intent = _extract_trace_intent(json_data, observations)
    return intent, trace_id, session_id, observations, traj_score


def _sort_and_filter_generations(
    observations: List[Dict[str, Any]],
    predicate: Callable[[Dict[str, Any]], bool],
) -> List[Dict[str, Any]]:
    """
    Sort observations by startTime and keep only GENERATION types that pass predicate.
    """
    obs_sorted = sorted(observations, key=lambda x: x.get("startTime") or "")
    gens = [o for o in obs_sorted if o.get("type") == "GENERATION"]
    return [o for o in gens if predicate(o)]


# ----------------- framework-specific -----------------

def _langgraph_agent_name(obs: Dict[str, Any]) -> str:
    md = safe_json(obs.get("metadata")) or {}
    return md.get("langgraph_node") or obs.get("name") or "unknown_node"


def _crewai_agent_name(obs: Dict[str, Any]) -> str:
    md = safe_json(obs.get("metadata")) or {}
    return md.get("agent") or md.get("agent_name") or obs.get("name") or "unknown_agent"


def _langgraph_filter(obs: Dict[str, Any]) -> bool:
    """Exclude internal/housekeeping nodes common in LangGraph traces."""
    md = safe_json(obs.get("metadata")) or {}
    node = (md.get("langgraph_node") or "").lower()
    obs_name = (obs.get("name") or "").lower()
    skip_nodes = {
        "extract", "del_tool_call", "validate", "handle_retries",
        "filter_state", "coerce_inputs", "__start__", "enter"
    }
    skip_names = {"chatwatsonx", "userprofile", "trustcall"}

    if node and node in skip_nodes:
        return False
    if obs_name and (obs_name in skip_names or any(s in obs_name for s in skip_nodes)):
        return False

    # If there's no node metadata, still allow if it has substantive content
    out = safe_json(obs.get("output"))
    if isinstance(out, dict):
        content = out.get("content", "")
        if content and len(content) > 100:
            return True
    return True


def _crewai_filter(_obs: Dict[str, Any]) -> bool:
    """CrewAI: keep all GENERATION observations."""
    return True


# ----------------- public extractors -----------------

def extract_llm_calls_from_langgraph_trace(
    json_data: Any,
    file_name: str = None,
    separate_tools: bool = True,
    system_trunc_limit: int = 50_000,
) -> List[Dict[str, Any]]:
    """
    Extract LLM calls from a Langfuse trace produced by LangGraph-based apps.

    Args:
        json_data: Trace dict or list of observations
        file_name: Fallback name for trace_id
        separate_tools: If True, emit separate rows for tool calls vs text
        system_trunc_limit: Max chars for system messages before truncation

    Returns:
        List of row dicts matching the unified CSV schema
    """
    intent, trace_id, session_id, observations, traj_score = _extract_common_from_trace_root(json_data, file_name)
    rows: List[Dict[str, Any]] = []

    gens = _sort_and_filter_generations(observations, _langgraph_filter)
    step_counter = {"v": 0}

    for idx, obs in enumerate(gens):
        agent_name = _langgraph_agent_name(obs)
        _build_llm_rows_for_observation(
            rows_out=rows,
            trace_id=trace_id,
            session_id=session_id,
            intent=intent,
            obs=obs,
            agent_name=agent_name,
            step_counter_ref=step_counter,
            node_idx=idx,
            separate_tools=separate_tools,
            system_trunc_limit=system_trunc_limit,
            traj_score=traj_score,
        )

    return rows


def extract_llm_calls_from_crewai_trace(
    json_data: Any,
    file_name: str = None,
    separate_tools: bool = True,
    system_trunc_limit: int = 50_000,
) -> List[Dict[str, Any]]:
    """
    Extract LLM calls from a Langfuse trace produced by CrewAI-based apps.

    Args:
        json_data: Trace dict or list of observations
        file_name: Fallback name for trace_id
        separate_tools: If True, emit separate rows for tool calls vs text
        system_trunc_limit: Max chars for system messages before truncation

    Returns:
        List of row dicts matching the unified CSV schema
    """
    intent, trace_id, session_id, observations, traj_score = _extract_common_from_trace_root(json_data, file_name)
    rows: List[Dict[str, Any]] = []

    gens = _sort_and_filter_generations(observations, _crewai_filter)
    step_counter = {"v": 0}

    for idx, obs in enumerate(gens):
        agent_name = _crewai_agent_name(obs)
        _build_llm_rows_for_observation(
            rows_out=rows,
            trace_id=trace_id,
            session_id=session_id,
            intent=intent,
            obs=obs,
            agent_name=agent_name,
            step_counter_ref=step_counter,
            node_idx=idx,
            separate_tools=separate_tools,
            system_trunc_limit=system_trunc_limit,
            traj_score=traj_score,
        )

    return rows
