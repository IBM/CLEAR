"""
Extract LLM calls from Langfuse traces.

Supports LangGraph and CrewAI frameworks.
Handles multiple providers: OpenAI, Anthropic, Gemini, etc.

Output fields match the unified CSV schema:
id, Name, intent, task_id, step_in_trace_general, llm_call_index,
  model_input, response, tool_or_agent, api_spec, meta_data, traj_score
"""

import json
from typing import Any, Dict, List, Callable

from .trace_utils import (
    safe_json,
    extract_messages_from_input,
    normalize_input_messages,
    normalize_response,
    extract_tool_calls,
    extract_api_spec,
    extract_intent_from_input,
    truncate_middle,
    build_csv_rows,
    _INTENT_LIMIT,
)


# ----------------- field extraction -----------------

def _extract_llm_call_record(
    *,
    trace_id: str,
    session_id: str,
    intent: str,
    obs: Dict[str, Any],
    agent_name: str,
    system_trunc_limit: int = 50_000,
    traj_score: Any = None,
) -> Dict[str, Any]:
    """
    Extract fields from a Langfuse GENERATION observation into a raw
    LLM-call record suitable for :func:`build_csv_rows`.

    Returns a single dict (one per LLM invocation) — row splitting and
    counter assignment happen in :func:`build_csv_rows`.
    """
    obs_id = obs.get("id")
    parent_id = obs.get("parentObservationId")
    model = obs.get("model", "")

    # Input - normalize and serialize
    input_data = safe_json(obs.get("input"))
    messages = input_data if isinstance(input_data, list) else extract_messages_from_input(input_data)

    if messages:
        model_input_str = json.dumps(normalize_input_messages(messages, system_trunc_limit), ensure_ascii=False)
    elif isinstance(input_data, str):
        model_input_str = input_data
    elif isinstance(input_data, dict) and input_data:
        model_input_str = json.dumps(input_data, ensure_ascii=False)
    else:
        model_input_str = ""

    # Output
    output_data = safe_json(obs.get("output"))
    response_text = normalize_response(output_data)
    tool_calls = extract_tool_calls(output_data)

    # Bound tools (API spec)
    api_spec = extract_api_spec(input_data)

    # Metadata
    meta_data = {
        "span_id": obs_id,
        "span_name": obs.get("name"),
        "span_type": obs.get("type"),
        "parent_span_id": parent_id,
        "duration_ms": obs.get("latency"),
        "status": obs.get("statusMessage"),
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

    return {
        "agent_name": agent_name,
        "task_id": trace_id,
        "intent": intent,
        "model_input": model_input_str,
        "response_text": response_text,
        "tool_calls": tool_calls,
        "api_spec": api_spec,
        "meta_data": meta_data,
        "traj_score": traj_score,
    }


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
    """
    Filter for LangGraph GENERATION observations.

    Since we already filter to type=='GENERATION', all observations here represent
    actual LLM API calls. Node names and observation names reflect graph structure
    and instrumentation, not whether the call is meaningful.

    Removed skip_nodes and skip_names filters to prevent data loss from legitimate
    LLM calls in nodes like "extract" or using clients like "ChatWatsonX".
    """
    # Accept all GENERATION observations - they represent real LLM calls
    return True


def _crewai_filter(_obs: Dict[str, Any]) -> bool:
    """CrewAI: keep all GENERATION observations."""
    return True


# ----------------- public extractors -----------------

def _extract_llm_call_records(
    json_data: Any,
    file_name: str,
    predicate: Callable[[Dict[str, Any]], bool],
    name_func: Callable[[Dict[str, Any]], str],
    system_trunc_limit: int = 50_000,
) -> List[Dict[str, Any]]:
    """
    Shared extraction logic for Langfuse traces.

    Filters GENERATION observations, extracts fields via *name_func* and
    *predicate*, and returns raw LLM-call records (no counters / row
    splitting — that is handled by :func:`build_csv_rows`).
    """
    intent, trace_id, session_id, observations, traj_score = _extract_common_from_trace_root(json_data, file_name)
    gens = _sort_and_filter_generations(observations, predicate)

    llm_calls: List[Dict[str, Any]] = []
    for obs in gens:
        agent_name = name_func(obs)
        llm_calls.append(_extract_llm_call_record(
            trace_id=trace_id,
            session_id=session_id or "",
            intent=intent,
            obs=obs,
            agent_name=agent_name,
            system_trunc_limit=system_trunc_limit,
            traj_score=traj_score,
        ))
    return llm_calls


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
    llm_calls = _extract_llm_call_records(
        json_data, file_name or "", _langgraph_filter, _langgraph_agent_name,
        system_trunc_limit=system_trunc_limit,
    )
    return build_csv_rows(llm_calls, separate_tools=separate_tools)


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
    llm_calls = _extract_llm_call_records(
        json_data, file_name or "", _crewai_filter, _crewai_agent_name,
        system_trunc_limit=system_trunc_limit,
    )
    return build_csv_rows(llm_calls, separate_tools=separate_tools)
