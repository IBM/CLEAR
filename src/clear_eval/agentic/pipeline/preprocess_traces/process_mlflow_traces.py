"""
Extract LLM calls from MLflow/OTel JSON traces.

Focus: LangGraph-style traces (agent/chain/node functions), using parent/child
relationships to reconstruct order and ownership.

Supports input files that contain:
  - a single trace dict
  - a list of trace dicts
  - a wrapper {"traces": [...]}

Output fields match the unified CSV schema:
  id, Name, intent, task_id, step_in_trace_general, step_in_trace_node,
  model_input, response, tool_or_agent, api_spec, meta_data, traj_score
"""

import json
from typing import Any, Dict, List, Optional

from .trace_utils import (
    normalize_input_messages,
    normalize_response,
    extract_tool_calls,
    extract_from_output_messages,
    extract_api_spec,
)


# ----------------- helpers -----------------

def _get(d: Dict[str, Any], path: List[Any], default=None):
    """Safely navigate nested dict by path."""
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def get_by_any_key(s: Dict[str, Any], keys: list[str]):
    for k in keys:
        if k in s:
            return s[k]
    return None

def get_parent_id(s):
     return get_by_any_key(s, ["parent_id", "parent_span_id"])

def get_start_time(s):
    return get_by_any_key(s, ["start_time_unix_nano", "start_time_ns"])

def get_end_time(s):
    return get_by_any_key(s, ["end_time_unix_nano", "end_time_ns"])

# ----------------- span semantics -----------------

def _span_type(s: Dict[str, Any]) -> Optional[str]:
    return _get(s, ["attributes", "mlflow.spanType"]) or _get(s, ["attributes", "span.type"])


_WRAPPER_TYPES = {"CHAIN", "AGENT", "TOOL"}
_MODEL_TYPES = {"CHAT_MODEL", "MODEL", "GENERATION"}


def _is_wrapper_span(s: Dict[str, Any]) -> bool:
    return (_span_type(s) in _WRAPPER_TYPES) or bool(_get(s, ["attributes", "langgraph.node"]))


def _is_model_call_span(s: Dict[str, Any]) -> bool:
    st = _span_type(s)
    if st in _MODEL_TYPES:
        return True
    if _get(s, ["attributes", "gen_ai.operation.name"]):
        return True
    outputs_obj = s.get("outputs")
    if isinstance(outputs_obj, dict) and ("choices" in outputs_obj or "content" in outputs_obj or "usage" in outputs_obj):
        return True
    if _get(s, ["attributes", "gen_ai.output.messages"]):
        return True
    return False


def _get_span_name(s: Dict[str, Any]) -> Optional[str]:
    return (
        _get(s, ["attributes", "langgraph.node"]) or
        _get(s, ["attributes", "mlflow.spanFunctionName"]) or
        s.get("name")
    )


# ----------------- payload extraction -----------------

def _extract_model_name(attrs: Dict[str, Any], inputs: Dict[str, Any], outputs_obj: Any) -> Optional[str]:
    return (
        attrs.get("model") or
        inputs.get("model") or
        attrs.get("gen_ai.request.model") or
        attrs.get("gen_ai.response.model") or
        (outputs_obj.get("model") if isinstance(outputs_obj, dict) else None)
    )


def _extract_input_output_from_span(
    s: Dict[str, Any],
    system_trunc_limit: int = 50_000
) -> tuple:
    """
    Extract model_input, response_text, tool_calls, api_spec, and meta_data from a span.

    Returns:
        (model_input_str, response_text, tool_calls, api_spec, meta_data)
    """
    attrs = s.get("attributes", {})
    inputs = s.get("inputs", {}) or {}
    outputs_obj = s.get("outputs", {}) or {}

    # Extract bound tools (API spec)
    api_spec = extract_api_spec(inputs)

    # Extract input messages (OpenAI: messages, Gemini: contents)
    messages = (
        inputs.get("messages") or
        inputs.get("contents") or
        _get(attrs, ["gen_ai.input.messages"]) or
        _get(attrs, ["gen_ai.prompt"]) or
        inputs.get("input")
    )
    if messages is None and isinstance(s.get("events"), list):
        for ev in s["events"]:
            if ev.get("name") == "gen_ai.client.inference.operation.details":
                ev_attrs = ev.get("attributes", {})
                messages = ev_attrs.get("gen_ai.input.messages") or ev_attrs.get("gen_ai.prompt")
                if messages is not None:
                    break

    # Normalize input and serialize
    if messages:
        model_input_normalized = normalize_input_messages(messages, system_trunc_limit)
    else:
        model_input_normalized = normalize_input_messages(inputs, system_trunc_limit) if inputs else []

    # Serialize: if already string keep as-is, otherwise JSON encode
    if isinstance(model_input_normalized, str):
        model_input_str = model_input_normalized
    else:
        model_input_str = json.dumps(model_input_normalized, ensure_ascii=False)

    # Extract response and tool calls
    response_text = ""
    tool_calls = []

    if isinstance(outputs_obj, dict) and outputs_obj:
        response_text = normalize_response(outputs_obj)
        tool_calls = extract_tool_calls(outputs_obj)

    # Fallback: check gen_ai.output.messages in attributes
    if not response_text:
        out_msgs = _get(attrs, ["gen_ai.output.messages"]) or _get(attrs, ["gen_ai.completion"])
        if out_msgs is not None:
            response_text, extra_tool_calls = extract_from_output_messages(out_msgs)
            if not tool_calls and extra_tool_calls:
                tool_calls = extra_tool_calls

    # Fallback: check events (OTel-style instrumentation)
    if not response_text and isinstance(s.get("events"), list):
        for ev in s["events"]:
            if ev.get("name") == "gen_ai.client.inference.operation.details":
                ev_attrs = ev.get("attributes", {})
                out_msgs = ev_attrs.get("gen_ai.output.messages") or ev_attrs.get("gen_ai.completion")
                if out_msgs is not None:
                    response_text, extra_tool_calls = extract_from_output_messages(out_msgs)
                    if not tool_calls and extra_tool_calls:
                        tool_calls = extra_tool_calls
                    break

    # Build metadata (model-level, span-level added separately)
    meta_data = {
        "model": _extract_model_name(attrs, inputs, outputs_obj),
        "provider": attrs.get("gen_ai.provider.name") or attrs.get("gen_ai.system") or attrs.get("provider"),
        "operation": attrs.get("gen_ai.operation.name") or attrs.get("operation"),
        "response_format": attrs.get("response_format") or inputs.get("response_format"),
        "tokens": {
            "prompt": attrs.get("gen_ai.usage.input_tokens") or _get(outputs_obj, ["usage", "prompt_tokens"]),
            "completion": attrs.get("gen_ai.usage.output_tokens") or _get(outputs_obj, ["usage", "completion_tokens"]),
            "total": attrs.get("gen_ai.usage.total_tokens") or _get(outputs_obj, ["usage", "total_tokens"]),
        },
    }

    return model_input_str, response_text, tool_calls, api_spec, meta_data


def _build_span_metadata(s: Dict[str, Any], model_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build complete metadata including span-level information.

    Args:
        s: The span dict
        model_meta: Model-level metadata from _extract_input_output_from_span

    Returns:
        Complete metadata dict with span info
    """
    attrs = s.get("attributes", {})

    # Calculate duration if timestamps available
    duration_ms = s.get("duration_ms")
    if duration_ms is None:
        start_ns = get_start_time(s)
        end_ns = get_end_time(s)
        if start_ns is not None and end_ns is not None:
            duration_ms = (end_ns - start_ns) / 1_000_000

    span_meta = {
        "span_id": s.get("span_id"),
        "span_name": s.get("name"),
        "span_type": _span_type(s),
        "parent_span_id": get_parent_id(s),
        "duration_ms": duration_ms,
        "status": s.get("status", {}).get("status_code"),
    }

    # Merge span metadata with model metadata
    return {**span_meta, **model_meta}


# ----------------- core extraction -----------------

# Names to skip when walking up parent chain for calling context
_SKIP_NAMES = {"get_llm", "Completions", "ChatCompletions", "openai_call", "llm_invoke"}


def extract_llm_calls_from_mlflow_trace(
    trace: Dict[str, Any],
    file_name: str = None,
    separate_tools: bool = True,
    system_trunc_limit: int = 50_000
) -> List[Dict[str, Any]]:
    """
    Extract LLM calls from an MLflow trace.

    Uses parent_id to:
    1. Find calling node name by walking up the parent chain
    2. Order LLM calls by start_time, with parent hierarchy as context

    Args:
        trace: The trace dict containing spans
        file_name: Fallback name for trace_id
        separate_tools: If True, emit separate rows for tool calls vs text responses
                       If False, emit single row per model call with combined response
        system_trunc_limit: Max chars for system messages before truncation

    Returns:
        List of row dicts matching the unified CSV schema
    """
    trace_id = trace.get("trace_id") or trace.get("id") or file_name or "unknown"
    spans: List[Dict[str, Any]] = trace.get("spans", [])

    if not spans:
        return []

    # Build span index
    by_id: Dict[str, Dict[str, Any]] = {}
    for i, s in enumerate(spans):
        sid = s.get("span_id") or f"_idx_{i}"
        s["span_id"] = sid
        by_id[sid] = s

    def get_parent(s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get parent span if exists."""
        pid = get_parent_id(s)
        return by_id.get(pid) if pid else None

    def find_calling_node_name(s: Dict[str, Any]) -> str:
        """
        Walk up parent chain to find the first meaningful calling node name.
        Skips generic names like 'Completions', 'get_llm', etc.
        """
        current = s
        visited = set()

        while current:
            sid = current.get("span_id")
            if sid in visited:
                break
            visited.add(sid)

            # Get candidate name
            name = _get(current, ["attributes", "mlflow.spanFunctionName"]) or current.get("name")

            # Check if this is a meaningful name (not a generic LLM wrapper)
            if name and name not in _SKIP_NAMES:
                span_type = _span_type(current)
                # Prefer LLM-type spans (these are the decorated functions)
                # or any span with a function name that's not a model call
                if span_type == "LLM" or (span_type not in _MODEL_TYPES and name):
                    return name

            current = get_parent(current)

        # Fallback to span's own name
        return _get_span_name(s) or "unknown"

    # Extract intent from trace metadata
    trace_intent = (
        _get(trace, ["tags", "intent"]) or
        _get(trace, ["metadata", "intent"]) or
        _get(trace, ["metadata", "user_query"]) or
        ""
    )
    traj_score = trace.get("traj_score") or _get(trace, ["tags", "traj_score"])

    # Find all model call spans
    model_spans = [s for s in spans if _is_model_call_span(s)]

    # Sort by start_time if available, otherwise by original order
    def sort_key(s: Dict[str, Any]) -> tuple:
        start_time = get_start_time(s)
        if start_time is not None:
            return (0, start_time)
        return (1, spans.index(s) if s in spans else 0)

    model_spans.sort(key=sort_key)

    # Process each model call span
    rows: List[Dict[str, Any]] = []
    per_node_counter: Dict[str, int] = {}
    step_counter = 0

    for s in model_spans:
        # Find calling node by walking up parent chain
        agent_name = find_calling_node_name(s)

        per_node_counter.setdefault(agent_name, 0)
        per_node_counter[agent_name] += 1

        model_input_str, response_text, tool_calls, api_spec, model_meta = _extract_input_output_from_span(
            s, system_trunc_limit
        )
        api_spec_str = json.dumps(api_spec) if api_spec else ""

        # Build complete metadata with span info
        meta_data = _build_span_metadata(s, model_meta)

        attrs = s.get("attributes", {})
        intent = (
            attrs.get("langgraph.intent") or
            attrs.get("intent") or
            attrs.get("mlflow.intent") or
            attrs.get("gen_ai.intent") or
            trace_intent
        )

        if separate_tools:
            # Emit separate rows for tool calls
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    step_counter += 1
                    rows.append({
                        "id": f"{trace_id}_{step_counter}",
                        "Name": agent_name,
                        "intent": intent,
                        "task_id": trace_id,
                        "step_in_trace_general": step_counter,
                        "step_in_trace_node": per_node_counter[agent_name],
                        "model_input": model_input_str,
                        "response": json.dumps(tc, indent=2),
                        "tool_or_agent": "tool",
                        "api_spec": api_spec_str,
                        "meta_data": json.dumps(meta_data),
                        "traj_score": traj_score,
                    })

            # Emit row for text response
            if response_text and response_text.strip() not in ("null", "None", ""):
                step_counter += 1
                rows.append({
                    "id": f"{trace_id}_{step_counter}",
                    "Name": agent_name,
                    "intent": intent,
                    "task_id": trace_id,
                    "step_in_trace_general": step_counter,
                    "step_in_trace_node": per_node_counter[agent_name],
                    "model_input": model_input_str,
                    "response": response_text,
                    "tool_or_agent": "agent",
                    "api_spec": api_spec_str,
                    "meta_data": json.dumps(meta_data),
                    "traj_score": traj_score,
                })
        else:
            # Single row mode: combine everything
            step_counter += 1
            combined_response = response_text
            if tool_calls:
                tool_parts = [json.dumps(tc, indent=2) for tc in tool_calls]
                if tool_parts:
                    combined_response = "\n---\n".join(tool_parts)
                    if response_text:
                        combined_response += f"\n---\n{response_text}"

            rows.append({
                "id": f"{trace_id}_{step_counter}",
                "Name": agent_name,
                "intent": intent,
                "task_id": trace_id,
                "step_in_trace_general": step_counter,
                "step_in_trace_node": per_node_counter[agent_name],
                "model_input": model_input_str,
                "response": combined_response,
                "tool_or_agent": "agent",
                "api_spec": api_spec_str,
                "meta_data": json.dumps(meta_data),
                "traj_score": traj_score,
            })

    return rows
