"""
Shared utilities for trace preprocessing.

Common functions for normalizing LLM inputs/outputs across different
observability platforms (Langfuse, MLflow) and providers (OpenAI, Gemini, Anthropic).
"""

import json
from typing import Any, Dict, List


def truncate_middle(text: str, max_len: int) -> str:
    """Truncate text in the middle, keeping start and end visible."""
    if len(text) <= max_len:
        return text
    if max_len < 25:
        return text[:max_len] + "..."
    keep_each = (max_len - 17) // 2
    return f"{text[:keep_each]}[...truncated...]{text[-keep_each:]}"


def normalize_content(content: Any, include_function_calls: bool = True) -> str:
    """
    Normalize content which could be string, list of blocks, or other.
    Handles OpenAI, Anthropic, and Gemini formats.

    Args:
        content: The content to normalize
        include_function_calls: If False, skip functionCall/functionResponse parts
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")

                # Text blocks: OpenAI/Anthropic {"text": "..."} or {"content": "..."}
                text = block.get("text") or block.get("content") or block.get("input_text")
                if text and block_type not in ("tool_use", "tool_result"):
                    parts.append(str(text))

                # Anthropic tool_use: {"type": "tool_use", "name": "...", "input": {...}}
                elif block_type == "tool_use":
                    if include_function_calls:
                        parts.append(f"[Tool Call: {block.get('name', '')}({json.dumps(block.get('input', {}), ensure_ascii=False)})]")

                # Anthropic tool_result: {"type": "tool_result", "content": "..."}
                elif block_type == "tool_result":
                    if include_function_calls:
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = normalize_content(result_content, include_function_calls)
                        parts.append(f"[Tool Result: {result_content}]")

                # Gemini function call
                elif "functionCall" in block:
                    if include_function_calls:
                        fc = block["functionCall"]
                        parts.append(f"[Tool Call: {fc.get('name', '')}({json.dumps(fc.get('args', {}), ensure_ascii=False)})]")

                # Gemini function response
                elif "functionResponse" in block:
                    if include_function_calls:
                        fr = block["functionResponse"]
                        parts.append(f"[Tool Result: {fr.get('name', '')} -> {json.dumps(fr.get('response', {}), ensure_ascii=False)}]")

                # Gemini inline data (images, etc.)
                elif "inlineData" in block:
                    parts.append(f"[Inline Data: {block['inlineData'].get('mimeType', 'unknown')}]")

                # Fallback: unknown dict block — serialize rather than silently drop
                elif not text:
                    parts.append(json.dumps(block, ensure_ascii=False))

            elif block is not None:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)


def _extract_tool_calls_from_content(msg: dict) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a single message/content dict, in full OpenAI format.
    Returns list of {"id": str, "type": "function", "function": {"name": str, "arguments": str}}.

    Handles:
    - OpenAI: msg["tool_calls"] -> kept in OpenAI format
    - Anthropic: content blocks with type=="tool_use" -> converted to OpenAI format
    - Gemini: parts with "functionCall" -> converted to OpenAI format
    """
    tool_calls = []

    # OpenAI: tool_calls field on the message
    tc_list = msg.get("tool_calls")
    if isinstance(tc_list, list):
        for tc in tc_list:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                if not isinstance(func, dict):
                    func = {}
                name = func.get("name", tc.get("name", ""))
                args = func.get("arguments", tc.get("arguments", "{}"))
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })

    # Anthropic / Gemini: scan content blocks
    content = msg.get("content") or msg.get("parts") or []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            # Anthropic tool_use
            if block.get("type") == "tool_use":
                args = block.get("input", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": args,
                    },
                })
            # Gemini functionCall
            elif "functionCall" in block:
                fc = block["functionCall"]
                args = fc.get("args", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append({
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": args,
                    },
                })

    return tool_calls


def extract_messages_from_input(input_data: Any) -> Any:
    """
    Extract the messages list from an LLM input, including any separate system prompt.

    Handles:
    - List input: returned as-is (already a messages array)
    - Dict with "messages" (OpenAI/Anthropic) or "contents" (Gemini): extracts them
      and prepends the system prompt from "system" (Anthropic) or
      "system_instruction" (Gemini) if present
    - String / other: returned as-is

    Returns:
        The messages list (possibly with a system message prepended), or the
        original value if it's not a dict/list.
    """
    if not isinstance(input_data, dict):
        return input_data

    messages = input_data.get("messages") or input_data.get("contents")
    if messages is None:
        return None

    # Anthropic: "system" (str or list of blocks)
    # Gemini: "system_instruction" (dict with "parts")
    system = input_data.get("system") or input_data.get("system_instruction")
    if system:
        system_text = normalize_content(system, include_function_calls=False)
        if system_text:
            messages = [{"role": "system", "content": system_text}] + list(messages)

    return messages


def normalize_input_messages(messages: Any, system_trunc_limit: int = 50_000) -> Any:
    """
    Normalize input messages (what goes INTO the LLM).

    Behavior:
    - If string: return as-is
    - If list/dict of messages: transform to structured list

    Returns:
        str (if input was string) or List[{"role": str, "content": str, "tool_calls": list (full OpenAI format)}]
    """
    if messages is None:
        return []

    if isinstance(messages, str):
        return messages

    if isinstance(messages, dict):
        messages = [messages]

    if not isinstance(messages, list):
        return str(messages)

    result = []
    system_chars_used = 0

    for msg in messages:
        if msg is None:
            continue
        if isinstance(msg, str):
            result.append({"role": "unknown", "content": msg, "tool_calls": []})
            continue
        if not isinstance(msg, dict):
            result.append({"role": "unknown", "content": str(msg), "tool_calls": []})
            continue

        role = (msg.get("role") or msg.get("type") or "unknown").lower()
        if role == "model":
            role = "assistant"
        raw_content = msg.get("content") or msg.get("parts") or ""

        # Normalize content to string (exclude tool calls from text to avoid duplication)
        content = normalize_content(raw_content, include_function_calls=False)

        # Extract tool calls from the message
        msg_tool_calls = _extract_tool_calls_from_content(msg)

        # Truncate system messages
        if role == "system" and content:
            remaining = max(0, system_trunc_limit - system_chars_used)
            if len(content) > remaining:
                content = truncate_middle(content, remaining)
            system_chars_used += len(content)

        # Keep messages that have content OR tool calls (don't drop tool-only messages)
        if not content and not msg_tool_calls:
            continue

        result.append({"role": role, "content": content, "tool_calls": msg_tool_calls})

    return result


def extract_from_output_messages(messages: Any) -> tuple:
    """
    Extract response text and tool calls from message-format output.

    Used for fallback when output is stored as gen_ai.output.messages.

    Args:
        messages: Output in message format [{"role": "assistant", "content": "...", "tool_calls": [...]}]

    Returns:
        (response_text: str, tool_calls: list in full OpenAI format)
    """
    if messages is None:
        return "", []
    if isinstance(messages, str):
        return messages, []
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return str(messages), []

    text_parts = []
    tool_calls = []

    for msg in messages:
        if isinstance(msg, str):
            text_parts.append(msg)
        elif isinstance(msg, dict):
            content = msg.get("content") or msg.get("parts") or ""
            if isinstance(content, str) and content:
                text_parts.append(content)
            elif content:
                text_parts.append(normalize_content(content, include_function_calls=False))

            # Extract tool calls via shared helper (handles OpenAI, Anthropic, Gemini)
            msg_tool_calls = _extract_tool_calls_from_content(msg)
            tool_calls.extend(msg_tool_calls)

    return "\n".join(text_parts), tool_calls


def normalize_response(output_data: Any) -> str:
    """Extract and normalize response text from output. Handles multiple provider formats."""
    if output_data is None:
        return ""
    if isinstance(output_data, str):
        return output_data

    if isinstance(output_data, dict):
        # OpenAI-style choices format
        if "choices" in output_data and isinstance(output_data["choices"], list):
            parts = []
            for ch in output_data["choices"]:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message", {})
                if isinstance(msg, dict):
                    if msg.get("parsed") is not None:
                        parts.append(json.dumps(msg["parsed"], ensure_ascii=False))
                    else:
                        parts.append(normalize_content(msg.get("content", ""), include_function_calls=False))
            return "\n---\n".join(p for p in parts if p)

        # Gemini-style candidates format
        elif "candidates" in output_data and isinstance(output_data["candidates"], list):
            parts = []
            for candidate in output_data["candidates"]:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content", {})
                if isinstance(content, dict):
                    parts.append(normalize_content(content.get("parts", []), include_function_calls=False))
            return "\n---\n".join(p for p in parts if p)

        # Direct content/parts format (Anthropic or unwrapped)
        elif "content" in output_data:
            return normalize_content(output_data["content"], include_function_calls=False)
        elif "parts" in output_data:
            return normalize_content(output_data["parts"], include_function_calls=False)

    return json.dumps(output_data, ensure_ascii=False) if isinstance(output_data, dict) else str(output_data)


def extract_tool_calls(output_data: Any) -> List[Dict[str, Any]]:
    """
    Extract tool calls from output data. Handles OpenAI, Anthropic, and Gemini formats.

    Returns list in full OpenAI format:
    [{"id": str, "type": "function", "function": {"name": str, "arguments": str}}, ...]
    """
    tool_calls = []

    if not isinstance(output_data, dict):
        return tool_calls

    # OpenAI choices format
    if "choices" in output_data and isinstance(output_data["choices"], list):
        for ch in output_data["choices"]:
            if isinstance(ch, dict):
                msg = ch.get("message", {})
                if isinstance(msg, dict):
                    tool_calls.extend(_extract_tool_calls_from_content(msg))

    # Gemini candidates format
    elif "candidates" in output_data and isinstance(output_data["candidates"], list):
        for candidate in output_data["candidates"]:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            if isinstance(content, dict):
                tool_calls.extend(_extract_tool_calls_from_content(content))

    # Anthropic direct output: {"content": [{"type": "tool_use", ...}, ...], "role": "assistant"}
    elif "content" in output_data and isinstance(output_data.get("content"), list):
        tool_calls.extend(_extract_tool_calls_from_content(output_data))

    # Direct tool_calls field (e.g. Langfuse-normalized output)
    elif "tool_calls" in output_data:
        tool_calls.extend(_extract_tool_calls_from_content(output_data))

    return tool_calls


def _normalize_tool_def(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a tool definition to full OpenAI format:
    {"type": "function", "function": {"name", "description", "parameters"}}.

    Handles:
    - OpenAI: already {"name", "description", "parameters"}
    - Anthropic: {"name", "description", "input_schema"} -> map input_schema to parameters
    - Gemini: {"name", "description", "parameters"}
    """
    return {
        "type": "function",
        "function": {
            "name": tool_def.get("name", ""),
            "description": tool_def.get("description", ""),
            "parameters": tool_def.get("parameters") or tool_def.get("input_schema") or {},
        },
    }


def extract_api_spec(input_data: Any) -> List[Dict[str, Any]]:
    """
    Extract bound tool definitions (API spec) from input data.
    These are the tools available to the LLM, not the tool calls it made.

    Returns list in full OpenAI format:
    [{"type": "function", "function": {"name": str, "description": str, "parameters": dict}}, ...]

    Handles:
    - OpenAI: input.tools or input.functions
    - Gemini: input.tools with functionDeclarations
    - Anthropic: input.tools (with input_schema -> parameters)
    """
    tools = []

    if not isinstance(input_data, dict):
        return tools

    # OpenAI/Anthropic style: tools array
    if "tools" in input_data:
        tools_list = input_data["tools"]
        if isinstance(tools_list, list):
            for tool in tools_list:
                if isinstance(tool, dict):
                    # OpenAI format: {"type": "function", "function": {...}}
                    if "function" in tool:
                        tools.append(_normalize_tool_def(tool["function"]))
                    # Gemini format: {"functionDeclarations": [...]}
                    elif "functionDeclarations" in tool:
                        decls = tool["functionDeclarations"]
                        if isinstance(decls, list):
                            for decl in decls:
                                if isinstance(decl, dict):
                                    tools.append(_normalize_tool_def(decl))
                    # Direct tool definition (Anthropic or other)
                    elif "name" in tool:
                        tools.append(_normalize_tool_def(tool))

    # Legacy OpenAI: functions array
    if "functions" in input_data:
        funcs = input_data["functions"]
        if isinstance(funcs, list):
            for func in funcs:
                if isinstance(func, dict):
                    tools.append(_normalize_tool_def(func))

    return tools


def safe_json(value: Any) -> Any:
    """Return parsed JSON if 'value' is a JSON string; otherwise return value."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


# ----- intent extraction (shared by MLflow & Langfuse processors) -----

_INTENT_LIMIT = 500  # max chars for extracted intent


def first_user_message(messages: Any) -> str:
    """
    Return the content of the first user/human message in a message list.

    Supports:
      - list of dicts with role/content keys (OpenAI / Anthropic / Gemini)
      - JSON-encoded string of the above
      - LangGraph tuple format: [["human", "text"], ...]
    """
    if messages is None:
        return ""

    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (json.JSONDecodeError, TypeError):
            return messages.strip()

    if not isinstance(messages, list) or not messages:
        return ""

    # Standard dict messages: [{role: "user", content: "..."}, ...]
    for msg in messages:
        if isinstance(msg, dict):
            role = (msg.get("role") or msg.get("type") or "").lower()
            if role in ("user", "human"):
                content = msg.get("content") or msg.get("text") or ""
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                if isinstance(content, str) and content.strip():
                    return content.strip()

    # LangGraph tuple format: [["human", "hello"], ...]
    for msg in messages:
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            role = str(msg[0]).lower()
            if role in ("user", "human", "humanmessage"):
                content = str(msg[1]).strip()
                if content:
                    return content

    return ""


def extract_intent_from_input(input_data: Any, attributes: Dict[str, Any] | None = None) -> str:
    """
    Try to extract the user's query / intent from a span/observation's input.

    Handles common patterns across MLflow and Langfuse:
      - Plain string input (e.g. CrewAI kickoff, simple chains)
      - Dict with scalar fields: input, query, question, task, goal, etc.
      - OpenAI-style messages with role=user
      - Gemini-style contents with role=user
      - LangGraph tuple format [["human", "text"], ...]
      - gen_ai.input.messages / mlflow.chat.messages in attributes
    """
    attributes = attributes or {}

    # Plain string input
    if isinstance(input_data, str) and input_data.strip():
        return truncate_middle(input_data.strip(), _INTENT_LIMIT)

    if isinstance(input_data, dict):
        # Direct scalar fields
        for key in ("input", "query", "question", "task", "goal",
                     "description", "user_input", "human_input", "prompt"):
            val = input_data.get(key)
            if isinstance(val, str) and val.strip():
                return truncate_middle(val.strip(), _INTENT_LIMIT)

        # OpenAI / Gemini style message lists
        messages = input_data.get("messages") or input_data.get("contents")
        intent = first_user_message(messages)
        if intent:
            return truncate_middle(intent, _INTENT_LIMIT)

    # Attributes: gen_ai.input.messages / mlflow.chat.messages
    for attr_key in ("gen_ai.input.messages", "mlflow.chat.messages",
                     "gen_ai.prompt"):
        raw = attributes.get(attr_key)
        if raw is not None:
            intent = first_user_message(raw)
            if intent:
                return truncate_middle(intent, _INTENT_LIMIT)

    return ""


# ----- CSV row construction (shared by MLflow & Langfuse processors) -----


def build_csv_rows(
    llm_calls: List[Dict[str, Any]],
    separate_tools: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build final CSV rows from a list of extracted LLM call records.

    Each record represents a single LLM invocation and contains fields
    extracted by a framework-specific processor (MLflow / Langfuse).  This
    function handles row splitting (tool vs agent), counter assignment, and
    ID generation — logic that is identical across observability backends.

    Args:
        llm_calls: One dict per LLM call with keys:
            - agent_name (str)
            - model_input (str, already serialized)
            - response_text (str)
            - tool_calls (list in full OpenAI format: {"id", "type", "function": {"name", "arguments"}})
            - api_spec (list of tool definitions)
            - meta_data (dict, not yet JSON-serialized)
            - intent (str)
            - task_id (str)
            - traj_score (Any)
        separate_tools: When True, emit one row per tool call plus an
            optional row for the text response.  When False, emit a single
            combined row per LLM call.

    Returns:
        List of row dicts matching the unified CSV schema.
    """
    rows: List[Dict[str, Any]] = []
    step_counter = 0
    llm_call_counter = 0

    for call in llm_calls:
        agent_name = call["agent_name"]
        task_id = call["task_id"]
        intent = call["intent"]
        model_input = call["model_input"]
        response_text = call["response_text"]
        tool_calls = call.get("tool_calls") or []
        api_spec = call.get("api_spec") or []
        meta_data = call.get("meta_data") or {}
        traj_score = call.get("traj_score")

        llm_call_counter += 1

        api_spec_str = json.dumps(api_spec) if api_spec else ""
        meta_data_str = json.dumps(meta_data)

        # Shared fields for every row produced from this LLM call
        base = {
            "Name": agent_name,
            "intent": intent,
            "task_id": task_id,
            "llm_call_index": llm_call_counter,
            "model_input": model_input,
            "api_spec": api_spec_str,
            "meta_data": meta_data_str,
            "traj_score": traj_score,
        }

        if separate_tools:
            # One row per tool call
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    step_counter += 1
                    rows.append({
                        **base,
                        "id": f"{task_id}_{step_counter}",
                        "step_in_trace_general": step_counter,
                        "response": json.dumps(tc, indent=2),
                        "tool_or_agent": "tool",
                    })

            # Row for text response (if non-empty)
            if response_text and response_text.strip() not in ("null", "None", ""):
                step_counter += 1
                rows.append({
                    **base,
                    "id": f"{task_id}_{step_counter}",
                    "step_in_trace_general": step_counter,
                    "response": response_text,
                    "tool_or_agent": "agent",
                })
        else:
            # Single combined row
            step_counter += 1
            if tool_calls:
                combined_response = json.dumps({
                    "content": response_text or "",
                    "tool_calls": tool_calls,
                })
            else:
                combined_response = response_text

            rows.append({
                **base,
                "id": f"{task_id}_{step_counter}",
                "step_in_trace_general": step_counter,
                "response": combined_response,
                "tool_or_agent": "agent",
            })

    return rows
