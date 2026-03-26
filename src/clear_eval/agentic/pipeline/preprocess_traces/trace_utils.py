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
                # OpenAI/Anthropic: {"text": "..."} or {"content": "..."}
                text = block.get("text") or block.get("content") or block.get("input_text")
                if text:
                    parts.append(str(text))
                # Gemini function call
                elif "functionCall" in block:
                    if include_function_calls:
                        fc = block["functionCall"]
                        parts.append(f"[Function: {fc.get('name', '')}({json.dumps(fc.get('args', {}))})]")
                # Gemini function response
                elif "functionResponse" in block:
                    if include_function_calls:
                        fr = block["functionResponse"]
                        parts.append(f"[Function Response: {fr.get('name', '')} -> {json.dumps(fr.get('response', {}))}]")
                # Gemini inline data (images, etc.)
                elif "inlineData" in block:
                    parts.append(f"[Inline Data: {block['inlineData'].get('mimeType', 'unknown')}]")
            elif block is not None:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)


def normalize_input_messages(messages: Any, system_trunc_limit: int = 50_000) -> Any:
    """
    Normalize input messages (what goes INTO the LLM).

    Behavior:
    - If string: return as-is
    - If list/dict of messages: transform to structured list

    Returns:
        str (if input was string) or List[{"role": str, "content": str, "is_tool_def": bool}]
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
            result.append({"role": "unknown", "content": msg, "is_tool_def": False})
            continue
        if not isinstance(msg, dict):
            result.append({"role": "unknown", "content": str(msg), "is_tool_def": False})
            continue

        role = (msg.get("role") or msg.get("type") or "unknown").lower()
        raw_content = msg.get("content") or msg.get("parts") or ""

        # Normalize content to string
        content = normalize_content(raw_content)

        # Truncate system messages
        if role == "system" and content:
            remaining = max(0, system_trunc_limit - system_chars_used)
            if len(content) > remaining:
                content = truncate_middle(content, remaining)
            system_chars_used += len(content)

        # Detect tool definitions
        is_tool_def = False
        if role == "tool" and isinstance(raw_content, (dict, str)):
            content_to_check = raw_content
            if isinstance(content_to_check, str):
                try:
                    content_to_check = json.loads(content_to_check)
                except:
                    pass
            if isinstance(content_to_check, dict) and "function" in content_to_check:
                is_tool_def = True

        if not content:
            continue

        result.append({"role": role, "content": content, "is_tool_def": is_tool_def})

    return result


def extract_from_output_messages(messages: Any) -> tuple:
    """
    Extract response text and tool calls from message-format output.

    Used for fallback when output is stored as gen_ai.output.messages.

    Args:
        messages: Output in message format [{"role": "assistant", "content": "...", "tool_calls": [...]}]

    Returns:
        (response_text: str, tool_calls: list)
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
                text_parts.append(normalize_content(content))

            tc = msg.get("tool_calls")
            if tc and isinstance(tc, list):
                tool_calls.extend(tc)

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
        if "candidates" in output_data and isinstance(output_data["candidates"], list):
            parts = []
            for candidate in output_data["candidates"]:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content", {})
                if isinstance(content, dict):
                    parts.append(normalize_content(content.get("parts", []), include_function_calls=False))
            return "\n---\n".join(p for p in parts if p)

        # Direct content/parts format
        if "content" in output_data:
            return normalize_content(output_data["content"], include_function_calls=False)
        if "parts" in output_data:
            return normalize_content(output_data["parts"], include_function_calls=False)

    return json.dumps(output_data, ensure_ascii=False) if isinstance(output_data, dict) else str(output_data)


def extract_tool_calls(output_data: Any) -> List[Dict[str, Any]]:
    """Extract tool calls from output data. Handles OpenAI and Gemini formats."""
    tool_calls = []

    if not isinstance(output_data, dict):
        return tool_calls

    # Direct tool_calls field (common in Langfuse)
    if "tool_calls" in output_data:
        tc_list = output_data["tool_calls"]
        if isinstance(tc_list, list):
            tool_calls.extend(tc_list)

    # OpenAI choices format
    if "choices" in output_data and isinstance(output_data["choices"], list):
        for ch in output_data["choices"]:
            if isinstance(ch, dict):
                msg = ch.get("message", {})
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tc_list = msg.get("tool_calls", [])
                    if tc_list:
                        tool_calls.extend(tc_list)

    # Gemini candidates format
    if "candidates" in output_data and isinstance(output_data["candidates"], list):
        for candidate in output_data["candidates"]:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            if isinstance(content, dict):
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append({
                            "name": fc.get("name", ""),
                            "args": fc.get("args", {})
                        })

    return tool_calls


def extract_api_spec(input_data: Any) -> List[Dict[str, Any]]:
    """
    Extract bound tool definitions (API spec) from input data.
    These are the tools available to the LLM, not the tool calls it made.

    Handles:
    - OpenAI: input.tools or input.functions
    - Gemini: input.tools with functionDeclarations
    - Anthropic: input.tools
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
                        tools.append(tool["function"])
                    # Gemini format: {"functionDeclarations": [...]}
                    elif "functionDeclarations" in tool:
                        decls = tool["functionDeclarations"]
                        if isinstance(decls, list):
                            tools.extend(decls)
                    # Direct tool definition
                    elif "name" in tool:
                        tools.append(tool)

    # Legacy OpenAI: functions array
    if "functions" in input_data:
        funcs = input_data["functions"]
        if isinstance(funcs, list):
            tools.extend(funcs)

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


def extract_intent_from_input(input_data: Any, attributes: Dict[str, Any] = None) -> str:
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
