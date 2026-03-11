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


def normalize_messages(messages: Any, system_trunc_limit: int = 50_000) -> str:
    """
    Convert messages/contents list into a flat text string.
    Handles OpenAI (role/content), Anthropic, and Gemini (role/parts) formats.
    Preserves original order; truncates long system messages in the middle.
    """
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return json.dumps(messages, ensure_ascii=False) if isinstance(messages, dict) else str(messages)

    lines = []
    system_chars_used = 0

    for msg in messages:
        if msg is None:
            continue
        if isinstance(msg, str):
            lines.append(f"unknown: {msg}")
            continue
        if not isinstance(msg, dict):
            lines.append(f"unknown: {str(msg)}")
            continue

        role = (msg.get("role") or msg.get("type") or "unknown").lower()
        # Handle OpenAI (content), Gemini (parts), and raw content
        content = normalize_content(msg.get("content") or msg.get("parts") or "")
        if not content:
            continue

        if role == "system":
            remaining = max(0, system_trunc_limit - system_chars_used)
            if len(content) > remaining:
                content = truncate_middle(content, remaining)
            system_chars_used += len(content)

        lines.append(f"{role}: {content}")

    return "\n\n".join(lines)


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
