"""
Create compact trace representations for LLM judges.

This module converts the unified CSV trace representation into a compact
format suitable for LLM-as-judge evaluation, significantly reducing token usage
while preserving the information needed for quality assessment.

INPUT: Unified CSV format (output of preprocessing step)
  Columns: id, Name, intent, task_id, step_in_trace_general, step_in_trace_node,
           model_input, response, tool_or_agent, api_spec, meta_data, traj_score

The CSV format is provider-agnostic (works with Langfuse, MLflow traces)
and framework-agnostic (works with LangGraph, CrewAI, etc.)

Key optimizations:
- model_input: Extract conversation context, strip tool definitions
- api_spec: Show tool names per step (not full schemas)
- response: Include full response (optionally truncated)
- meta_data: Extract key metrics (model, tokens, latency)
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd


# =============================================================================
# Parsing utilities for model_input field
# =============================================================================

def parse_model_input(model_input: str) -> List[Dict[str, Any]]:
    """
    Parse the model_input field from CSV into structured messages.

    The model_input can be:
    1. JSON-encoded list: [{"role": "...", "content": "...", "tool_calls": [...]}, ...]
    2. Plain string (legacy or raw text)

    Returns:
        List of {"role": str, "content": str, "tool_calls": list}
    """
    if not model_input or not isinstance(model_input, str):
        return []

    model_input = model_input.strip()

    # Try to parse as JSON first (new format)
    if model_input.startswith('['):
        try:
            parsed = json.loads(model_input)
            if isinstance(parsed, list):
                # Validate and normalize the structure
                result = []
                for msg in parsed:
                    if isinstance(msg, dict):
                        result.append({
                            "role": msg.get("role", "unknown"),
                            "content": msg.get("content", ""),
                            "tool_calls": msg.get("tool_calls", []),
                        })
                    elif isinstance(msg, str):
                        result.append({"role": "unknown", "content": msg, "tool_calls": []})
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: treat as plain text (legacy format or raw string)
    return [{"role": "unknown", "content": model_input, "tool_calls": []}]


def extract_input_context(
    model_input: str,
    max_system_len: int = 5000,
    max_total_len: int = 50000,
) -> str:
    """
    Extract relevant context from model_input for compact representation.

    Keeps:
    - User/human messages (the actual queries/inputs)
    - Assistant/ai messages (conversation history)
    - Tool results and tool calls

    Optionally strips or truncates:
    - System prompts (heavily truncated)

    Args:
        model_input: Raw model_input string from CSV
        max_system_len: Max chars to keep from system prompts
        max_total_len: Max total chars for the extracted context

    Returns:
        Compact input context string
    """
    messages = parse_model_input(model_input)

    if not messages:
        return ""

    output_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Truncate system prompts
        if role == "system":
            content = truncate_text(content, max_system_len, strategy="middle")

        # Format output
        output_parts.append(f"{role}: {content}")

    result = "\n\n".join(output_parts)

    # Final truncation if needed
    result = truncate_text(result, max_total_len, strategy="middle")

    return result


# =============================================================================
# Parsing utilities for response field
# =============================================================================

def parse_response(response: str) -> List[Dict[str, Any]]:
    """
    Parse the response field from CSV into structured messages.

    The response can be:
    1. JSON-encoded list: [{"role": "assistant", "content": "...", "tool_calls": [...]}]
    2. Plain string (text response)
    3. JSON object (tool call, etc.)

    Returns:
        List of {"role": str, "content": str, "tool_calls": list|None}
    """
    if not response or not isinstance(response, str):
        return []

    response = response.strip()

    # Try to parse as JSON list (new structured format)
    if response.startswith('['):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                result = []
                for msg in parsed:
                    if isinstance(msg, dict):
                        result.append({
                            "role": msg.get("role", "assistant"),
                            "content": msg.get("content", ""),
                            "tool_calls": msg.get("tool_calls")
                        })
                    elif isinstance(msg, str):
                        result.append({"role": "assistant", "content": msg, "tool_calls": None})
                return result
        except json.JSONDecodeError:
            pass

    # Try to parse as JSON object (single tool call or structured response)
    if response.startswith('{'):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                # Could be a tool call or single message
                if "role" in parsed:
                    return [{
                        "role": parsed.get("role", "assistant"),
                        "content": parsed.get("content", ""),
                        "tool_calls": parsed.get("tool_calls")
                    }]
                else:
                    # Treat as tool call or other structured response
                    return [{"role": "assistant", "content": "", "tool_calls": [parsed]}]
        except json.JSONDecodeError:
            pass

    # Fallback: plain text response
    return [{"role": "assistant", "content": response, "tool_calls": None}]


def format_response_compact(response: str, max_len: int = 1000) -> str:
    """
    Format response for compact representation.

    Handles both structured (JSON) and plain text responses.
    """
    messages = parse_response(response)

    if not messages:
        return ""

    parts = []
    for msg in messages:
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if content:
            parts.append(content)

        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    # Format tool call compactly
                    func = tc.get("function", tc)
                    name = func.get("name", tc.get("name", "unknown"))
                    args = func.get("arguments", tc.get("args", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    parts.append(f"[Tool Call: {name}({args_str})]")

    result = "\n".join(parts)

    # Truncate if needed
    result = truncate_text(result, max_len, strategy="middle")

    return result


# =============================================================================
# Parsing utilities for other CSV fields
# =============================================================================

def extract_tool_names(api_spec: str) -> List[str]:
    """
    Extract tool names from api_spec JSON.

    api_spec contains the full tool definitions, but we only need names
    for the compact representation.
    """
    if not api_spec or not isinstance(api_spec, str) or api_spec == "":
        return []

    try:
        tools = json.loads(api_spec)
        if isinstance(tools, list):
            return [t.get("name", "unknown") for t in tools if isinstance(t, dict) and t.get("name")]
    except:
        pass

    return []


def parse_metadata(meta_data: str) -> Dict[str, Any]:
    """Parse meta_data JSON string into dict."""
    if not meta_data or not isinstance(meta_data, str):
        return {}

    try:
        return json.loads(meta_data)
    except:
        return {}


def truncate_text(text: str, max_len: int, strategy: str = "middle") -> str:
    """
    Truncate text to max_len.

    Args:
        text: Text to truncate
        max_len: Maximum length
        strategy: "end" (keep start), "middle" (keep start+end), "start" (keep end)
    """
    if not text or len(text) <= max_len:
        return text or ""

    if strategy == "end":
        return text[:max_len] + "..."
    elif strategy == "start":
        return "..." + text[-max_len:]
    else:  # middle
        keep_each = (max_len - 15) // 2
        return f"{text[:keep_each]}...[truncated]...{text[-keep_each:]}"


# =============================================================================
# Main compact formatting function
# =============================================================================

def format_compact_trace(
    df: pd.DataFrame,
    max_input_context: int = 10000,
    max_response_len: int = 10000,
    max_system_prompt: int = 10000,
    include_tools_per_step: bool = True,
    include_input_context: bool = True,
    include_metadata: bool = True,
) -> str:
    """
    Convert a trace DataFrame to compact text format for LLM judge.

    Args:
        df: DataFrame with unified CSV schema
        max_input_context: Max chars for extracted input context per step
        max_response_len: Max chars for response per step
        max_system_prompt: Max chars to keep from system prompts
        include_tools_per_step: Show available tools at each step
        include_input_context: Include extracted input context
        include_metadata: Include model/tokens/latency info

    Returns:
        Compact text representation of the trace
    """
    if df.empty:
        return "Empty trace"

    # Get trace-level info from first row
    first_row = df.iloc[0]
    task_id = first_row.get("task_id", "unknown")
    intent = first_row.get("intent", "")
    traj_score = first_row.get("traj_score")
    if pd.isna(traj_score):
        traj_score = "N/A"

    lines = []
    lines.append(f"## Trace: {task_id}")
    lines.append(f"**User Query:** {intent}")
    lines.append(f"**Total Steps:** {len(df)} | **Trajectory Score:** {traj_score}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Process each step
    for idx, row in df.iterrows():
        step_num = row.get("step_in_trace_general", idx + 1)
        agent_name = row.get("Name", "unknown")
        action_type = row.get("tool_or_agent", "agent")

        # Header line
        header_parts = [f"### Step {step_num}", f"Agent: {agent_name}", f"Type: {action_type}"]

        # Add metadata if requested
        if include_metadata:
            meta = parse_metadata(row.get("meta_data", ""))
            model = meta.get("model", "")
            tokens = meta.get("tokens", {})
            total_tokens = tokens.get("total", 0) if isinstance(tokens, dict) else 0
            latency = meta.get("latency", 0)

            if model:
                header_parts.append(f"Model: {model}")
            if total_tokens:
                header_parts.append(f"Tokens: {total_tokens}")
            if latency:
                header_parts.append(f"Latency: {latency}ms")

        lines.append(" | ".join(header_parts))
        lines.append("")

        # Available tools at this step
        if include_tools_per_step:
            api_spec = row.get("api_spec", "")
            if pd.notna(api_spec) and api_spec:
                tool_names = extract_tool_names(api_spec)
                if tool_names:
                    if len(tool_names) <= 5:
                        lines.append(f"**Available Tools:** {', '.join(tool_names)}")
                    else:
                        lines.append(f"**Available Tools:** {', '.join(tool_names[:5])} (+{len(tool_names)-5} more)")
                    lines.append("")

        # Input context (extracted from model_input)
        if include_input_context:
            model_input = row.get("model_input", "")
            if pd.notna(model_input) and model_input:
                input_context = extract_input_context(
                    str(model_input),
                    max_system_len=max_system_prompt,
                    max_total_len=max_input_context,
                )
                if input_context:
                    lines.append("**Input Context:**")
                    lines.append("```")
                    lines.append(input_context)
                    lines.append("```")
                    lines.append("")

        # Response
        response = row.get("response", "")
        if pd.notna(response) and response:
            formatted_response = format_response_compact(str(response), max_len=max_response_len)

            if formatted_response:
                lines.append("**Response:**")
                lines.append("```")
                lines.append(formatted_response)
                lines.append("```")
                lines.append("")

        lines.append("---")
        lines.append("")

    res = "\n".join(lines)
    print(len(res)/4)
    return res


def format_compact_trace_from_csv(
    csv_path: str,
    **kwargs
) -> str:
    """
    Load CSV and format as compact trace.

    Args:
        csv_path: Path to the trace CSV file
        **kwargs: Passed to format_compact_trace()

    Returns:
        Compact text representation
    """
    df = pd.read_csv(csv_path)
    return format_compact_trace(df, **kwargs)


# =============================================================================
# Batch processing
# =============================================================================

def process_trace_directory(
    input_dir: str,
    output_dir: str,
    **kwargs
) -> Dict[str, str]:
    """
    Process all trace CSVs in a directory.

    Args:
        input_dir: Directory containing trace CSV files
        output_dir: Directory to save compact representations
        **kwargs: Passed to format_compact_trace()

    Returns:
        Dict mapping trace_id to output file path
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for csv_file in input_path.glob("*.csv"):
        trace_id = csv_file.stem

        try:
            compact = format_compact_trace_from_csv(str(csv_file), **kwargs)

            output_file = output_path / f"{trace_id}_compact.txt"
            with open(output_file, 'w') as f:
                f.write(compact)

            results[trace_id] = str(output_file)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compact_trace_formatter.py <trace.csv> [output.txt]")
        print("\nConverts unified CSV trace to compact text format for LLM judges.")
        sys.exit(1)

    csv_path = sys.argv[1]
    compact = format_compact_trace_from_csv(csv_path)

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        with open(output_path, 'w') as f:
            f.write(compact)
        print(f"Saved to {output_path}")
    else:
        print(compact)
