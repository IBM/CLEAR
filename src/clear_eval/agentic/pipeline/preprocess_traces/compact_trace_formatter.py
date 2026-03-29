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
    #print(f"truncate_text: {len(text)} > {max_len}")
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
NO_LIMIT = 10 ** 9
def _precompute_steps(
    df: pd.DataFrame,
    include_tools_per_step: bool,
    include_input_context: bool,
    include_metadata: bool,
    truncate_content: bool,
    max_input_context: int = 10000,
    max_response_len: int = 10000,
    max_system_prompt: int = 10000,
) -> List[Dict[str, str]]:
    """
    Build per-step data (header, tools line, input context, response).

    When *truncate_content* is False the input/response fields are extracted
    at full length so the caller can measure them before deciding how much to
    cut.  When True the fixed per-field limits are applied immediately.
    """

    steps: List[Dict[str, str]] = []

    for idx, row in df.iterrows():
        step_num = row.get("step_in_trace_general", idx + 1)
        agent_name = row.get("Name", "unknown")
        action_type = row.get("tool_or_agent", "agent")

        # --- step header ---
        header_parts = [
            f"### Step {step_num}",
            f"Agent: {agent_name}",
            f"Type: {action_type}",
        ]
        if include_metadata:
            meta = parse_metadata(row.get("meta_data", ""))
            model = meta.get("model", "")
            tokens_info = meta.get("tokens", {})
            total_tokens = (
                tokens_info.get("total", 0) if isinstance(tokens_info, dict) else 0
            )
            latency = meta.get("latency", 0)
            if model:
                header_parts.append(f"Model: {model}")
            if total_tokens:
                header_parts.append(f"Tokens: {total_tokens}")
            if latency:
                header_parts.append(f"Latency: {latency}ms")
        step_header = " | ".join(header_parts)

        # --- tool names ---
        tools_line = ""
        if include_tools_per_step:
            api_spec = row.get("api_spec", "")
            if pd.notna(api_spec) and api_spec:
                tool_names = extract_tool_names(api_spec)
                if tool_names:
                    if len(tool_names) <= 5:
                        tools_line = f"**Available Tools:** {', '.join(tool_names)}"
                    else:
                        tools_line = (
                            f"**Available Tools:** "
                            f"{', '.join(tool_names[:5])} "
                            f"(+{len(tool_names) - 5} more)"
                        )

        # --- input context ---
        input_text = ""
        if include_input_context:
            model_input = row.get("model_input", "")
            if pd.notna(model_input) and model_input:
                if truncate_content:
                    input_text = extract_input_context(
                        str(model_input),
                        max_system_len=max_system_prompt,
                        max_total_len=max_input_context,
                    )
                else:
                    input_text = extract_input_context(
                        str(model_input),
                        max_system_len=NO_LIMIT,
                        max_total_len=NO_LIMIT,
                    )

        # --- response ---
        response_text = ""
        response_raw = row.get("response", "")
        if pd.notna(response_raw) and response_raw:
            limit = max_response_len if truncate_content else NO_LIMIT
            response_text = format_response_compact(str(response_raw), max_len=limit)

        steps.append({
            "header": step_header,
            "tools": tools_line,
            "input": input_text,
            "response": response_text,
        })

    return steps


def _assemble_trace(header_text: str, steps: List[Dict[str, str]]) -> str:
    """Join the trace header and per-step blocks into the final string."""
    lines = [header_text]
    for s in steps:
        lines.append(s["header"])
        lines.append("")

        if s["tools"]:
            lines.append(s["tools"])
            lines.append("")

        if s["input"]:
            lines.append("**Input Context:**")
            lines.append("```")
            lines.append(s["input"])
            lines.append("```")
            lines.append("")

        if s["response"]:
            lines.append("**Response:**")
            lines.append("```")
            lines.append(s["response"])
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _estimate_skeleton_size(header_text: str, steps: List[Dict[str, str]]) -> int:
    """
    Estimate the character count of everything *except* input/response content.

    This includes the trace header, per-step headers, tool lines, markdown
    fences, separators, and newlines.
    """
    size = len(header_text)
    for s in steps:
        size += len(s["header"]) + 2          # header + "\n\n"
        if s["tools"]:
            size += len(s["tools"]) + 2       # tools + "\n\n"
        if s["input"]:
            # "**Input Context:**\n```\n" ... "\n```\n\n"
            size += len("**Input Context:**\n```\n") + len("\n```\n\n")
        if s["response"]:
            size += len("**Response:**\n```\n") + len("\n```\n\n")
        size += len("---\n\n")
    return size


def format_compact_trace(
    df: pd.DataFrame,
    max_tokens: Optional[int] = None,
    chars_per_token: float = 3.5,
    max_input_context: int = 10000,
    max_response_len: int = 10000,
    max_system_prompt: int = 10000,
    include_tools_per_step: bool = True,
    include_input_context: bool = True,
    include_metadata: bool = True,
) -> str:
    """
    Convert a trace DataFrame to compact text format for LLM judge.

    Two modes of operation:

    **Fixed limits** (``max_tokens is None``, the default):
        Each field is truncated to the fixed per-field character limits
        (``max_input_context``, ``max_response_len``, ``max_system_prompt``).
        Behaviour is identical to the original implementation.

    **Adaptive** (``max_tokens`` is set):
        The formatter first extracts all content *without* truncation,
        measures the total size, and – only if it exceeds the token budget –
        applies the *minimal* proportional truncation needed to fit.
        The per-field fixed limits are ignored in this mode.

    Args:
        df: DataFrame with unified CSV schema
        max_tokens: Total token budget for the output.  When set, adaptive
            truncation is used and the fixed per-field limits are ignored.
        chars_per_token: Approximate characters per token used to convert
            *max_tokens* to a character budget (default 3.5).
        max_input_context: Max chars for input context per step (fixed mode)
        max_response_len: Max chars for response per step (fixed mode)
        max_system_prompt: Max chars for system prompts (fixed mode)
        include_tools_per_step: Show available tools at each step
        include_input_context: Include extracted input context
        include_metadata: Include model/tokens/latency info

    Returns:
        Compact text representation of the trace
    """
    if df.empty:
        return "Empty trace"

    adaptive = max_tokens is not None

    # ── trace header ──────────────────────────────────────────────────
    first_row = df.iloc[0]
    task_id = first_row.get("task_id", "unknown")
    intent = first_row.get("intent", "")
    traj_score = first_row.get("traj_score")
    if pd.isna(traj_score):
        traj_score = "N/A"

    header_text = "\n".join([
        f"## Trace: {task_id}",
        f"**User Query:** {intent}",
        f"**Total Steps:** {len(df)} | **Trajectory Score:** {traj_score}",
        "", "---", "",
    ])

    # ── fixed-limit mode (original behaviour) ────────────────────────
    if not adaptive:
        steps = _precompute_steps(
            df,
            include_tools_per_step=include_tools_per_step,
            include_input_context=include_input_context,
            include_metadata=include_metadata,
            truncate_content=True,
            max_input_context=max_input_context,
            max_response_len=max_response_len,
            max_system_prompt=max_system_prompt,
        )
        res = _assemble_trace(header_text, steps)
        return res

    # ── adaptive mode ─────────────────────────────────────────────────
    # Pass 1: extract all content untruncated
    steps = _precompute_steps(
        df,
        include_tools_per_step=include_tools_per_step,
        include_input_context=include_input_context,
        include_metadata=include_metadata,
        truncate_content=False,
    )

    # Measure
    char_budget = int(max_tokens * chars_per_token)
    skeleton_size = _estimate_skeleton_size(header_text, steps)
    total_content = sum(len(s["input"]) + len(s["response"]) for s in steps)
    available = char_budget - skeleton_size

    # Pass 2: apply proportional truncation only if needed
    if available > 0 and total_content > available:
        ratio = available / total_content
        for s in steps:
            if s["input"]:
                limit = max(int(len(s["input"]) * ratio), 50)
                s["input"] = truncate_text(s["input"], limit, strategy="middle")
            if s["response"]:
                limit = max(int(len(s["response"]) * ratio), 50)
                s["response"] = truncate_text(s["response"], limit, strategy="middle")
    elif available <= 0:
        # Budget too tight even for skeleton – truncate everything to minimum
        for s in steps:
            if s["input"]:
                s["input"] = truncate_text(s["input"], 50, strategy="middle")
            if s["response"]:
                s["response"] = truncate_text(s["response"], 50, strategy="middle")

    res = _assemble_trace(header_text, steps)
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
    output_dir: str = None,
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
    output_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for csv_file in input_path.glob("*.csv"):
        trace_id = csv_file.stem

        try:
            full_trace = format_compact_trace_from_csv(str(csv_file), max_input_context=NO_LIMIT, max_response_len=NO_LIMIT, max_system_prompt=NO_LIMIT)
            compact = format_compact_trace_from_csv(str(csv_file), max_tokens=128_000, **kwargs)
            prev = format_compact_trace_from_csv(str(csv_file), **kwargs)

            def tok_len(s): return int(len(s)/3.5)
            print(f"{trace_id},{tok_len(full_trace)},{tok_len(prev)},{tok_len(compact)}")
            if output_dir:
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
    output_path = Path("/Users/lilache/Documents/agentic/data/paper_experiments/HAL/gaia_hal_generalist_agent_gpt4120250414_1744652581/new_processed_traces")
    res = process_trace_directory("/Users/lilache/Documents/agentic/data/paper_experiments/HAL/gaia_hal_generalist_agent_gpt4120250414_1744652581/csvs", output_dir=  output_path)
    sys.exit(0)
    if len(sys.argv) < 2:
        print("Usage: python compact_trace_formatter.py <trace.csv> [output.txt]")
        print("\nConverts unified CSV trace to compact text format for LLM judges.")
        sys.exit(1)

    csv_path = sys.argv[1]
    import os
    if os.path.isdir(csv_path):
        csv_paths = os.listdir(csv_path)
    else:
        csv_paths = [csv_path]
        compact = format_compact_trace_from_csv(csv_path)

        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            with open(output_path, 'w') as f:
                f.write(compact)
            print(f"Saved to {output_path}")
        else:
            print(compact)
