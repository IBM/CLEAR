"""
Create compact trace representations for LLM judges.

This module converts the unified CSV trace representation into a compact
format suitable for LLM-as-judge evaluation, significantly reducing token usage
while preserving the information needed for quality assessment.

INPUT: Unified CSV format (output of preprocessing step)
  Columns: id, Name, intent, task_id, step_in_trace_general, llm_call_index,
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


def deduplicate_model_input(
    model_input: str,
    previous_input: Optional[str] = None,
) -> str:
    """
    Remove prefix messages from model_input that were already seen in previous_input.
    
    This deduplication happens BEFORE any truncation or formatting to ensure
    accurate comparison of message content.
    
    Args:
        model_input: Current model_input string from CSV
        previous_input: Previous model_input for this agent (for deduplication)
    
    Returns:
        Deduplicated model_input string (as JSON) with only delta messages
    """
    if not previous_input:
        return model_input
    
    messages = parse_model_input(model_input)
    if not messages:
        return model_input
    
    previous_messages = parse_model_input(previous_input)
    if not previous_messages:
        return model_input
    
    # Find the longest common prefix
    common_prefix_len = 0
    for i, (curr_msg, prev_msg) in enumerate(zip(messages, previous_messages)):
        # Compare role and content to determine if messages are identical
        if (curr_msg.get("role") == prev_msg.get("role") and
            curr_msg.get("content") == prev_msg.get("content")):
            common_prefix_len = i + 1
        else:
            break
    
    # Keep only the delta (new messages after the common prefix)
    if common_prefix_len > 0:
        messages = messages[common_prefix_len:]
    
    # Return as JSON string to maintain format consistency
    if not messages:
        return "[]"
    
    return json.dumps(messages)


def _format_tool_call(tc: Any, role: str = "") -> str:
    """
    Format a single tool call for display.
    
    Handles multiple formats:
    - OpenAI format: {"function": {"name": "...", "arguments": "..."}}
    - Direct format: {"name": "...", "args": "..."}
    - Non-standard: dumps as JSON
    
    Args:
        tc: Tool call object (dict or other)
        role: Role prefix for the output (optional)
    
    Returns:
        Formatted tool call string
    """
    prefix = f"{role} [Tool Call]: " if role else "[Tool Call]: "
    
    if not isinstance(tc, dict):
        # Non-dict tool call - dump as is
        return f"{prefix}{str(tc)}"
    
    # Try to extract name and args from various formats
    func = tc.get("function", tc)
    name = func.get("name", tc.get("name"))
    
    if not name:
        # No name found - dump entire dict as JSON
        return f"{prefix}{json.dumps(tc)}"
    
    # Extract arguments from various possible locations
    args = func.get("arguments", tc.get("arguments", tc.get("args", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            pass
    
    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
    return f"{prefix}{name}({args_str})"


def extract_input_context(
    model_input: str,
    max_system_len: int = 10000,
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
        model_input: Raw model_input string from CSV (should be deduplicated first)
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
        tool_calls = msg.get("tool_calls", [])

        # Truncate system prompts
        if role == "system":
            content = truncate_text(content, max_system_len, strategy="middle")

        # Format output with content
        if content:
            output_parts.append(f"{role}: {content}")
        
        # Add tool calls if present
        if tool_calls:
            for tc in tool_calls:
                output_parts.append(_format_tool_call(tc, role=role))

    result = "\n\n".join(output_parts)

    # Final truncation if needed
    result = truncate_text(result, max_total_len, strategy="middle")

    return result


# =============================================================================
# Formatting utilities for response field
# =============================================================================

def format_response_compact(response: str, max_len: int = 10000) -> str:
    """
    Format response for compact representation.
    
    Simply returns the response as-is with truncation applied.
    The response field already contains the formatted output from the agent.
    """
    if not response or not isinstance(response, str):
        return ""
    
    return truncate_text(response.strip(), max_len, strategy="middle")


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
            names = []
            for t in tools:
                if not isinstance(t, dict):
                    continue
                # Full OpenAI format: {"type": "function", "function": {"name": ...}}
                func = t.get("function", t)
                name = func.get("name") if isinstance(func, dict) else t.get("name")
                if name:
                    names.append(name)
            return names
    except:
        pass

    return []


def collect_all_tools(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Collect all unique tools from the entire trace.
    
    Returns a dict mapping tool name to tool definition.
    """
    all_tools = {}
    
    for idx, row in df.iterrows():
        api_spec = row.get("api_spec", "")
        if pd.notna(api_spec) and api_spec:
            try:
                tools = json.loads(str(api_spec))
                if isinstance(tools, list):
                    for tool in tools:
                        if not isinstance(tool, dict):
                            continue
                        # Full OpenAI format: {"type": "function", "function": {"name": ...}}
                        func = tool.get("function", tool)
                        name = func.get("name") if isinstance(func, dict) else tool.get("name")
                        if name and name not in all_tools:
                            all_tools[name] = tool
            except:
                pass
    
    return all_tools


def format_tools_section(all_tools: Dict[str, Dict[str, Any]]) -> str:
    """
    Format all tools into a section for the trace header.
    
    Simply dumps the tool definitions as JSON to preserve exact format.
    """
    if not all_tools:
        return ""
    
    lines = ["## Available Tools", "", "```json"]
    
    # Convert to list and dump as JSON
    tools_list = [tool_def for tool_def in all_tools.values()]
    lines.append(json.dumps(tools_list, indent=2))
    
    lines.extend(["```", "", "---", ""])
    
    return "\n".join(lines)

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

def _collect_and_deduplicate_steps(
    df: pd.DataFrame,
    include_tools_per_step: bool,
    include_input_context: bool,
) -> List[Dict[str, Any]]:
    """
    Phase 1: Collect step data and deduplicate input messages.
    
    This phase ONLY:
    - Extracts headers and tools (formatted)
    - Deduplicates model_input per agent (stores raw JSON strings)
    - Stores raw response strings
    - Does NOT format or truncate input/response content
    
    Returns steps with raw deduplicated content ready for processing.
    """
    steps: List[Dict[str, Any]] = []
    # Track previous model_input per agent for deduplication
    agent_previous_input: Dict[str, str] = {}

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

        # --- deduplicate model_input (store RAW JSON, no formatting yet) ---
        deduplicated_input_raw = ""
        if include_input_context:
            model_input = row.get("model_input", "")
            if pd.notna(model_input) and model_input:
                # Get previous input for this agent (if any)
                previous_input = agent_previous_input.get(agent_name)

                # Deduplicate and store RAW JSON string
                deduplicated_input_raw = deduplicate_model_input(
                    str(model_input),
                    previous_input=previous_input,
                )
                
                # Update the previous input for this agent (store the ORIGINAL, not deduplicated)
                agent_previous_input[agent_name] = str(model_input)

        # --- store raw response ---
        response_raw = row.get("response", "")
        if not pd.notna(response_raw):
            response_raw = ""

        steps.append({
            "header": step_header,
            "tools": tools_line,
            "input_raw": deduplicated_input_raw,  # Raw JSON string
            "response_raw": str(response_raw) if response_raw else "",  # Raw string
        })

    return steps


def _process_and_truncate_fixed(
    steps: List[Dict[str, Any]],
) -> None:
    """
    Phase 2a: Process raw content and apply fixed per-field truncation.
    
    This matches the original behavior:
    - Extracts and formats input with per-message system prompt truncation
    - Formats response
    - Applies final truncation limits
    
    Modifies steps in-place, converting raw fields to formatted text.
    """
    for s in steps:
        # Process input with per-message truncation (original behavior)
        if s["input_raw"]:
            s["input"] = extract_input_context(
                s["input_raw"],
                max_system_len=NO_LIMIT,
                max_total_len=NO_LIMIT,
            )
        else:
            s["input"] = ""
        
        # Process response with truncation
        if s["response_raw"]:
            s["response"] = format_response_compact(
                s["response_raw"],
                max_len=NO_LIMIT,
            )
        else:
            s["response"] = ""
        
        # Remove raw fields (no longer needed)
        del s["input_raw"]
        del s["response_raw"]


def _process_and_truncate_adaptive(
    steps: List[Dict[str, Any]],
    header_text: str,
    max_tokens: int,
    chars_per_token: float,
) -> None:
    """
    Phase 2b: Process raw content and apply adaptive proportional truncation.
    
    - First extracts and formats all content WITHOUT truncation
    - Then applies minimal proportional truncation to fit token budget
    
    Modifies steps in-place, converting raw fields to formatted text.
    """
    # First pass: extract and format without truncation
    for s in steps:
        if s["input_raw"]:
            s["input"] = extract_input_context(
                s["input_raw"],
                max_system_len=NO_LIMIT,
                max_total_len=NO_LIMIT,
            )
        else:
            s["input"] = ""
        
        if s["response_raw"]:
            s["response"] = format_response_compact(
                s["response_raw"],
                max_len=NO_LIMIT,
            )
        else:
            s["response"] = ""
    
    # Second pass: apply proportional truncation if needed
    char_budget = int(max_tokens * chars_per_token)
    skeleton_size = _estimate_skeleton_size(header_text, steps)
    total_content = sum(len(s["input"]) + len(s["response"]) for s in steps)
    available = char_budget - skeleton_size

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
    
    # Remove raw fields (no longer needed)
    for s in steps:
        del s["input_raw"]
        del s["response_raw"]


def _apply_adaptive_truncation(
    steps: List[Dict[str, str]],
    header_text: str,
    max_tokens: int,
    chars_per_token: float,
) -> None:
    """
    Phase 2b: Apply adaptive proportional truncation.
    
    Modifies steps in-place by applying minimal proportional truncation
    to fit within the token budget.
    """
    char_budget = int(max_tokens * chars_per_token)
    skeleton_size = _estimate_skeleton_size(header_text, steps)
    total_content = sum(len(s["input"]) + len(s["response"]) for s in steps)
    available = char_budget - skeleton_size

    # Apply proportional truncation only if needed
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

    Returns:
        Compact text representation of the trace
    """
    if df.empty:
        return "Empty trace"

    adaptive = max_tokens is not None

    # ── trace header with tools ───────────────────────────────────────
    first_row = df.iloc[0]
    task_id = first_row.get("task_id", "unknown")
    intent = first_row.get("intent", "")
    traj_score = first_row.get("traj_score")
    if pd.isna(traj_score):
        traj_score = "N/A"

    # Collect all unique tools from the trace
    all_tools = collect_all_tools(df)
    tools_section = format_tools_section(all_tools)

    header_parts = [
        f"## Trace: {task_id}",
        f"**User Query:** {intent}",
        f"**Total Steps:** {len(df)} | **Trajectory Score:** {traj_score}",
        "",
    ]
    
    # Add tools section if there are any tools
    if tools_section:
        header_parts.append(tools_section)
    else:
        header_parts.extend(["---", ""])
    
    header_text = "\n".join(header_parts)

    # ── Phase 1: Collect and deduplicate (no processing) ─────────────
    steps = _collect_and_deduplicate_steps(
        df,
        include_tools_per_step=include_tools_per_step,
        include_input_context=include_input_context,
    )

    # ── Phase 2: Process and truncate ────────────────────────────────
    if not adaptive:
        # Fixed-limit mode: process with per-message truncation (original behavior)
        _process_and_truncate_fixed(
            steps,
        )
    else:
        # Adaptive mode: process then apply proportional truncation
        _process_and_truncate_adaptive(
            steps,
            header_text=header_text,
            max_tokens=max_tokens,
            chars_per_token=chars_per_token,
        )

    # ── Phase 3: Assemble final output ───────────────────────────────
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
