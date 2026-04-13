# Trajectory CSV Format Reference

This document describes the CSV format consumed by the CLEAR agentic evaluation
pipeline.  If you use a supported observability platform (MLflow, Langfuse),
this format is produced automatically via `--from-raw-traces true`.
If your platform is not supported, produce these CSVs yourself and pass them
with `--from-raw-traces false`.

---

## File layout

Place one CSV file per trajectory in a directory and point `--data-dir` at it:

```
my_data/
  trace_001_id.csv
  trace_002_id.csv
  ...
```

Each file contains all the rows for a single trajectory (one `task_id`).

---

## Columns

Each row represents **one LLM invocation**.

### Required

| Column                  | Type     | Description                                                                                                          |
|-------------------------|----------|----------------------------------------------------------------------------------------------------------------------|
| `Name`                  | str      | Agent or node name.  CLEAR groups and evaluates rows by this value, so set it to the component that invoked the LLM. |
| `id` | str | Unique row identifier. Defaults to `{task_id}_{step_in_trace_general}`. |

| `task_id`               | str      | Trajectory identifier.  All rows from the same trace must share this value.                                          |
| `step_in_trace_general` | int      | Row ordering (1-indexed, unique per row within the file).                                                            |
| `model_input`           | json/str | The messages sent to the LLM (see [format](#model_input-format)).                                                    |
| `response`              | str      | The LLM output (see [format](#response-format)).                                                                     |

### Optional

| Column | Type | Description |
|--------|------|-------------|
| `intent` | str | The user's original query / goal for this trajectory. |
| `api_spec` | str | JSON string of tool definitions available to the LLM call, in OpenAI format (see [format](#api_spec-format)). When the response contains tool calls, this is automatically appended to `model_input` at evaluation time so the judge can assess tool selection quality. |
| `meta_data` | json/str | Free-form JSON string for any metadata you want to log (model name, token counts, latency, span IDs, etc.). |
| `traj_score` | float | Ground-truth trajectory score (0-1).  All rows in a trajectory should share the same value. |

---

## `model_input` format

A JSON string containing the messages sent to the LLM.  Use this structure:

```json
[
  {"role": "system",    "content": "You are a helpful assistant...", "tool_calls": []},
  {"role": "user",      "content": "What is the weather in Paris?",  "tool_calls": []},
  {"role": "assistant", "content": "",
   "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
  {"role": "tool",      "content": "{\"temp\": 18, \"condition\": \"cloudy\"}", "tool_calls": []}
]
```

Tips:
- Put tool calls on assistant messages in `tool_calls`, not in `content` (avoids duplication).
- If your input is a plain string rather than a message list, that works too.
- Very long system messages may be truncated by the pipeline (default limit: 50 000 chars).

---

## `response` format

Free-form text — any string representation of the LLM output works.  The
built-in preprocessors use a structured JSON format when tool calls are present:

```json
{"content": "I'll search for that.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"query\": \"climate\"}"}}]}
```

This format is recommended as it helps the LLM judge clearly distinguish text
from tool invocations, but it is not required.  If you produce CSVs yourself,
you may represent tool calls inline, appended, or in any other format.

---

## `api_spec` format

A JSON string listing the tools **available** to the LLM at call time (not the
calls it made).  Use OpenAI function-calling format:

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
      }
    }
  }
]
```

Set to empty string (`""`) when no tools are bound to the call.

The built-in preprocessors automatically normalise Anthropic (`input_schema`)
and Gemini (`functionDeclarations`) formats to this shape.  When producing
CSVs yourself, use the full OpenAI format directly.

---

## Example

A trace where agent A calls the LLM twice and agent B calls it once:

| Name | task_id | step_in_trace_general | llm_call_index | tool_or_agent | response (abbreviated) |
|------|---------|-----------------------|----------------|---------------|------------------------|
| A | tr-abc | 1 | 1 | agent | I'll search for that. [called search("climate")] |
| B | tr-abc | 2 | 2 | agent | The answer is 42. |
| A | tr-abc | 3 | 3 | agent | Done. |

---

## Experimental: separate tool-call modes

The `--separate-tools` flag supports additional modes for per-tool-call
evaluation.  These are **experimental** and not yet fully validated.
See [EXPERIMENTAL_SEPARATE_TOOLS.md](EXPERIMENTAL_SEPARATE_TOOLS.md) for
details.
