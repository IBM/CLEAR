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

## Modes: `--separate-tools`

The `--separate-tools` flag controls how you represent LLM calls that produce
tool calls.  Accepts bool (`true`/`false`) for backward compatibility.

| Mode | What each row represents | `tool_or_agent` | `response` content |
|------|--------------------------|-----------------|--------------------|
| `combined` (default) | One LLM invocation | Always `"agent"` | Free-form text (may include tool calls however you choose to represent them) |
| `separate` | One tool call **or** one text response | `"tool"` or `"agent"` | Single tool-call JSON **or** text |
| `tools_with_reasoning` | One tool call (reasoning text appended to tool rows' `model_input` as context) | `"tool"` or `"agent"` | Single tool-call JSON **or** text (standalone text only) |

All modes use the same columns.

## Shared columns

These columns appear in both modes.

### Required

| Column                  | Type     | Description                                                                                                          |
|-------------------------|----------|----------------------------------------------------------------------------------------------------------------------|
| `Name`                  | str      | Agent or node name.  CLEAR groups and evaluates rows by this value, so set it to the component that invoked the LLM. |
| `task_id`               | str      | Trajectory identifier.  All rows from the same trace must share this value.                                          |
| `step_in_trace_general` | int      | Row ordering (1-indexed, unique per row within the file).                                                            |
| `model_input`           | json/str | The messages sent to the LLM (see [format](#model_input-format)).                                                    |
| `response`              | str      | The LLM output (format differs by mode — see below).                                                                 |
| `tool_or_agent`         | str      | Differs by mode — see below                                                                                          |
| `llm_call_index`        | int      | LLM invocation counter (1-indexed) - See below                                                                       

### Recommended

| Column | Type | Description |
|--------|------|-------------|
| `intent` | str | The user's original query / goal for this trajectory. |
| `api_spec` | str | JSON string of tool definitions available to the LLM call, in OpenAI format (see [format](#api_spec-format)). |
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

## Mode: `--separate-tools combined` (default)

Each row represents **one LLM invocation**.  This is the simpler mode — use it
when you don't need per-tool-call evaluation.

### Mode-specific columns

| Column | Type | Description |
|--------|------|-------------|
| `tool_or_agent` | str | Always `"agent"`. |
| `llm_call_index` | int | LLM invocation counter (1-indexed).  Equals `step_in_trace_general` in this mode (one row per call). |

### `response` format

Free-form text — any string representation of the LLM output works.  The
built-in preprocessors use a structured JSON format when tool calls are present:

```json
{"content": "I'll search for that.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"query\": \"climate\"}"}}]}
```

This format is recommended as it helps the LLM judge clearly distinguish text
from tool invocations, but it is not required.  If you produce CSVs yourself,
you may represent tool calls inline, appended, or in any other format.

### Example

A trace where agent A calls the LLM twice and agent B calls it once:

| Name | task_id | step_in_trace_general | llm_call_index | tool_or_agent | response (abbreviated) |
|------|---------|-----------------------|----------------|---------------|------------------------|
| A | tr-abc | 1 | 1 | agent | I'll search for that. [called search("climate")] |
| B | tr-abc | 2 | 2 | agent | The answer is 42. |
| A | tr-abc | 3 | 3 | agent | Done. |

---

## Mode: `--separate-tools separate`

Each LLM invocation is split into **separate rows**: one per tool call, plus an
optional row for the text response.  Use this mode when you want per-tool-call
evaluation (e.g. via SPARC).

### Mode-specific columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `tool_or_agent` | str | Yes | `"tool"` for a tool-call row, `"agent"` for a text-response row. |
| `llm_call_index` | int | Yes | LLM invocation counter (1-indexed).  All rows from the same LLM call share this value. |

### `response` format

| Row type | `tool_or_agent` | `response` |
|----------|-----------------|------------|
| Tool call | `"tool"` | A single tool-call JSON in full OpenAI format: `{"id": str, "type": "function", "function": {"name": str, "arguments": str}}` |
| Text response | `"agent"` | The text as-is |

Omit rows for empty text responses.

### Example

A trace where LLM call 1 produced 2 tool calls + text, call 2 produced text
only, and call 3 produced 1 tool call with no text:

| Name | task_id | step_in_trace_general | llm_call_index | tool_or_agent | response (abbreviated) |
|------|---------|-----------------------|----------------|---------------|------------------------|
| A | tr-abc | 1 | 1 | tool | `{"id":"call_1","type":"function","function":{"name":"search","arguments":"{...}"}}` |
| A | tr-abc | 2 | 1 | tool | `{"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{...}"}}` |
| A | tr-abc | 3 | 1 | agent | Based on the search results... |
| B | tr-abc | 4 | 2 | agent | The answer is 42. |
| A | tr-abc | 5 | 3 | tool | `{"id":"call_3","type":"function","function":{"name":"submit","arguments":"{...}"}}` |

Key rules:
- `step_in_trace_general` increments for every **row** (must be unique within the file).
- `llm_call_index` increments per **LLM invocation** (shared by all rows from that invocation).

---

## Mode: `--separate-tools tools_with_reasoning`

Like `separate`, but when an LLM call produces both tool calls and text, the
reasoning text is **not** emitted as its own row.  Instead it is appended to
each tool row's `model_input` as an assistant message, giving SPARC richer
context for evaluating the tool call.

Text-only LLM calls (no tool calls) still produce an `"agent"` row as usual.

### Example

Same trace as the `separate` example above.  Note that row 3 (the agent text
from LLM call 1) is gone — its content is folded into the `model_input` of
rows 1 and 2:

| Name | task_id | step_in_trace_general | llm_call_index | tool_or_agent | response (abbreviated) | model_input note |
|------|---------|-----------------------|----------------|---------------|------------------------|------------------|
| A | tr-abc | 1 | 1 | tool | `{"id":"call_1",...}` | includes `"Based on the search results..."` as assistant message |
| A | tr-abc | 2 | 1 | tool | `{"id":"call_2",...}` | includes `"Based on the search results..."` as assistant message |
| B | tr-abc | 3 | 2 | agent | The answer is 42. | |
| A | tr-abc | 4 | 3 | tool | `{"id":"call_3",...}` | |

- Tool rows come before the agent row within each `llm_call_index` group.
