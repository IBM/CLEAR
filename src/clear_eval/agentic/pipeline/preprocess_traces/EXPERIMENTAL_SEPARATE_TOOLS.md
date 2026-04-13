# Experimental: Separate Tool-Call Modes

> **Status: Experimental** — these modes are functional but not yet fully validated.
> The default `combined` mode is recommended for production use.

The `--separate-tools` flag controls how LLM calls that produce tool calls are
represented in the trajectory CSV and evaluated downstream.  It accepts
`combined` (default), `separate`, or `tools_with_reasoning`.  Boolean values
(`true`/`false`) are accepted for backward compatibility (`true` maps to
`separate`, `false` maps to `combined`).

---

## Modes

| Mode | What each row represents | `tool_or_agent` | `response` content |
|------|--------------------------|-----------------|--------------------|
| `combined` (default) | One LLM invocation | Always `"agent"` | Free-form text (may include tool calls however you choose to represent them) |
| `separate` | One tool call **or** one text response | `"tool"` or `"agent"` | Single tool-call JSON **or** text |
| `tools_with_reasoning` | One tool call (reasoning text appended to tool rows' `model_input` as context) | `"tool"` or `"agent"` | Single tool-call JSON **or** text (standalone text only) |

All modes use the same columns (see [INTERMEDIATE_REPR.md](INTERMEDIATE_REPR.md)).

---

## Additional columns required by `separate` and `tools_with_reasoning`

| Column | Type | Description |
|--------|------|-------------|
| `tool_or_agent` | str | **Required.** `"tool"` for a tool-call row, `"agent"` for a text-response row. |
| `llm_call_index` | int | **Required.** LLM invocation counter (1-indexed). All rows from the same LLM call share this value. |

---

## Mode: `separate`

Each LLM invocation is split into **separate rows**: one per tool call, plus an
optional row for the text response.  Use this mode when you want per-tool-call
evaluation (e.g. via SPARC).

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

## Mode: `tools_with_reasoning`

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

---

## Evaluation behavior

When `separate` or `tools_with_reasoning` is used:

- **Tool rows** (`tool_or_agent="tool"`) are evaluated by SPARC (tool-call quality). Results are stored in a `tool_calls/` subdirectory under each agent.
- **Reasoning rows** (`tool_or_agent="agent"`) are evaluated by standard CLEAR. Results are stored at the agent root directory.
- The JSON output wraps per-agent data in `reasoning_eval` and/or `tools_eval` keys.
- The dashboard workflow graph uses `llm_call_index` for deduplication (one node visit per LLM call, not per row).

---

## Output structure (when tool rows are present)

```
clear_results/
  agent_1/                    # reasoning results at agent root
    analysis_results_*.csv
    tool_calls/               # tool eval results in subdir
      analysis_results_*.csv
  agent_2/                    # reasoning-only agent, no subdir
    analysis_results_*.csv
```

JSON `clear_results.json` per-agent structure:

```json
{
  "agents": {
    "agent_1": {
      "reasoning_eval": { "agent_summary": {}, "issues_catalog": {}, "issues": [], "no_issues": [] },
      "tools_eval": { "agent_summary": {}, "issues_catalog": {}, "issues": [], "no_issues": [] }
    },
    "agent_2": {
      "reasoning_eval": { "agent_summary": {}, "issues_catalog": {}, "issues": [], "no_issues": [] }
    }
  }
}
```
