# Separate Tool-Call Evaluation

> **Status: Experimental** — functional but not yet fully validated.
> The default combined mode (`--separate-tools false`) is recommended for production use.

The `--separate-tools` flag controls whether tool calls are evaluated
individually (per-tool-call SPARC evaluation) or as part of the combined
LLM response.  It is a boolean flag:

- `false` (default): combined mode — one evaluation per LLM call
- `true`: tools-with-reasoning mode — one evaluation per tool call

For backward compatibility, `combined` maps to `false` and
`tools_with_reasoning` / `separate` map to `true`.

---

## How it works

The intermediate representation (IR) always stores **one row per LLM call**
with `response` (text only) and `tool_calls` (JSON list) as separate columns.
See [INTERMEDIATE_REPR.md](INTERMEDIATE_REPR.md).

Row splitting happens at **analysis time** in `convert_to_clear_format()`,
not during preprocessing.  This means you can switch between modes without
re-preprocessing your traces.

---

## Mode: `false` (combined, default)

Each LLM call is a single evaluation record.  When tool calls exist, the
`response` field is reconstructed as a JSON object:

```json
{"content": "I'll search for that.", "tool_calls": [{...}, {...}]}
```

All rows go to the agent's main CSV for standard CLEAR evaluation.

---

## Mode: `true` (tools_with_reasoning)

When an LLM call produces tool calls, each tool call becomes a separate row
for SPARC evaluation.  The reasoning text (from `response`) is appended to
each tool row's `model_input` as an assistant message, giving SPARC richer
context.

Text-only LLM calls (no tool calls) are evaluated normally via CLEAR.

---

## Evaluation behavior

When `--separate-tools true`:

- **Tool rows** are evaluated by SPARC (tool-call quality). Results are stored in a `tool_calls/` subdirectory under each agent.
- **Reasoning rows** (text-only LLM calls) are evaluated by standard CLEAR. Results are stored at the agent root directory.
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
