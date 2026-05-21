# Running CLEAR with SPARC

CLEAR's tool-calling task delegates per-call evaluation to SPARC (the
Semantic Pre-Action Reflection Component shipped from `altk-boost`).
This directory contains the helpers needed to convert raw OpenTelemetry
traces into CLEAR's unified-row CSV format and run the pipeline against
them under the SPARC prompts.

## Prerequisites

1. **Python 3.10+** with a venv. Install CLEAR and and agent-lifecycle-toolkit

2. A `.env` file at the CLEAR repo root with watsonx credentials:

   ```
   WATSONX_APIKEY=...
   WATSONX_URL=...
   WATSONX_PROJECT_ID=...        # or WATSONX_SPACE_ID
   ```

3. Tool specs JSON files for each benchmark live in `scripts/tool_specs/`
   (e.g. `appworld.json`, `tau2_retail.json`). The converter merges them
   with any tool names declared inside each trace.

## 1. Convert traces to the unified-row CSV

The converter (`scripts/traces_to_clear_csv.py`) emits one row per LLM
call in the format CLEAR's pipeline now expects:

- `model_input`: original chat history at the time of the call
- `response`: when the call produced tool calls,
  `{"content": "<text>", "tool_calls": [...]}` JSON;
  otherwise just the assistant text
- `tool_or_agent`: always `"agent"`
- SPARC-specific reformatting (per-tool-call splitting, reasoning
  injection) happens later inside the pipeline when `separate_tools` is
  enabled — the converter does not pre-split.

```bash
python scripts/traces_to_clear_csv.py \
    /path/to/traces/*.json \
    --output-dir scripts/runs/<bench>/input \
    --specs-dir scripts/tool_specs
```

## 2. Configure the run

[`scripts/sparc_config.yaml`](sparc_config.yaml) holds the defaults. The
SPARC-relevant knobs are:

| Key                | Values                                                                                                  | Effect |
|--------------------|--------------------------------------------------------------------------------------------------------|--------|
| `track`            | `slow_track` \| `fast_track` \| `spec_free` \| `syntax` \| `transformations_only`                       | Which SPARC pipeline track to run. `slow_track` is the full semantic pipeline; `spec_free` is used automatically when no tool specs are available. |
| `runtime_pipeline` | `true` (default) / `false`                                                                              | `true` = fast prompts, no actionable recommendations. `false` = evaluation-time prompts that emit unified-diff `actionable_recommendations` with `importance ∈ [0, 1]`. |
| `separate_tools`   | `true` (default) / `false`                                                                              | `true` = pipeline splits each unified row's bundled tool calls into per-call analysis rows, folds reasoning into each row's `model_input`, and routes them to CLEAR's `task="tool_call"` use case (SPARC). `false` skips SPARC entirely and only runs the standard reasoning analysis on the bundled rows. |
| `eval_model_name`  | e.g. `openai/gpt-oss-120b`                                                                              | The judge model. |

## 3. Run the pipeline

`scripts/run_sparc.sh` wraps the CLI so you can mix `--track` × `--mode`:

```bash
# Default: slow_track, runtime mode, all 3 benches
bash scripts/run_sparc.sh

# Single bench, eval mode (emits actionable recommendations)
bash scripts/run_sparc.sh --track slow_track --mode eval appworld

# Fast track, multiple benches
bash scripts/run_sparc.sh --track fast_track tau2_retail appworld
```

Output lands at `scripts/runs/<bench>/output_<track>_<mode>/`.

## 4. Read the results

For each `<bench>` and `<track>_<mode>` combination the pipeline writes:

- **Per-record CSV** —
  `clear_results/tool_calls/analysis_results_*.csv`. Adds three SPARC
  columns to every row:
  - `sparc_decision` — `True` iff SPARC approved the call
  - `sparc_score_1_to_5` — raw 1-5 rubric mean (null on static-only)
  - `sparc_recommendations` — JSON array of `{target, diff, rationale,
    importance, tool_name?, parameter_name?}` entries (always `[]` in
    runtime mode)

- **Aggregated JSON** —
  `clear_results.json` →
  `agents.<name>.tools_eval.recommendations`:
  ```jsonc
  {
    "by_tool": {
      "<tool>": [
        { "target": "tool_description" | "parameter_description" | "parameter_examples",
          "diff": "...",
          "rationale": "...",
          "tool_name": "...",
          "parameter_name": "...",
          "count": <int>,
          "importance_mean": <float>,
          "importance_max": <float> }, ...
      ], ...
    },
    "by_system_prompt": [
      { "target": "system_prompt", "diff": "...", "count": <int>,
        "importance_mean": <float>, "importance_max": <float> }, ...
    ],
    "total": <int>
  }
  ```
  - `system_prompt` recommendations are aggregated globally across every
    judgment; tool / parameter recommendations are aggregated per tool
    (and per parameter inside the tool entry).
  - The same diff text from multiple judgments dedupes to one entry; the
    `count` field shows how many judgments emitted it, and
    `importance_mean` / `importance_max` summarize the importances.
  - The key is **absent** in runtime mode (no recs were emitted) — the
    dashboard guards on its presence.

- **Static dashboard** —
  `clear_results.html` (or unzip `ui_results.zip`). When recommendations
  are present, a **"SPARC Recommendations"** section appears below the
  per-agent issues table, organized into a System Prompt block and a
  Per-Tool Suggestions block.

## Directory layout (after a run)

```
scripts/
├── runs/
│   └── <bench>/
│       ├── input/
│       │   └── *.csv                       # converted traces
│       └── output_<track>_<mode>/
│           └── <bench>/
│               ├── clear_results.html
│               ├── clear_results.json
│               └── clear_results/
│                   └── <agent>/
│                       └── tool_calls/
│                           └── analysis_results_*.csv
└── tool_specs/
    └── <slug>[_<subset>].json
```

## Common pitfalls

- **`response` field too large for csv.DictReader** — the unified row
  format embeds the full assistant message + tool calls into a single
  cell, so consumers must call `csv.field_size_limit(sys.maxsize)` before
  reading. CLEAR's own readers already do this; only ad-hoc scripts care.
- **`appworld 0.1.3.post1 requires pydantic<2.0.0` warning during install**
  — comes from a transitive dep that pins old pydantic; ignore. CLEAR and
  altk-boost both run on pydantic 2.x.
- **`runtime_pipeline: true` with empty `recommendations` key in JSON** —
  expected. Recs only travel in eval mode (`runtime_pipeline: false`).
