# Tool specs for the trace converter

Each file in this directory is a JSON array of OpenAI function-calling tool
definitions used by `../traces_to_clear_csv.py` to populate the `api_spec`
column of the emitted CLEAR CSVs.

## File naming

Filename is `<benchmark_slug>_<subset>.json` when both exist, or just
`<benchmark_slug>.json` otherwise. The converter reads
`exgentic.benchmark.slug_name` and `exgentic.benchmark.subset` from each trace's
session span and looks up a spec with this precedence:

1. `<slug>_<subset>.json`
2. `<slug>.json`
3. Fall back to session-attribute-derived specs (name + description only; empty
   parameters object).

Tool names that appear in a trace's `exgentic.session.action.*` attributes but
are missing from the loaded spec file are merged in with a minimal
name/description/empty-params entry, so the spec covers every tool the agent
actually had access to.

## Bundled files

- `tau2_retail.json` — from
  https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/domains/retail/tools.py
- `tau2_airline.json` — from
  https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/domains/airline/tools.py

## Adding a new benchmark

### AppWorld

AppWorld ships ready-made OpenAI function-calling JSON at
`data/api_docs/function_calling/<app_name>.json` inside the
[StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld) repo. To use
it here, concatenate all `<app_name>.json` arrays from the apps the agent had
access to into a single top-level array and save as
`appworld.json` (or `appworld_<subset>.json`).

### Any other tool-using benchmark

Convert each tool into the OpenAI function-calling schema:

```json
{
  "type": "function",
  "function": {
    "name": "<tool_name>",
    "description": "<one-line or paragraph description>",
    "parameters": {
      "type": "object",
      "properties": { "<arg>": { "type": "string", "description": "..." } },
      "required": ["<arg>"]
    }
  }
}
```

and drop the JSON array into this directory with the matching benchmark slug.

The converter intentionally does NOT fetch specs from the network at runtime —
everything it needs is in this folder.
