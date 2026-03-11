# CLEAR Pipeline for Agentic Workflows

This directory contains two pipelines for running CLEAR analysis on agentic workflows:

1. **Full Pipeline** (`run_clear_pipeline.py`): Process raw traces → trajectory data → CLEAR results
2. **Analysis Pipeline** (`run_clear_on_traj_data.py`): Run CLEAR on preprocessed trajectory data

## Quick Start

### Using a Config File (Recommended)

Create a YAML config file (see `setup/config_example.yaml`) and run:

```bash
# Full pipeline (from raw traces)
python -m clear_eval.agentic.pipeline.run_clear_pipeline \
    --agentic-config-path my_config.yaml

# Analysis pipeline (from preprocessed trajectory data)
python -m clear_eval.agentic.pipeline.run_clear_on_traj_data \
    --agentic-config-path my_config.yaml
```

### Config File with CLI Overrides

CLI arguments override config file values:

```bash
python -m clear_eval.agentic.pipeline.run_clear_pipeline \
    --agentic-config-path my_config.yaml \
    --eval-model-name Azure/gpt-4.1
```

## Configuration

Use `--agentic-config-path` to specify a YAML configuration file. All parameters are at the top level.

### Example Config File

```yaml
# Agentic pipeline configuration
traces_input_dir: /path/to/raw/traces       # Required for full pipeline
agentic_output_dir: /path/to/output         # Required
agent_framework: langgraph                  # langgraph | crewai
separate_tools: false                       # Separate tool calls from text responses
overwrite: true                             # Overwrite existing results

# CLEAR evaluation configuration
provider: openai                            # LLM provider
eval_model_name: Azure/gpt-4.1              # Model for evaluation
eval_model_params:
  temperature: 0.0
  max_tokens: 8096

resume_enabled: false                       # Reuse cached results
evaluation_criteria: null                   # Custom criteria dict
run_name: null                              # Unique run identifier
```

### Configuration Parameters

| Parameter | CLI Argument | Default | Description |
|-----------|--------------|---------|-------------|
| `traces_input_dir` | `--traces-input-dir` | *required* | Directory containing raw trace JSON files (full pipeline only) |
| `traces_data_dir` | `--traces-data-dir` | *required* | Directory containing trajectory CSVs (analysis pipeline only) |
| `agentic_output_dir` | `--agentic-output-dir` | *required* | Output directory for all results |
| `agent_framework` | `--agent-framework` | `langgraph` | Agent framework: `langgraph` or `crewai` |
| `separate_tools` | `--separate-tools` | `false` | Emit separate rows for tool calls vs text responses |
| `overwrite` | `--overwrite` | `true` | Overwrite existing CLEAR results |
| `provider` | `--provider` | `watsonx` | LLM provider (e.g., `openai`, `watsonx`, `rits`) |
| `eval_model_name` | `--eval-model-name` | `openai/gpt-oss-120b` | Model name for CLEAR evaluation |
| `eval_model_params` | `--eval-model-params` | `{temperature: 0.0, max_tokens: 8096}` | Model parameters |
| `resume_enabled` | `--resume-enabled` | `false` | Reuse intermediate results from previous runs |
| `evaluation_criteria` | `--evaluation-criteria` | `null` | Custom evaluation criteria dict |
| `run_name` | `--run-name` | `null` | Unique run identifier (appears in file names) |

**Note:** Use `snake_case` in YAML files and `--kebab-case` for CLI arguments.

### Configuration Precedence

1. `setup/default_config.yaml` (lowest)
2. User config file via `--agentic-config-path`
3. CLI arguments (highest)

## Output Structure

```
agentic_output_dir/
├── traces_data/            # Trajectory CSV files (one per trace)
├── clear_data/             # CLEAR format data (grouped by agent)
│   ├── agent_1.csv
│   ├── agent_2.csv
│   └── statistics.json
└── clear_results/          # CLEAR analysis results
    └── <judge-model>/
        ├── <agent_1>/
        │   ├── analysis_results_*.csv
        │   └── shortcoming_list_*_dedup.json
        ├── <agent_2>/
        └── clear_results.json  # Comprehensive JSON output
```

## JSON Output Format

The pipeline generates `clear_results.json` with all issues mapped to spans:

```json
{
  "metadata": {
    "pipeline_version": "1.0",
    "created_at": "2024-03-11T10:30:00",
    "statistics": {
      "total_traces": 10,
      "total_agents": 3,
      "total_issues_discovered": 12,
      "total_interactions_analyzed": 150,
      "total_interactions_with_issues": 45,
      "total_interactions_no_issues": 105
    }
  },
  "agents": {
    "classify_log": {
      "agent_summary": {
        "total_interactions": 50,
        "avg_score": 0.72,
        "interactions_with_issues": 15,
        "interactions_no_issues": 35,
        "issues_count": {"issue_1": 8, "issue_2": 7}
      },
      "issues_catalog": {
        "issue_1": "Incomplete reasoning in classification",
        "issue_2": "Missing confidence scores"
      },
      "issues": [
        {
          "issue_id": "issue_1",
          "issue_text": "Incomplete reasoning in classification",
          "occurrence_count": 8,
          "occurrences": [
            {
              "trace_id": "tr-7d2e6a006ccb9332bd146903ff2a7cba",
              "span_reference": {
                "span_id": "42918ea30be390fb",
                "span_name": "Completions",
                "span_type": "CHAT_MODEL",
                "tool_or_agent": "agent",
                "parent_span_id": "7f8d1c10018586dc",
                "step_in_trace": 3
              },
              "input_output_pair": {
                "id": "tr-7d2e6a006ccb9332bd146903ff2a7cba_3",
                "model_input": "Analyze this server log...",
                "response": "{\"classification\":\"warning\"...}",
                "score": 0.65
              },
              "evaluation": {
                "evaluation_text": "The model correctly classified...",
                "evaluation_summary": "Correct classification but incomplete reasoning"
              },
              "span_metadata": {
                "duration_ms": 1200,
                "status": "OK",
                "model": "Azure/gpt-4.1",
                "tokens": {"prompt": 272, "completion": 34, "total": 306}
              }
            }
          ]
        }
      ],
      "no_issues": [
        { /* same span structure for interactions without issues */ }
      ]
    }
  }
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `agents.<name>.issues_catalog` | Issues discovered for this agent (unique per agent) |
| `agents.<name>.issues` | List of issues with all occurrences mapped to spans |
| `agents.<name>.no_issues` | Spans that had no issues detected |
| `span_reference.span_type` | Original MLflow/Langfuse span type (CHAT_MODEL, LLM, AGENT, TOOL, etc.) |
| `span_reference.tool_or_agent` | Preprocessing classification: "tool" (tool call) or "agent" (text response) |
| `span_metadata` | Span-level info from trace (duration, model, tokens, etc.) |

## Building JSON Results Separately

You can also build the JSON output from existing CLEAR results:

```bash
python -m clear_eval.agentic.pipeline.build_json_results \
    --judge-results-dir /path/to/clear_results/<judge-model> \
    --traces-data-dir /path/to/traces_data \
    --output-dir /path/to/output \
    --output-filename clear_results.json
```

## Environment Setup

The pipeline uses [LiteLLM](https://docs.litellm.ai/) for model inference, which supports 100+ LLM providers.

Set environment variables according to your chosen provider. You are responsible for configuring credentials based on your provider's requirements. See the [LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) for details.

#### Examples:

OpenAI:
```bash
export OPENAI_API_KEY=your_key
```

**WatsonX:**
```bash
export WATSONX_APIKEY=your_key
export WATSONX_PROJECT_ID=your_project_id
export WATSONX_URL=your_url
```
