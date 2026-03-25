# CLEAR for Agentic Workflows

This module provides evaluation pipelines for multi-agent systems, from raw traces to interactive analysis dashboards.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Input Data Formats](#input-data-formats)
4. [Running the Pipeline](#running-the-pipeline)
5. [Configuration Reference](#configuration-reference)
6. [Output Structure](#output-structure)
7. [Dashboard](#dashboard)
8. [Examples](#examples)

---

## Overview

CLEAR for Agentic Workflows provides two complementary analysis modes:

| Analysis Type | Description | Use Case |
|---------------|-------------|----------|
| **Step-by-Step** | Evaluates individual agent interactions using CLEAR methodology | Understanding agent-level quality issues |
| **Full Trajectory** | Evaluates complete task trajectories (success, quality, rubric-based) | Assessing overall task completion and quality |

Both analyses share a centralized CSV intermediate representation and produce results viewable in an interactive dashboard.

### Supported Frameworks

The pipeline includes built-in preprocessors for the following combinations:

| Agent Framework | Observability Platform | Status |
|-----------------|------------------------|--------|
| LangGraph | MLflow | Supported |
| LangGraph | Langfuse | Supported |
| CrewAI | Langfuse | Supported |

For other frameworks or observability platforms, preprocess your traces to the CSV format described below and use `--from-raw-traces false`.

---

## Quick Start

### 1. Install from Source

```bash
git clone <repository-url>
cd CLEAR
pip install -e .
```

### 2. Set Up Credentials

Configure your LLM provider credentials. See the main [README](../../README.md#supported-providers-and-credentials) for provider-specific setup instructions.

### 3. Run the Pipeline

**Using Python module:**
```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --data-dir data/my_traces \
    --results-dir results \
    --from-raw-traces true \
    --eval-model-name gpt-4o \
    --provider openai
```

**Using CLI command:**
```bash
run-clear-agentic-eval \
    --data-dir data/my_traces \
    --results-dirr results \
    --from-raw-traces true \
    --eval-model-name gpt-4o \
    --provider openai
```

### 4. Launch the Dashboard

**Using Python module:**
```bash
python -m clear_eval.agentic.dashboard.launch_dashboard
```

**Using CLI command:**
```bash
run-clear-agentic-dashboard
```

Upload the generated `unified_ui_results.zip` from your results directory.

---

## Input Data Formats

### Option 1: Raw JSON Traces

Use `--from-raw-traces true` to process JSON trace files from your observability platform.

**Directory Structure:**
```
my_traces/
├── trace_001.json
├── trace_002.json
└── ...
```

**Trace Format (MLflow example):**
```json
{
  "trace_id": "tr-abc123",
  "spans": [
    {
      "span_id": "span-001",
      "name": "agent_name",
      "span_type": "CHAT_MODEL",
      "parent_span_id": null,
      "inputs": {"messages": [...]},
      "outputs": {"content": "..."},
      "attributes": {
        "llm.model_name": "gpt-4o",
        "llm.token_count.prompt": 100,
        "llm.token_count.completion": 50
      }
    }
  ]
}
```

### Option 2: Preprocessed CSV Files

Use `--from-raw-traces false` to use existing CSV files directly.

**CSV Schema:**

Each row represents a single LLM call within a trajectory. The `Name` column identifies the component (agent/tool) that made the call—CLEAR analysis is performed separately for each unique component.

| Column | Type | Required | Description                                               |
|--------|------|----------|-----------------------------------------------------------|
| `id` | str | Yes      | Unique row identifier: `{task_id}_{step}`                 |
| `Name` or `agent_name` | str | Yes      | Component name (CLEAR analyzes each component separately) |
| `task_id` | str | Yes      | Trajectory identifier                                     |
| `step_in_trace_general` | int | Yes      | Step number in trajectory (0-indexed)                     |
| `step_in_trace_node` | int | No       | Step number within this component                         |
| `model_input` | str | Yes      | Input to the LLM call                                     |
| `response` | str | Yes      | Output from the LLM call                                  |
| `tool_or_agent` | str | No       | Type: `tool` or `agent`                                   |
| `intent` | str | No       | Original user query                                       |
| `meta_data` | json | No       | Additional metadata (tokens, latency)                     |
| `traj_score` | float | No       | Ground truth trajectory score (0-1)                       |

---

## Running the Pipeline

### Both Analyses (Default)

Run both step-by-step and full trajectory analysis:

```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --data-dir data/my_traces \
    --results-dir results \
    --from-raw-traces true \
    --eval-model-name gpt-4o \
    --provider openai
```

### Step-by-Step Analysis Only

```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --data-dir data/my_traces \
    --results-dir results \
    --from-raw-traces true \
    --run-step-by-step true \
    --run-full-trajectory false \
    --eval-model-name gpt-4o \
    --provider openai
```

### Full Trajectory Evaluation Only

```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --data-dir data/my_traces \
    --results-dir results \
    --from-raw-traces true \
    --run-step-by-step false \
    --run-full-trajectory true \
    --eval-types task_success full_trajectory \
    --eval-model-name gpt-4o \
    --provider openai
```

### Using a Configuration File

```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --agentic-config-path my_config.yaml
```

---

## Configuration Reference

See [`pipeline/setup/default_agentic_config.yaml`](pipeline/setup/default_agentic_config.yaml) for all available options.

### Required Parameters

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| `data_dir` | `--data-dir` | Input directory (raw traces or trajectory CSVs) |
| `results_dir` | `--results-dir` | Output directory for results |
| `eval_model_name` | `--eval-model-name` | Model identifier (e.g., `gpt-4o`) |

### Pipeline Control

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `run_step_by_step` | `--run-step-by-step` | `true` | Enable step-by-step analysis |
| `run_full_trajectory` | `--run-full-trajectory` | `true` | Enable trajectory evaluation |
| `from_raw_traces` | `--from-raw-traces` | `false` | `true` = process JSON traces, `false` = use CSV files |

### Preprocessing (when `from_raw_traces=true`)

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `agent_framework` | `--agent-framework` | `langgraph` | Agent framework (`langgraph`, `crewai`) |
| `observability_framework` | `--observability-framework` | `langfuse` | Platform (`mlflow`, `langfuse`) |

### Full Trajectory Evaluation

| Parameter | CLI Flag | Default                      | Description |
|-----------|----------|------------------------------|-------------|
| `eval_types` | `--eval-types` | `full_trajectory`, `rubrics` | Evaluations: `task_success`, `full_trajectory`, `rubric`, `all` |
| `generate_rubrics` | `--generate-rubrics` | `true`                       | Generate rubrics before evaluation |
| `rubric_dir` | `--rubric-dir` | None                         | Path to existing rubrics |
| `clear_analysis_types` | `--clear-analysis-types` | `issues`                     | CLEAR analyses: `root_cause`, `issues`, `all`, `none` |

**Note on Rubric Evaluation:**
- Requires trajectories to have a valid `intent` field
- Automatically skipped if no trajectories have intent data
- Runs only on trajectories with valid (non-empty) intent

### Model Configuration

| Parameter | CLI Flag | Default | Description                                                                |
|-----------|----------|---------|----------------------------------------------------------------------------|
| `provider` | `--provider` | `openai` | LLM provider                                                               |
| `inference_backend` | `--inference-backend` | `litellm` | Backend: `litellm`, `langchain`, `endpoint`                                |
| `eval_model_params` | `--eval-model-params` | `{}` | Model parameters (JSON)                                                    |
| `context_tokens` | `--context-tokens` | None | Context window; truncates long trajectories in full trajectory evaluation  |

### Execution

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `run_name` | `--run-name` | timestamp | Unique identifier for this run |
| `overwrite` | `--overwrite` | `true` | Overwrite existing results |
| `max_workers` | `--max-workers` | `10` | Parallel workers |

### Configuration Precedence

1. Default config ([`pipeline/setup/default_agentic_config.yaml`](pipeline/setup/default_agentic_config.yaml))
2. User config file (`--agentic-config-path`)
3. CLI arguments (highest priority)

---

## Output Structure

```
results/
└── <run_name>/
    ├── traces_data/              # Preprocessed CSV files (if from_raw_traces=true)
    │   ├── trace_001.csv
    │   └── ...
    │
    ├── step_by_step/             # Step-by-step CLEAR results
    │   ├── clear_data/           # CLEAR format grouped by agent
    │   │   ├── agent_1.csv
    │   │   └── statistics.json
    │   ├── clear_results/        # CLEAR analysis per agent
    │   │   ├── agent_1/
    │   │   │   ├── analysis_results_*.csv
    │   │   │   ├── per_record_evaluations_*.csv
    │   │   │   └── shortcoming_list_*_dedup.json
    │   │   └── agent_2/
    │   └── clear_results.json    # Comprehensive JSON output
    │
    ├── full_trajectory/          # Full trajectory results
    │   ├── task_success/         # Binary success evaluation
    │   ├── full_trajectory/      # Multi-dimensional quality scores
    │   ├── rubric_generation/    # Generated rubrics (if enabled)
    │   ├── rubric/               # Rubric-based evaluation
    │   └── clear_analysis/       # CLEAR analysis of results
    │       ├── root_cause/       # Analysis of failure causes
    │       └── issues/           # Analysis of quality issues
    │
    ├── unified_ui_results.zip    # Dashboard input
    └── pipeline_summary.json     # Execution metadata
```

### Key Output Files

| File | Description |
|------|-------------|
| `unified_ui_results.zip` | Dashboard-ready package containing all results |
| `clear_results.json` | Comprehensive JSON with issues mapped to spans |
| `pipeline_summary.json` | Execution summary and configuration |
| `analysis_results_*.csv` | Aggregated CLEAR analysis results |
| `shortcoming_list_*_dedup.json` | Deduplicated list of discovered issues |

---

## Dashboard

See [dashboard/README_DASHBOARD.md](dashboard/README_DASHBOARD.md) for detailed documentation.

### Quick Launch

```bash
python -m clear_eval.agentic.dashboard.launch_dashboard
```

Upload `unified_ui_results.zip` from your results directory.

### Features

- **Workflow View**: Interactive graph of agents and transitions
- **Node Analysis**: CLEAR analysis results per agent
- **Trajectory Explorer**: Browse individual trajectories with filtering
- **Path Analysis**: Common path patterns and success/failure analysis
- **Temporal Analysis**: Agent position and score progression
- **Score Prediction**: ROC analysis for trajectory success prediction

**Note:** The dashboard requires at least step-by-step analysis results. Full trajectory analysis adds trajectory-level scores and evaluations.

---

## Examples

### Complete Analysis from Raw Traces

```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --data-dir data/mlflow_traces \
    --results-dir results \
    --from-raw-traces true \
    --agent-framework langgraph \
    --observability-framework mlflow \
    --run-name my_experiment \
    --eval-model-name gpt-4o \
    --provider openai

# Launch dashboard
python -m clear_eval.agentic.dashboard.launch_dashboard
# Upload: results/my_experiment/unified_ui_results.zip
```

### Rubric-Based Evaluation

```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --data-dir data/traces \
    --results-dir results \
    --from-raw-traces true \
    --run-step-by-step false \
    --run-full-trajectory true \
    --eval-types rubric \
    --generate-rubrics true \
    --eval-model-name gpt-4o \
    --provider openai
```

### Using Existing Rubrics

```bash
python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --data-dir data/traces \
    --results-dir results \
    --eval-types rubric \
    --rubric-dir path/to/rubrics \
    --eval-model-name gpt-4o
```

### Using a Configuration File

```bash
cp src/clear_eval/agentic/pipeline/setup/default_agentic_config.yaml my_config.yaml
# Edit my_config.yaml

python -m clear_eval.agentic.pipeline.run_clear_agentic_eval \
    --agentic-config-path my_config.yaml
```
