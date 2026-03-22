# CLEAR for Agentic Workflows

This module provides evaluation pipelines for multi-agent systems, from raw traces to interactive analysis dashboards.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Input Data Formats](#input-data-formats)
4. [Running the Pipeline](#running-the-pipeline)
5. [Output Structure](#output-structure)
6. [Dashboard](#dashboard)
7. [Configuration Reference](#configuration-reference)
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

| Agent Framework | Observability Platform | Status |
|-----------------|------------------------|--------|
| LangGraph | MLflow | Supported |
| LangGraph | Langfuse | Supported |
| CrewAI | Langfuse | Supported |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install clear-eval
```

### 2. Set Up Credentials

Configure your LLM provider credentials. See the main [README](../../README.md#supported-providers-and-credentials) for provider-specific setup instructions.

### 3. Run the Unified Pipeline

```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-input-dir data/my_traces \
    --agentic-output-dir results \
    --from-raw-traces true \
    --eval-model-name gpt-4o \
    --provider openai
```

### 4. Launch the Dashboard

```bash
python -m clear_eval.agentic.dashboard.launch_dashboard
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

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `id` | str | Yes | Unique identifier: `{task_id}_{step}` |
| `Name` or `agent_name` | str | Yes | Agent/node name |
| `task_id` | str | Yes | Trajectory identifier |
| `step_in_trace_general` | int | Yes | Step number in trajectory (0-indexed) |
| `step_in_trace_node` | int | Yes | Step number within this agent |
| `model_input` | str | Yes | Input to the agent |
| `response` | str | Yes | Output from the agent |
| `tool_or_agent` | str | Yes | Type: `tool` or `agent` |
| `intent` | str | No | Original user query |
| `meta_data` | json | No | Additional metadata (tokens, latency) |
| `traj_score` | float | No | Ground truth trajectory score (0-1) |

---

## Running the Pipeline

### Unified Pipeline (Recommended)

The unified pipeline runs both step-by-step and full trajectory analysis, creating a shared CSV intermediate representation.

```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-config-path config.yaml
```

**Example config.yaml:**
```yaml
# Input/Output
agentic_input_dir: data/my_traces
agentic_output_dir: results
run_name: experiment_001

# Pipeline Control
run_step_by_step: true
run_full_trajectory: true
from_raw_traces: true

# Preprocessing (when from_raw_traces=true)
agent_framework: langgraph       # langgraph | crewai
observability_framework: mlflow  # mlflow | langfuse

# Full Trajectory Evaluation
eval_types:
  - task_success
  - full_trajectory
  - rubric
generate_rubrics: true
clear_analysis_types:
  - root_cause    # Analyze task_success failures
  - issues        # Analyze full_trajectory issues

# Model Configuration
provider: openai
eval_model_name: gpt-4o
eval_model_params:
  temperature: 0.0
  max_tokens: 8096

# Execution
concurrency: 10
overwrite: true
```

### Step-by-Step Analysis Only

```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-input-dir data/my_traces \
    --agentic-output-dir results \
    --from-raw-traces true \
    --run-step-by-step \
    --no-run-full-trajectory \
    --eval-model-name gpt-4o \
    --provider openai
```

### Full Trajectory Evaluation Only

```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-input-dir data/my_traces \
    --agentic-output-dir results \
    --from-raw-traces true \
    --no-run-step-by-step \
    --run-full-trajectory \
    --eval-types task_success full_trajectory \
    --eval-model-name gpt-4o \
    --provider openai
```

### Alternative: Direct Step-by-Step from CSVs

For running step-by-step analysis directly on preprocessed CSV files:

```bash
python -m clear_eval.agentic.pipeline.run_clear_on_traj_data \
    --traces-data-dir data/preprocessed_csvs \
    --agentic-output-dir results \
    --eval-model-name gpt-4o \
    --provider openai
```

---

## Output Structure

```
results/
└── clear_results/
    └── <run_name>/
        ├── traces_data/              # Centralized CSV intermediate representation
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
        │   ├── task_success/
        │   │   └── results.csv
        │   ├── full_trajectory/
        │   │   └── results.csv
        │   ├── rubric_generation/    # Generated rubrics (if enabled)
        │   ├── rubric/
        │   │   └── results.csv
        │   └── clear_analysis/
        │       ├── root_cause/       # CLEAR analysis of failures
        │       └── issues/           # CLEAR analysis of quality issues
        │
        ├── unified_ui_results.zip    # Dashboard input (both analyses)
        └── pipeline_summary.json     # Execution metadata
```

### Key Output Files

| File | Description |
|------|-------------|
| `unified_ui_results.zip` | Dashboard-ready package containing all results |
| `clear_results.json` | Comprehensive JSON with issues mapped to spans |
| `pipeline_summary.json` | Execution summary and configuration |
| `analysis_results_*.csv` | Aggregated CLEAR analysis results |
| `per_record_evaluations_*.csv` | Individual evaluation records |
| `shortcoming_list_*_dedup.json` | Deduplicated list of discovered issues |

---

## Dashboard

See [README_DASHBOARD.md](dashboard/README_DASHBOARD.md) for detailed dashboard documentation.

### Quick Launch

```bash
python -m clear_eval.agentic.dashboard.launch_dashboard
```

Then upload `unified_ui_results.zip` from your results directory.

### Dashboard Features

- **Workflow View**: Interactive graph of agents and transitions
- **Node Analysis**: CLEAR analysis results per agent
- **Trajectory Explorer**: Browse individual trajectories with filtering
- **Path Analysis**: Common path patterns and success/failure analysis
- **Temporal Analysis**: Agent position and score progression
- **Score Prediction**: ROC analysis for trajectory success prediction

---

## Configuration Reference

### Pipeline Control

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `run_step_by_step` | `--run-step-by-step` | `true` | Enable step-by-step analysis |
| `run_full_trajectory` | `--run-full-trajectory` | `true` | Enable trajectory evaluation |
| `from_raw_traces` | `--from-raw-traces` | `false` | Process JSON traces vs. use CSV files |

### Preprocessing (when `from_raw_traces=true`)

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `agent_framework` | `--agent-framework` | `langgraph` | Agent framework (`langgraph`, `crewai`) |
| `observability_framework` | `--observability-framework` | `mlflow` | Platform (`mlflow`, `langfuse`) |
| `separate_tools` | `--separate-tools` | `false` | Emit separate rows for tool calls |

### Full Trajectory Evaluation

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `eval_types` | `--eval-types` | `all` | Evaluations: `task_success`, `full_trajectory`, `rubric`, `all` |
| `generate_rubrics` | `--generate-rubrics` | `false` | Generate rubrics before evaluation |
| `rubric_dir` | `--rubric-dir` | None | Path to existing rubrics |
| `clear_analysis_types` | `--clear-analysis-types` | `all` | CLEAR analyses: `root_cause`, `issues`, `all`, `none` |

### Model Configuration

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `provider` | `--provider` | `openai` | LLM provider |
| `eval_model_name` | `--eval-model-name` | Required | Model identifier |
| `eval_model_params` | `--eval-model-params` | `{}` | Model parameters (JSON) |
| `context_tokens` | `--context-tokens` | None | Model context window |

### Execution

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `overwrite` | `--overwrite` | `true` | Overwrite existing results |
| `concurrency` | `--concurrency` | `10` | Parallel workers |
| `max_files` | `--max-files` | None | Limit files (for testing) |

### Configuration Precedence

1. Default config (`setup/unified_config.yaml`)
2. User config file (`--agentic-config-path`)
3. CLI arguments (highest priority)

---

## Examples

### Complete Analysis from Raw MLflow Traces

```bash
# Run pipeline
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-input-dir data/mlflow_traces \
    --agentic-output-dir results \
    --from-raw-traces true \
    --agent-framework langgraph \
    --observability-framework mlflow \
    --run-name mlflow_experiment \
    --eval-model-name gpt-4o \
    --provider openai

# Launch dashboard
python -m clear_eval.agentic.dashboard.launch_dashboard
# Upload: results/clear_results/mlflow_experiment/unified_ui_results.zip
```

### Rubric-Based Evaluation with Generated Rubrics

```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-input-dir data/traces \
    --agentic-output-dir results \
    --from-raw-traces true \
    --no-run-step-by-step \
    --run-full-trajectory \
    --eval-types rubric \
    --generate-rubrics \
    --eval-model-name gpt-4o \
    --provider openai
```

### Using Existing Rubrics

```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-input-dir data/traces \
    --agentic-output-dir results \
    --from-raw-traces true \
    --eval-types rubric \
    --rubric-dir path/to/rubrics \
    --eval-model-name gpt-4o
```

### Test Run with Limited Files

```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-input-dir data/traces \
    --agentic-output-dir results \
    --from-raw-traces true \
    --max-files 10 \
    --eval-model-name gpt-4o
```

### Full Configuration File

```yaml
# config.yaml - Complete configuration example

# Input/Output
agentic_input_dir: data/my_experiment
agentic_output_dir: results
run_name: full_analysis_001

# Pipeline Control
run_step_by_step: true
run_full_trajectory: true
from_raw_traces: true

# Preprocessing
agent_framework: langgraph
observability_framework: mlflow
separate_tools: false

# Full Trajectory Evaluation
eval_types:
  - task_success
  - full_trajectory
  - rubric
generate_rubrics: true
clear_analysis_types:
  - root_cause
  - issues

# Model Configuration
provider: openai
eval_model_name: gpt-4o
eval_model_params:
  temperature: 0.0
  max_tokens: 8096

# CLEAR Settings
agent_mode: true
success_threshold: 0.7
pass_criteria: avg

# Execution
overwrite: true
concurrency: 10
context_tokens: 128000
```

Run with:
```bash
python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \
    --agentic-config-path config.yaml
```
