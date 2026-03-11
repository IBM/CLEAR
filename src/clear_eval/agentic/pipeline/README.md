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
        ├── <agent_2>/
        └── ui_results.zip  # Upload this to the dashboard
```

## Viewing Results

After running the pipeline:

1. Start the dashboard:
   ```bash
   streamlit run src/clear_eval/agentic/dashboard/launch_dashboard.py
   ```

2. Upload the generated `ui_results.zip` file

## Environment Setup

Set environment variables for your provider:

**OpenAI:**
```bash
export OPENAI_API_KEY=your_key
```

**WatsonX:**
```bash
export WATSONX_APIKEY=your_key
export WATSONX_PROJECT_ID=your_project_id
export WATSONX_URL=your_url
```

**Azure:**
```bash
export AZURE_OPENAI_API_KEY=your_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
```
