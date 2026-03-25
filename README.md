# CLEAR: Comprehensive LLM Error Analysis and Reporting

**CLEAR** is an open-source toolkit for **LLM error analysis** using an LLM-as-a-Judge approach.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/clear-eval.svg)](https://pypi.org/project/clear-eval/)

---

## 🎯 What is CLEAR?

CLEAR provides systematic error analysis for:
- **Single LLM Responses** — Analyze quality issues in model outputs for tasks like Q&A, summarization, and generation
- **Agentic Workflows** — Evaluate complex workflows with multiple components, tool usage, and multi-step task trajectories

It combines automated LLM-as-a-judge evaluation with interactive dashboards to help you:
- Identify recurring error patterns across your dataset
- Quantify issue frequencies and severity
- Drill down into specific failure cases
- Prioritize improvements based on data-driven insights

## ⚙️ How It Works

CLEAR operates in two phases:

1. **Analysis** — Generates per-instance textual feedback, identifies system-level error categories, and quantifies their frequencies.
2. **Interactive Dashboard** — Explore aggregate visualizations, apply dynamic filters, and drill down into individual failure examples.

---

## 🔀 Two Analysis Modes

CLEAR supports two distinct analysis modes, each with its own pipeline, dashboard, and documentation:

### 📝 LLM Analysis

Evaluate standard LLM outputs — generation quality, correctness, and recurring error patterns. Provide a CSV with prompts and responses, and CLEAR will score each instance, generate textual critiques, and surface system-level issues.

| | |
|---|---|
| **Input** | CSV with model inputs and responses |
| **Output** | Per-record scores, evaluation text, aggregated issue categories |
| **Dashboard** | Streamlit-based interactive explorer |

> 📖 **[Full LLM Analysis Guide →](docs/ANALYSIS_README.md)**

### 🤖 Agentic Analysis

Evaluate multi-agent system trajectories — step-by-step agent interactions and full trajectory analysis.
Supports traces from LangGraph, CrewAI, and other frameworks via MLflow or Langfuse.

| | |
|---|---|
| **Input** | Raw JSON traces or preprocessed trajectory CSVs |
| **Output** | Per-step CLEAR analysis, trajectory-level scores, rubric evaluations |
| **Dashboard** | NiceGUI-based workflow visualization with path and temporal analysis |

> 📖 **[Agentic Workflows Guide →](src/clear_eval/agentic/README.md)** | **[Agentic Dashboard Guide →](src/clear_eval/agentic/dashboard/README_DASHBOARD.md)**

---

## ✨ Key Features

| | |
|---|---|
| 🧑‍⚖️ **LLM-as-a-Judge** | Automated evaluation for any text generation task |
| 🤖 **Agentic Workflows** | Evaluate multi-agent trajectories, tool usage, and task completion |
| 🔌 **Multiple Backends** | LangChain, LiteLLM (100+ providers), or direct HTTP endpoints |
| 🧩 **External Judges** | Plug in custom evaluation functions |
| 📊 **Interactive Dashboards** | Standard and agentic-specific visualizations |
| 🛠️ **Flexible Configuration** | YAML config files, CLI flags, or Python API |

---

## 📦 Installation

**Requires Python 3.10+**

#### Option 1: pip (recommended)

```bash
pip install clear-eval
```

#### Option 2: From source (for development)

```bash
git clone https://github.com/IBM/CLEAR.git
cd CLEAR
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## 🚀 Quick Start

### 1. Set your provider credentials

CLEAR requires a supported LLM provider. Set the appropriate environment variables for your provider (e.g., `OPENAI_API_KEY` for OpenAI). See the [Providers and Credentials Guide](docs/PROVIDERS.md) for all supported providers and backends.

### 2. Run on sample data

With no data path specified, CLEAR runs on a built-in [GSM8K sample dataset](https://github.com/IBM/CLEAR/blob/main/src/clear_eval/sample_data/gsm8k/gsm8k_default_predictions.csv) using [default settings](https://github.com/IBM/CLEAR/blob/main/src/clear_eval/pipeline/setup/default_config.yaml):

```bash
run-clear-eval-analysis --provider openai --eval-model-name gpt-4o
```

Results are saved to `results/gsm8k/sample_output/`.

### 3. Run on your own data

```bash
run-clear-eval-analysis \
    --provider openai \
    --eval-model-name gpt-4o \
    --data-path path/to/your_data.csv \
    --output-dir results/my_run/ \
    --run-name my_run
```

Your CSV should contain at minimum `id`, `model_input`, and `response` columns. See the [LLM Analysis Guide](docs/ANALYSIS_README.md#input-data-format) for the full input format specification.

### 4. View results

```bash
run-clear-eval-dashboard
```

Upload the generated ZIP file from the results directory to explore issues, scores, and individual examples.

---

## 🔍 Usage Overview

### 📝 LLM Analysis (CLI)

```bash
# Full pipeline
run-clear-eval-analysis --provider openai --eval-model-name gpt-4o --config_path path/to/config.yaml

# Evaluation only (using existing responses)
run-clear-eval-evaluation --provider openai --eval-model-name gpt-4o --config_path path/to/config.yaml
```

### 📝 LLM Analysis (Python)

```python
from clear_eval.analysis_runner import run_clear_eval_analysis

run_clear_eval_analysis(
    run_name="my_run",
    provider="openai",
    data_path="my_data.csv",
    eval_model_name="gpt-4o",
    output_dir="results/",
)
```

### 🤖 Agentic Analysis

```bash
run-clear-agentic-eval \
    --data-dir data/my_traces \
    --results-dir results \
    --from-raw-traces true \
    --eval-model-name gpt-4o \
    --provider openai

# Launch agentic dashboard
run-clear-agentic-dashboard
```

See the [Agentic Workflows Guide](src/clear_eval/agentic/README.md) for full details.

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| 📝 [**LLM Analysis Guide**](docs/ANALYSIS_README.md) | Full pipeline reference — input formats, CLI arguments, configuration, and external judges |
| 🤖 [**Agentic Workflows Guide**](src/clear_eval/agentic/README.md) | Multi-agent evaluation — trace preprocessing, step-by-step and trajectory analysis, configuration reference |
| 📊 [**Agentic Dashboard Guide**](src/clear_eval/agentic/dashboard/README_DASHBOARD.md) | Dashboard features — workflow view, node analysis, trajectory explorer, path and temporal analysis |
| 🔑 [**Providers and Credentials**](docs/PROVIDERS.md) | Inference backends (LangChain, LiteLLM, Endpoint), provider setup, and configuration examples |

---

## 🔑 Supported Providers

| Provider | Backend | Credentials |
|----------|---------|-------------|
| OpenAI | LangChain, LiteLLM, Endpoint | `OPENAI_API_KEY` |
| WatsonX | LangChain, LiteLLM, Endpoint | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID` |
| Anthropic | LiteLLM | `ANTHROPIC_API_KEY` |
| AWS Bedrock | LiteLLM | AWS credentials |
| Google Vertex AI | LiteLLM | GCP credentials |
| [100+ more](https://docs.litellm.ai/docs/providers) | LiteLLM | Provider-specific |

See the **[Providers and Credentials Guide](docs/PROVIDERS.md)** for backend configuration details and examples.

---

## 🗂️ Project Structure

```
CLEAR/
├── README.md                              # This file
├── src/clear_eval/
│   ├── pipeline/                          # LLM analysis pipeline
│   ├── dashboard/                         # LLM dashboard (Streamlit)
│   ├── agentic/
│   │   ├── README.md                      # Agentic Workflows Guide
│   │   ├── pipeline/                      # Agentic pipeline
│   │   └── dashboard/
│   │       ├── README_DASHBOARD.md        # Agentic Dashboard Guide
│   │       └── ...
│   └── sample_data/                       # Sample datasets
├── docs/
│   ├── ANALYSIS_README.md                 # LLM Analysis Guide
│   └── PROVIDERS.md                       # Providers and Credentials Guide
├── examples/                              # Configuration examples
└── tests/                                 # Test suite
```

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.
