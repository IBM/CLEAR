# CLEAR: Comprehensive LLM Error Analysis and Reporting

**CLEAR** is an open-source toolkit for **LLM error analysis** using an LLM-as-a-Judge approach.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/clear-eval.svg)](https://pypi.org/project/clear-eval/)

---

## What is CLEAR?

CLEAR provides systematic error analysis for LLM-based systems. It combines automated LLM-as-a-judge evaluation with interactive dashboards to help you:
- Identify recurring error patterns across your dataset
- Quantify issue frequencies and severity
- Drill down into specific failure cases
- Prioritize improvements based on data-driven insights

CLEAR operates in two phases:

1. **Analysis** — Generates per-instance textual feedback, identifies system-level error categories, and quantifies their frequencies.
2. **Interactive Dashboard** — Explore aggregate visualizations, apply dynamic filters, and drill down into individual failure examples.

---

## Two Analysis Modes

### LLM Analysis

Evaluate standard LLM outputs — generation quality, correctness, and recurring error patterns. Provide a CSV with prompts and responses, and CLEAR will score each instance, generate textual critiques, and surface system-level issues.

- **Input:** CSV with model inputs and responses
- **Output:** Per-record scores, evaluation text, aggregated issue categories
- **Dashboard:** Streamlit-based interactive explorer

> **[LLM Analysis Guide →](docs/llm-analysis.md)**

### Agentic Analysis

Evaluate multi-agent system trajectories — step-by-step agent interactions and full trajectory analysis.
Supports traces from LangGraph, CrewAI, and other frameworks via MLflow or Langfuse.

- **Input:** Raw JSON traces or preprocessed trajectory CSVs (each trace captures one complete agent task execution)
- **Output:** Per-step CLEAR analysis, trajectory-level scores, rubric evaluations
- **Dashboard:** NiceGUI-based workflow visualization with path and temporal analysis

> **[Agentic Workflows Guide →](src/clear_eval/agentic/README.md)**

---

## Installation

**Requires Python 3.10+**

#### Option 1: pip

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

Verify the installation:

```bash
run-clear-eval-analysis --help
```

---

## Quick Start

### 1. Set your provider credentials

CLEAR requires a supported LLM provider. Set the appropriate environment variables for your provider (e.g., `OPENAI_API_KEY` for OpenAI). Adjust `--provider` and `--eval-model-name` in the commands below to match your setup. See [Provider Configuration](#provider-configuration) for details.

### 2. LLM Analysis

This evaluates GSM8K math problem responses and surfaces recurring quality issues:

```bash
run-clear-eval-analysis --provider openai --eval-model-name gpt-4o
```

Results are saved to `results/gsm8k/sample_output/`. View them:

```bash
run-clear-eval-dashboard
```

> **[Full LLM Analysis Guide →](docs/llm-analysis.md)** — input formats, CLI arguments, configuration, Python API, and external judges.

### 3. Agentic Analysis

These two modes are independent — this section does not require step 2.

Run CLEAR on sample agent traces (3 traces, each capturing one complete agent task execution, ~2 minutes):

```bash
run-clear-agentic-eval \
    --data-dir src/clear_eval/sample_data/agentic/research_agent_traces/mlflow \
    --results-dir my_smoke_test_results \
    --from-raw-traces true \
    --agent-framework langgraph \
    --observability-framework mlflow \
    --run-name smoke_test \
    --max-files 3 \
    --eval-model-name gpt-4o \
    --provider openai
```

View pre-computed results (all 20 traces) without re-running:

```bash
run-clear-agentic-dashboard
# Upload: src/clear_eval/sample_data/agentic/research_agent_results/mlflow/my_experiment/unified_ui_results.zip
```

> **[Full Agentic Guide →](src/clear_eval/agentic/README.md)** — trace generation, configuration, output structure, and dashboard features.

---

## Provider Configuration

CLEAR uses [LiteLLM](https://docs.litellm.ai/docs/providers) as its inference backend, supporting 100+ LLM providers (OpenAI, Anthropic, WatsonX, AWS Bedrock, Google Vertex AI, and more).

**Parameters:**

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| `provider` | `--provider` | LiteLLM provider name (e.g., `openai`, `anthropic`, `bedrock`, `vertex_ai`) |
| `eval_model_name` | `--eval-model-name` | Model identifier (e.g., `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `eval_model_params` | `--eval-model-params` | Additional model parameters as JSON (e.g., `{"temperature": 0}`) |
| `endpoint_url` | `--endpoint-url` | Custom endpoint URL for local/self-hosted models |

**Cloud provider example:**

```bash
export OPENAI_API_KEY="..."
run-clear-eval-analysis --provider openai --eval-model-name gpt-4o
```

**Local model example (vLLM, llama.cpp, Ollama, etc.):**

```bash
run-clear-eval-analysis \
    --provider openai \
    --eval-model-name my-local-model \
    --endpoint-url http://localhost:8000/v1
```

No credentials are needed when using `--endpoint-url` with a local server.

Set the required environment variables for your provider according to [LiteLLM's documentation](https://docs.litellm.ai/docs/providers).

---

## Documentation

| Guide | Description |
|-------|-------------|
| [**Agentic Workflows Guide**](src/clear_eval/agentic/README.md) | Multi-agent evaluation — trace preprocessing, step-by-step and trajectory analysis, configuration reference |
| [**Agentic Dashboard Guide**](docs/agentic/dashboard.md) | Dashboard features — workflow view, node analysis, trajectory explorer, path and temporal analysis |
| [**LLM Analysis Guide**](docs/llm-analysis.md) | Full pipeline reference — input formats, CLI arguments, configuration, and external judges |

---

## Citation

If you use CLEAR in your research, please cite the relevant paper(s):

**LLM Analysis** ([AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/42398)):
```bibtex
@inproceedings{yehudai2026clear,
  title={CLEAR: Error analysis via llm-as-a-judge made easy},
  author={Yehudai, Asaf and Eden, Lilach and Perlitz, Yotam and Bar-Haim, Roy and Shmueli-Scheuer, Michal},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={48},
  pages={41736--41738},
  year={2026}
}
```

**Agentic Analysis** (ACL 2026, to appear — [preprint](https://arxiv.org/abs/2605.22608)):
```bibtex
@article{yehudai2026agentic,
  title={Agentic CLEAR: Automating Multi-Level Evaluation of LLM Agents},
  author={Yehudai, Asaf and Eden, Lilach and Shmueli-Scheuer, Michal},
  journal={arXiv preprint arXiv:2605.22608},
  year={2026}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
