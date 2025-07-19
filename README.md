# CLEAR: Error Analysis via LLM-as-a-Judge Made Easy

**CLEAR** is an interactive, open-source package for **LLM-based error analysis**. It helps surface meaningful, recurring issues in model outputs by combining automated evaluation with powerful visualization tools.

The workflow consists of two main phases:

1. **Analysis**  
   CLEAR first generates per-instance textual feedback for each model response. It then identifies system-level error categories from these critiques and quantifies how frequently each issue occurs across the dataset.

2. **Interactive Dashboard**  
   An intuitive dashboard provides a comprehensive view of model behavior. Users can:  
   - Explore aggregate visualizations of identified issues  
   - Apply dynamic filters to focus on specific error types or score ranges  
   - Drill down into individual examples that illustrate specific failure patterns

CLEAR makes it easier to diagnose model shortcomings and prioritize targeted improvements.


## 🛠 Installation

```bash
git clone https://github.com/IBM/CLEAR.git
cd CLEAR
pip install -e .
```

Requires Python 3.10+ and the necessary credentials for a supported provider.

---

## 📄 Input Data Format

CLEAR takes a **CSV file** as input, with each row representing a single instance to be evaluated.

### Required Columns

| Column        | Used When                         | Description                                                                 |
|---------------|-----------------------------------|-----------------------------------------------------------------------------|
| `id`          | Always                            | Unique identifier for the instance                                          |
| `model_input` | Always                            | Prompt provided to the generation model                                     |
| `response`    | `--perform-generations=False`     | Pre-generated model response (ignored if generation is enabled)             |
| `reference`   | `--is-reference-based=True`       | Ground-truth answer for evaluation (optional)                               |
| _others_      | `--input_columns` is used         | Additional input fields to show in dashboard (e.g. `question`, `document`)  |

---

## 🚀 Usage

You can run CLEAR either by specifying a **YAML config file** (`--config_path`) or by providing **individual CLI arguments**.  
If both are used, CLI arguments take precedence and override values in the config.

### 📦 Option 1: Using a Config File

#### Command Line

```bash
run-analysis --config_path=configs/sample_demo.yaml
```

#### Python

```python
from clear_ai.analysis_runner import run_analysis_pipeline

run_analysis_pipeline(config_path="configs/sample_demo.yaml")
```

### 📊 Launching the Dashboard

```bash
run-clear-ai-dashboard
```

Upload the ZIP file generated in your `--output-dir` when prompted.


## 🎛 CLI Arguments

CLEAR supports two ways to provide parameters:  
- A **YAML config file** (`--config_path`)  
- Individual **CLI arguments**  

If `--config_path` is provided, all parameters are loaded from the config file by default.  
However, any parameters also specified directly via the CLI will **override** their values from the config.

### Supported CLI Arguments

| Argument               | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| `--data-path`          | Path to input CSV file                                                                          |
| `--output-dir`         | Output directory to write results                                                               |
| `--provider`           | Model provider: `openai`, `azure`, `watsonx`, `rits`                                            |
| `--eval-model-name`    | Name of judge model (e.g. `gpt-4o`)                                                              |
| `--gen-model-name`     | Name of generation model (used if `--perform-generations=True`)                                |
| `--config_path`        | Path to a YAML config file (all values loaded unless overridden by CLI args)                    |
| `--perform-generations`| Whether to generate responses (`True`) or use existing `response` column                        |
| `--is-reference-based` | Use reference-based evaluation (requires `reference` column)                                    |
| `--resume-enabled`     | Whether to reuse intermediate outputs from previous runs                                        |
| `--run-name`           | Unique run ID (tag)                                                                             |
| `--evaluation_criteria`| Custom criteria for scoring, passed as a JSON string                                            |
| `--input_columns`      | Comma-separated list of additional input fields to show in the dashboard (e.g. `question`)      |

---

## 🔑 Required Environment Variables

Depending on your selected `--provider`:

| Provider   | Required Environment Variables                                      |
|------------|---------------------------------------------------------------------|
| `openai`   | `OPENAI_API_KEY`                                                    |
| `azure`    | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`                     |
| `watsonx`  | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_SPACE_ID` or `PROJECT_ID` |
| `rits`     | `RITS_API_KEY`                                                      |

---

## 📚 Citation

If you use CLEAR, please cite:

```bibtex
@misc{yehudai2025clear,
  title={CLEAR: Error Analysis via LLM-as-a-Judge Made Easy},
  author={Yehudai, Asaf and Eden, Lilach and Perlitz, Yotam and Bar-Haim, Roy and Shmueli-Scheuer, Michal},
  year={2025},
  url={https://openreview.net/forum?id=EQ0FEiuo81}
}
```

---

## 🤝 Acknowledgments

Developed by IBM Research
