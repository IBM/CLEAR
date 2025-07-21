# CLEAR: Error Analysis via LLM-as-a-Judge Made Easy

**CLEAR** (Comprehensive LLM Error Analysis and Reporting) is an interactive, open-source package for **LLM-based error analysis**. It helps surface meaningful, recurring issues in model outputs by combining automated evaluation with powerful visualization tools.

The workflow consists of two main phases:

1. **Analysis**  
    Generates textual feedback for each instance; Identifies system-level error categories from these critiques and quantifies their frequencies.

2. **Interactive Dashboard**  
   An intuitive dashboard provides a comprehensive view of model behavior. Users can:  
   - Explore aggregate visualizations of identified issues  
   - Apply dynamic filters to focus on specific error types or score ranges  
   - Drill down into individual examples that illustrate specific failure patterns

CLEAR makes it easier to diagnose model shortcomings and prioritize targeted improvements.

You can run CLEAR as a full pipeline, or reuse specific stages (generation, evaluation, or just UI).



## 🚀 Quickstart

Requires Python 3.10+ and the necessary credentials for a supported provider.

1. ### **Clone the repo and set up a virtual environment:**

```bash
git clone https://github.com/IBM/CLEAR.git
cd CLEAR
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Set provider type and credentials
CLEAR requires a supported LLM provider and credentials to run analysis. [See supported providers ↓](#supported-providers-and-credentials)
> ⚠️ Using a private proxy or openai deployment? You must configure your model names explicitly (see below). Otherwise, default model names will be used automatically for supported providers (`openai`, `watsonx`, `ritz`).

### 3. **Run on sample data:**

The sample dataset is a small subset of the **GSM8K math problems**.
For running on the sample data and default configuration, you simpy have to set your provider and run
```bash
run-analysis --provider=openai # or rits, watsonx
```

This will:
- Run the full CLEAR pipeline
- Save results under: `results/gsm8k/sample_output/`

### 4. **View results in the interactive dashboard:**

```bash
run-clear-ai-dashboard
```

Then:
- Upload the generated ZIP file from `results/gsm8k/sample_output/`
- Explore issues, scores, filters, and drill into examples

### 5. **To explore the dashboard without running any analysis:**
Run the dashboard:
```bash
run-clear-ai-dashboard
```

Then you can load the pre-generated sample output zip from [here](https://github.com/IBM/CLEAR/tree/main/results/input_for_ui), without running any analysis. 


---


## 📂 Running on your own data

## 📄 Input Data Format

CLEAR takes a **CSV file** as input, with each row representing a single instance to be evaluated.

### Required Columns

| Column        | Used When                           | Description                                                                |
|---------------|-------------------------------------|----------------------------------------------------------------------------|
| `id`          | Always                              | Unique identifier for the instance                                         |
| `model_input` | Always                              | Prompt provided to the generation model                                    |
| `response`    | Using pre-generated responses       | Pre-generated model response (ignored if generation is enabled)            |
| `reference`   | Performing reference based analysis | Ground-truth answer for evaluation (optional)                              |
| _others_      | `--input_columns` is used           | Additional input columns to show in dashboard (e.g. `question`) |

---

## 🚀 Usage

You can run CLEAR either by specifying a **YAML config file** (`--config_path`) or by providing **individual CLI arguments**.  
If both are used, CLI arguments take precedence and override values in the config.

### 📦 Option 1: Using a Config File

#### Command Line

```bash
run-clear-ai-analysis --config_path=configs/sample_demo.yaml
```

#### Python

```python
from clear_ai.analysis_runner import run_analysis

run_analysis(config_path="configs/sample_demo.yaml")
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

| Argument               | Description                                                                                    |
|------------------------|------------------------------------------------------------------------------------------------|
| `--data-path`          | Path to input CSV file                                                                         |
| `--output-dir`         | Output directory to write results                                                              |
| `--provider`           | Model provider: `openai`, `watsonx`, `rits`                                            |
| `--eval-model-name`    | Name of judge model (e.g. `gpt-4o`)                                                             |
| `--gen-model-name`     | Name of generation model (used if `--perform-generations=True`)                               |
| `--config_path`        | Path to a YAML config file (all values loaded unless overridden by CLI args)                   |
| `--perform-generations`| Whether to generate responses (`True`) or use existing `response` column                       |
| `--is-reference-based` | Use reference-based evaluation (requires `reference` column)                                   |
| `--resume-enabled`     | Whether to reuse intermediate outputs from previous runs                                       |
| `--run-name`           | Unique run ID (tag)                                                                            |
| `--evaluation_criteria`| Custom criteria for scoring, passed as a JSON string                                           |
| `--input_columns`      | Comma-separated list of additional input fields to show in the dashboard (e.g. `question`)     |

---

## 🔑Supported providers and credentials

Depending on your selected `--provider`:

| Provider   | Required Environment Variables                                      |
|------------|---------------------------------------------------------------------|
| `openai`   | `OPENAI_API_KEY`,  [`OPENAI_API_BASE` if using proxy ]                                                  |
| `watsonx`  | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_SPACE_ID` or `PROJECT_ID` |
| `rits`     | `RITS_API_KEY`                                                      |

---

## 🤝 Acknowledgments

Developed by IBM Research
