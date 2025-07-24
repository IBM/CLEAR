# CLEAR: Error Analysis via LLM-as-a-Judge Made Easy

**CLEAR (Comprehensive LLM Error Analysis and Reporting)** is an interactive, open-source package for **LLM-based error analysis**. It helps surface meaningful, recurring issues in model outputs by combining automated evaluation with powerful visualization tools.

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

### 1. Installation 
#### Option 1 (Recommended for development): **Clone the repo and set up a virtual environment:**

```bash
git clone https://github.com/IBM/CLEAR.git
cd CLEAR
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

#### 📦 Option 2: Install via pip (Latest Release)

```bash
pip install clear-eval
```
`
 2. ### Set provider type and credentials
CLEAR requires a supported LLM provider and credentials to run analysis. [See supported providers ↓](#supported-providers-and-credentials)
> ⚠️ Using a private proxy or openai deployment? You must configure your model names explicitly (see below). Otherwise, default model names will be used automatically for supported providers.

3. ### **Run on sample data:**

The sample dataset is a small subset of the **GSM8K math problems**.
For running on the sample data and default configuration, you simpy have to set your provider and run
```bash
run-clear-eval-analysis --provider=openai # or rits, watsonx
```

This will:
- Run the full CLEAR pipeline
- Save results under: `results/gsm8k/sample_output/`

4. ###  **View results in the interactive dashboard:**

```bash
run-clear-eval-dashboard
```

Or set the port with 
```bash
run-clear-eval-dashboard --port <port>
```
Then:
- Upload the generated ZIP file from `results/gsm8k/sample_output/`
- Explore issues, scores, filters, and drill into examples

5. ###  **To explore the dashboard without running any analysis:**
Run the dashboard:
```bash
run-clear-eval-dashboard
```

Then you can load the pre-generated sample output zip.
you can manually upload a sample `.zip` file located at:
```
<your-env>/site-packages/clear_eval/sample_data/gsm8k/analysis_results_gsm8k_default.zip
```

📁 Or just download it directly from the [GitHub repo](https://github.com/IBM/CLEAR/blob/main/src/clear_eval/sample_data/gsm8k/analysis_results_gsm8k_default.zip).


---


## 📂 Analyzing your own data

### 📄 Input Data Format

CLEAR takes a **CSV file** as input, with each row representing a single instance to be evaluated.

#### Required Columns

| Column         | Used When                           | Description                                                       |
|----------------|-------------------------------------|-------------------------------------------------------------------|
| `id`           | Always                              | Unique identifier for the instance                                |
| `model_input`  | Always                              | Prompt provided to the generation model                           |
| `response`     | Using pre-generated responses       | Pre-generated model response (ignored if generation is enabled)   |
| `ground_truth` | Performing reference based analysis | Ground-truth answer for evaluation (optional)                     |
| _others_       | `--input_columns` is used           | Additional input columns to show in dashboard (e.g. `question`)   |

---

### 🚀 Running the analysis

CLEAR can be run via the CLI or Python API.

#### Option 1: CLI commands

Each stage has its own entry point:

```bash
run-clear-eval-analysis --config_path path/to/config.yaml    # run full pypeline
run-clear-eval-generation --config_path path/to/config.yaml  # run generation only
run-clear-eval-evaluation --config_path path/to/config.yaml  # Assume generation responses are given, run evaluation
```

- If `--config_path` is specified, **all parameters are taken from the config** unless explicitly overridden
- CLI flags passed directly override corresponding config values

#### Option 2: Python API

```python
from clear_eval.analysis_runner import run_clear_eval_analysis, run_clear_eval_generation, run_clear_eval_evaluation

run_clear_eval_analysis(
    config_path="configs/sample_run_config.yaml"
)
```

You may also pass overrides instead of using a config file:

```python
from clear_eval.analysis_runner import run_clear_eval_analysis

run_clear_eval_analysis(
    run_name="my_data",
    provider="openai",
    data_path="my_data.csv",
    gen_model_name="gpt-3.5-turbo",
    eval_model_name="gpt-4",
    output_dir="results/gsm8k/",
    perform_generation=False,
    input_columns=["question"]
)
```
### 📊 Launching the Dashboard

```bash
run-clear-eval-dashboard
```

Upload the ZIP file generated in your `--output-dir` when prompted.

### 🎛 Supported CLI Arguments

Arguments can be provided via:
- A YAML config file (`--config_path`)
- CLI flags
- Python function parameters (when using the API)

> ⚠️ **Boolean arguments** (`perform_generation`, `is_reference_based`, `resume_enabled`)  
> These must be set explicitly to `true` or `false` in YAML, CLI, or Python.  
> On the CLI, use `--flag True` or `--flag False` (case-insensitive).

> ⚠️ **Naming Convention**  
> Parameter names use `snake_case` in YAML and Python, but use `--kebab-case` in CLI.  
> For example:  
> - YAML: `perform_generation: true`  
> - Python: `perform_generation=True`  
> - CLI: `--perform-generation True`

| Argument                | Description                                                                                                                                | Default |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `--config_path`         | Path to a YAML config file (all values loaded unless overridden by CLI args)                                                               |         |
| `--run_name`            | Unique run name (used in result file names)                                                                                                |         |
| `--data_path`           | Path to input CSV file                                                                                                                     |         |
| `--output_dir`          | Output directory to write results                                                                                                          |         |
| `--provider`            | Model provider: `openai`, `watsonx`, `rits`                                                                                                |         |
| `--eval_model_name`     | Name of judge model (e.g. `gpt-4o`)                                                                                                        |         |
| `--gen_model_name`      | Name of the generator model to evaluate. If not running generations - the generator name to display.                                       |         |
| `--perform_generation`  | Whether to generate responses or use existing `response` column                                                                            | True    |  
| `--is_reference_based`  | Use reference-based evaluation (requires `ground_truth` column in input)                                                                   | False   |
| `--resume_enabled`      | Whether to reuse intermediate outputs from previous runs stored in output_dir                                                              | True    |
| `--evaluation_criteria` | Custom criteria dictionary for scoring individual records: `{"criteria_name1":"criteria_desc1", ...}`supported for yaml config and python. | None    |
| `--input_columns`       | Comma-separated list of additional input fields (other than `model_input`) to appear in the results and dashboard (e.g. `question`)        | None    | 

---

## 🔑Supported providers and credentials

Depending on your selected `--provider`:

| Provider   | Required Environment Variables                                      |
|------------|---------------------------------------------------------------------|
| `openai`   | `OPENAI_API_KEY`,  [`OPENAI_API_BASE` if using proxy ]              |                                   |
| `watsonx`  | `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_SPACE_ID` or `PROJECT_ID` |
| `rits`     | `RITS_API_KEY`                                                      |

---

