# Custom External Judges for CLEAR

This directory contains example implementations of external judges that can be used with CLEAR's evaluation pipeline.

## What is an External Judge?

An external judge is a Python function that evaluates model outputs without using an LLM. This is useful when:
- You have a deterministic evaluation metric (e.g., exact match, numeric tolerance)
- You want to reduce evaluation costs by avoiding LLM API calls
- You need faster evaluation for large datasets
- You want to integrate existing evaluation metrics

## Judge Interface

All external judges must implement the following function signature:

```python
def evaluate(row: pd.Series, config: dict) -> tuple[str, float]:
    """
    Evaluate a single record.
    
    Args:
        row: pandas Series containing the record data (includes model_input, response, etc.)
        config: Configuration dictionary with all settings
        
    Returns:
        Tuple of (evaluation_text: str, score: float)
        - evaluation_text: Textual feedback/analysis
        - score: Numerical score between 0.0 and 1.0, or pd.NA if evaluation failed
    """
    pass
```

## Available Examples

### 1. Exact Match Judge (`exact_match_judge.py`)

Simple string matching evaluator that checks if the model response exactly matches the ground truth.

**Usage:**
```yaml
judge_type: external
external_judge_path: examples/custom_judges/exact_match_judge.py
external_judge_function: evaluate
```

**CLI:**
```bash
run-clear-eval-analysis \
  --judge-type external \
  --external-judge-path examples/custom_judges/exact_match_judge.py \
  --data-path your_data.csv
```

### 2. Numeric Tolerance Judge (`numeric_tolerance_judge.py`)

Evaluates numeric answers with configurable tolerance, useful for math problems.

**Usage:**
```yaml
judge_type: external
external_judge_path: examples/custom_judges/numeric_tolerance_judge.py
external_judge_function: evaluate
external_judge_config:
  tolerance: 0.01  # 1% tolerance
  extract_last_number: true  # Extract last number from response
```

**CLI:**
```bash
run-clear-eval-analysis \
  --judge-type external \
  --external-judge-path examples/custom_judges/numeric_tolerance_judge.py \
  --external-judge-config '{"tolerance": 0.01, "extract_last_number": true}' \
  --data-path your_data.csv
```

## Creating Your Own Judge

1. **Create a Python file** with your judge function:

```python
import pandas as pd

def evaluate(row: pd.Series, config: dict) -> tuple[str, float]:
    # Get data from row
    response = row.get(config['model_output_column'], '')
    ground_truth = row.get(config['reference_column'], '')
    
    # Your evaluation logic here
    score = 1.0 if response == ground_truth else 0.0
    eval_text = f"Match: {score == 1.0}"
    
    return eval_text, score
```

2. **Configure CLEAR to use your judge:**

```yaml
judge_type: external
external_judge_path: path/to/your_judge.py
external_judge_function: evaluate  # or your function name
external_judge_config:
  # Any custom configuration your judge needs
  custom_param: value
```

3. **Run the evaluation:**

```bash
run-clear-eval-analysis --config-path your_config.yaml
```

## Best Practices

1. **Handle Missing Data**: Always check for `pd.isna()` and return appropriate error messages
2. **Return Valid Scores**: Scores should be between 0.0 and 1.0, or `pd.NA` for failures
3. **Provide Informative Text**: The evaluation text helps users understand why a score was assigned
4. **Use Configuration**: Access judge-specific settings via `config['external_judge_config']`
5. **Error Handling**: Wrap risky operations in try-except blocks and return error messages

## Accessing Row Data

The `row` parameter contains all columns from your input CSV:

```python
# Standard columns (if present)
model_input = row.get(config['model_input_column'], '')
response = row.get(config['model_output_column'], '')
ground_truth = row.get(config['reference_column'], '')
question_id = row.get(config['qid_column'], '')

# Custom columns from your data
custom_field = row.get('your_column_name', '')
```

## Accessing Configuration

The `config` dictionary contains all CLEAR configuration:

```python
# Standard config
max_workers = config.get('max_workers', 1)
is_reference_based = config.get('is_reference_based', False)

# Judge-specific config
judge_config = config.get('external_judge_config', {})
custom_setting = judge_config.get('custom_param', default_value)
```

## Testing Your Judge

Test your judge function before running the full pipeline:

```python
import pandas as pd
from your_judge import evaluate

# Create test data
test_row = pd.Series({
    'model_input': 'What is 2+2?',
    'response': '4',
    'ground_truth': '4',
    'id': 'test_1'
})

test_config = {
    'model_output_column': 'response',
    'reference_column': 'ground_truth',
    'external_judge_config': {}
}

# Test evaluation
eval_text, score = evaluate(test_row, test_config)
print(f"Score: {score}")
print(f"Text: {eval_text}")
```

## Troubleshooting

**Error: "External judge file not found"**
- Check that the path to your judge file is correct
- Use absolute paths or paths relative to where you run the command

**Error: "Function 'X' not found"**
- Verify the function name matches `external_judge_function` in config
- Ensure the function is defined at the module level (not inside a class)

**Error: "Judge function must accept at least 2 parameters"**
- Your function signature must be: `def evaluate(row, config)`

**Scores not in expected range**
- Ensure your judge returns scores between 0.0 and 1.0
- Use `pd.NA` for failed evaluations, not `None`