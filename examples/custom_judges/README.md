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
def evaluate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Evaluate all records in the dataset.
    
    Args:
        df: pandas DataFrame containing all records to evaluate
        config: Configuration dictionary with all settings
        
    Returns:
        DataFrame with added columns:
        - 'evaluation_text': Textual feedback for each record
        - 'score': Numerical score between 0.0 and 1.0, or pd.NA if evaluation failed
    
    The judge receives the entire dataset and can process it however it wants
    (sequentially, in parallel, in batches, etc.). It must return a DataFrame
    with the same number of rows and the required evaluation columns added.
    """
    pass
```

**Important**: External judges must be **completely standalone** - they should not import anything from `clear_eval`. Use standard column names `'evaluation_text'` and `'score'` directly as strings.

## Available Examples

### 1. Exact Match Judge (`exact_match_judge.py`)

Simple string matching evaluator that checks if the model response exactly matches the ground truth.

**Usage:**
```yaml
task: external
external_judge_path: examples/custom_judges/exact_match_judge.py
external_judge_function: evaluate
```

**CLI:**
```bash
run-clear-eval-analysis \
  --task external \
  --external-judge-path examples/custom_judges/exact_match_judge.py \
  --data-path your_data.csv
```

### 2. Numeric Tolerance Judge (`numeric_tolerance_judge.py`)

Evaluates numeric answers with configurable tolerance, useful for math problems.

**Usage:**
```yaml
task: external
external_judge_path: examples/custom_judges/numeric_tolerance_judge.py
external_judge_function: evaluate
external_judge_config:
  tolerance: 0.01  # 1% tolerance
  extract_last_number: true  # Extract last number from response
```

**CLI:**
```bash
run-clear-eval-analysis \
  --task external \
  --external-judge-path examples/custom_judges/numeric_tolerance_judge.py \
  --external-judge-config '{"tolerance": 0.01, "extract_last_number": true}' \
  --data-path your_data.csv
```

## Creating Your Own Judge

1. **Create a Python file** with your judge function:

```python
import pandas as pd

def evaluate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Evaluate all records in the dataset."""
    # Get configuration
    response_col = config.get('model_output_column', 'response')
    reference_col = config.get('reference_column', 'ground_truth')
    
    # Initialize result lists
    evaluation_texts = []
    scores = []
    
    # Process each row (or use vectorized operations, parallel processing, etc.)
    for idx, row in df.iterrows():
        response = row.get(response_col, '')
        ground_truth = row.get(reference_col, '')
        
        # Your evaluation logic here
        score = 1.0 if response == ground_truth else 0.0
        eval_text = f"Match: {score == 1.0}"
        
        evaluation_texts.append(eval_text)
        scores.append(score)
    
    # Add results to DataFrame using standard column names
    df['evaluation_text'] = evaluation_texts
    df['score'] = scores
    
    return df
```

2. **Configure CLEAR to use your judge:**

```yaml
task: external
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

## Accessing DataFrame Data

The `df` parameter is a pandas DataFrame containing all records with all columns from your input CSV:

```python
# Get column names from config
response_col = config.get('model_output_column', 'response')
reference_col = config.get('reference_column', 'ground_truth')
qid_col = config.get('qid_column', 'id')

# Access data for all records
responses = df[response_col]
ground_truths = df[reference_col]
question_ids = df[qid_col]

# Access custom columns
custom_data = df['your_column_name']

# Process row by row if needed
for idx, row in df.iterrows():
    response = row[response_col]
    ground_truth = row[reference_col]
    # ... your logic
```

## Processing Flexibility

Since the judge receives the entire DataFrame, you have complete control over how to process it:

**Sequential Processing:**
```python
for idx, row in df.iterrows():
    # Process one row at a time
    pass
```

**Vectorized Operations (fastest for simple logic):**
```python
# Use pandas vectorized operations
df['score'] = (df[response_col] == df[reference_col]).astype(float)
df['evaluation_text'] = df['score'].apply(lambda x: 'Match' if x == 1.0 else 'No match')
```

**Parallel Processing:**
```python
from multiprocessing import Pool

def process_row(row):
    # Your logic
    return eval_text, score

with Pool() as pool:
    results = pool.map(process_row, [row for _, row in df.iterrows()])
```

**Batch Processing:**
```python
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    # Process batch
    pass
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
test_df = pd.DataFrame([
    {
        'id': 'test_1',
        'model_input': 'What is 2+2?',
        'response': '4',
        'ground_truth': '4'
    },
    {
        'id': 'test_2',
        'model_input': 'What is 3+3?',
        'response': '6',
        'ground_truth': '6'
    }
])

test_config = {
    'model_output_column': 'response',
    'reference_column': 'ground_truth',
    'external_judge_config': {}
}

# Test evaluation
result_df = evaluate(test_df, test_config)
print(result_df[['evaluation_text', 'score']])
```

Or use the provided test script:
```bash
python examples/test_external_judge.py
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