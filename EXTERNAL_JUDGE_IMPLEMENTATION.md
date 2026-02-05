# External Judge Implementation Summary

## Overview

This document summarizes the implementation of external judge support for the CLEAR evaluation pipeline. External judges allow users to plug in custom Python functions to evaluate model outputs, replacing LLM-based evaluation with complete control over the evaluation process.

## Implementation Date
February 2026

## Key Design Decision: Batch Interface

**The external judge receives the entire DataFrame at once**, not individual records. This design gives users complete flexibility in how they process the data:
- Sequential processing (row by row)
- Vectorized operations (pandas operations on entire columns)
- Parallel processing (multiprocessing, threading)
- Batch processing (process in chunks)
- External API calls (batch requests)
- GPU acceleration (for ML-based judges)

## What Was Added

### 1. Core Infrastructure (`src/clear_eval/pipeline/external_judge.py`)

A new module providing:
- **`load_external_judge()`**: Dynamically loads judge functions from Python files
- **`validate_judge_function()`**: Validates judge function signatures
- **`call_external_judge()`**: Safely executes judge functions with comprehensive validation
- **`ExternalJudgeError`**: Custom exception for judge-related errors
- **`get_judge_info()`**: Extracts judge configuration from config dict

### 2. New Use Case (`src/clear_eval/pipeline/use_cases/ExternalJudgeUseCase.py`)

Created `ExternalJudgeUseCase` class that:
- Inherits from `EvalUseCase`
- Overrides `eval_records()` to load and call external judge
- Receives entire DataFrame and returns it with evaluation columns
- No required input fields - judge defines what it needs
- Handles all external judge logic in one place

### 3. Configuration Support

**Default Config (`src/clear_eval/pipeline/setup/default_config.yaml`)**:
```yaml
# External judge configuration (used when task is "external")
external_judge_path: null
external_judge_function: evaluate
external_judge_config: {}
```

**CLI Arguments (`src/clear_eval/args.py`)**:
- `--task`: Set to 'external' to use external judge
- `--external-judge-path`: Path to judge Python file
- `--external-judge-function`: Function name to call
- `--external-judge-config`: JSON dict of judge-specific config

### 4. Example Judges (`examples/custom_judges/`)

**Exact Match Judge** (`exact_match_judge.py`):
- Simple string comparison
- Case-insensitive matching
- Demonstrates basic batch processing

**Numeric Tolerance Judge** (`numeric_tolerance_judge.py`):
- Numeric comparison with configurable tolerance
- Extracts numbers from text
- Demonstrates configurable judge parameters

**Documentation** (`README.md`):
- Comprehensive guide for creating custom judges
- Usage examples with different processing strategies
- Best practices and troubleshooting

### 5. Testing & Examples

**Test Script** (`examples/test_external_judge.py`):
- Validates both example judges
- Demonstrates batch interface usage
- Can be run standalone: `python examples/test_external_judge.py`

**Example Config** (`examples/config_external_judge_example.yaml`):
- Complete configuration example
- Demonstrates all external judge parameters
- Ready to use with minimal modifications

### 6. Documentation Updates

**Main README.md**:
- New section: "🔌 Using External Judges"
- Updated CLI arguments table
- Quick start guide with batch interface
- Links to detailed documentation

## Judge Interface Specification

All external judges must implement:

```python
def evaluate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Evaluate all records in the dataset.
    
    Args:
        df: pandas DataFrame with all records to evaluate
        config: Full configuration dictionary
        
    Returns:
        DataFrame with added columns:
        - 'evaluation_text': Textual feedback for each record
        - 'score': Numerical score 0.0-1.0, or pd.NA for failures
    
    The judge receives the entire dataset and can process it however
    it wants (sequentially, in parallel, in batches, etc.). It must
    return a DataFrame with the same number of rows and the required
    evaluation columns added.
    """
    pass
```

## Usage Examples

### Via Configuration File

```yaml
task: external
external_judge_path: path/to/my_judge.py
external_judge_function: evaluate
data_path: data.csv
output_dir: results/
```

```bash
run-clear-eval-analysis --config-path config.yaml
```

### Via CLI

```bash
run-clear-eval-analysis \
  --task external \
  --external-judge-path my_judge.py \
  --data-path data.csv \
  --output-dir results/
```

### Via Python API

```python
from clear_eval.analysis_runner import run_clear_eval_analysis

run_clear_eval_analysis(
    task='external',
    external_judge_path='my_judge.py',
    data_path='data.csv',
    output_dir='results/'
)
```

## Key Features

1. **Batch Interface**: Judge receives entire DataFrame, enabling flexible processing strategies
2. **Backward Compatible**: Default behavior unchanged (uses LLM)
3. **Flexible Processing**: Users control how data is processed (sequential, parallel, vectorized, etc.)
4. **Comprehensive Validation**: Validates function signature, return type, and output format
5. **Error Handling**: Clear error messages and graceful failure handling
6. **Configurable**: Judge-specific config via `external_judge_config`
7. **Well Documented**: Examples, guides, and inline documentation

## Benefits

1. **Complete Control**: Users decide how to process data (no assumptions about parallelization)
2. **Performance**: Can use vectorized operations, GPU acceleration, or external APIs
3. **Cost Reduction**: Avoid LLM API costs for deterministic metrics
4. **Flexibility**: Integrate any evaluation logic or existing metrics
5. **Transparency**: Full visibility into evaluation process
6. **Compatibility**: Works with all CLEAR features (aggregation, dashboard, etc.)

## Processing Strategies Enabled

The batch interface enables various processing strategies:

**Sequential (Simple)**:
```python
for idx, row in df.iterrows():
    # Process one at a time
```

**Vectorized (Fastest for simple logic)**:
```python
df['score'] = (df['response'] == df['ground_truth']).astype(float)
```

**Parallel (For complex per-record logic)**:
```python
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(process_func, df.iterrows())
```

**Batch API Calls**:
```python
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    results = external_api.evaluate_batch(batch)
```

## Files Modified

- `src/clear_eval/pipeline/eval_utils.py` - Core evaluation logic
- `src/clear_eval/pipeline/setup/default_config.yaml` - Default configuration
- `src/clear_eval/args.py` - CLI argument parsing
- `README.md` - Main documentation

## Files Created

- `src/clear_eval/pipeline/external_judge.py` - Judge infrastructure
- `examples/custom_judges/exact_match_judge.py` - Example judge
- `examples/custom_judges/numeric_tolerance_judge.py` - Example judge
- `examples/custom_judges/README.md` - Judge documentation
- `examples/test_external_judge.py` - Test script
- `examples/config_external_judge_example.yaml` - Example config
- `EXTERNAL_JUDGE_IMPLEMENTATION.md` - This document

## Testing

Run the test script to verify functionality:

```bash
cd examples
python test_external_judge.py
```

Expected output:
```
✓ All tests passed!
```

## Notes

- External judges still require an LLM provider for issue synthesis/aggregation
- The `eval_model_name` config is ignored when using external judges
- Type hints in example judges may show warnings but are compatible with the system
- External judges have access to the full config dictionary for maximum flexibility
- The batch interface gives users complete control over processing strategy

---

**Implementation Status**: ✅ Complete and Ready for Use