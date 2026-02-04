# External Judge Implementation Summary

## Overview

This document summarizes the implementation of external judge support for the CLEAR evaluation pipeline. External judges allow users to plug in custom Python functions to evaluate model outputs, replacing or supplementing LLM-based evaluation.

## Implementation Date
February 2026

## What Was Added

### 1. Core Infrastructure (`src/clear_eval/pipeline/external_judge.py`)

A new module providing:
- **`load_external_judge()`**: Dynamically loads judge functions from Python files
- **`validate_judge_function()`**: Validates judge function signatures
- **`call_external_judge()`**: Safely executes judge functions with error handling
- **`ExternalJudgeError`**: Custom exception for judge-related errors
- **`get_judge_info()`**: Extracts judge configuration from config dict

### 2. Modified Core Evaluation (`src/clear_eval/pipeline/eval_utils.py`)

Updated `evaluate_single_records()` to:
- Check `judge_type` configuration parameter
- Load and validate external judge if `judge_type == 'external'`
- Route evaluation through external judge or LLM based on configuration
- Maintain parallel execution support for both judge types

Added `evaluate_row_external()` function for external judge evaluation.

### 3. Configuration Support

**Default Config (`src/clear_eval/pipeline/setup/default_config.yaml`)**:
```yaml
judge_type: llm  # or 'external'
external_judge_path: null
external_judge_function: evaluate
external_judge_config: {}
```

**CLI Arguments (`src/clear_eval/args.py`)**:
- `--judge-type`: Choose between 'llm' or 'external'
- `--external-judge-path`: Path to judge Python file
- `--external-judge-function`: Function name to call
- `--external-judge-config`: JSON dict of judge-specific config

### 4. Example Judges (`examples/custom_judges/`)

**Exact Match Judge** (`exact_match_judge.py`):
- Simple string comparison
- Case-insensitive matching
- Useful for classification tasks

**Numeric Tolerance Judge** (`numeric_tolerance_judge.py`):
- Numeric comparison with configurable tolerance
- Extracts numbers from text
- Useful for math problems

**Documentation** (`README.md`):
- Comprehensive guide for creating custom judges
- Usage examples
- Best practices
- Troubleshooting guide

### 5. Testing & Examples

**Test Script** (`examples/test_external_judge.py`):
- Validates both example judges
- Demonstrates judge testing methodology
- Can be run standalone: `python examples/test_external_judge.py`

**Example Config** (`examples/config_external_judge_example.yaml`):
- Complete configuration example
- Demonstrates all external judge parameters
- Ready to use with minimal modifications

### 6. Documentation Updates

**Main README.md**:
- New section: "🔌 Using External Judges"
- Updated CLI arguments table
- Quick start guide
- Links to detailed documentation

## Judge Interface Specification

All external judges must implement:

```python
def evaluate(row: pd.Series, config: dict) -> tuple[str, float]:
    """
    Evaluate a single record.
    
    Args:
        row: pandas Series with record data (model_input, response, ground_truth, etc.)
        config: Full configuration dictionary
        
    Returns:
        (evaluation_text: str, score: float)
        - evaluation_text: Textual feedback about the evaluation
        - score: Numerical score 0.0-1.0, or pd.NA for failures
    """
    pass
```

## Usage Examples

### Via Configuration File

```yaml
judge_type: external
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
  --judge-type external \
  --external-judge-path my_judge.py \
  --data-path data.csv \
  --output-dir results/
```

### Via Python API

```python
from clear_eval.analysis_runner import run_clear_eval_analysis

run_clear_eval_analysis(
    judge_type='external',
    external_judge_path='my_judge.py',
    data_path='data.csv',
    output_dir='results/'
)
```

## Key Features

1. **Backward Compatible**: Default behavior unchanged (uses LLM)
2. **Flexible**: Supports any Python callable with correct signature
3. **Parallel Execution**: External judges run in parallel like LLM evaluation
4. **Error Handling**: Comprehensive validation and error messages
5. **Configurable**: Judge-specific config via `external_judge_config`
6. **Well Documented**: Examples, guides, and inline documentation

## Benefits

1. **Cost Reduction**: Avoid LLM API costs for deterministic metrics
2. **Speed**: Faster evaluation for simple metrics
3. **Flexibility**: Integrate existing evaluation functions
4. **Transparency**: Full control over evaluation logic
5. **Compatibility**: Works with all CLEAR features (aggregation, dashboard, etc.)

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

## Future Enhancements (Optional)

Potential improvements for future versions:

1. **Built-in Judges**: Add common judges (BLEU, ROUGE, etc.) to the package
2. **Judge Registry**: Allow registering judges by name instead of file path
3. **Async Support**: Add async judge support for I/O-bound operations
4. **Judge Composition**: Allow combining multiple judges
5. **Fallback Mechanism**: Automatic fallback to LLM if external judge fails

## Notes

- External judges still require an LLM provider for issue synthesis/aggregation
- The `eval_model_name` config is ignored when using external judges
- Type hints in example judges may show warnings but are compatible with the system
- External judges have access to the full config dictionary for maximum flexibility

## Support

For questions or issues:
1. Check `examples/custom_judges/README.md` for detailed documentation
2. Review example judges for implementation patterns
3. Run `examples/test_external_judge.py` to verify setup
4. Refer to main README.md for general CLEAR usage

---

**Implementation Status**: ✅ Complete and Ready for Use