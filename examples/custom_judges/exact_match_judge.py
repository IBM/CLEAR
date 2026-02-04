"""
Example External Judge: Exact Match Evaluator

This is a simple example of an external judge that performs exact string matching
between the model response and ground truth.

Usage:
    In your config file or CLI, set:
    - judge_type: external
    - external_judge_path: examples/custom_judges/exact_match_judge.py
    - external_judge_function: evaluate
"""

import pandas as pd


def evaluate(row: pd.Series, config: dict) -> tuple[str, float]:
    """
    Evaluate a model response using exact string matching.
    
    Args:
        row: pandas Series containing the record data with columns like:
            - response: model's generated response
            - ground_truth: expected correct answer
            - id: unique identifier for the record
        config: Configuration dictionary containing:
            - model_output_column: name of the column with model responses
            - reference_column: name of the column with ground truth
            - external_judge_config: additional judge-specific configuration
    
    Returns:
        Tuple of (evaluation_text: str, score: float)
        - evaluation_text: Textual feedback about the evaluation
        - score: Numerical score between 0.0 and 1.0, or pd.NA if evaluation failed
    """
    # Get column names from config
    response_col = config.get('model_output_column', 'response')
    reference_col = config.get('reference_column', 'ground_truth')
    
    # Get the response and ground truth
    response = row.get(response_col, '')
    ground_truth = row.get(reference_col, '')
    
    # Handle missing data
    if pd.isna(response) or pd.isna(ground_truth):
        return "Missing data: response or ground truth is not available", pd.NA
    
    # Normalize strings for comparison
    response_normalized = str(response).strip().lower()
    ground_truth_normalized = str(ground_truth).strip().lower()
    
    # Perform exact match
    is_match = response_normalized == ground_truth_normalized
    score = 1.0 if is_match else 0.0
    
    # Generate evaluation text
    if is_match:
        eval_text = f"✓ Exact match: The response exactly matches the ground truth."
    else:
        eval_text = (
            f"✗ No match: The response does not match the ground truth.\n"
            f"Expected: '{ground_truth}'\n"
            f"Got: '{response}'"
        )
    
    return eval_text, score

# Made with Bob
