"""
Example External Judge: Numeric Tolerance Evaluator

This judge evaluates numeric answers with a configurable tolerance level.
Useful for math problems where slight variations in decimal places are acceptable.

The judge receives the entire DataFrame and returns it with added evaluation columns.

Usage:
    In your config file, set:
    - task: external
    - external_judge_path: examples/custom_judges/numeric_tolerance_judge.py
    - external_judge_function: evaluate
    - external_judge_config:
        tolerance: 0.01  # Allow 1% difference (default)
        extract_last_number: true  # Extract last number from response (default: false)
"""

import re
import pandas as pd
from typing import Optional
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL


def extract_number(text: str) -> Optional[float]:
    """Extract a numeric value from text."""
    if not text:
        return None
    
    # Remove common formatting (commas, dollar signs, etc.)
    text = str(text).replace(',', '').replace('$', '').strip()
    
    # Try to find numbers in the text
    # Match integers, decimals, and scientific notation
    number_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(number_pattern, text)
    
    if not matches:
        return None
    
    # Return the last number found (common in step-by-step solutions)
    try:
        return float(matches[-1])
    except (ValueError, IndexError):
        return None


def evaluate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Evaluate numeric responses with tolerance.
    
    Args:
        df: pandas DataFrame containing all records
        config: Configuration dictionary
    
    Returns:
        DataFrame with added evaluation_text and score columns
    """
    # Get configuration
    response_col = config.get('model_output_column', 'response')
    reference_col = config.get('reference_column', 'ground_truth')
    judge_config = config.get('external_judge_config', {})
    
    tolerance = judge_config.get('tolerance', 0.01)  # 1% default tolerance
    extract_last = judge_config.get('extract_last_number', False)
    
    # Initialize result columns
    evaluation_texts = []
    scores = []
    
    # Process each row
    for idx, row in df.iterrows():
        response = row.get(response_col, '')
        ground_truth = row.get(reference_col, '')
        
        # Handle missing data
        if pd.isna(response) or pd.isna(ground_truth):
            evaluation_texts.append("Missing data: response or ground truth is not available")
            scores.append(pd.NA)
            continue
        
        # Extract numbers
        if extract_last:
            response_num = extract_number(str(response))
            ground_truth_num = extract_number(str(ground_truth))
        else:
            try:
                response_num = float(str(response).strip())
                ground_truth_num = float(str(ground_truth).strip())
            except ValueError:
                response_num = None
                ground_truth_num = None
        
        # Check if we could extract numbers
        if response_num is None or ground_truth_num is None:
            eval_text = (
                f"Could not extract numeric values.\n"
                f"Response: '{response}'\n"
                f"Ground truth: '{ground_truth}'"
            )
            evaluation_texts.append(eval_text)
            scores.append(0.0)
            continue
        
        # Calculate relative error
        if ground_truth_num == 0:
            # Handle division by zero - use absolute difference
            error = abs(response_num - ground_truth_num)
            is_correct = error <= tolerance
        else:
            relative_error = abs(response_num - ground_truth_num) / abs(ground_truth_num)
            is_correct = relative_error <= tolerance
            error = relative_error
        
        # Determine score and evaluation text
        if is_correct:
            score = 1.0
            eval_text = (
                f"✓ Correct: Response is within tolerance.\n"
                f"Expected: {ground_truth_num}\n"
                f"Got: {response_num}\n"
                f"Error: {error:.4f} (tolerance: {tolerance})"
            )
        else:
            score = 0.0
            eval_text = (
                f"✗ Incorrect: Response exceeds tolerance.\n"
                f"Expected: {ground_truth_num}\n"
                f"Got: {response_num}\n"
                f"Error: {error:.4f} (tolerance: {tolerance})"
            )
        
        evaluation_texts.append(eval_text)
        scores.append(score)
    
    # Add results to DataFrame
    df[EVALUATION_TEXT_COL] = evaluation_texts
    df[SCORE_COL] = scores
    
    return df

