"""
Example External Judge: Exact Match Evaluator

This is a simple example of an external judge that performs exact string matching
between the model response and ground truth.

The judge receives the entire DataFrame and returns it with added evaluation columns.

Usage:
    In your config file or CLI, set:
    - judge_type: external
    - external_judge_path: examples/custom_judges/exact_match_judge.py
    - external_judge_function: evaluate
"""

import pandas as pd
from unitxt.task import Task
from unitxt.templates import NullTemplate

from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL


def evaluate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Evaluate model responses using exact string matching.

    Args:
        df: pandas DataFrame containing all records with columns like:
            - response: model's generated response
            - ground_truth: expected correct answer
            - id: unique identifier for the record
        config: Configuration dictionary containing:
            - model_output_column: name of the column with model responses
            - reference_column: name of the column with ground truth
            - external_judge_config: additional judge-specific configuration

    Returns:
        DataFrame with added columns:
        - evaluation_text: Textual feedback about the evaluation
        - score: Numerical score between 0.0 and 1.0, or pd.NA if evaluation failed
    """
    # Get column names from config
    response_col = config.get('model_output_column', 'response')
    reference_col = config.get('reference_column', 'ground_truth')

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

        evaluation_texts.append(eval_text)
        scores.append(score)

    # Add results to DataFrame
    df[EVALUATION_TEXT_COL] = evaluation_texts
    df[SCORE_COL] = scores

    return df

# Made with Bob

from unitxt.api import create_dataset, evaluate as evaluate_with_unitxt
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect

def evaluate_unitxt(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Get column names from config
    response_col = config.get('model_output_column', 'response')
    model_input_col = config.get('model_input_col', 'response')
    data = list(df[model_input_col])
    criteria = "metrics.llm_as_judge.direct.criteria.answer_relevance"
    metric = f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria={criteria},context_fields=[{model_input_col}]]"
    task=Task(
        input_fields={"model_input_field": str},
        reference_fields={},
        prediction_type=str,
        default_template=NullTemplate(),
        metrics=[metric],
    )
    dataset = create_dataset(
        task=task, test_set=data, metrics=[metric], split="test"
    )

    predictions = list(df[response_col])

    results = evaluate_with_unitxt(predictions=predictions, data=dataset)

    print("Global Scores:")
    print(results.global_scores.summary)

    print("Instance Scores:")
    print(results.instance_scores)
