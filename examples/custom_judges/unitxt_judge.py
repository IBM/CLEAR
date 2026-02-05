"""
Example External Judge: Unitxt judge Evaluator

This is a simple example of an external judge that uses a unitxt judge

The judge receives the entire DataFrame and returns it with added evaluation columns.

Usage:
    In your config file or CLI, set:
    - task: external
    - external_judge_path: examples/custom_judges/unitxt_judge.py
    - external_judge_function: evaluate
"""

import pandas as pd
from unitxt.task import Task
from unitxt.templates import NullTemplate

from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL
from unitxt.api import create_dataset, evaluate as evaluate_with_unitxt

def evaluate(df: pd.DataFrame, config: dict) -> pd.DataFrame:

    response_col = config.get('model_output_column', 'response')
    model_input_col = config.get('model_input_column', 'model_input')
    data = [{model_input_col:d} for d in list(df[model_input_col])]
    criteria = "metrics.llm_as_judge.direct.criteria.answer_relevance"
    metric = f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria={criteria},context_fields=[{model_input_col}]]"
    task=Task(
        input_fields={model_input_col: str},
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
    df[EVALUATION_TEXT_COL] = [r["score"] for r in results.instance_scores]
    df[SCORE_COL] = [r[f"{r['score_name']}_assessment"] for r in results.instance_scores]

    return df
