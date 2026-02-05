"""
Example External Judge: Unitxt Judge Evaluator

This is an example of an external judge that uses Unitxt's LLM-as-judge functionality.

The judge receives the entire DataFrame and returns it with added evaluation columns:
- 'evaluation_text': Text explanation of the evaluation
- 'score': Numeric score between 0.0 and 1.0

Usage:
    In your config file or CLI, set:
    - task: external
    - external_judge_path: examples/custom_judges/unitxt_judge.py
    - external_judge_function: evaluate
    
Note: This requires the unitxt package to be installed.
"""

import pandas as pd
from unitxt.task import Task
from unitxt.templates import NullTemplate
from unitxt.api import create_dataset, evaluate as evaluate_with_unitxt


def evaluate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Evaluate model responses using Unitxt's LLM-as-judge.
    
    Args:
        df: DataFrame with model inputs and responses
        config: Configuration dictionary
        
    Returns:
        DataFrame with 'evaluation_text' and 'score' columns added
    """
    response_col = config.get('model_output_column', 'response')
    model_input_col = config.get('model_input_column', 'model_input')
    
    # Prepare data for Unitxt
    data = [{model_input_col: d} for d in list(df[model_input_col])]
    criteria = "metrics.llm_as_judge.direct.criteria.answer_relevance"
    metric = f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria={criteria},context_fields=[{model_input_col}]]"
    
    task = Task(
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
    
    # Evaluate with Unitxt
    results = evaluate_with_unitxt(predictions=predictions, data=dataset)
    
    # Add results using standard column names
    df['score'] = [r["score"] for r in results.instance_scores]
    df['evaluation_text'] = [r[f"{r['score_name']}_assessment"] for r in results.instance_scores]
    
    return df
