"""
External Judge Use Case for CLEAR Evaluation Pipeline.

This use case allows users to plug in their own custom evaluation functions
that receive the entire dataset and return evaluation results.
"""

import logging
import pandas as pd

from clear_eval.pipeline.use_cases.EvalUseCase import EvalUseCase
from clear_eval.pipeline.constants import SCORE_COL, EVALUATION_TEXT_COL
from clear_eval.pipeline.external_judge import (
    load_external_judge, call_external_judge, ExternalJudgeError
)

logger = logging.getLogger(__name__)


class ExternalJudgeUseCase(EvalUseCase):
    """
    Use case for external judge evaluation.
    
    External judges receive the entire DataFrame and return it with added
    evaluation columns. This bypasses all LLM-based evaluation logic and
    task-specific prompt generation.
    """
    
    # No required input fields - external judge defines what it needs
    required_input_fields = []
    
    def eval_records(self, df, llm, config, score_col=SCORE_COL):
        """
        Evaluate records using an external judge function.
        
        Args:
            df: DataFrame with records to evaluate
            llm: LLM instance (ignored for external judges, but kept for interface compatibility)
            config: Configuration dictionary
            score_col: Name of the score column to add
            
        Returns:
            DataFrame with added evaluation_text and score columns
        """
        logger.info(f"\n--- Evaluating Predictions with External Judge ---")
        
        # Get external judge configuration
        external_judge_path = config.get('external_judge_path')
        external_judge_function = config.get('external_judge_function', 'evaluate')
        
        if not external_judge_path:
            logger.error("Error: external_judge_path not provided for external judge task.")
            df[EVALUATION_TEXT_COL] = "Error: External judge path not configured"
            df[score_col] = pd.NA
            return df
        
        try:
            # Load the external judge
            external_judge_func = load_external_judge(external_judge_path, external_judge_function)
            logger.info(f"Using external judge: {external_judge_path}::{external_judge_function}")
        except ExternalJudgeError as e:
            logger.error(f"Error loading external judge: {e}")
            df[EVALUATION_TEXT_COL] = f"Error: {str(e)}"
            df[score_col] = pd.NA
            return df
        
        # Call external judge with entire DataFrame
        try:
            logger.info(f"Calling external judge with {len(df)} records...")
            result_df = call_external_judge(external_judge_func, df.copy(), config)
            
            # Copy evaluation results back to original DataFrame
            df[EVALUATION_TEXT_COL] = result_df[EVALUATION_TEXT_COL]
            df[score_col] = result_df[score_col]
            
            logger.info("Finished evaluating predictions with external judge.")
            
            # Convert score column to nullable float type
            df[score_col] = df[score_col].astype('Float64')
            
            return df
            
        except ExternalJudgeError as e:
            logger.error(f"Error executing external judge: {e}")
            df[EVALUATION_TEXT_COL] = f"Error: {str(e)}"
            df[score_col] = pd.NA
            return df

