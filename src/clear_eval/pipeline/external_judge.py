"""
External Judge Support for CLEAR Evaluation Pipeline.

This module provides functionality to load and use custom external judges
for evaluating model outputs, as an alternative to LLM-based evaluation.
"""

import importlib.util
import inspect
import logging
import os
from typing import Callable, Tuple, Any
import pandas as pd
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL

logger = logging.getLogger(__name__)


class ExternalJudgeError(Exception):
    """Exception raised for errors in external judge loading or execution."""
    pass


def load_external_judge(judge_path: str, function_name: str = "evaluate") -> Callable:
    """
    Dynamically load an external judge function from a Python file.
    
    Args:
        judge_path: Path to the Python file containing the judge function.
                   Supports absolute paths, relative paths, and ~ expansion.
        function_name: Name of the function to load (default: "evaluate")
        
    Returns:
        The loaded judge function
        
    Raises:
        ExternalJudgeError: If the judge cannot be loaded or validated
    """
    # Expand user path (~ to home directory) and convert to absolute path
    judge_path = os.path.abspath(os.path.expanduser(judge_path))
    
    # Validate file exists
    if not os.path.exists(judge_path):
        raise ExternalJudgeError(f"External judge file not found: {judge_path}")
    
    if not judge_path.endswith('.py'):
        raise ExternalJudgeError(f"External judge file must be a Python file (.py): {judge_path}")
    
    try:
        # Load the module
        module_name = os.path.splitext(os.path.basename(judge_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, judge_path)
        if spec is None or spec.loader is None:
            raise ExternalJudgeError(f"Failed to load module spec from: {judge_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        if not hasattr(module, function_name):
            raise ExternalJudgeError(
                f"Function '{function_name}' not found in {judge_path}. "
                f"Available functions: {[name for name in dir(module) if not name.startswith('_')]}"
            )
        
        judge_func = getattr(module, function_name)
        
        # Validate the function
        validate_judge_function(judge_func, function_name)
        
        logger.info(f"Successfully loaded external judge: {function_name} from {judge_path}")
        return judge_func
        
    except ExternalJudgeError:
        raise
    except Exception as e:
        raise ExternalJudgeError(f"Error loading external judge from {judge_path}: {str(e)}")


def validate_judge_function(judge_func: Callable, function_name: str) -> None:
    """
    Validate that a judge function has the correct signature.
    
    Expected signature: (df: pd.DataFrame, config: dict) -> pd.DataFrame
    
    The returned DataFrame must contain 'evaluation_text' and 'score' columns.
    
    Args:
        judge_func: The function to validate
        function_name: Name of the function (for error messages)
        
    Raises:
        ExternalJudgeError: If the function signature is invalid
    """
    if not callable(judge_func):
        raise ExternalJudgeError(f"'{function_name}' is not callable")
    
    # Check signature
    try:
        sig = inspect.signature(judge_func)
        params = list(sig.parameters.keys())
        
        # Should have at least 2 parameters (df, config)
        if len(params) < 2:
            raise ExternalJudgeError(
                f"Judge function '{function_name}' must accept at least 2 parameters (df, config). "
                f"Found: {params}"
            )
        
        logger.debug(f"Judge function '{function_name}' signature validated: {sig}")
        
    except Exception as e:
        raise ExternalJudgeError(f"Error validating judge function signature: {str(e)}")


def call_external_judge(
    judge_func: Callable,
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Call an external judge function and validate its output.
    
    The judge function must return a DataFrame with 'evaluation_text' and 'score' columns.
    
    Args:
        judge_func: The judge function to call
        df: The DataFrame with all records to evaluate
        config: Configuration dictionary
        
    Returns:
        DataFrame with added 'evaluation_text' and 'score' columns
        
    Raises:
        ExternalJudgeError: If the judge execution fails or returns invalid output
    """
    try:
        result_df = judge_func(df, config)
        
        # Validate return type
        if not isinstance(result_df, pd.DataFrame):
            raise ExternalJudgeError(
                f"Judge function must return a pandas DataFrame. Got: {type(result_df)}"
            )

        # Check if columns need to be renamed (support both 'evaluation_text' and 'evaluation text')
        if 'evaluation_text' in result_df.columns and EVALUATION_TEXT_COL not in result_df.columns:
            result_df = result_df.rename(columns={'evaluation_text': EVALUATION_TEXT_COL})
        if 'score' in result_df.columns and SCORE_COL not in result_df.columns:
            result_df = result_df.rename(columns={'score': SCORE_COL})
        
        # Check that required columns exist
        if EVALUATION_TEXT_COL not in result_df.columns:
            raise ExternalJudgeError(
                f"Judge function must add '{EVALUATION_TEXT_COL}' column to the DataFrame. "
                f"Found columns: {list(result_df.columns)}"
            )
        
        if SCORE_COL not in result_df.columns:
            raise ExternalJudgeError(
                f"Judge function must add '{SCORE_COL}' column to the DataFrame. "
                f"Found columns: {list(result_df.columns)}"
            )

        # Validate that DataFrame has same number of rows
        if len(result_df) != len(df):
            raise ExternalJudgeError(
                f"Judge function must return DataFrame with same number of rows. "
                f"Expected: {len(df)}, Got: {len(result_df)}"
            )
        
        # Validate evaluation text column
        if not result_df[EVALUATION_TEXT_COL].dtype == 'object':
            logger.warning(
                f"'{EVALUATION_TEXT_COL}' column should contain strings. "
                f"Got dtype: {result_df[EVALUATION_TEXT_COL].dtype}"
            )
        
        # Validate score column (should be numeric or contain pd.NA)
        score_col = result_df[SCORE_COL]
        non_na_scores = score_col.dropna()
        if len(non_na_scores) > 0:
            try:
                # Check if scores are numeric
                numeric_scores = pd.to_numeric(non_na_scores, errors='coerce')
                if pd.isna(numeric_scores).any():
                    raise ExternalJudgeError(
                        f"'{SCORE_COL}' column must contain numeric values or pd.NA"
                    )
                
                # Warn if scores are outside expected range
                out_of_range = (numeric_scores < 0.0) | (numeric_scores > 1.0)
                if out_of_range.any():
                    logger.warning(
                        f"Some scores in '{SCORE_COL}' are outside the expected range [0.0, 1.0]. "
                        f"This may affect analysis results."
                    )
            except Exception as e:
                raise ExternalJudgeError(
                    f"Error validating '{SCORE_COL}' column: {str(e)}"
                )
        
        return result_df
        
    except ExternalJudgeError:
        raise
    except Exception as e:
        raise ExternalJudgeError(f"Error executing external judge: {str(e)}")


def get_judge_info(config: dict) -> dict:
    """
    Extract judge configuration information.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with judge type and parameters
    """
    return {
        'external_judge_path': config.get('external_judge_path'),
        'external_judge_function': config.get('external_judge_function', 'evaluate'),
        'external_judge_config': config.get('external_judge_config', {})
    }
