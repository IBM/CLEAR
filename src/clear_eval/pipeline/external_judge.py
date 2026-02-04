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

logger = logging.getLogger(__name__)


class ExternalJudgeError(Exception):
    """Exception raised for errors in external judge loading or execution."""
    pass


def load_external_judge(judge_path: str, function_name: str = "evaluate") -> Callable:
    """
    Dynamically load an external judge function from a Python file.
    
    Args:
        judge_path: Path to the Python file containing the judge function
        function_name: Name of the function to load (default: "evaluate")
        
    Returns:
        The loaded judge function
        
    Raises:
        ExternalJudgeError: If the judge cannot be loaded or validated
    """
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
    
    Expected signature: (row: pd.Series, config: dict) -> Tuple[str, float]
    
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
        
        # Should have at least 2 parameters (row, config)
        if len(params) < 2:
            raise ExternalJudgeError(
                f"Judge function '{function_name}' must accept at least 2 parameters (row, config). "
                f"Found: {params}"
            )
        
        logger.debug(f"Judge function '{function_name}' signature validated: {sig}")
        
    except Exception as e:
        raise ExternalJudgeError(f"Error validating judge function signature: {str(e)}")


def call_external_judge(
    judge_func: Callable,
    row: pd.Series,
    config: dict
) -> Tuple[str, Any]:
    """
    Call an external judge function and validate its output.
    
    Args:
        judge_func: The judge function to call
        row: The data row to evaluate
        config: Configuration dictionary
        
    Returns:
        Tuple of (evaluation_text, score)
        
    Raises:
        ExternalJudgeError: If the judge execution fails or returns invalid output
    """
    try:
        result = judge_func(row, config)
        
        # Validate return type
        if not isinstance(result, tuple) or len(result) != 2:
            raise ExternalJudgeError(
                f"Judge function must return a tuple of (str, float). Got: {type(result)}"
            )
        
        eval_text, score = result
        
        # Validate evaluation text
        if not isinstance(eval_text, str):
            raise ExternalJudgeError(
                f"Evaluation text must be a string. Got: {type(eval_text)}"
            )
        
        # Validate score (allow pd.NA for missing scores)
        if not pd.isna(score):
            try:
                score = float(score)
                if not (0.0 <= score <= 1.0):
                    logger.warning(
                        f"Score {score} is outside the expected range [0.0, 1.0]. "
                        f"This may affect analysis results."
                    )
            except (TypeError, ValueError):
                raise ExternalJudgeError(
                    f"Score must be a number or pd.NA. Got: {type(score)}"
                )
        
        return eval_text, score
        
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
        'judge_type': config.get('judge_type', 'llm'),
        'external_judge_path': config.get('external_judge_path'),
        'external_judge_function': config.get('external_judge_function', 'evaluate'),
        'external_judge_config': config.get('external_judge_config', {})
    }

# Made with Bob
