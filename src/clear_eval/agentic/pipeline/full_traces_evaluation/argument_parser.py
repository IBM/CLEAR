"""
Shared Argument Parsing
=======================

Common argument parsing utilities for full trajectory evaluation scripts.
Uses unified argument names consistent with the rest of the agentic pipeline.
"""

import argparse
import json
from typing import Optional

from clear_eval.args import add_clear_args_to_parser, str2bool


def parse_dict(arg: str) -> dict:
    """Parse JSON string to dictionary."""
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}")


def add_preprocessing_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """
    Add trace preprocessing arguments to parser.
    
    These arguments control how raw traces are processed into trajectory data.
    Only relevant when --from-raw-traces is True.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Trace Preprocessing (only used when --from-raw-traces=True)")
    
    group.add_argument(
        "--agent-framework",
        type=str,
        choices=['langgraph', 'crewai'],
        default=None,
        help="Agent framework used to generate traces (default: langgraph)",
    )
    group.add_argument(
        "--observability-framework",
        type=str,
        choices=['mlflow', 'langfuse'],
        default=None,
        help="Observability framework used to capture traces (default: mlflow)",
    )
    group.add_argument(
        "--separate-tools",
        type=str2bool,
        default=None,
        help="Separate tool calls in preprocessing (default: false, keep false for now)",
    )


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create base argument parser with unified argument names.
    
    This parser uses the same argument names as the unified pipeline and other
    agentic scripts for consistency. It also includes CLEAR configuration arguments.
    
    Args:
        description: Script description
        
    Returns:
        ArgumentParser with common arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Input/Output directories (unified names)
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Input directory (JSON traces if from-raw-traces=True, else CSV files)",
    )
    parser.add_argument(
        "--from-raw-traces",
        type=str2bool,
        default=None,
        help="If True, process JSON traces; if False, use CSV files directly (default: false)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Base directory for saving evaluation results",
    )
    
    # Add preprocessing arguments
    add_preprocessing_args_to_parser(parser)
    
    # Add all CLEAR configuration arguments (includes provider, eval_model_name, etc.)
    add_clear_args_to_parser(parser, group_name="CLEAR Configuration")
    
    # Full trajectory specific arguments
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help="Model context window size in tokens (default: None)",
    )
    
    # Execution control
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=None,
        help="Re-evaluate even if results already exist (default: true)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max files to evaluate per dataset/model combo",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of concurrent API requests (default: 10)",
    )

    return parser

