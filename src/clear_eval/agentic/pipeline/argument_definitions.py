"""
Centralized Argument Definitions for Agentic Pipeline
======================================================

All argparse argument definitions for the agentic pipeline in one place.
Each function adds a specific group of related arguments.

This module eliminates duplication across multiple pipeline scripts by providing
reusable argument groups that can be composed based on each script's needs.
"""

import argparse
from clear_eval.args import str2bool


def add_agentic_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """
    Add core agentic pipeline arguments.
    
    These are the fundamental arguments used by most pipeline scripts for
    input/output configuration, execution control, and pipeline behavior.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Agentic Pipeline")
    
    group.add_argument(
        "--agentic-config-path",
        help="Path to config file (JSON or YAML)"
    )
    group.add_argument(
        "--data-dir",
        help="Input directory (JSON traces if from-raw-traces=True, else CSV files)"
    )
    group.add_argument(
        "--results-dir",
        help="Output directory for pipeline results"
    )
    group.add_argument(
        "--from-raw-traces",
        type=str2bool,
        help="If True, process JSON traces; if False, use CSV files directly"
    )
    group.add_argument(
        "--overwrite",
        type=str2bool,
        help="Whether to overwrite existing results"
    )
    group.add_argument(
        "--memory-only",
        type=str2bool,
        help="If true, use temporary directories and save only ui_input and json_result to results_dir"
    )
    group.add_argument(
        "--separate-tools",
        type=str2bool,
        default=None,
        help="Enable per-tool-call evaluation (tools_with_reasoning mode). "
             "If false: single combined evaluation per LLM call. "
             "If true: one evaluation per tool call with reasoning in input."
    )
    # Note: --max-workers is defined in add_clear_args_to_parser() in src/clear_eval/args.py
    # Do not add it here to avoid conflicts


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add trace preprocessing arguments (raw traces -> trajectory CSVs).
    
    These arguments control how raw traces from observability platforms are
    converted into trajectory CSV files. Only relevant when --from-raw-traces=True.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Trace Preprocessing (only used when --from-raw-traces=True)")
    
    group.add_argument(
        "--agent-framework",
        choices=['langgraph', 'crewai', 'atif'],
        help="Agent framework used to generate traces"
    )
    group.add_argument(
        "--observability-framework",
        choices=['mlflow', 'langfuse'],
        help="Observability framework used to capture traces"
    )


def add_full_trajectory_args(parser: argparse.ArgumentParser) -> None:
    """
    Add full trajectory evaluation arguments.
    
    These arguments control the full trajectory evaluation pipeline, including
    task success evaluation, rubric-based evaluation, and CLEAR analysis.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Full Trajectory Evaluation")
    
    group.add_argument(
        "--eval-types",
        nargs='+',
        choices=['task_success', 'full_trajectory', 'rubric', 'all'],
        help="Evaluations to run. Options: task_success, full_trajectory, rubric, all"
    )
    group.add_argument(
        "--generate-rubrics",
        type=str2bool,
        help="Generate rubrics before evaluation"
    )
    group.add_argument(
        "--rubric-dir",
        help="Path to existing rubrics (if not generating)"
    )
    group.add_argument(
        "--clear-analysis-types",
        nargs='+',
        choices=['root_cause', 'issues', 'all', 'none'],
        help="CLEAR analyses to run on full trajectory results. Options: root_cause, issues, all, none"
    )
    group.add_argument(
        "--context-tokens",
        type=int,
        help="Model context window size (for full trajectory)"
    )
    group.add_argument(
        "--max-files",
        type=int,
        help="Limit files to process (for testing)"
    )


def add_unified_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for unified pipeline mode control.
    
    These arguments control which pipelines to run when using the unified
    pipeline script (step-by-step and/or full trajectory).
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    group = parser.add_argument_group("Pipeline Mode Control")
    
    group.add_argument(
        "--run-step-by-step",
        type=str2bool,
        help="Enable step-by-step CLEAR analysis"
    )
    group.add_argument(
        "--run-full-trajectory",
        type=str2bool,
        help="Enable full trajectory evaluation"
    )