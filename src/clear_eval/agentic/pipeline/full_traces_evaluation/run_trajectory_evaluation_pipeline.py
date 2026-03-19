#!/usr/bin/env python3
"""
Unified Trajectory Evaluation Pipeline
=======================================

Orchestrates multiple evaluation types and CLEAR analysis in a single run.

This script:
1. Parses arguments to determine which evaluations to run
2. Optionally generates rubrics (if needed)
3. Runs selected evaluations on trajectory data
4. Runs selective CLEAR analysis on evaluation results

Features:
- Single model/provider configuration for all inference
- Automatic rubric generation
- Selective CLEAR analysis (root_cause, issues, all, or none)
- Parallel execution with configurable concurrency
- Comprehensive error handling and logging

Usage Examples:

    # Run all evaluations with all CLEAR analyses (default)
    python run_trajectory_evaluation_pipeline.py \\
        --traj-input-dir ./trajectories \\
        --output-dir ./results \\
        --model-id gpt-4o \\
        --provider openai

    # Run specific evaluations
    python run_trajectory_evaluation_pipeline.py \\
        --traj-input-dir ./trajectories \\
        --output-dir ./results \\
        --model-id gpt-4o \\
        --provider openai \\
        --eval-types task_success full_trajectory

    # Generate rubrics and run rubric evaluation
    python run_trajectory_evaluation_pipeline.py \\
        --traj-input-dir ./trajectories \\
        --output-dir ./results \\
        --model-id gpt-4o \\
        --provider openai \\
        --eval-types rubric \\
        --generate-rubrics

    # Use existing rubrics
    python run_trajectory_evaluation_pipeline.py \\
        --traj-input-dir ./trajectories \\
        --output-dir ./results \\
        --model-id gpt-4o \\
        --provider openai \\
        --eval-types rubric \\
        --rubric-dir ./rubrics/gpt-4o

    # Run with selective CLEAR analysis
    python run_trajectory_evaluation_pipeline.py \\
        --traj-input-dir ./trajectories \\
        --output-dir ./results \\
        --model-id gpt-4o \\
        --provider openai \\
        --eval-types task_success \\
        --clear-analysis-types root_cause

    # Skip CLEAR analysis entirely
    python run_trajectory_evaluation_pipeline.py \\
        --traj-input-dir ./trajectories \\
        --output-dir ./results \\
        --model-id gpt-4o \\
        --provider openai \\
        --clear-analysis-types none

Arguments:
    Required:
        --traj-input-dir: Directory containing trajectory JSON files
        --output-dir: Base directory for saving results
        --model-id: Model identifier (e.g., gpt-4o, llama-3.3-70b)
        --provider: LLM provider (openai, watsonx, anthropic, etc.)

    Evaluation Control:
        --eval-types: Evaluations to run (default: all)
                     Choices: task_success, full_trajectory, rubric, all

    Rubric Options:
        --generate-rubrics: Generate rubrics before evaluation
        --rubric-dir: Use existing rubrics from directory

    CLEAR Analysis Control:
        --clear-analysis-types: CLEAR analyses to run (default: all)
                               Choices: root_cause, issues, all, none
                               - root_cause: Analyze task_success failures
                               - issues: Analyze full_trajectory issues
                               - all: Run all applicable analyses
                               - none: Skip CLEAR analysis

    Optional:
        --context-tokens: Override auto-detected context window
        --eval-model-params: JSON dict of model parameters
        --overwrite: Re-run even if results exist
        --concurrency: Number of parallel workers (default: 7)
        --max-files: Limit files to process (for testing)

Output Structure:
    output_dir/
    ├── task_success/           # Task success evaluation results
    ├── full_trajectory/        # Full trajectory evaluation results
    ├── rubric_generation/      # Generated rubrics (if --generate-rubrics)
    │   └── <model_id>/
    ├── rubric/                 # Rubric evaluation results
    └── clear_analysis/         # CLEAR analysis results
        ├── root_cause/         # From task_success (if requested)
        └── issues/             # From full_trajectory (if requested)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from clear_eval.logging_config import setup_logging
from agentic.pipeline.full_traces_evaluation.argument_parser import create_base_parser
from agentic.pipeline.full_traces_evaluation.trace_evaluation.task_success_evaluator import TaskSuccessEvaluator
from agentic.pipeline.full_traces_evaluation.trace_evaluation.full_trajectory_evaluator import FullTrajectoryEvaluator
from agentic.pipeline.full_traces_evaluation.trace_evaluation.rubric_evaluator import RubricEvaluator
from agentic.pipeline.full_traces_evaluation.trace_evaluation.rubric_generator import RubricGenerator
from agentic.pipeline.full_traces_evaluation.clear_analysis.root_cause_clear_runner import RootCauseClearRunner
from agentic.pipeline.full_traces_evaluation.clear_analysis.issues_clear_runner import IssuesClearRunner


setup_logging()
logger = logging.getLogger(__name__)


# Evaluation type constants
EVAL_TYPE_TASK_SUCCESS = "task_success"
EVAL_TYPE_FULL_TRAJECTORY = "full_trajectory"
EVAL_TYPE_RUBRIC = "rubric"
ALL_EVAL_TYPES = [EVAL_TYPE_TASK_SUCCESS, EVAL_TYPE_FULL_TRAJECTORY, EVAL_TYPE_RUBRIC]

# CLEAR analysis type constants
CLEAR_TYPE_ROOT_CAUSE = "root_cause"
CLEAR_TYPE_ISSUES = "issues"
ALL_CLEAR_TYPES = [CLEAR_TYPE_ROOT_CAUSE, CLEAR_TYPE_ISSUES]


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the pipeline."""
    parser = create_base_parser(
        description="Run trajectory evaluation pipeline with optional CLEAR analysis"
    )

    # Evaluation type selection
    parser.add_argument(
        "--eval-types",
        type=str,
        nargs="+",
        choices=ALL_EVAL_TYPES + ["all"],
        default=["all"],
        help=(
            "Evaluation types to run. Options: task_success, full_trajectory, rubric, all. "
            "Default: all (runs all three; rubric skipped if no rubrics available)"
        ),
    )

    # Rubric-specific arguments
    parser.add_argument(
        "--rubric-dir",
        type=str,
        default=None,
        help="Directory containing rubric JSON files (for rubric evaluation, or where to save generated rubrics)",
    )

    parser.add_argument(
        "--generate-rubrics",
        action="store_true",
        help="Generate rubrics before running rubric evaluation (saves to --rubric-dir or output_dir/rubrics/<model>)",
    )

    # CLEAR analysis control
    parser.add_argument(
        "--clear-analysis-types",
        type=str,
        nargs="+",
        choices=ALL_CLEAR_TYPES + ["all", "none"],
        default=["all"],
        help=(
            "CLEAR analysis types to run. Options: root_cause (from task_success), "
            "issues (from full_trajectory), all, none. Default: all"
        ),
    )

    return parser


def resolve_eval_types(eval_types: List[str]) -> List[str]:
    """
    Resolve evaluation types, expanding 'all' to actual types.
    
    Args:
        eval_types: List of evaluation type strings
        
    Returns:
        List of resolved evaluation types (including rubric in 'all')
    """
    if "all" in eval_types:
        # 'all' means all three evaluation types (including rubric)
        return [EVAL_TYPE_TASK_SUCCESS, EVAL_TYPE_FULL_TRAJECTORY, EVAL_TYPE_RUBRIC]
    return eval_types


def resolve_clear_analysis_types(
    clear_types: List[str], completed_evals: List[str]
) -> List[str]:
    """
    Resolve CLEAR analysis types based on user selection and completed evaluations.
    
    Args:
        clear_types: List of CLEAR analysis type strings from user
        completed_evals: List of successfully completed evaluation types
        
    Returns:
        List of CLEAR analysis types to run
    """
    if "none" in clear_types:
        return []
    
    if "all" in clear_types:
        # Run all applicable CLEAR analyses based on completed evaluations
        result = []
        if EVAL_TYPE_TASK_SUCCESS in completed_evals:
            result.append(CLEAR_TYPE_ROOT_CAUSE)
        if EVAL_TYPE_FULL_TRAJECTORY in completed_evals:
            result.append(CLEAR_TYPE_ISSUES)
        return result
    
    # User specified specific types - filter by what's available
    result = []
    if CLEAR_TYPE_ROOT_CAUSE in clear_types and EVAL_TYPE_TASK_SUCCESS in completed_evals:
        result.append(CLEAR_TYPE_ROOT_CAUSE)
    elif CLEAR_TYPE_ROOT_CAUSE in clear_types:
        logger.warning(
            "Cannot run root_cause CLEAR analysis: task_success evaluation not completed"
        )
    
    if CLEAR_TYPE_ISSUES in clear_types and EVAL_TYPE_FULL_TRAJECTORY in completed_evals:
        result.append(CLEAR_TYPE_ISSUES)
    elif CLEAR_TYPE_ISSUES in clear_types:
        logger.warning(
            "Cannot run issues CLEAR analysis: full_trajectory evaluation not completed"
        )
    
    return result


def run_task_success_evaluation(
    traj_input_dir: Path,
    output_dir: Path,
    judge_model_id: str,
    provider: str,
    context_tokens: Optional[int],
    overwrite: bool,
    concurrency: int,
    eval_model_params: dict,
    max_files: Optional[int],
) -> bool:
    """Run task success evaluation."""
    logger.info("=" * 80)
    logger.info("RUNNING TASK SUCCESS EVALUATION")
    logger.info("=" * 80)

    try:
        evaluator = TaskSuccessEvaluator(
            judge_model_id=judge_model_id,
            provider=provider,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            concurrency=concurrency,
            eval_model_params=eval_model_params,
            max_files=max_files,
        )
        evaluator.run_pipeline()
        logger.info("Task success evaluation completed successfully")
        return True
    except Exception as e:
        logger.error("Task success evaluation failed: %s", e, exc_info=True)
        return False


def run_full_trajectory_evaluation(
    traj_input_dir: Path,
    output_dir: Path,
    judge_model_id: str,
    provider: str,
    context_tokens: Optional[int],
    overwrite: bool,
    concurrency: int,
    eval_model_params: dict,
    max_files: Optional[int],
) -> bool:
    """Run full trajectory evaluation."""
    logger.info("=" * 80)
    logger.info("RUNNING FULL TRAJECTORY EVALUATION")
    logger.info("=" * 80)

    try:
        evaluator = FullTrajectoryEvaluator(
            judge_model_id=judge_model_id,
            provider=provider,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            concurrency=concurrency,
            eval_model_params=eval_model_params,
            max_files=max_files,
        )
        evaluator.run_pipeline()
        logger.info("Full trajectory evaluation completed successfully")
        return True
    except Exception as e:
        logger.error("Full trajectory evaluation failed: %s", e, exc_info=True)
        return False


def run_rubric_generation(
    traj_input_dir: Path,
    output_dir: Path,
    judge_model_id: str,
    provider: str,
    context_tokens: Optional[int],
    overwrite: bool,
    concurrency: int,
    eval_model_params: dict,
    max_files: Optional[int],
) -> tuple[bool, Optional[Path]]:
    """
    Run rubric generation.
    
    Returns:
        Tuple of (success, rubrics_dir_path)
    """
    logger.info("=" * 80)
    logger.info("RUNNING RUBRIC GENERATION")
    logger.info("=" * 80)

    try:
        generator = RubricGenerator(
            judge_model_id=judge_model_id,
            provider=provider,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            context_tokens=context_tokens or 128000,  # Default if not specified
            overwrite=overwrite,
            concurrency=concurrency,
            eval_model_params=eval_model_params,
            max_files=max_files,
        )
        generator.run_pipeline()
        
        # Rubrics are saved in output_dir/rubric_generation/<model_id>/
        rubrics_dir = generator.results_dir
        logger.info("Rubric generation completed successfully")
        logger.info("Rubrics saved to: %s", rubrics_dir)
        return True, rubrics_dir
    except Exception as e:
        logger.error("Rubric generation failed: %s", e, exc_info=True)
        return False, None


def run_rubric_evaluation(
    traj_input_dir: Path,
    output_dir: Path,
    rubrics_dir: Path,
    judge_model_id: str,
    provider: str,
    context_tokens: Optional[int],
    overwrite: bool,
    concurrency: int,
    eval_model_params: dict,
    max_files: Optional[int],
) -> bool:
    """Run rubric-based evaluation."""
    logger.info("=" * 80)
    logger.info("RUNNING RUBRIC EVALUATION")
    logger.info("=" * 80)

    try:
        evaluator = RubricEvaluator(
            judge_model_id=judge_model_id,
            provider=provider,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            rubrics_dir=rubrics_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            concurrency=concurrency,
            eval_model_params=eval_model_params,
            max_files=max_files,
        )
        evaluator.run_pipeline()
        logger.info("Rubric evaluation completed successfully")
        return True
    except Exception as e:
        logger.error("Rubric evaluation failed: %s", e, exc_info=True)
        return False


def run_clear_analysis(
    eval_results_dir: Path,
    clear_output_dir: Path,
    clear_types: List[str],
    model_id: str,
    provider: str,
    eval_model_params: dict,
    overwrite: bool,
) -> bool:
    """Run CLEAR analysis on evaluation results."""
    if not clear_types:
        logger.info("Skipping CLEAR analysis (none requested)")
        return True
    
    logger.info("=" * 80)
    logger.info("RUNNING CLEAR ANALYSIS")
    logger.info("=" * 80)
    logger.info("CLEAR analysis types: %s", clear_types)

    success = True

    # Run root cause analysis if requested
    if CLEAR_TYPE_ROOT_CAUSE in clear_types:
        logger.info("\nRunning CLEAR analysis on task success results (root causes)...")
        try:
            runner = RootCauseClearRunner(
                eval_results_dir=eval_results_dir,
                output_dir=clear_output_dir,
                clear_model_id=model_id,
                provider=provider,
                overwrite=overwrite,
                eval_model_params=eval_model_params,
            )
            runner.run_analysis()
            logger.info("Root cause CLEAR analysis completed successfully")
        except Exception as e:
            logger.error("Root cause CLEAR analysis failed: %s", e, exc_info=True)
            success = False

    # Run issues analysis if requested
    if CLEAR_TYPE_ISSUES in clear_types:
        logger.info("\nRunning CLEAR analysis on full trajectory results (issues)...")
        try:
            runner = IssuesClearRunner(
                eval_results_dir=eval_results_dir,
                output_dir=clear_output_dir,
                clear_model_id=model_id,
                provider=provider,
                overwrite=overwrite,
                eval_model_params=eval_model_params,
            )
            runner.run_analysis()
            logger.info("Issues CLEAR analysis completed successfully")
        except Exception as e:
            logger.error("Issues CLEAR analysis failed: %s", e, exc_info=True)
            success = False

    return success


def main():
    """Main pipeline orchestration."""
    parser = create_parser()
    args = parser.parse_args()

    # Resolve evaluation types
    eval_types = resolve_eval_types(args.eval_types)
    logger.info("Evaluation types to run: %s", eval_types)

    # Convert paths
    traj_input_dir = Path(args.traj_input_dir)
    output_dir = Path(args.output_dir)

    # Prepare common evaluation arguments
    eval_kwargs = {
        "traj_input_dir": traj_input_dir,
        "output_dir": output_dir,
        "judge_model_id": args.model_id,
        "provider": args.provider,
        "context_tokens": args.context_tokens,
        "overwrite": args.overwrite,
        "concurrency": args.concurrency,
        "eval_model_params": args.eval_model_params,
        "max_files": args.max_files,
    }

    # Track which evaluations succeeded
    completed_evals = []
    failed_evals = []

    # Handle rubric generation/validation if rubric evaluation is requested
    rubric_dir = None
    if EVAL_TYPE_RUBRIC in eval_types:
        if args.generate_rubrics:
            # Generate rubrics
            logger.info("Generating rubrics...")
            success, generated_rubric_dir = run_rubric_generation(**eval_kwargs)
            if success and generated_rubric_dir:
                rubric_dir = generated_rubric_dir
                logger.info("Using generated rubrics from: %s", rubric_dir)
            else:
                logger.error("Rubric generation failed")
                failed_evals.append("rubric_generation")
                eval_types.remove(EVAL_TYPE_RUBRIC)  # Skip rubric evaluation
        elif args.rubric_dir:
            # Use existing rubrics
            rubric_dir = Path(args.rubric_dir)
            if not rubric_dir.exists():
                logger.warning("Rubric directory does not exist: %s", rubric_dir)
                logger.warning("Skipping rubric evaluation")
                eval_types.remove(EVAL_TYPE_RUBRIC)  # Skip rubric evaluation
            else:
                logger.info("Using existing rubrics from: %s", rubric_dir)
        else:
            # No rubrics available - warn and skip
            logger.warning("Rubric evaluation requested but no rubrics available")
            logger.warning("Use --rubric-dir to specify existing rubrics or --generate-rubrics to create them")
            logger.warning("Skipping rubric evaluation")
            eval_types.remove(EVAL_TYPE_RUBRIC)  # Skip rubric evaluation

    # Run evaluations
    if EVAL_TYPE_TASK_SUCCESS in eval_types:
        if run_task_success_evaluation(**eval_kwargs):
            completed_evals.append(EVAL_TYPE_TASK_SUCCESS)
        else:
            failed_evals.append(EVAL_TYPE_TASK_SUCCESS)

    if EVAL_TYPE_FULL_TRAJECTORY in eval_types:
        if run_full_trajectory_evaluation(**eval_kwargs):
            completed_evals.append(EVAL_TYPE_FULL_TRAJECTORY)
        else:
            failed_evals.append(EVAL_TYPE_FULL_TRAJECTORY)

    if EVAL_TYPE_RUBRIC in eval_types and rubric_dir:
        if run_rubric_evaluation(rubrics_dir=rubric_dir, **eval_kwargs):
            completed_evals.append(EVAL_TYPE_RUBRIC)
        else:
            failed_evals.append(EVAL_TYPE_RUBRIC)

    # Run CLEAR analysis if requested and evaluations succeeded
    if completed_evals:
        # Resolve which CLEAR analyses to run
        clear_types = resolve_clear_analysis_types(
            args.clear_analysis_types, completed_evals
        )
        
        if clear_types:
            clear_output_dir = output_dir / "clear_analysis"
            clear_output_dir.mkdir(parents=True, exist_ok=True)

            run_clear_analysis(
                eval_results_dir=output_dir,
                clear_output_dir=clear_output_dir,
                clear_types=clear_types,
                model_id=args.model_id,
                provider=args.provider,
                eval_model_params=args.eval_model_params,
                overwrite=args.overwrite,
            )

    # Print summary
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info("Completed evaluations: %s", completed_evals or "None")
    logger.info("Failed evaluations: %s", failed_evals or "None")
    logger.info("Results directory: %s", output_dir)

    if failed_evals:
        logger.warning("Some evaluations failed. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All evaluations completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
