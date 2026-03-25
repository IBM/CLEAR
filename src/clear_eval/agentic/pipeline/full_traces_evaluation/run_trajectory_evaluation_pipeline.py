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
- Parallel execution with configurable max_workers
- Comprehensive error handling and logging

Usage Examples:

    # Run all evaluations with all CLEAR analyses (default)
    python run_trajectory_evaluation_pipeline.py \\
        --data-dir ./trajectories \\
        --results-dirr ./results \\
        --eval-model-name gpt-4o \\
        --provider openai

    # Run specific evaluations
    python run_trajectory_evaluation_pipeline.py \\
        --data-dir ./trajectories \\
        --results-dirr ./results \\
        --eval-model-name gpt-4o \\
        --provider openai \\
        --eval-types task_success full_trajectory

    # Generate rubrics and run rubric evaluation
    python run_trajectory_evaluation_pipeline.py \\
        --data-dir ./trajectories \\
        --results-dirr ./results \\
        --eval-model-name gpt-4o \\
        --provider openai \\
        --eval-types rubric \\
        --generate-rubrics

    # Use existing rubrics
    python run_trajectory_evaluation_pipeline.py \\
        --data-dir ./trajectories \\
        --results-dirr ./results \\
        --eval-model-name gpt-4o \\
        --provider openai \\
        --eval-types rubric \\
        --rubric-dir ./rubrics/gpt-4o

    # Run with selective CLEAR analysis
    python run_trajectory_evaluation_pipeline.py \\
        --data-dir ./trajectories \\
        --results-dirr ./results \\
        --eval-model-name gpt-4o \\
        --provider openai \\
        --eval-types task_success \\
        --clear-analysis-types root_cause

    # Skip CLEAR analysis entirely
    python run_trajectory_evaluation_pipeline.py \\
        --data-dir ./trajectories \\
        --results-dirr ./results \\
        --eval-model-name gpt-4o \\
        --provider openai \\
        --clear-analysis-types none

Arguments:
    Required:
        --data-dir: Directory containing trajectory JSON files
        --results-dirr: Base directory for saving results
        --eval-model-name: Model identifier (e.g., gpt-4o, llama-3.3-70b)
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
        --max-workers: Number of parallel workers (default: 7)
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
import pandas as pd

from clear_eval.logging_config import setup_logging
from clear_eval.agentic.pipeline.utils import (
    build_cli_overrides,
    load_pipeline_config,
    get_run_output_dir,
    validate_required_config,
    InferenceConfig,
)
from clear_eval.agentic.pipeline.preprocess_traces.preprocess_traces import process_traces_to_traj_data
from clear_eval.agentic.pipeline.full_traces_evaluation.argument_parser import create_base_parser
from clear_eval.agentic.pipeline.full_traces_evaluation.trace_evaluation.task_success_evaluator import TaskSuccessEvaluator
from clear_eval.agentic.pipeline.full_traces_evaluation.trace_evaluation.full_trajectory_evaluator import FullTrajectoryEvaluator
from clear_eval.agentic.pipeline.full_traces_evaluation.trace_evaluation.rubric_evaluator import RubricEvaluator
from clear_eval.agentic.pipeline.full_traces_evaluation.trace_evaluation.rubric_generator import RubricGenerator
from clear_eval.agentic.pipeline.full_traces_evaluation.clear_analysis.root_cause_clear_runner import RootCauseClearRunner
from clear_eval.agentic.pipeline.full_traces_evaluation.clear_analysis.issues_clear_runner import IssuesClearRunner


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
    
    # Add optional config file argument
    parser.add_argument(
        "--agentic-config-path",
        type=str,
        default=None,
        help="Path to config file (JSON or YAML). CLI args override config values."
    )

    # Evaluation type selection
    parser.add_argument(
        "--eval-types",
        type=str,
        nargs="+",
        choices=ALL_EVAL_TYPES + ["all"],
        default=None,
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
        default=None,
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
    inference_config: InferenceConfig,
    context_tokens: Optional[int],
    overwrite: bool,
    max_workers: int,
    max_files: Optional[int],
) -> bool:
    """Run task success evaluation."""
    logger.info("=" * 80)
    logger.info("RUNNING TASK SUCCESS EVALUATION")
    logger.info("=" * 80)

    try:
        evaluator = TaskSuccessEvaluator(
            inference_config=inference_config,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            max_workers=max_workers,
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
    inference_config: InferenceConfig,
    context_tokens: Optional[int],
    overwrite: bool,
    max_workers: int,
    max_files: Optional[int],
) -> bool:
    """Run full trajectory evaluation."""
    logger.info("=" * 80)
    logger.info("RUNNING FULL TRAJECTORY EVALUATION")
    logger.info("=" * 80)

    try:
        evaluator = FullTrajectoryEvaluator(
            inference_config=inference_config,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            max_workers=max_workers,
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
    inference_config: InferenceConfig,
    context_tokens: Optional[int],
    overwrite: bool,
    max_workers: int,
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
            inference_config=inference_config,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            max_workers=max_workers,
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
    inference_config: InferenceConfig,
    context_tokens: Optional[int],
    overwrite: bool,
    max_workers: int,
    max_files: Optional[int],
) -> bool:
    """Run rubric-based evaluation."""
    logger.info("=" * 80)
    logger.info("RUNNING RUBRIC EVALUATION")
    logger.info("=" * 80)

    try:
        evaluator = RubricEvaluator(
            inference_config=inference_config,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            rubrics_dir=rubrics_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            max_workers=max_workers,
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
    inference_config: InferenceConfig,
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
                inference_config=inference_config,
                overwrite=overwrite,
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
                inference_config=inference_config,
                overwrite=overwrite,
            )
            runner.run_analysis()
            logger.info("Issues CLEAR analysis completed successfully")
        except Exception as e:
            logger.error("Issues CLEAR analysis failed: %s", e, exc_info=True)
            success = False

    return success


def preprocess_traces_if_needed(
    input_dir: Path,
    output_dir: Path,
    from_raw_traces: bool,
    agent_framework: str = "langgraph",
    observability_framework: str = "mlflow",
    separate_tools: bool = False,
) -> Path:
    """
    Preprocess raw traces to CSV if needed, otherwise return input directory.
    
    Args:
        input_dir: Input directory (JSON traces or CSV files)
        output_dir: Base output directory
        from_raw_traces: If True, process JSON traces; if False, use CSV files directly
        agent_framework: Agent framework (for JSON processing)
        observability_framework: Observability framework (for JSON processing)
        separate_tools: Separate tool calls (for JSON processing)
        
    Returns:
        Path to directory containing CSV files
    """
    if from_raw_traces:
        # Process JSON traces to CSV
        logger.info("=" * 80)
        logger.info("PREPROCESSING: Converting raw traces to CSV")
        logger.info("=" * 80)
        logger.info(f"Input: {input_dir}")
        logger.info(f"Agent framework: {agent_framework}")
        logger.info(f"Observability framework: {observability_framework}")
        
        traces_data_dir = output_dir / "traces_data"
        traces_data_dir.mkdir(parents=True, exist_ok=True)
        
        process_traces_to_traj_data(
            input_dir=str(input_dir),
            output_dir=str(traces_data_dir),
            agent_framework=agent_framework,
            observability_framework=observability_framework,
            separate_tools=separate_tools
        )
        logger.info(f"✓ Processed traces to: {traces_data_dir}")
        return traces_data_dir
    else:
        # Use CSV files directly from input directory
        logger.info(f"Using existing CSV files from: {input_dir}")
        return input_dir


def check_intent_availability(traj_input_dir: Path) -> tuple[bool, int, int]:
    """
    Check if all trajectory files have valid intent data.
    
    Intent is considered valid if it exists, is not None, and is not empty/whitespace.
    
    Args:
        traj_input_dir: Directory containing CSV trajectory files
        
    Returns:
        Tuple of (all_have_intent, total_files, files_with_intent)
    """
    csv_files = list(traj_input_dir.glob("*.csv"))
    total_files = len(csv_files)
    files_with_intent = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                first_row = df.iloc[0]
                intent = first_row.get("intent", "")
                # Check if intent is valid (not None, not empty, not just whitespace)
                if intent is not None and not pd.isna(intent) and str(intent).strip():
                    files_with_intent += 1
        except Exception as e:
            logger.warning(f"Failed to check intent in {csv_file}: {e}")
            continue
    
    all_have_intent = files_with_intent == total_files
    return all_have_intent, total_files, files_with_intent


def run_trajectory_evaluation_pipeline(
    traj_input_dir: Path,
    output_dir: Path,
    inference_config: InferenceConfig,
    eval_types: List[str],
    generate_rubrics: bool = False,
    rubric_dir: Optional[Path] = None,
    clear_analysis_types: Optional[List[str]] = None,
    context_tokens: Optional[int] = None,
    overwrite: bool = False,
    max_workers: int = 10,
    max_files: Optional[int] = None,
) -> tuple[List[str], List[str]]:
    """
    Run trajectory evaluation pipeline on CSV trajectory data.

    This function expects CSV files to already be available in traj_input_dir.
    For preprocessing raw traces, use the main() function or call process_traces_to_traj_data() first.

    Args:
        traj_input_dir: Directory containing CSV trajectory files
        output_dir: Base directory for saving results
        inference_config: LLM inference configuration
        eval_types: Evaluations to run (task_success, full_trajectory, rubric, all)
        generate_rubrics: Generate rubrics before evaluation
        rubric_dir: Path to existing rubrics
        clear_analysis_types: CLEAR analyses to run (root_cause, issues, all, none)
        context_tokens: Model context window size
        overwrite: Re-run even if results exist
        max_workers: Number of parallel workers
        max_files: Limit files to process

    Returns:
        Tuple of (completed_evals, failed_evals)
    """
    if clear_analysis_types is None:
        clear_analysis_types = ["all"]

    logger.info(f"Using CSV trajectory files from: {traj_input_dir}")
    
    # Resolve evaluation types
    eval_types = resolve_eval_types(eval_types)
    logger.info("Evaluation types to run: %s", eval_types)
    
    # Check if rubric evaluation is requested and verify intent availability
    if EVAL_TYPE_RUBRIC in eval_types:
        logger.info("Checking intent availability for rubric evaluation...")
        all_have_intent, total_files, files_with_intent = check_intent_availability(traj_input_dir)
        
        if files_with_intent == 0:
            # No trajectories have intent - skip rubric evaluation entirely
            logger.warning("=" * 80)
            logger.warning("RUBRIC EVALUATION SKIPPED")
            logger.warning("=" * 80)
            logger.warning(
                f"None of the {total_files} trajectory files have intent data. "
                "Rubric evaluation requires intent to generate/evaluate rubrics."
            )
            logger.warning("Removing rubric evaluation from the pipeline.")
            logger.warning("=" * 80)
            eval_types.remove(EVAL_TYPE_RUBRIC)
            
            # Also skip rubric generation if it was requested
            if generate_rubrics:
                logger.warning("Skipping rubric generation as well.")
                generate_rubrics = False
        elif not all_have_intent:
            # Some trajectories have intent, some don't - run on those with intent
            logger.info("=" * 80)
            logger.info(
                f"Intent data found in {files_with_intent}/{total_files} trajectory files. "
                "Rubric evaluation will run on trajectories with intent data only. "
                "Trajectories without intent will be skipped automatically."
            )
            logger.info("=" * 80)

    # Prepare common evaluation arguments
    eval_kwargs = {
        "traj_input_dir": traj_input_dir,
        "output_dir": output_dir,
        "inference_config": inference_config,
        "context_tokens": context_tokens,
        "overwrite": overwrite,
        "max_workers": max_workers,
        "max_files": max_files,
    }

    # Track which evaluations succeeded
    completed_evals = []
    failed_evals = []

    # Handle rubric generation/validation if rubric evaluation is requested
    rubric_dir_path = None
    if EVAL_TYPE_RUBRIC in eval_types:
        if generate_rubrics:
            # Generate rubrics
            logger.info("Generating rubrics...")
            success, generated_rubric_dir = run_rubric_generation(**eval_kwargs)
            if success and generated_rubric_dir:
                rubric_dir_path = generated_rubric_dir
                logger.info("Using generated rubrics from: %s", rubric_dir_path)
            else:
                logger.error("Rubric generation failed")
                failed_evals.append("rubric_generation")
                eval_types.remove(EVAL_TYPE_RUBRIC)  # Skip rubric evaluation
        elif rubric_dir:
            # Use existing rubrics
            rubric_dir_path = Path(rubric_dir) if not isinstance(rubric_dir, Path) else rubric_dir
            if not rubric_dir_path.exists():
                logger.warning("Rubric directory does not exist: %s", rubric_dir_path)
                logger.warning("Skipping rubric evaluation")
                eval_types.remove(EVAL_TYPE_RUBRIC)  # Skip rubric evaluation
            else:
                logger.info("Using existing rubrics from: %s", rubric_dir_path)
        else:
            # No rubrics available - warn and skip
            logger.warning("Rubric evaluation requested but no rubrics available")
            logger.warning("Use rubric_dir to specify existing rubrics or generate_rubrics=True to create them")
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

    if EVAL_TYPE_RUBRIC in eval_types and rubric_dir_path:
        if run_rubric_evaluation(rubrics_dir=rubric_dir_path, **eval_kwargs):
            completed_evals.append(EVAL_TYPE_RUBRIC)
        else:
            failed_evals.append(EVAL_TYPE_RUBRIC)

    # Run CLEAR analysis if requested and evaluations succeeded
    if completed_evals:
        # Resolve which CLEAR analyses to run
        clear_types = resolve_clear_analysis_types(
            clear_analysis_types, completed_evals
        )
        
        if clear_types:
            clear_output_dir = output_dir / "clear_analysis"
            clear_output_dir.mkdir(parents=True, exist_ok=True)

            run_clear_analysis(
                eval_results_dir=output_dir,
                clear_output_dir=clear_output_dir,
                clear_types=clear_types,
                inference_config=inference_config,
                overwrite=overwrite,
            )

    # Print summary
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info("Completed evaluations: %s", completed_evals or "None")
    logger.info("Failed evaluations: %s", failed_evals or "None")
    logger.info("Results directory: %s", output_dir)

    return completed_evals, failed_evals


def main():
    """Main pipeline orchestration (CLI entry point)."""
    parser = create_parser()
    args = parser.parse_args()

    # Build CLI overrides (only include non-None arguments)
    cli_overrides = build_cli_overrides(args)

    # Load configuration
    config = load_pipeline_config(args.agentic_config_path, **cli_overrides)

    # Validate required parameters
    validate_required_config(config, ['data_dir', 'results_dir'], parser)

    # Get run output directory
    output_dir, run_name = get_run_output_dir(
        config['results_dir'],
        config.get('run_name')
    )

    # Convert paths
    traj_input_dir = Path(config['data_dir'])
    rubric_dir = Path(config['rubric_dir']) if config.get('rubric_dir') else None

    # Create inference config
    inference_config = InferenceConfig.from_config(config)

    # Validate input directory
    if not traj_input_dir.exists():
        logger.error(f"Input directory does not exist: {traj_input_dir}")
        sys.exit(1)

    # Preprocess traces if needed
    csv_input_dir = preprocess_traces_if_needed(
        input_dir=traj_input_dir,
        output_dir=output_dir,
        from_raw_traces=config.get('from_raw_traces'),
        agent_framework=config.get('agent_framework'),
        observability_framework=config.get('observability_framework'),
        separate_tools=config.get('separate_tools')
    )

    # Run the evaluation pipeline
    logger.info("=" * 80)
    logger.info("TRAJECTORY EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input: {traj_input_dir}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {inference_config.model_id} ({inference_config.provider})")

    completed_evals, failed_evals = run_trajectory_evaluation_pipeline(
        traj_input_dir=csv_input_dir,
        output_dir=output_dir,
        inference_config=inference_config,
        eval_types=config.get('eval_types'),
        generate_rubrics=config.get('generate_rubrics'),
        rubric_dir=rubric_dir,
        clear_analysis_types=config.get('clear_analysis_types'),
        context_tokens=config.get('context_tokens'),
        overwrite=config.get('overwrite'),
        max_workers=config.get('max_workers'),
        max_files=config.get('max_files'),
    )

    if failed_evals:
        logger.warning("Some evaluations failed. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All evaluations completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
