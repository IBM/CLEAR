#!/usr/bin/env python3
"""
Unified Agentic Pipeline Script
================================

This script provides a unified interface to run both:
1. Step-by-step CLEAR analysis (on trajectory data)
2. Full trajectory evaluation (task success, full trajectory, rubric, CLEAR analysis)

All results are organized under: clear_results/<judge-model>/<run-name>/

Configuration Precedence (lowest to highest):
    1. Default config: setup/default_agentic_config.yaml
    2. User config file: --agentic-config-path (if provided)
    3. CLI arguments (override both config files)

Usage Examples:

    # Run both pipelines with config file
    python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \\
        --agentic-config-path my_config.yaml

    # Run only step-by-step from raw traces
    python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \\
        --agentic-input-dir data/experiment_001 \\
        --agentic-output-dir results \\
        --run-step-by-step true \\
        --run-full-trajectory false \\
        --from-raw-traces true \\
        --eval-model-name openai/gpt-oss-120b \\
        --provider watsonx

    # Run only step-by-step from preprocessed CSVs
    python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \\
        --agentic-input-dir data/experiment_001 \\
        --agentic-output-dir results \\
        --run-step-by-step true \\
        --run-full-trajectory false \\
        --from-raw-traces false \\
        --eval-model-name openai/gpt-oss-120b

    # Run both with custom run name
    python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \\
        --agentic-config-path my_config.yaml \\
        --run-name experiment_gpt4_001 \\
        --eval-model-name gpt-4o
"""

import argparse
import json
import logging
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from clear_eval.agentic.pipeline.run_clear_step_analysis import (
    run_step_analysis_pipeline,
)
from clear_eval.agentic.pipeline.preprocess_traces.preprocess_traces import process_traces_to_traj_data
from clear_eval.agentic.pipeline.utils import build_cli_overrides
from clear_eval.args import add_clear_args_to_parser, str2bool
from clear_eval.logging_config import setup_logging
from clear_eval.pipeline.config_loader import load_config
from clear_eval.agentic.pipeline.full_traces_evaluation.run_trajectory_evaluation_pipeline import run_trajectory_evaluation_pipeline
from clear_eval.agentic.pipeline.full_traces_evaluation.argument_parser import add_preprocessing_args_to_parser
FULL_TRAJ_AVAILABLE = True

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Path to unified default config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "setup", "default_agentic_config.yaml")


def add_agentic_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add agentic pipeline arguments to the parser."""
    group = parser.add_argument_group("Agentic Pipeline Arguments")
    
    group.add_argument(
        "--agentic-config-path",
        help="Path to unified config file (JSON or YAML)"
    )
    group.add_argument(
        "--agentic-input-dir",
        help="Input directory (JSON traces if from-raw-traces=True, else CSV files)"
    )
    group.add_argument(
        "--from-raw-traces",
        type=str2bool,
        help="If True, process JSON traces; if False, use CSV files directly (default: false)"
    )
    group.add_argument(
        "--agentic-output-dir",
        help="Base output directory (required)"
    )
    
    # Pipeline mode control
    group.add_argument(
        "--run-step-by-step",
        type=str2bool,
        help="Enable step-by-step CLEAR analysis (default: true)"
    )
    group.add_argument(
        "--run-full-trajectory",
        type=str2bool,
        help="Enable full trajectory evaluation (default: true)"
    )
    
    # Add preprocessing arguments (agent-framework, observability-framework, separate-tools)
    add_preprocessing_args_to_parser(parser)
    
    # Full trajectory options
    group.add_argument(
        "--eval-types",
        nargs='+',
        choices=['task_success', 'full_trajectory', 'rubric', 'all'],
        help="Evaluations to run (default: all)"
    )
    group.add_argument(
        "--generate-rubrics",
        type=str2bool,
        help="Generate rubrics before evaluation"
    )
    group.add_argument(
        "--rubric-dir",
        help="Path to existing rubrics"
    )
    group.add_argument(
        "--clear-analysis-types",
        nargs='+',
        choices=['root_cause', 'issues', 'all', 'none'],
        help="CLEAR analyses to run on full trajectory results (default: all)"
    )
    
    # Execution control
    group.add_argument(
        "--overwrite",
        type=str2bool,
        help="Overwrite existing results (default: true)"
    )
    group.add_argument(
        "--max-files",
        type=int,
        help="Limit files to process (for testing)"
    )
    group.add_argument(
        "--context-tokens",
        type=int,
        help="Model context window (for full trajectory)"
    )
    group.add_argument(
        "--memory-only",
        type=str2bool,
        help="If true, use temporary directories and save only ui_input and json_result to agentic_output_dir (default: false)"
    )


def create_output_structure(
    output_dir: Path,
    run_name: str
) -> Dict[str, Path]:
    """
    Create organized output directory structure.
    
    Returns paths:
        {
            'base': output_dir/clear_results/<judge-model>/<run-name>/,
            'step_by_step': .../step_by_step/,
            'full_trajectory': .../full_trajectory/,
        }
    """
    base_dir = output_dir / 'clear_results' / run_name
    
    paths = {
        'base': base_dir,
        'step_by_step': base_dir / 'step_by_step',
        'full_trajectory': base_dir / 'full_trajectory',
    }
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def prepare_traces_data(
    agentic_input_dir: Path,
    from_raw_traces: bool,
    output_paths: Dict[str, Path],
    config: dict
) -> Optional[Path]:
    """
    Prepare centralized traces_data directory.
    
    Creates run_name/traces_data by either:
    1. Processing raw JSON traces if from_raw_traces=True
    2. Copying existing CSV files if from_raw_traces=False
    
    Args:
        agentic_input_dir: Input directory (JSON traces or CSV files)
        from_raw_traces: If True, process JSON traces; if False, use CSV files
        output_paths: Output directory structure
        config: Configuration dict
        
    Returns:
        Path to traces_data directory, or None if preparation failed
    """
    traces_data_dir = output_paths['base'] / 'traces_data'
    
    logger.info("=" * 80)
    logger.info("PREPARING CENTRALIZED TRACES_DATA")
    logger.info("=" * 80)
    
    if from_raw_traces:
        # Process raw JSON traces
        logger.info(f"Processing raw JSON traces from: {agentic_input_dir}")
        logger.info(f"Output directory: {traces_data_dir}")
        
        try:
            process_traces_to_traj_data(
                input_dir=str(agentic_input_dir),
                output_dir=str(traces_data_dir),
                agent_framework=config.get('agent_framework'),
                observability_framework=config.get('observability_framework'),
                separate_tools=config.get('separate_tools', False)
            )
            logger.info(f"✓ Processed traces successfully")
            return traces_data_dir
        except Exception as e:
            logger.error(f"Failed to process traces: {e}", exc_info=True)
            return None
    else:
        # Copy existing CSV files
        logger.info(f"Using existing CSV files from: {agentic_input_dir}")
        logger.info(f"Copying to: {traces_data_dir}")
        
        try:
            if traces_data_dir.exists():
                shutil.rmtree(traces_data_dir)
            shutil.copytree(agentic_input_dir, traces_data_dir)
            logger.info(f"✓ Copied CSV files successfully")
            return traces_data_dir
        except Exception as e:
            logger.error(f"Failed to copy CSV files: {e}", exc_info=True)
            return None


def run_step_by_step_pipeline(
    traces_data_dir: Path,
    output_dir: Path,
    config: dict
) -> bool:
    """
    Run step-by-step CLEAR analysis pipeline using centralized traces_data.
    
    Args:
        traces_data_dir: Path to centralized traces_data directory
        output_dir: Output directory for step-by-step results
        config: Configuration dict
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("RUNNING STEP-BY-STEP CLEAR ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Using traces_data from: {traces_data_dir}")
    
    try:
        # Call run_step_analysis_pipeline for steps 2-4
        logger.info("Calling run_step_analysis_pipeline")
        run_step_analysis_pipeline(
            traces_data_dir=str(traces_data_dir),
            agentic_output_dir=str(output_dir),
            config_dict=config,
            overwrite=config.get('overwrite', True)
        )
        
        logger.info("Step-by-step pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Step-by-step pipeline failed: {e}", exc_info=True)
        return False


def run_full_trajectory_pipeline(
    traces_data_dir: Path,
    output_dir: Path,
    config: dict
) -> bool:
    """
    Run full trajectory evaluation pipeline using centralized traces_data.
    
    Args:
        traces_data_dir: Path to centralized traces_data directory (CSV files)
        output_dir: Output directory for full trajectory results
        config: Configuration dict
        
    Returns:
        True if successful, False otherwise
    """
    if not FULL_TRAJ_AVAILABLE or run_trajectory_evaluation_pipeline is None:
        logger.error("Full trajectory evaluation components not available")
        return False
    
    logger.info("=" * 80)
    logger.info("RUNNING FULL TRAJECTORY EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Using traces_data from: {traces_data_dir}")
    
    try:
        
        # Prepare rubric_dir
        rubric_dir = None
        if config.get('rubric_dir'):
            rubric_dir = Path(config['rubric_dir'])
        
        # Call the trajectory evaluation pipeline with CSV files
        # Note: traces_data_dir already contains CSV files, no preprocessing needed
        completed_evals, failed_evals = run_trajectory_evaluation_pipeline(
            traj_input_dir=traces_data_dir,
            output_dir=output_dir,
            model_id=config.get('eval_model_name', 'openai/gpt-oss-120b'),
            provider=config.get('provider'),
            inference_backend=config.get('inference_backend'),
            eval_types=config.get('eval_types', ['all']),
            generate_rubrics=config.get('generate_rubrics', False),
            rubric_dir=rubric_dir,
            clear_analysis_types=config.get('clear_analysis_types', ['all']),
            context_tokens=config.get('context_tokens'),
            overwrite=config.get('overwrite', True),
            max_workers=config.get('max_workers'),
            eval_model_params=config.get('eval_model_params', {}),
            max_files=config.get('max_files'),
        )
        
        logger.info("=" * 80)
        logger.info("FULL TRAJECTORY PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Completed evaluations: {completed_evals or 'None'}")
        logger.info(f"Failed evaluations: {failed_evals or 'None'}")
        
        return len(failed_evals) == 0
        
    except Exception as e:
        logger.error(f"Full trajectory pipeline failed: {e}", exc_info=True)
        return False


def create_pipeline_summary(base_dir: Path, config: dict, results: dict):
    """Create pipeline execution summary."""
    summary = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "run_name": config.get('run_name'),
            "eval_model_name": config.get('eval_model_name'),
            "provider": config.get('provider'),
            "run_step_by_step": config.get('run_step_by_step', True),
            "run_full_trajectory": config.get('run_full_trajectory', True),
        },
        "results": results,
        "output_structure": {
            "step_by_step/": "Step-by-step CLEAR analysis results" if results.get('step_by_step_success') else "Not run",
            "full_trajectory/": "Full trajectory evaluation results" if results.get('full_trajectory_success') else "Not run",
        }
    }
    
    summary_path = base_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Created pipeline summary: {summary_path}")

def main():
    """Main entry point for unified agentic pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified Agentic Pipeline: Run step-by-step and/or full trajectory analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add agentic pipeline arguments
    add_agentic_args_to_parser(parser)
    
    # Add CLEAR configuration arguments
    add_clear_args_to_parser(parser, group_name="CLEAR Configuration")
    
    args = parser.parse_args()

    cli_overrides = build_cli_overrides(args)
    
    # Load configuration with precedence: default -> user config -> CLI overrides
    config = load_config(
        DEFAULT_CONFIG_PATH,
        args.agentic_config_path,
        **cli_overrides
    )
    
    # Validate required parameters
    if not config.get('agentic_output_dir'):
        parser.error("agentic_output_dir is required (set in config or use --agentic-output-dir)")
    
    if not config.get('agentic_input_dir'):
        parser.error("agentic_input_dir is required (set in config or use --agentic-input-dir)")
    
    # Extract parameters
    agentic_input_dir = Path(config['agentic_input_dir'])
    output_dir = Path(config['agentic_output_dir'])
    from_raw_traces = config.get('from_raw_traces', False)
    run_name = config.get('run_name') or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Validate input directory exists
    if not agentic_input_dir.exists():
        parser.error(f"Input directory does not exist: {agentic_input_dir}")
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("UNIFIED AGENTIC PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input directory: {agentic_input_dir}")
    logger.info(f"From raw traces: {from_raw_traces}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Run name: {run_name}")
    
    # Create output structure
    output_paths = create_output_structure(output_dir, run_name)
    logger.info(f"Output base directory: {output_paths['base']}")
    
    # Prepare centralized traces_data directory
    traces_data_dir = prepare_traces_data(
        agentic_input_dir,
        from_raw_traces,
        output_paths,
        config
    )
    
    if not traces_data_dir:
        logger.error("Failed to prepare traces_data. Cannot proceed with pipelines.")
        sys.exit(1)
    
    logger.info(f"✓ Centralized traces_data ready at: {traces_data_dir}")
    
    # Track results
    results = {}
    
    # Run step-by-step if enabled
    if config.get('run_step_by_step', True):
        results['step_by_step_success'] = run_step_by_step_pipeline(
            traces_data_dir,
            output_paths['step_by_step'],
            config
        )
    else:
        logger.info("Step-by-step pipeline disabled")
        results['step_by_step_success'] = None
    
    # Run full trajectory if enabled
    if config.get('run_full_trajectory', True):
        results['full_trajectory_success'] = run_full_trajectory_pipeline(
            traces_data_dir,
            output_paths['full_trajectory'],
            config
        )
    else:
        logger.info("Full trajectory pipeline disabled")
        results['full_trajectory_success'] = None
    
    # Create pipeline summary
    create_pipeline_summary(output_paths['base'], config, results)
    
    # Create unified UI zip if both pipelines ran
    logger.info("=" * 80)
    logger.info("Creating unified UI zip...")
    logger.info("=" * 80)
    
    from clear_eval.agentic.pipeline.create_ui_input import create_unified_ui_zip
    
    # Determine paths for unified zip
    step_by_step_results_path = None
    full_traj_results_path = None
    
    # Check for step-by-step results
    if results.get('step_by_step_success'):
        step_by_step_results_path = output_paths['step_by_step'] / 'clear_results'
    
    # Check for full trajectory results
    if results.get('full_trajectory_success'):
        full_traj_results_path = output_paths['full_trajectory']
    
    # Create unified zip
    try:
        unified_zip = create_unified_ui_zip(
            output_dir=output_paths['base'],
            traces_data_dir=traces_data_dir,
            step_by_step_clear_results_dir=step_by_step_results_path,
            full_trajectory_results_dir=full_traj_results_path,
            output_zip_name="unified_ui_results.zip"
        )
        logger.info(f"✓ Created unified UI zip: {unified_zip}")
    except Exception as e:
        logger.error(f"Failed to create unified UI zip: {e}", exc_info=True)
    
    # Final summary
    logger.info("=" * 80)
    logger.info("UNIFIED PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results directory: {output_paths['base']}")
    
    if results.get('step_by_step_success') is not None:
        status = "✓ SUCCESS" if results['step_by_step_success'] else "✗ FAILED"
        logger.info(f"Step-by-step pipeline: {status}")
    
    if results.get('full_trajectory_success') is not None:
        status = "✓ SUCCESS" if results['full_trajectory_success'] else "✗ FAILED"
        logger.info(f"Full trajectory pipeline: {status}")
    
    # Exit with appropriate code
    if any(v is False for v in results.values()):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

