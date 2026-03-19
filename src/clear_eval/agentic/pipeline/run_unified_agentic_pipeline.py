#!/usr/bin/env python3
"""
Unified Agentic Pipeline Script
================================

This script provides a unified interface to run both:
1. Step-by-step CLEAR analysis (on trajectory data)
2. Full trajectory evaluation (task success, full trajectory, rubric, CLEAR analysis)

All results are organized under: clear_results/<judge-model>/<run-name>/

Configuration Precedence (lowest to highest):
    1. Default config: setup/unified_config.yaml
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
        --run-step-by-step \\
        --no-run-full-trajectory \\
        --process-from-traces \\
        --eval-model-name openai/gpt-oss-120b \\
        --provider watsonx

    # Run only step-by-step from preprocessed CSVs
    python -m clear_eval.agentic.pipeline.run_unified_agentic_pipeline \\
        --agentic-input-dir data/experiment_001 \\
        --agentic-output-dir results \\
        --run-step-by-step \\
        --no-run-full-trajectory \\
        --no-process-from-traces \\
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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from clear_eval.agentic.pipeline.run_clear_pipeline import run_full_pipeline
from clear_eval.agentic.pipeline.run_clear_on_traj_data import (
    run_traj_data_pipeline,
    get_judge_model_folder_name,
)
from clear_eval.args import add_clear_args_to_parser, str2bool
from clear_eval.logging_config import setup_logging
from clear_eval.pipeline.config_loader import load_config

# Import full trajectory evaluation components
try:
    from clear_eval.agentic.pipeline.full_traces_evaluation.run_trajectory_evaluation_pipeline import (
        run_trajectory_evaluation_pipeline,
    )
    FULL_TRAJ_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Full trajectory evaluation not available: {e}")
    FULL_TRAJ_AVAILABLE = False
    run_trajectory_evaluation_pipeline = None

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Path to unified default config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED_DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "setup", "unified_config.yaml")


def add_agentic_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """Add agentic pipeline arguments to the parser."""
    group = parser.add_argument_group("Agentic Pipeline Arguments")
    
    group.add_argument(
        "--agentic-config-path",
        help="Path to unified config file (JSON or YAML)"
    )
    group.add_argument(
        "--agentic-input-dir",
        help="Base input directory (required)"
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
    group.add_argument(
        "--process-from-traces",
        type=str2bool,
        help="For step-by-step: process from raw traces vs traces_data/ (default: true)"
    )
    
    # Input processing options
    group.add_argument(
        "--agent-framework",
        choices=['langgraph', 'crewai'],
        help="Agent framework (default: langgraph)"
    )
    group.add_argument(
        "--observability-framework",
        choices=['mlflow', 'langsmith'],
        help="Observability framework (default: mlflow)"
    )
    group.add_argument(
        "--separate-tools",
        type=str2bool,
        help="Separate tool calls (default: false, keep false for now)"
    )
    
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
        "--concurrency",
        type=int,
        help="Number of parallel workers (default: 10)"
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


def detect_input_structure(input_dir: Path) -> Dict[str, bool]:
    """
    Detect available input types in directory.
    
    Returns:
        {
            'has_traces': bool,           # Raw traces for step-by-step
            'has_traces_compact': bool,   # Compact traces for full trajectory
            'has_traces_data': bool       # Preprocessed CSVs for step-by-step
        }
    """
    return {
        'has_traces': (input_dir / 'traces').exists(),
        'has_traces_compact': (input_dir / 'traces_compact').exists(),
        'has_traces_data': (input_dir / 'traces_data').exists(),
    }


def create_output_structure(
    output_dir: Path,
    judge_model: str,
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


def run_step_by_step_pipeline(
    input_dir: Path,
    output_dir: Path,
    config: dict,
    input_structure: dict
) -> bool:
    """
    Run step-by-step CLEAR analysis pipeline.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("RUNNING STEP-BY-STEP CLEAR ANALYSIS")
    logger.info("=" * 80)
    
    try:
        if config.get('process_from_traces', True):
            # Process from raw traces using run_full_pipeline
            if not input_structure['has_traces']:
                logger.error("No traces/ directory found in input directory")
                return False
            
            traces_dir = input_dir / 'traces'
            logger.info(f"Processing traces from: {traces_dir}")
            
            # Prepare config for run_full_pipeline
            # run_full_pipeline expects all config at top level
            pipeline_config = config.copy()
            pipeline_config['traces_input_dir'] = str(traces_dir)
            pipeline_config['agentic_output_dir'] = str(output_dir)
            
            # Call run_full_pipeline directly
            logger.info("Calling run_full_pipeline from run_clear_pipeline.py")
            run_full_pipeline(pipeline_config)
            
        else:
            # Use existing traces_data
            if not input_structure['has_traces_data']:
                logger.error("No traces_data/ directory found in input directory")
                return False
            
            traces_data_dir = input_dir / 'traces_data'
            logger.info(f"Using existing trajectory data from: {traces_data_dir}")
            
            # Call run_traj_data_pipeline for steps 2-4
            logger.info("Calling run_traj_data_pipeline from run_clear_pipeline.py")
            run_traj_data_pipeline(
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
    input_dir: Path,
    output_dir: Path,
    config: dict,
    input_structure: dict
) -> bool:
    """
    Run full trajectory evaluation pipeline.
    
    Returns:
        True if successful, False otherwise
    """
    if not FULL_TRAJ_AVAILABLE or run_trajectory_evaluation_pipeline is None:
        logger.error("Full trajectory evaluation components not available")
        return False
    
    logger.info("=" * 80)
    logger.info("RUNNING FULL TRAJECTORY EVALUATION")
    logger.info("=" * 80)
    
    try:
        # Check for traces_compact
        if not input_structure['has_traces_compact']:
            logger.error("No traces_compact/ directory found in input directory")
            return False
        
        traj_input_dir = input_dir / 'traces_compact'
        logger.info(f"Using compact traces from: {traj_input_dir}")
        
        # Prepare rubric_dir
        rubric_dir = None
        if config.get('rubric_dir'):
            rubric_dir = Path(config['rubric_dir'])
        
        # Call the refactored function
        completed_evals, failed_evals = run_trajectory_evaluation_pipeline(
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            model_id=config.get('eval_model_name', 'openai/gpt-oss-120b'),
            provider=config.get('provider', 'watsonx'),
            eval_types=config.get('eval_types', ['all']),
            generate_rubrics=config.get('generate_rubrics', False),
            rubric_dir=rubric_dir,
            clear_analysis_types=config.get('clear_analysis_types', ['all']),
            context_tokens=config.get('context_tokens', 128000),
            overwrite=config.get('overwrite', True),
            concurrency=config.get('concurrency', 10),
            eval_model_params=config.get('eval_model_params', {}),
            max_files=config.get('max_files'),
        )
        
        # Create UI input for full trajectory results
        logger.info("Creating UI input for full trajectory results...")
        create_full_trajectory_ui_input(output_dir)
        
        logger.info("=" * 80)
        logger.info("FULL TRAJECTORY PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Completed evaluations: {completed_evals or 'None'}")
        logger.info(f"Failed evaluations: {failed_evals or 'None'}")
        
        return len(failed_evals) == 0
        
    except Exception as e:
        logger.error(f"Full trajectory pipeline failed: {e}", exc_info=True)
        return False


def create_full_trajectory_ui_input(output_dir: Path):
    """
    Create UI input zip for full trajectory results.
    Includes: full_trajectory/, rubric/, and clear_analysis/ subdirectories.
    """
    import zipfile
    
    ui_zip_path = output_dir / "ui_input.zip"
    
    with zipfile.ZipFile(ui_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Add full_trajectory results
        full_traj_dir = output_dir / 'full_trajectory'
        if full_traj_dir.exists():
            for file_path in full_traj_dir.rglob('*'):
                if file_path.is_file():
                    arcname = f"full_trajectory/{file_path.relative_to(full_traj_dir)}"
                    zf.write(file_path, arcname=arcname)
        
        # Add rubric results
        rubric_dir = output_dir / 'rubric'
        if rubric_dir.exists():
            for file_path in rubric_dir.rglob('*'):
                if file_path.is_file():
                    arcname = f"rubric/{file_path.relative_to(rubric_dir)}"
                    zf.write(file_path, arcname=arcname)
        
        # Add CLEAR analysis results
        clear_dir = output_dir / 'clear_analysis'
        if clear_dir.exists():
            for file_path in clear_dir.rglob('*'):
                if file_path.is_file():
                    arcname = f"clear_analysis/{file_path.relative_to(clear_dir)}"
                    zf.write(file_path, arcname=arcname)
        
        # Add metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "type": "full_trajectory_results",
            "structure": {
                "full_trajectory/": "Full trajectory evaluation results",
                "rubric/": "Rubric evaluation results",
                "clear_analysis/": "CLEAR analysis on full trajectory results"
            }
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    logger.info(f"Created UI input: {ui_zip_path}")


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


def build_cli_overrides(args: argparse.Namespace) -> dict:
    """Build CLI overrides dictionary from parsed arguments."""
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None and key != 'agentic_config_path'
    }


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
    
    # Build CLI overrides
    cli_overrides = build_cli_overrides(args)
    
    # Load configuration with precedence: default -> user config -> CLI overrides
    config = load_config(
        UNIFIED_DEFAULT_CONFIG_PATH,
        args.agentic_config_path,
        **cli_overrides
    )
    
    # Validate required parameters
    if not config.get('agentic_input_dir'):
        parser.error("agentic_input_dir is required (set in config or use --agentic-input-dir)")
    if not config.get('agentic_output_dir'):
        parser.error("agentic_output_dir is required (set in config or use --agentic-output-dir)")
    
    # Extract parameters
    input_dir = Path(config['agentic_input_dir'])
    output_dir = Path(config['agentic_output_dir'])
    run_name = config.get('run_name') or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detect input structure
    logger.info("=" * 80)
    logger.info("UNIFIED AGENTIC PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Run name: {run_name}")
    
    input_structure = detect_input_structure(input_dir)
    logger.info("Input structure detected:")
    logger.info(f"  - traces/: {input_structure['has_traces']}")
    logger.info(f"  - traces_compact/: {input_structure['has_traces_compact']}")
    logger.info(f"  - traces_data/: {input_structure['has_traces_data']}")
    
    # Create output structure
    judge_model = get_judge_model_folder_name(config.get('eval_model_name', 'unknown'))
    output_paths = create_output_structure(output_dir, judge_model, run_name)
    logger.info(f"Output base directory: {output_paths['base']}")
    
    # Map concurrency to max_workers for CLEAR compatibility
    if 'concurrency' in config:
        config['max_workers'] = config['concurrency']
    
    # Track results
    results = {}
    
    # Run step-by-step if enabled
    if config.get('run_step_by_step', True):
        results['step_by_step_success'] = run_step_by_step_pipeline(
            input_dir,
            output_paths['step_by_step'],
            config,
            input_structure
        )
    else:
        logger.info("Step-by-step pipeline disabled")
        results['step_by_step_success'] = None
    
    # Run full trajectory if enabled
    if config.get('run_full_trajectory', True):
        results['full_trajectory_success'] = run_full_trajectory_pipeline(
            input_dir,
            output_paths['full_trajectory'],
            config,
            input_structure
        )
    else:
        logger.info("Full trajectory pipeline disabled")
        results['full_trajectory_success'] = None
    
    # Create pipeline summary
    create_pipeline_summary(output_paths['base'], config, results)
    
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

