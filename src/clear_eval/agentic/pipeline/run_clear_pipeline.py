"""
CLEAR Full Pipeline for Agentic Workflows

This module runs the complete pipeline from raw traces to CLEAR analysis results:
traces (LangGraph/CrewAI) -> trajectory data -> CLEAR format -> CLEAR results

For running CLEAR on already-preprocessed trajectory data, use run_clear_on_traj_data.py instead.

Configuration Precedence (lowest to highest):
    1. Default config: setup/default_config.yaml
    2. User config file: --agentic-config-path (if provided)
    3. CLI arguments (override both config files)

Arguments are split into two groups:
    - Agentic Pipeline Arguments: Pipeline-specific (traces_input_dir, etc.)
    - CLEAR Configuration: Evaluation args from clear_eval.args (provider, eval_model_name, etc.)
"""

import argparse
import logging
import os
import tempfile

from clear_eval.agentic.pipeline.build_json_results import build_comprehensive_json_results, \
    save_comprehensive_json_results, save_json_to_file
from clear_eval.agentic.pipeline.preprocess_traces.preprocess_traces import process_traces_to_traj_data
from clear_eval.agentic.pipeline.run_clear_on_traj_data import (
    convert_to_clear_format,
    run_clear_analysis,
    create_comprehensive_ui_results,
)
from clear_eval.args import add_clear_args_to_parser, str2bool
from clear_eval.logging_config import setup_logging
from clear_eval.pipeline.config_loader import load_config

# Initialize logging
setup_logging()

logger = logging.getLogger(__name__)

# Path to agentic pipeline default config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTIC_DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "setup", "default_config.yaml")


def add_agentic_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """
    Add agentic pipeline arguments to the parser.

    These correspond to the agentic pipeline section in default_config.yaml.
    """
    group = parser.add_argument_group("Agentic Pipeline Arguments")

    group.add_argument(
        "--agentic-config-path",
        help="Path to config file (JSON or YAML) that overrides defaults"
    )
    group.add_argument(
        "--traces-input-dir",
        help="Directory containing raw traces (required)"
    )
    group.add_argument(
        "--agentic-output-dir",
        help="Output directory for pipeline results (required)"
    )
    group.add_argument(
        "--agent-framework",
        choices=['langgraph', 'crewai'],
        help="Agent framework used to generate traces (default: langgraph)"
    )
    group.add_argument(
        "--observability-framework",
        choices=['mlflow', 'langsmith'],
        help="Observability framework used for tracing (default: mlflow)"
    )
    group.add_argument(
        "--separate-tools",
        type=str2bool,
        help="Whether to separate tool calls in processing (default: false)"
    )
    group.add_argument(
        "--overwrite",
        type=str2bool,
        help="Whether to overwrite existing results (default: true)"
    )
    group.add_argument(
        "--success-threshold",
        type=float,
        help="Threshold for pass/fail determination (default: 0.7)"
    )
    group.add_argument(
        "--pass-criteria",
        choices=['avg', 'min'],
        help="Score type for pass/fail: 'avg' or 'min' (default: avg)"
    )
    group.add_argument(
        "--memory-only",
        type=str2bool,
        help="If true, use temporary directories and return only JSON results (no files saved) (default: false)"
    )

def run_full_pipeline(config_dict: dict) -> dict:
    """
    Complete pipeline: traces -> trajectory data -> CLEAR results.

    Args:
        config_dict: Configuration dictionary with agentic and CLEAR params (all top-level)

    Returns:
        Dictionary with final JSON results (always returns dict)
        If memory_only=True in config_doct, results are only returned (not saved)
    """
    # Extract agentic-specific parameters
    traces_input_dir = config_dict.get('traces_input_dir')
    agentic_output_dir = config_dict.get('agentic_output_dir')
    agent_framework = config_dict.get('agent_framework', 'langgraph')
    observability_framework = config_dict.get('observability_framework', 'mlflow')
    separate_tools = config_dict.get('separate_tools', False)
    overwrite = config_dict.get('overwrite', True)
    memory_only = config_dict.get('memory_only', False)
    if not traces_input_dir:
        raise ValueError("traces_input_dir is required")
    if not agentic_output_dir:
        raise ValueError("agentic_output_dir is required")

    # Use temporary directory for intermediate files if memory_only mode
    if memory_only:
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_context.__enter__()
        intermediate_output_dir = temp_dir
        logger.info("=" * 80)
        logger.info("CLEAR FULL PIPELINE: MEMORY-ONLY MODE")
        logger.info("=" * 80)
        logger.info("Using temporary directory for intermediate files")
        logger.info(f"Final JSON results will be saved to: {agentic_output_dir}")
    else:
        temp_dir_context = None
        intermediate_output_dir = agentic_output_dir
        logger.info("=" * 80)
        logger.info("CLEAR FULL PIPELINE: FROM RAW TRACES")
        logger.info("=" * 80)

    try:
        # Intermediate files go to temp dir (memory-only) or agentic_output_dir (normal)
        traces_data_dir = os.path.join(intermediate_output_dir, 'traces_data')
        clear_data_dir = os.path.join(intermediate_output_dir, 'clear_data')
        clear_results_dir = os.path.join(intermediate_output_dir, 'clear_results')

        logger.info(f"Input traces: {traces_input_dir}")
        logger.info(f"Agent framework: {agent_framework}")
        logger.info(f"Observability: {observability_framework}")
        logger.info(f"Separate tools: {separate_tools}")
        logger.info(f"Output: {agentic_output_dir}")
        if memory_only:
            logger.info("  ├── clear_results/  (CLEAR analysis results)")
            logger.info("  └──── clear_results.json  (Final JSON results only)")
        else:
            logger.info("  ├── traces_data/    (Trajectory CSVs)")
            logger.info("  ├── clear_data/     (CLEAR format by agent)")
            logger.info("  ├── clear_results/  (CLEAR analysis results)")
            logger.info("  └──── clear_results.json  (Final JSON results)")
        logger.info("=" * 80)

        logger.info("STEP 1: Processing traces to trajectory data")
        process_traces_to_traj_data(
            traces_input_dir,
            traces_data_dir,
            agent_framework=agent_framework,
            observability_framework=observability_framework,
            separate_tools=separate_tools
        )

        logger.info("STEP 2: Converting trajectory data to CLEAR format")
        convert_to_clear_format(traces_data_dir, clear_data_dir)

        logger.info("STEP 3: Running CLEAR analysis for each agent")

        judge_results_dir = run_clear_analysis(
            clear_data_dir,
            clear_results_dir,
            config_dict,
            overwrite=overwrite,
        )

        logger.info("STEP 4: Creating comprehensive UI results")
        ui_results_path = create_comprehensive_ui_results(
            judge_results_dir,
            traces_data_dir
        )

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("Building JSON results")
        if memory_only:
            final_json_dir = os.path.join(agentic_output_dir, 'clear_results')
        else:
            final_json_dir = judge_results_dir
        json_results = build_comprehensive_json_results(
            judge_results_dir=judge_results_dir,
            traces_data_dir=traces_data_dir,
            config_dict=config_dict,
        )
        save_json_to_file(
            results=json_results,
            output_dir=final_json_dir,
        )
        logger.info("=" * 80)

    finally:
        # Clean up temporary directory if used
        if temp_dir_context is not None:
            temp_dir_context.__exit__(None, None, None)
            logger.info("Temporary intermediate files cleaned up")
    
    return json_results

def build_cli_overrides(args: argparse.Namespace) -> dict:
    """
    Build CLI overrides dictionary from parsed arguments.

    All arguments are at top level (both agentic and CLEAR).

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of CLI overrides ready for load_config()
    """
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None and key != 'agentic_config_path'
    }


def main():
    """CLI entry point for running the full CLEAR pipeline from raw traces."""
    parser = argparse.ArgumentParser(
        description="CLEAR Full Pipeline: Process Raw Traces and Run Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended) - supports JSON or YAML
  python -m clear_eval.agentic.pipeline.run_clear_pipeline \\
      --agentic-config-path my_config.yaml

  # Using JSON config file
  python -m clear_eval.agentic.pipeline.run_clear_pipeline \\
      --agentic-config-path my_config.json

  # Config file with CLI overrides
  python -m clear_eval.agentic.pipeline.run_clear_pipeline \\
      --agentic-config-path my_config.yaml \\
      --eval-model-name meta-llama/llama-3-1-70b-instruct

  # CLI only (all parameters)
  python -m clear_eval.agentic.pipeline.run_clear_pipeline \\
      --traces-input-dir data/traces \\
      --agentic-output-dir output/analysis \\
      --provider watsonx \\
      --eval-model-name meta-llama/llama-3-3-70b-instruct

  # Memory-only mode (no files saved, only JSON results returned)
  python -m clear_eval.agentic.pipeline.run_clear_pipeline \\
      --traces-input-dir data/traces \\
      --provider watsonx \\
      --eval-model-name meta-llama/llama-3-3-70b-instruct \\
      --memory-only true

Config file structure (YAML format - see setup/default_config.yaml):
  # Agentic pipeline arguments
  traces_input_dir: data/traces
  agentic_output_dir: output/analysis  # Optional if memory_only: true
  agent_framework: langgraph
  observability_framework: mlflow
  separate_tools: false
  overwrite: true
  memory_only: false  # Set to true to skip saving files

  # CLEAR arguments
  provider: watsonx
  eval_model_name: meta-llama/llama-3-3-70b-instruct
  agent_mode: true
  # ... other CLEAR parameters

Argument Precedence (lowest to highest):
  1. Default config (setup/default_config.yaml)
  2. User config file (--agentic-config-path)
  3. CLI arguments

Note: For running CLEAR on already-preprocessed trajectory data,
use run_clear_on_traj_data.py instead.
        """
    )

    # Add agentic pipeline arguments (includes --agentic-config-path)
    add_agentic_args_to_parser(parser)

    # Add CLEAR configuration arguments
    add_clear_args_to_parser(parser, group_name="CLEAR Configuration")

    args = parser.parse_args()

    # Build CLI overrides from non-None args
    cli_overrides = build_cli_overrides(args)

    # Load configuration with precedence: default -> user config -> CLI overrides
    config_dict = load_config(
        AGENTIC_DEFAULT_CONFIG_PATH,
        args.agentic_config_path,
        **cli_overrides
    )

    # Extract memory_only flag
    memory_only = config_dict.get('memory_only', False)
    
    # Validate required parameters
    if not config_dict.get('traces_input_dir'):
        parser.error("traces_input_dir is required (set in config or use --traces-input-dir)")
    if not memory_only and not config_dict.get('agentic_output_dir'):
        parser.error("agentic_output_dir is required when memory_only=False (set in config or use --agentic-output-dir)")

    # Run the full pipeline (always returns dict)
    result = run_full_pipeline(config_dict)
    
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info(f"Total agents: {len(result.get('agents', {}))}")
    logger.info(f"Total traces: {result.get('metadata', {}).get('statistics', {}).get('total_traces', 0)}")
    logger.info("=" * 80)
    return result


if __name__ == "__main__":
    main()
