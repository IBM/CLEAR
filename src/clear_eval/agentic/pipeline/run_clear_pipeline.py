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

from clear_eval.agentic.pipeline.build_json_results import build_comprehensive_json_results, \
    save_comprehensive_json_results
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

def run_full_pipeline(config_dict: dict) -> str:
    """
    Complete pipeline: traces -> trajectory data -> CLEAR results.

    Args:
        config_dict: Configuration dictionary with agentic and CLEAR params (all top-level)

    Returns:
        Path to the ui_results.zip file
    """
    # Extract agentic-specific parameters
    traces_input_dir = config_dict.get('traces_input_dir')
    agentic_output_dir = config_dict.get('agentic_output_dir')
    agent_framework = config_dict.get('agent_framework', 'langgraph')
    observability_framework = config_dict.get('observability_framework', 'mlflow')
    separate_tools = config_dict.get('separate_tools', False)
    overwrite = config_dict.get('overwrite', True)

    if not traces_input_dir:
        raise ValueError("traces_input_dir is required")
    if not agentic_output_dir:
        raise ValueError("agentic_output_dir is required")

    traces_data_dir = os.path.join(agentic_output_dir, 'traces_data')
    clear_data_dir = os.path.join(agentic_output_dir, 'clear_data')
    clear_results_dir = os.path.join(agentic_output_dir, 'clear_results')

    logger.info("=" * 80)
    logger.info("CLEAR FULL PIPELINE: FROM RAW TRACES")
    logger.info("=" * 80)
    logger.info(f"Input traces: {traces_input_dir}")
    logger.info(f"Agent framework: {agent_framework}")
    logger.info(f"Observability: {observability_framework}")
    logger.info(f"Separate tools: {separate_tools}")
    logger.info(f"Output: {agentic_output_dir}")
    logger.info("  ├── traces_data/    (Trajectory CSVs)")
    logger.info("  ├── clear_data/     (CLEAR format by agent)")
    logger.info("  └── clear_results/  (CLEAR analysis results)")
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

    json_results = save_comprehensive_json_results(
        judge_results_dir = judge_results_dir,
        traces_data_dir = traces_data_dir,
        config_dict = config_dict,
    )

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

Config file structure (YAML format - see setup/default_config.yaml):
  # Agentic pipeline arguments
  traces_input_dir: data/traces
  agentic_output_dir: output/analysis
  agent_framework: langgraph
  observability_framework: mlflow
  separate_tools: false
  overwrite: true

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

    # Validate required parameters
    if not config_dict.get('traces_input_dir'):
        parser.error("traces_input_dir is required (set in config or use --traces-input-dir)")
    if not config_dict.get('agentic_output_dir'):
        parser.error("agentic_output_dir is required (set in config or use --agentic-output-dir)")

    # Run the full pipeline
    run_full_pipeline(config_dict)


if __name__ == "__main__":
    main()
