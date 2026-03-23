"""
Run CLEAR Analysis on Preprocessed Trajectory Data

This script runs CLEAR analysis starting from already-preprocessed trajectory CSV files.
For the full pipeline from raw traces, use run_clear_pipeline.py instead.

Configuration Precedence (lowest to highest):
    1. Default config: setup/default_config.yaml
    2. User config file: --agentic-config-path (if provided)
    3. CLI arguments (override both config files)
"""

import argparse
import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from clear_eval.agentic.pipeline.utils import build_cli_overrides
from clear_eval.agentic.pipeline.build_json_results import save_comprehensive_json_results
from clear_eval.agentic.pipeline.create_ui_input import create_ui_input_zip
from clear_eval.args import add_clear_args_to_parser, str2bool
from clear_eval.logging_config import setup_logging
from clear_eval.pipeline.config_loader import load_config
from clear_eval.pipeline.full_pipeline import run_eval_pipeline

# Initialize logging
setup_logging()

logger = logging.getLogger(__name__)

# Path to agentic pipeline default config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTIC_DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "setup", "default_config.yaml")


def add_agentic_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """
    Add agentic pipeline arguments for trajectory data processing.

    These correspond to the agentic pipeline section in default_config.yaml.
    """
    group = parser.add_argument_group("Agentic Pipeline Arguments")

    group.add_argument(
        "--agentic-config-path",
        help="Path to config file (JSON or YAML) that overrides defaults"
    )
    group.add_argument(
        "--traces-data-dir",
        help="Directory containing preprocessed trajectory CSV files (required)"
    )
    group.add_argument(
        "--agentic-output-dir",
        help="Output directory for pipeline results (required)"
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
        help="If true, use temporary directories and save only ui_input and json_result to agentic_output_dir (default: false)"
    )


##########################################
## Convert shared data to clear format ###
##########################################
def convert_to_clear_format(input_dir: str, output_dir: str):
    """
    Convert CSV files to CLEAR format grouped by agent.

    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save CLEAR format files
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    agent_data = defaultdict(list)
    total_rows = 0
    task_counter = Counter()
    agent_counter = Counter()

    input_path = Path(input_dir)
    csv_files = list(input_path.glob('*.csv'))

    logger.info(f"Processing {len(csv_files)} CSV files from {input_dir}")

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                df = pd.read_csv(f)
                df = df.rename(columns={"trace_id": "task_id"}, inplace=False)
                if 'Name' not in df.columns:
                    df["Name"] = df["agent_name"]
                df.loc[:, "id"] = df.apply(
                    lambda row: f"{row['task_id']}_{row['step_in_trace_general']}", axis=1
                )
                for i, row in df.iterrows():
                    agent_name = row['Name']
                    task_id = row['task_id']

                    total_rows += 1
                    agent_counter[agent_name] += 1
                    task_counter[task_id] += 1

                    row_with_agent = {'agent_name': agent_name, **row}
                    agent_data[agent_name].append(row_with_agent)

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue

    logger.info(f"Writing {len(agent_data)} agent CSV files")

    for agent_name, rows in agent_data.items():
        output_file = output_dir / f"{agent_name}.csv"
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_file, index=False)
        logger.info(f"  {agent_name}.csv ({len(rows)} rows)")

    statistics = {
        "total_rows": total_rows,
        "unique_agents": len(agent_counter),
        "unique_tasks": len(task_counter),
        "agent_counts": dict(agent_counter.most_common()),
        "task_counts": dict(task_counter.most_common())
    }

    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2)

    logger.info("statistics.json created")
    logger.info("=" * 80)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total interactions: {total_rows}")
    logger.info(f"Unique agents: {len(agent_counter)}")
    logger.info(f"Unique tasks: {len(task_counter)}")
    logger.info("Agent distribution:")
    for agent, count in agent_counter.most_common():
        logger.info(f"  {agent:40} : {count:4} interactions")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


def get_judge_model_folder_name(eval_model_name: str) -> str:
    """Convert eval model name to a clean folder name."""
    if "/" in eval_model_name:
        judge_name = eval_model_name.split("/")[-1]
    else:
        judge_name = eval_model_name

    judge_name = judge_name.replace(".", "-").replace("_", "-").lower()
    return judge_name


def run_analysis_for_agent(
    csv_path: Path,
    results_dir: str,
    overwrite: bool,
    config_dict: dict,
) -> Optional[bool]:
    """
    Run CLEAR analysis for a single agent CSV file.

    Args:
        csv_path: Path to the CSV file
        results_dir: Directory to save CLEAR analysis results
        overwrite: Whether to overwrite existing results
        config_dict: Pre-loaded configuration dictionary

    Returns:
        True if analysis was run, False if skipped, None if error
    """
    # Make a copy to avoid modifying the original
    config_dict = config_dict.copy()

    agent_name = csv_path.stem
    eval_model_name = config_dict.get("eval_model_name", "unknown")

    output_dir = Path(results_dir) / agent_name

    logger.info("=" * 80)
    logger.info(f"Processing: {agent_name}")
    logger.info(f"Output folder: {output_dir}")
    logger.info("=" * 80)

    if output_dir.exists() and not overwrite and list(Path(output_dir).glob('*.zip')):
        logger.info(f"Skipping: output directory already exists: {output_dir}")
        logger.info("Use --overwrite to re-run analysis")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Running CLEAR analysis...")
        logger.info(f"  Data: {csv_path}")
        logger.info(f"  Output: {output_dir}")

        # Add computed paths to config
        config_dict["data_path"] = str(csv_path)
        config_dict["output_dir"] = str(output_dir)
        config_dict["resume_enabled"] = not overwrite

        # Run evaluation pipeline
        run_eval_pipeline(config_dict)

        logger.info("Analysis completed!")
        logger.info(f"  Results saved to: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Error running analysis for {agent_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_clear_analysis(
    input_data_dir: str,
    results_dir: str,
    config_dict: dict,
    overwrite: bool = False,
) -> str:
    """
    Run CLEAR analysis for agent trajectories.

    Args:
        input_data_dir: Directory containing agent CSV files
        results_dir: Directory to save results
        config_dict: Pre-loaded configuration dictionary
        overwrite: If True, re-run analysis even if output exists

    Returns:
        Path to judge results directory
    """
    input_data_path = Path(input_data_dir)
    csv_files = sorted([f for f in input_data_path.glob("*.csv")])

    if not csv_files:
        logger.error(f"No CSV files found in {input_data_path}")
        return ""

    eval_model_name = config_dict.get("eval_model_name", "unknown")

    logger.info("=" * 80)
    logger.info(f"CLEAR Analysis - {input_data_path.name}")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Results directory: {results_dir}")
    logger.info(f"  Eval model: {eval_model_name}")
    logger.info(f"Found {len(csv_files)} agent CSV files:")
    for csv_file in csv_files:
        logger.info(f"  - {csv_file.name}")
    logger.info("=" * 80)

    stats = {"total": len(csv_files), "processed": 0, "skipped": 0, "errors": 0}

    for csv_file in csv_files:
        result = run_analysis_for_agent(
            csv_file,
            results_dir,
            overwrite,
            config_dict,
        )
        if result is True:
            stats["processed"] += 1
        elif result is False:
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    logger.info("=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Total agent types: {stats['total']}")
    logger.info(f"  Processed: {stats['processed']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    if stats['errors'] > 0:
        logger.warning(f"  Errors: {stats['errors']}")
    logger.info("=" * 80)

    judge_folder = get_judge_model_folder_name(eval_model_name)
    logger.info("Results Directory Structure:")
    logger.info(f"   {results_dir}/")
    logger.info(f"   └── {judge_folder}/")
    for csv_file in csv_files[:3]:
        logger.info(f"           ├── {csv_file.stem}/")
    if len(csv_files) > 3:
        logger.info(f"           └── ... ({len(csv_files) - 3} more agents)")
    logger.info("           └── ui_results.zip")
    logger.info("=" * 80)


def create_comprehensive_ui_results(
    clear_results_dir: str | Path,
    traj_data_dir: str | Path,
    result_zip_name: str = "ui_results.zip",
) -> Path:
    """
    Create a comprehensive UI results zip with INPUT DEDUPLICATION.

    This is a wrapper around create_ui_input_zip() for backward compatibility.
    All logic is now in create_ui_input.py to avoid code duplication.

    Parameters
    ----------
    clear_results_dir : str | Path
        Directory containing agent subdirectories with CLEAR results
    traj_data_dir : str | Path
        Directory containing trajectory CSV files
    result_zip_name : str
        Name of the output zip file (default: 'ui_results.zip')

    Returns
    -------
    Path
        Path to the created ui_results.zip
    """
    clear_results_path = Path(clear_results_dir).resolve()

    return create_ui_input_zip(
        output_dir=clear_results_path,
        traces_data_dir=traj_data_dir,
        clear_results_dir=clear_results_path,
        output_zip_name=result_zip_name
    )


def run_traj_data_pipeline(
    traces_data_dir: str,
    agentic_output_dir: str,
    config_dict: dict,
    overwrite: bool = True,
    intermediate_output_dir: Optional[str] = None
) -> dict:
    """
    Run pipeline from trajectory data to CLEAR results.
    
    This function handles the complete workflow from trajectory CSVs:
    - Convert trajectory data to CLEAR format
    - Run CLEAR analysis for each agent
    - Create comprehensive UI results
    - Save JSON results
    
    This is the shared function used by both:
    - run_clear_pipeline.py (after preprocessing traces)
    - run_unified_agentic_pipeline.py (when starting from traces_data)
    
    Args:
        traces_data_dir: Directory containing trajectory CSV files
        agentic_output_dir: Base output directory for final results
        config_dict: Configuration dictionary with CLEAR params
        overwrite: Whether to overwrite existing results
        intermediate_output_dir: Optional directory for intermediate files (if None, will create temp dir in memory-only mode)
        
    Returns:
        Dictionary with JSON results
    """
    memory_only = config_dict.get('memory_only', False)

    # Create output directory for final results
    os.makedirs(agentic_output_dir, exist_ok=True)
    # Use temporary directory for intermediate files if memory_only mode and no intermediate_output_dir provided
    if memory_only and intermediate_output_dir is None:
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_context.__enter__()
        intermediate_output_dir = temp_dir
        logger.info("Memory-only mode: Using temporary directory for intermediate files")
    elif intermediate_output_dir is None:
        temp_dir_context = None
        intermediate_output_dir = agentic_output_dir
    else:
        # intermediate_output_dir was provided (from run_full_pipeline)
        temp_dir_context = None
    
    try:
        clear_data_dir = os.path.join(intermediate_output_dir, 'clear_data')
        clear_results_dir = os.path.join(intermediate_output_dir, 'clear_results')
        
        logger.info("Converting trajectory data to CLEAR format...")
        convert_to_clear_format(traces_data_dir, clear_data_dir)

        logger.info("Running CLEAR analysis for each agent...")
        run_clear_analysis(
            clear_data_dir,
            clear_results_dir,
            config_dict,
            overwrite=overwrite,
        )

        # Build JSON results (returns dict)
        from clear_eval.agentic.pipeline.build_json_results import build_comprehensive_json_results, save_json_to_file
        json_results = save_comprehensive_json_results(
            clear_results_dir=clear_results_dir,
            traces_data_dir=traces_data_dir,
            config_dict=config_dict,
            output_dir = agentic_output_dir,
        )
        
        # Create UI results zip directly in final output directory
        from clear_eval.agentic.pipeline.create_ui_input import create_ui_input_zip
        ui_results_path = create_ui_input_zip(
            output_dir=Path(agentic_output_dir),
            traces_data_dir=Path(traces_data_dir),
            clear_results_dir=Path(clear_results_dir),
            output_zip_name="ui_results.zip"
        )
        logger.info(f"Saved UI results to: {ui_results_path}")

        if memory_only:
            logger.info("Memory-only mode: Intermediate files will be cleaned up")

        logger.info("Pipeline complete!")
        return json_results
        
    finally:
        # Clean up temporary directory if used
        if temp_dir_context is not None:
            temp_dir_context.__exit__(None, None, None)
            logger.info("Temporary intermediate files cleaned up")


def main():
    """CLI entry point for running CLEAR analysis on preprocessed trajectory data."""
    parser = argparse.ArgumentParser(
        description="Run CLEAR Analysis on Preprocessed Trajectory Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended) - supports JSON or YAML
  python -m clear_eval.agentic.pipeline.run_clear_on_traj_data \\
      --agentic-config-path my_config.yaml

  # Using JSON config file
  python -m clear_eval.agentic.pipeline.run_clear_on_traj_data \\
      --agentic-config-path my_config.json

  # Config file with CLI overrides
  python -m clear_eval.agentic.pipeline.run_clear_on_traj_data \\
      --agentic-config-path my_config.yaml \\
      --eval-model-name meta-llama/llama-3-1-70b-instruct

  # CLI only (all parameters)
  python -m clear_eval.agentic.pipeline.run_clear_on_traj_data \\
      --traces-data-dir output/traces_data \\
      --agentic-output-dir output/analysis \\
      --provider watsonx \\
      --eval-model-name meta-llama/llama-3-3-70b-instruct

Config file structure (YAML format - see setup/default_config.yaml):
  # Agentic pipeline arguments
  traces_data_dir: output/traces_data
  agentic_output_dir: output/analysis
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
    if not config_dict.get('traces_data_dir'):
        parser.error("traces_data_dir is required (set in config or use --traces-data-dir)")
    if not config_dict.get('agentic_output_dir'):
        parser.error("agentic_output_dir is required (set in config or use --agentic-output-dir)")

    # Extract parameters
    traces_data_dir = config_dict['traces_data_dir']
    agentic_output_dir = config_dict['agentic_output_dir']
    overwrite = config_dict.get('overwrite', True)

    logger.info("=" * 80)
    logger.info("CLEAR PIPELINE: TRAJECTORY DATA ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Trajectory data: {traces_data_dir}")
    logger.info(f"Output: {agentic_output_dir}")
    if memory_only:
        logger.info("  └── Memory-only mode: Only ui_results.zip and clear_results.json will be saved")
    else:
        logger.info("  ├── clear_data/     (CLEAR format by agent)")
        logger.info("  └── clear_results/  (CLEAR analysis results)")
    logger.info("=" * 80)

    # Call the shared pipeline function
    json_results = run_traj_data_pipeline(
        traces_data_dir=traces_data_dir,
        agentic_output_dir=agentic_output_dir,
        config_dict=config_dict,
        overwrite=overwrite
    )


if __name__ == "__main__":
    main()

