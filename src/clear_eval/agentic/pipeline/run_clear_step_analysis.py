"""
CLEAR Step-by-Step Analysis Pipeline for Agentic Workflows

This module runs the step-by-step CLEAR analysis pipeline:
- If from_raw_traces=True: traces (LangGraph/CrewAI) -> trajectory data -> CLEAR format -> CLEAR results
- If from_raw_traces=False: trajectory data (CSVs) -> CLEAR format -> CLEAR results

Configuration Precedence (lowest to highest):
    1. Default config: setup/default_agentic_config.yaml
    2. User config file: --agentic-config-path (if provided)
    3. CLI arguments (override both config files)

Arguments are split into two groups:
    - Agentic Pipeline Arguments: Pipeline-specific (data_dir, etc.)
    - CLEAR Configuration: Evaluation args from clear_eval.args (provider, eval_model_name, etc.)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from clear_eval.analysis_runner import run_clear_eval_evaluation
from clear_eval.agentic.dashboard.generate_static_dashboard import generate_html
from clear_eval.agentic.pipeline.utils import (
    build_cli_overrides,
    load_pipeline_config,
    get_run_output_dir,
    validate_required_config,
)
from clear_eval.agentic.pipeline.build_json_results import save_comprehensive_json_results
from clear_eval.agentic.pipeline.create_ui_input import create_ui_input_zip
from clear_eval.agentic.pipeline.preprocess_traces.preprocess_traces import process_traces_to_traj_data
from clear_eval.agentic.pipeline.full_traces_evaluation.argument_parser import add_preprocessing_args_to_parser
from clear_eval.args import add_clear_args_to_parser, str2bool
from clear_eval.logging_config import setup_logging

# Initialize logging
setup_logging()

logger = logging.getLogger(__name__)


def add_agentic_args_to_parser(parser: argparse.ArgumentParser) -> None:
    """
    Add agentic pipeline arguments to the parser.

    These correspond to the agentic pipeline section in default_agentic_config.yaml.
    """
    group = parser.add_argument_group("Agentic Pipeline Arguments")

    group.add_argument(
        "--agentic-config-path",
        help="Path to config file (JSON or YAML) that overrides defaults"
    )
    group.add_argument(
        "--data-dir",
        help="Input directory: raw traces (if from-raw-traces=true) or trajectory CSVs (if from-raw-traces=false)"
    )
    group.add_argument(
        "--results-dir",
        help="Output directory for pipeline results (required)"
    )
    group.add_argument(
        "--from-raw-traces",
        type=str2bool,
        help="If true, preprocess raw traces first; if false, use trajectory CSVs directly (default: false)"
    )

    # Add preprocessing arguments (agent-framework, observability-framework, separate-tools)
    add_preprocessing_args_to_parser(parser)

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
        help="If true, use temp directories and return only JSON results (no files saved) (default: false)"
    )


TOOL_CALLS_SUFFIX = "__tool_calls"


##########################################
## Convert shared data to clear format ###
##########################################
def convert_to_clear_format(input_dir: str, output_dir: str, overwrite: bool = True) -> None:
    """
    Convert CSV files to CLEAR format grouped by agent.

    When the data contains tool rows (``tool_or_agent == "tool"``), each agent
    produces two CSVs:
    - ``{agent_name}.csv`` — reasoning (agent) rows
    - ``{agent_name}__tool_calls.csv`` — tool-call rows

    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save CLEAR format files
        overwrite: If False, skip conversion if output files already exist
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check if conversion already done (statistics.json exists and has content)
    stats_file = output_dir / "statistics.json"
    if not overwrite and stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                if stats.get('unique_agents', 0) > 0:
                    logger.info(f"Skipping conversion (already exists): {output_dir}")
                    logger.info(f"  Found {stats['unique_agents']} agent types, {stats['total_rows']} interactions")
                    return
        except Exception:
            pass  # If we can't read stats, proceed with conversion

    agent_data = defaultdict(list)
    tool_data = defaultdict(list)
    total_rows = 0
    task_counter = Counter()
    agent_counter = Counter()

    input_path = Path(input_dir)
    csv_files = list(input_path.glob('*.csv'))

    logger.info(f"Converting {len(csv_files)} CSV files to CLEAR format...")

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
                    is_tool_row = (
                        'tool_or_agent' in row.index
                        and row.get('tool_or_agent') == 'tool'
                    )
                    if is_tool_row:
                        tool_data[agent_name].append(row_with_agent)
                    else:
                        agent_data[agent_name].append(row_with_agent)

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue

    # Write agent (reasoning) CSVs
    all_agents = set(agent_data.keys()) | set(tool_data.keys())

    for agent_name in sorted(all_agents):
        if agent_name in agent_data:
            out_df = pd.DataFrame(agent_data[agent_name])
            output_file = output_dir / f"{agent_name}.csv"
            out_df.to_csv(output_file, index=False)

        if agent_name in tool_data:
            out_df = pd.DataFrame(tool_data[agent_name])
            output_file = output_dir / f"{agent_name}{TOOL_CALLS_SUFFIX}.csv"
            out_df.to_csv(output_file, index=False)
            logger.info(f"  {agent_name}{TOOL_CALLS_SUFFIX}.csv ({len(out_df)} tool rows)")

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

    logger.info(f"✓ Converted to CLEAR format: {len(agent_counter)} agent types, {total_rows} interactions")
    logger.info("Agent distribution:")
    for agent, count in agent_counter.most_common():
        logger.info(f"  {agent:40} : {count:4} interactions")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


def run_analysis_for_agent(
    csv_path: Path,
    results_dir: str,
    overwrite: bool,
    config_dict: dict,
) -> Optional[bool]:
    """
    Run CLEAR analysis for a single agent CSV file.

    Tool-call CSVs (filename ending with ``__tool_calls``) are routed to
    ``task="tool_call"`` (SPARC evaluation) and stored in a ``tool_calls/``
    subdirectory under the parent agent's results directory.  All other CSVs
    use ``task="general"`` (standard CLEAR evaluation) at the agent root.

    Directory layout::

        clear_results/
          agent_1/                    # reasoning results at agent root
            analysis_results_*.csv
            tool_calls/               # tool eval results in subdir
              analysis_results_*.csv
          agent_2/                    # reasoning-only agent, no subdir
            analysis_results_*.csv

    Args:
        csv_path: Path to the CSV file
        results_dir: Directory to save CLEAR analysis results
        overwrite: Whether to overwrite existing results
        config_dict: Pre-loaded configuration dictionary

    Returns:
        True if analysis was run, False if skipped, None if error
    """
    config_dict = config_dict.copy()

    stem = csv_path.stem
    is_tool_csv = stem.endswith(TOOL_CALLS_SUFFIX)
    if is_tool_csv:
        agent_name = stem[: -len(TOOL_CALLS_SUFFIX)]
        output_dir = Path(results_dir) / agent_name / "tool_calls"
        task = "tool_call"
        label = "tool-call"
    else:
        agent_name = stem
        output_dir = Path(results_dir) / agent_name
        task = "general"
        label = "reasoning"

    logger.info("=" * 80)
    logger.info(f"Processing: {agent_name} ({label})")
    logger.info(f"Output folder: {output_dir}")
    logger.info("=" * 80)

    if output_dir.exists() and not overwrite and list(Path(output_dir).glob('*.zip')):
        logger.info(f"Skipping: output directory already exists: {output_dir}")
        logger.info("Use --overwrite to re-run analysis")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Running {label} analysis...")
        logger.info(f"  Data: {csv_path}")
        logger.info(f"  Output: {output_dir}")

        config_dict["data_path"] = str(csv_path)
        config_dict["output_dir"] = str(output_dir)
        config_dict["resume_enabled"] = not overwrite
        config_dict["task"] = task

        run_clear_eval_evaluation(None, **config_dict)

        logger.info("Analysis completed!")
        logger.info(f"  Results saved to: {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Error running analysis for {agent_name} ({label}): {str(e)}")
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
    logger.info(f"Running CLEAR analysis on {len(csv_files)} agent types (eval model: {eval_model_name})")

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

    logger.info(f"✓ Analysis complete: {stats['processed']} processed, {stats['skipped']} skipped" +
               (f", {stats['errors']} errors" if stats['errors'] > 0 else ""))


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


def run_step_analysis_pipeline(
    traces_data_dir: str,
    results_dir: str,
    config_dict: dict,
    overwrite: bool = True,
    intermediate_output_dir: Optional[str] = None,
    create_ui_zip: bool = True
) -> dict:
    """
    Run pipeline from trajectory data to CLEAR results.

    This function handles the complete workflow from trajectory CSVs:
    - Convert trajectory data to CLEAR format
    - Run CLEAR analysis for each agent
    - Create comprehensive UI results
    - Save JSON results

    Args:
        traces_data_dir: Directory containing trajectory CSV files
        results_dir: Base output directory for final results
        config_dict: Configuration dictionary with CLEAR params
        overwrite: Whether to overwrite existing results
        intermediate_output_dir: Optional directory for intermediate files
        create_ui_zip: Whether to create ui_results.zip (default: True).
                      Set to False when called from unified pipeline to avoid duplicate zips.

    Returns:
        Dictionary with JSON results
    """
    memory_only = config_dict.get('memory_only')

    # Create output directory for final results
    os.makedirs(results_dir, exist_ok=True)

    # Use temporary directory for intermediate files if memory_only mode and no intermediate_output_dir provided
    if memory_only and intermediate_output_dir is None:
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_context.__enter__()
        intermediate_output_dir = temp_dir
        logger.info("Memory-only mode: Using temporary directory for intermediate files")
    elif intermediate_output_dir is None:
        temp_dir_context = None
        intermediate_output_dir = results_dir
    else:
        # intermediate_output_dir was provided (from run_full_pipeline)
        temp_dir_context = None

    try:
        clear_data_dir = os.path.join(intermediate_output_dir, 'clear_data')
        clear_results_dir = os.path.join(intermediate_output_dir, 'clear_results')

        logger.info("Converting trajectory data to CLEAR format...")
        convert_to_clear_format(traces_data_dir, clear_data_dir, overwrite=overwrite)

        run_clear_analysis(
            clear_data_dir,
            clear_results_dir,
            config_dict,
            overwrite=overwrite,
        )

        # Build JSON results (returns dict)
        from clear_eval.agentic.pipeline.build_json_results import build_comprehensive_json_results, save_json_to_file
        json_results_path = save_comprehensive_json_results(
            clear_results_dir=clear_results_dir,
            traces_data_dir=traces_data_dir,
            config_dict=config_dict,
            output_dir=results_dir,
        )

        # generate static HTML report of results
        generate_html(json_results_path)


        # Create UI results zip directly in final output directory (if requested)
        if create_ui_zip:
            ui_results_path = create_ui_input_zip(
                output_dir=Path(results_dir),
                traces_data_dir=Path(traces_data_dir),
                clear_results_dir=Path(clear_results_dir),
                output_zip_name="ui_results.zip"
            )
            logger.info(f"Saved UI results to: {ui_results_path}")

        if memory_only:
            logger.info("Memory-only mode: Intermediate files will be cleaned up")

        logger.info("Pipeline complete!")
        return json_results_path

    finally:
        # Clean up temporary directory if used
        if temp_dir_context is not None:
            temp_dir_context.__exit__(None, None, None)
            logger.info("Temporary intermediate files cleaned up")


# Backward compatibility alias
run_traj_data_pipeline = run_step_analysis_pipeline


def run_full_pipeline(config_dict: dict) -> dict:
    """
    Complete pipeline: raw traces -> trajectory data -> CLEAR results.

    Args:
        config_dict: Configuration dictionary with agentic and CLEAR params (all top-level)

    Returns:
        Dictionary with JSON results
    """
    # Extract agentic-specific parameters
    traces_input_dir = config_dict.get('data_dir')
    results_dir = config_dict.get('results_dir')
    agent_framework = config_dict.get('agent_framework', 'langgraph')
    observability_framework = config_dict.get('observability_framework', 'mlflow')
    separate_tools = config_dict.get('separate_tools')
    overwrite = config_dict.get('overwrite')
    memory_only = config_dict.get('memory_only')

    if not traces_input_dir:
        raise ValueError("data_dir is required")
    if not results_dir:
        raise ValueError("results_dir is required")

    # Use temporary directory for intermediate files if memory_only mode
    if memory_only:
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_context.__enter__()
        intermediate_output_dir = temp_dir
        logger.info("Using temporary directory for intermediate files")
    else:
        temp_dir_context = None
        intermediate_output_dir = results_dir

    try:
        # Intermediate files go to temp dir (memory-only) or results_dir (normal)
        traces_data_dir = os.path.join(intermediate_output_dir, 'traces_data')

        logger.info("Processing traces to trajectory data")
        process_traces_to_traj_data(
            traces_input_dir,
            traces_data_dir,
            agent_framework=agent_framework,
            observability_framework=observability_framework,
            separate_tools=separate_tools
        )

        # Call the step analysis pipeline
        json_results = run_step_analysis_pipeline(
            traces_data_dir=traces_data_dir,
            results_dir=results_dir,
            config_dict=config_dict,
            overwrite=overwrite,
            intermediate_output_dir=intermediate_output_dir
        )

        logger.info("=" * 80)

    finally:
        # Clean up temporary directory if used
        if temp_dir_context is not None:
            temp_dir_context.__exit__(None, None, None)
            logger.info("Temporary intermediate files cleaned up")

    return json_results


def main():
    """CLI entry point for running CLEAR step-by-step analysis."""
    parser = argparse.ArgumentParser(
        description="CLEAR Step-by-Step Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From preprocessed trajectory CSVs (default)
  python -m clear_eval.agentic.pipeline.run_clear_step_analysis \\
      --data-dir data/trajectory_csvs \\
      --results-dir output/analysis \\
      --provider openai \\
      --eval-model-name gpt-4o

  # From raw traces (with preprocessing)
  python -m clear_eval.agentic.pipeline.run_clear_step_analysis \\
      --data-dir data/raw_traces \\
      --results-dir output/analysis \\
      --from-raw-traces true \\
      --agent-framework langgraph \\
      --observability-framework langfuse \\
      --provider openai \\
      --eval-model-name gpt-4o

  # Using config file (recommended)
  python -m clear_eval.agentic.pipeline.run_clear_step_analysis \\
      --agentic-config-path my_config.yaml

  # Config file with CLI overrides
  python -m clear_eval.agentic.pipeline.run_clear_step_analysis \\
      --agentic-config-path my_config.yaml \\
      --eval-model-name meta-llama/llama-3-1-70b-instruct

  # Memory-only mode (no intermediate files saved)
  python -m clear_eval.agentic.pipeline.run_clear_step_analysis \\
      --data-dir data/traces \\
      --results-dir output/analysis \\
      --from-raw-traces true \\
      --memory-only true

Config file structure (YAML format - see setup/default_agentic_config.yaml):
  # Input/Output
  data_dir: data/traces
  results_dir: output/analysis
  from_raw_traces: false  # Set to true to preprocess raw traces first

  # Preprocessing options (only used when from_raw_traces=true)
  agent_framework: langgraph
  observability_framework: langfuse
  separate_tools: combined

  # Execution options
  overwrite: true
  memory_only: false

  # CLEAR arguments
  provider: openai
  eval_model_name: gpt-4o
  agent_mode: true
  # ... other CLEAR parameters

Argument Precedence (lowest to highest):
  1. Default config (setup/default_agentic_config.yaml)
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

    # Load configuration
    config_dict = load_pipeline_config(args.agentic_config_path, **cli_overrides)

    # Validate required parameters
    validate_required_config(config_dict, ['data_dir', 'results_dir'], parser)

    # Get run output directory
    results_dir, run_name = get_run_output_dir(
        config_dict['results_dir'],
        config_dict.get('run_name')
    )

    # Update config with resolved values
    config_dict['results_dir'] = str(results_dir)
    config_dict['run_name'] = run_name

    # Extract parameters
    from_raw_traces = config_dict.get('from_raw_traces')
    data_dir = config_dict['data_dir']
    overwrite = config_dict.get('overwrite')
    memory_only = config_dict.get('memory_only')

    logger.info("=" * 80)
    if from_raw_traces:
        logger.info("CLEAR STEP ANALYSIS: FROM RAW TRACES")
    else:
        logger.info("CLEAR STEP ANALYSIS: FROM TRAJECTORY DATA")
    logger.info("=" * 80)
    logger.info(f"Input: {data_dir}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Output: {results_dir}")
    if memory_only:
        logger.info("  └── Memory-only mode: Only ui_results.zip and clear_results.json will be saved")
    else:
        if from_raw_traces:
            logger.info("  ├── traces_data/    (Trajectory CSVs)")
        logger.info("  ├── clear_data/     (CLEAR format by agent)")
        logger.info("  └── clear_results/  (CLEAR analysis results)")
    logger.info("=" * 80)

    if from_raw_traces:
        # Run full pipeline: raw traces -> trajectory data -> CLEAR
        run_full_pipeline(config_dict)
    else:
        # Run from trajectory data directly
        run_step_analysis_pipeline(
            traces_data_dir=data_dir,
            results_dir=results_dir,
            config_dict=config_dict,
            overwrite=overwrite
        )


if __name__ == "__main__":
    main()
