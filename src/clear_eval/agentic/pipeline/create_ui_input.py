#!/usr/bin/env python3
"""
Create optimized UI input zip from CLEAR results and trajectory data.
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path
from io import BytesIO, StringIO
from datetime import datetime
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def convert_csv_to_parquet_buffer(csv_file: Path) -> tuple[str, bytes, int]:
    """
    Convert a single CSV file to Parquet format in memory.

    Args:
        csv_file: Path to CSV file

    Returns:
        Tuple of (filename, parquet_bytes, csv_size)
    """
    df = pd.read_csv(csv_file)

    # Convert to Parquet in memory
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, engine='pyarrow', compression='snappy', index=False)
    parquet_bytes = parquet_buffer.getvalue()

    # Calculate sizes for logging
    csv_size = csv_file.stat().st_size
    parquet_size = len(parquet_bytes)
    reduction = ((csv_size - parquet_size) / csv_size * 100) if csv_size > 0 else 0

    logger.debug(f"  {csv_file.name}: {csv_size/1024:.1f}KB -> {parquet_size/1024:.1f}KB ({reduction:.1f}% reduction)")

    return f"{csv_file.stem}.parquet", parquet_bytes, csv_size


def deduplicate_agent_result_zip(agent_zip_path: Path, output_buffer: BytesIO) -> dict:
    """
    Remove model_input and response columns from agent result zip CSVs.

    Returns:
        Dictionary with size statistics
    """
    original_size = 0
    deduplicated_size = 0
    files_processed = 0

    with zipfile.ZipFile(agent_zip_path, 'r') as source_zip:
        with zipfile.ZipFile(output_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as target_zip:
            for item in source_zip.namelist():
                file_data = source_zip.read(item)
                original_size += len(file_data)

                if item.endswith('.csv') or item.endswith('.parquet'):
                    try:
                        # Read the file
                        if item.endswith('.csv'):
                            csv_content = file_data.decode('utf-8')
                            df = pd.read_csv(StringIO(csv_content))
                        else:
                            df = pd.read_parquet(BytesIO(file_data))

                        # Remove model_input and response columns
                        columns_to_drop = []
                        if 'model_input' in df.columns:
                            columns_to_drop.append('model_input')
                        if 'response' in df.columns:
                            columns_to_drop.append('response')

                        if columns_to_drop:
                            df = df.drop(columns=columns_to_drop)
                            files_processed += 1

                            # Write back in same format
                            if item.endswith('.csv'):
                                csv_buffer = StringIO()
                                df.to_csv(csv_buffer, index=False)
                                file_data = csv_buffer.getvalue().encode('utf-8')
                            else:
                                parquet_buffer = BytesIO()
                                df.to_parquet(parquet_buffer, engine='pyarrow', compression='snappy', index=False)
                                file_data = parquet_buffer.getvalue()

                    except Exception as e:
                        logger.warning(f"Could not process {item}: {e}")

                target_zip.writestr(item, file_data)
                deduplicated_size += len(file_data)

    return {
        'original_size': original_size,
        'deduplicated_size': deduplicated_size,
        'files_processed': files_processed,
        'reduction_pct': ((original_size - deduplicated_size) / original_size * 100) if original_size > 0 else 0
    }


def _add_trajectory_data_to_zip(zf: zipfile.ZipFile, traces_data_dir: Path) -> int:
    """
    Add trajectory data to zip file (converted to Parquet).

    Args:
        zf: ZipFile object to write to
        traces_data_dir: Directory containing trajectory CSV files

    Returns:
        Number of trajectory files added
    """
    traj_count = 0
    csv_files = list(traces_data_dir.glob("*.csv"))

    if csv_files:
        logger.info(f"Converting {len(csv_files)} trajectory files to Parquet...")

        # Create trajectory data zip with Parquet files (all in memory)
        traj_zip_buffer = BytesIO()
        with zipfile.ZipFile(traj_zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as traj_zf:
            for csv_file in sorted(csv_files):
                try:
                    filename, parquet_bytes, _ = convert_csv_to_parquet_buffer(csv_file)
                    traj_zf.writestr(filename, parquet_bytes)
                    traj_count += 1
                except Exception as e:
                    logger.warning(f"Error converting {csv_file.name}: {e}")

        traj_zip_buffer.seek(0)
        zf.writestr("trajectory_data.zip", traj_zip_buffer.read())
        logger.info(f"  ✓ Trajectory data: {traj_count} files")
    else:
        logger.warning(f"No CSV files found in {traces_data_dir}")

    return traj_count


def _process_and_add_results_from_dir(
    zf: zipfile.ZipFile,
    results_dir: Path,
    agent_label: str,
    total_agent_stats: dict
) -> bool:
    """
    Find, process, and add CLEAR results zip from a directory.

    This helper function eliminates code duplication between processing
    regular agent results and tool-call results.

    Args:
        zf: ZipFile object to write to
        results_dir: Directory containing analysis_results_*.zip file
        agent_label: Label for logging (e.g., "agent_name" or "agent_name__tool_calls")
        total_agent_stats: Dictionary to accumulate statistics

    Returns:
        True if a zip file was found and processed, False otherwise
    """
    # Find the analysis results zip file (at most one per directory)
    zip_files = list(results_dir.glob("analysis_results_*.zip"))
    if not zip_files:
        return False
    
    # Each directory produces exactly one analysis_results_*.zip file
    zip_file = zip_files[0]
    arcname = f"agent_results/{agent_label}.zip"
    dedup_buffer = BytesIO()
    
    try:
        stats = deduplicate_agent_result_zip(zip_file, dedup_buffer)
        dedup_buffer.seek(0)
        zf.writestr(arcname, dedup_buffer.read())

        # Accumulate stats
        total_agent_stats['original_size'] += stats['original_size']
        total_agent_stats['deduplicated_size'] += stats['deduplicated_size']
        total_agent_stats['files_processed'] += stats['files_processed']

        logger.debug(f"  {agent_label}: {stats['files_processed']} files, "
                    f"{stats['original_size']/1024:.1f}KB -> {stats['deduplicated_size']/1024:.1f}KB")
        return True

    except Exception as e:
        logger.warning(f"Deduplication failed for {agent_label}, using original: {e}")
        zf.write(zip_file, arcname=arcname)
        return True


def _add_agent_results_to_zip(zf: zipfile.ZipFile, clear_results_dir: Path) -> tuple[int, dict]:
    """
    Add deduplicated agent CLEAR results to zip file.

    Args:
        zf: ZipFile object to write to
        clear_results_dir: Directory containing agent CLEAR result subdirectories

    Returns:
        Tuple of (agent_count, stats_dict)
    """
    agent_dirs = [d for d in clear_results_dir.iterdir() if d.is_dir()]
    agent_count = 0
    total_agent_stats = {
        'original_size': 0,
        'deduplicated_size': 0,
        'files_processed': 0
    }

    for agent_dir in sorted(agent_dirs):
        agent_name = agent_dir.name
        
        # Process main agent results
        if _process_and_add_results_from_dir(zf, agent_dir, agent_name, total_agent_stats):
            agent_count += 1

        # Process tool-calls subdir results
        tool_calls_dir = agent_dir / "tool_calls"
        if tool_calls_dir.is_dir():
            agent_label = f"{agent_name}__tool_calls"
            if _process_and_add_results_from_dir(zf, tool_calls_dir, agent_label, total_agent_stats):
                agent_count += 1

    if agent_count > 0:
        logger.info(f"  ✓ Agent results: {agent_count} agents")

    return agent_count, total_agent_stats


def create_ui_input_zip(
    output_dir: Path,
    traces_data_dir: Path,
    clear_results_dir: Path,
    output_zip_name: str = "ui_input.zip"
) -> Path:
    """
    Create optimized UI input zip from trajectory data and CLEAR results.

    This is a convenience wrapper around create_unified_ui_zip() for backward compatibility.
    It enforces that both traces_data_dir and clear_results_dir are required and exist.

    Args:
        output_dir: Directory to save the output zip
        traces_data_dir: Directory containing trajectory CSV files (required)
        clear_results_dir: Directory containing agent CLEAR result subdirectories (required)
        output_zip_name: Name of the output zip file

    Returns:
        Path to the created zip file

    Raises:
        FileNotFoundError: If traces_data_dir or clear_results_dir don't exist
    """
    traj_data_dir = Path(traces_data_dir)
    clear_results_dir = Path(clear_results_dir)

    if not traj_data_dir.exists():
        raise FileNotFoundError(f"Trajectory data directory not found: {traj_data_dir}")
    if not clear_results_dir.exists():
        raise FileNotFoundError(f"CLEAR results directory not found: {clear_results_dir}")

    # Delegate to the unified function
    return create_unified_ui_zip(
        output_dir=output_dir,
        traces_data_dir=traj_data_dir,
        step_by_step_clear_results_dir=clear_results_dir,
        full_trajectory_results_dir=None,
        output_zip_name=output_zip_name
    )


def create_unified_ui_zip(
    output_dir: Path,
    traces_data_dir: Optional[Path] = None,
    step_by_step_clear_results_dir: Optional[Path] = None,
    full_trajectory_results_dir: Optional[Path] = None,
    output_zip_name: str = "ui_results.zip"
) -> Path:
    """
    Create a unified UI input zip containing both step-by-step and full trajectory results.
    
    This combines:
    - Step-by-step CLEAR analysis results (agent_results/)
    - Full trajectory evaluation results (full_traj_results/)
    - Trajectory data (trajectory_data.zip and traces_data/)
    
    Args:
        output_dir: Directory to save the output zip
        traces_data_dir: Directory containing trajectory CSV files
        step_by_step_clear_results_dir: Directory containing step-by-step CLEAR results
        full_trajectory_results_dir: Directory containing full trajectory evaluation results
        output_zip_name: Name of the output zip file
        
    Returns:
        Path to the created unified zip file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_zip = output_dir / output_zip_name
    
    if result_zip.exists():
        result_zip.unlink()

    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Add trajectory data if provided
        if traces_data_dir and Path(traces_data_dir).exists():
            traj_data_dir = Path(traces_data_dir)
            _add_trajectory_data_to_zip(zf, traj_data_dir)

        # 2. Add step-by-step CLEAR results if provided
        if step_by_step_clear_results_dir and Path(step_by_step_clear_results_dir).exists():
            clear_dir = Path(step_by_step_clear_results_dir)
            _add_agent_results_to_zip(zf, clear_dir)
        
        # 3. Add full trajectory results if provided
        if full_trajectory_results_dir and Path(full_trajectory_results_dir).exists():
            full_traj_dir = Path(full_trajectory_results_dir)
            
            # Map actual directory names to dashboard-expected names
            dir_mapping = {
                'task_success': 'task_success',
                'full_trajectory': 'per_traj_results',  # Map to expected name
                'rubric_evaluation': 'rubric_eval_results',  # Map to expected name
                'clear_analysis': 'clear_results'
            }
            
            for actual_dir_name, expected_dir_name in dir_mapping.items():
                subdir = full_traj_dir / actual_dir_name
                if subdir.exists():
                    file_count = 0
                    for file_path in subdir.rglob('*'):
                        if file_path.is_file():
                            arcname = f"full_traj_results/{expected_dir_name}/{file_path.relative_to(subdir)}"
                            zf.write(file_path, arcname=arcname)
                            file_count += 1
                    if file_count > 0:
                        logger.info(f"Added {file_count} files from {actual_dir_name}/ as {expected_dir_name}/")
        
        # 4. Create metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "type": "unified_results",
            "structure": {
                "trajectory_data.zip": "Trajectory data as Parquet (compressed)",
#                "traces_data/": "Original trajectory CSV files",
                "agent_results/": "Step-by-step CLEAR analysis results",
                "full_traj_results/": "Full trajectory evaluation results",
                "metadata.json": "This file - information about the zip contents"
            },
            "format_version": "5.0"
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    final_size = result_zip.stat().st_size
    logger.info(f"  ✓ Zip size: {final_size / (1024*1024):.2f} MB")
    
    return result_zip


def main():
    parser = argparse.ArgumentParser(
        description="Create optimized UI input zip from CLEAR results and trajectory data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Root directory containing traces_data/ and clear_results/ subdirectories",
        )
    parser.add_argument(
        "--traces-data",
        type=str,
        help="Path to trajectory data directory (default: output_dir/traces_data)"
    )
    parser.add_argument(
        "--clear-results",
        type=str,
        help="Path to CLEAR results directory (default: output_dir/clear_results)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="ui_input.zip",
        help="Name of the output zip file (default: ui_input.zip)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    traj_data_dir = Path(args.traces_data) if args.traces_data else output_dir / "traces_data"
    if args.clear_results:
        clear_results_dir = Path(args.clear_results)
    else:
        clear_results_dir = output_dir / "clear_results"
        d = Path(clear_results_dir)
        subs = [p for p in d.iterdir() if p.is_dir()]
        clear_results_dir =  subs[0] if len(subs) == 1 else clear_results_dir

    try:
        result_zip = create_ui_input_zip(
            output_dir=output_dir,
            traces_data_dir=traj_data_dir,
            clear_results_dir=clear_results_dir,
            output_zip_name=args.output_name
        )
        logger.info("To view the results, run: run-clear-ai-dashboard")
        logger.info(f"   Then upload: {result_zip}")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
