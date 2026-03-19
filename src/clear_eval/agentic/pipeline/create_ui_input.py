#!/usr/bin/env python3
"""
Create optimized UI input zip from CLEAR results and trajectory data.

NEW APPROACH:
- Trajectory data: Keep model_input and response, store as Parquet (compressed)
- CLEAR results: Remove model_input and response columns
- Dashboard: Join data using (task_id, step_in_trace_general) as key
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
import tempfile

logger = logging.getLogger(__name__)


def convert_traj_csvs_to_parquet(csv_files: list, temp_dir: Path) -> list:
    """
    Convert trajectory CSV files to Parquet format.
    Keep model_input and response columns.

    Returns:
        List of Parquet file paths
    """
    parquet_files = []

    logger.info(f"Converting {len(csv_files)} CSV files to Parquet")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Convert to Parquet (much better compression than CSV)
            parquet_path = temp_dir / f"{csv_file.stem}.parquet"
            df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
            parquet_files.append(parquet_path)

            # Show size reduction
            csv_size = csv_file.stat().st_size / 1024  # KB
            parquet_size = parquet_path.stat().st_size / 1024  # KB
            reduction = ((csv_size - parquet_size) / csv_size * 100) if csv_size > 0 else 0
            logger.debug(f"  {csv_file.name}: {csv_size:.1f}KB -> {parquet_size:.1f}KB ({reduction:.1f}% reduction)")

        except Exception as e:
            logger.warning(f"Error converting {csv_file.name}: {e}")

    logger.info(f"Converted {len(parquet_files)} files to Parquet")
    return parquet_files


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


def create_ui_input_zip(
    output_dir: Path,
    traces_data_dir: Path,
    clear_results_dir: Path,
    output_zip_name: str = "ui_input.zip"
) -> Path:
    """
    Create optimized UI input zip from trajectory data and CLEAR results.

    NEW APPROACH:
    - Trajectory data stored as Parquet (keeps model_input & response)
    - CLEAR results have model_input & response removed
    - Dashboard joins them using (task_id, step_in_trace_general)

    Args:
        output_dir: Directory to save the output zip
        traces_data_dir: Directory containing trajectory CSV files
        clear_results_dir: Directory containing agent CLEAR result subdirectories
        output_zip_name: Name of the output zip file

    Returns:
        Path to the created zip file
    """
    output_dir = Path(output_dir)
    traj_data_dir = Path(traces_data_dir)
    clear_results_dir = Path(clear_results_dir)

    if not traj_data_dir.exists():
        raise FileNotFoundError(f"Trajectory data directory not found: {traj_data_dir}")
    if not clear_results_dir.exists():
        raise FileNotFoundError(f"CLEAR results directory not found: {clear_results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_zip = output_dir / output_zip_name

    if result_zip.exists():
        result_zip.unlink()

    logger.info("=" * 80)
    logger.info("CREATING OPTIMIZED UI INPUT ZIP")
    logger.info("=" * 80)
    logger.info(f"Trajectory data: {traj_data_dir}")
    logger.info(f"CLEAR results: {clear_results_dir}")
    logger.info(f"Output: {result_zip}")
    logger.info("NEW APPROACH:")
    logger.info("  - Trajectory data: Parquet format (keeps model_input & response)")
    logger.info("  - CLEAR results: Remove model_input & response columns")
    logger.info("  - Dashboard: Joins data by (task_id, step_in_trace_general)")

    total_original_size = 0
    total_compressed_size = 0

    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Process trajectory data - convert to Parquet
        traj_count = 0
        logger.info("Processing trajectory data (CSV -> Parquet)")
        csv_files = list(traj_data_dir.glob("*.csv"))

        if csv_files:
            with tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir = Path(temp_dir_str)

                # Convert CSVs to Parquet
                parquet_files = convert_traj_csvs_to_parquet(csv_files, temp_dir)

                # Create trajectory data zip with Parquet files
                traj_zip_buffer = BytesIO()
                with zipfile.ZipFile(traj_zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as traj_zf:
                    for parquet_file in sorted(parquet_files):
                        traj_zf.write(parquet_file, arcname=parquet_file.name)
                        traj_count += 1
                        total_compressed_size += parquet_file.stat().st_size

                # Calculate original CSV size
                for csv_file in csv_files:
                    total_original_size += csv_file.stat().st_size

                # Add the trajectory zip to the main zip
                traj_zip_buffer.seek(0)
                zf.writestr("trajectory_data.zip", traj_zip_buffer.read())
                logger.info(f"Compressed {traj_count} trajectory files as Parquet")
        else:
            logger.warning(f"No CSV files found in {traj_data_dir}")

        # 2. Add deduplicated agent CLEAR results
        logger.info("Processing agent CLEAR results (removing model_input & response)")
        agent_dirs = [d for d in clear_results_dir.iterdir() if d.is_dir()]
        agent_count = 0
        total_agent_stats = {
            'original_size': 0,
            'deduplicated_size': 0,
            'files_processed': 0
        }

        for agent_dir in sorted(agent_dirs):
            agent_name = agent_dir.name
            zip_files = list(agent_dir.glob("analysis_results_*.zip"))

            if zip_files:
                for zip_file in zip_files:
                    arcname = f"agent_results/{agent_name}.zip"

                    logger.debug(f"Processing {agent_name}.zip")
                    dedup_buffer = BytesIO()
                    try:
                        stats = deduplicate_agent_result_zip(zip_file, dedup_buffer)
                        dedup_buffer.seek(0)
                        zf.writestr(arcname, dedup_buffer.read())

                        # Accumulate stats
                        total_agent_stats['original_size'] += stats['original_size']
                        total_agent_stats['deduplicated_size'] += stats['deduplicated_size']
                        total_agent_stats['files_processed'] += stats['files_processed']

                        logger.debug(f"  Removed model_input/response from {stats['files_processed']} files")
                        logger.debug(f"  Size: {stats['original_size']/1024:.1f}KB -> {stats['deduplicated_size']/1024:.1f}KB ({stats['reduction_pct']:.1f}% reduction)")
                    except Exception as e:
                        logger.warning(f"Deduplication failed for {agent_name}, using original: {e}")
                        zf.write(zip_file, arcname=arcname)

                    agent_count += 1

        logger.info(f"Total agent results: {agent_count}")
        if total_agent_stats['files_processed'] > 0:
            logger.info(f"Total files deduplicated: {total_agent_stats['files_processed']}")
            logger.info(f"Total size reduction: {total_agent_stats['original_size']/1024/1024:.1f}MB -> {total_agent_stats['deduplicated_size']/1024/1024:.1f}MB")

        # 3. Create and add metadata
        logger.info("Creating metadata")
        metadata = {
            "created_at": datetime.now().isoformat(),
            "traj_data_dir": str(traj_data_dir),
            "clear_results_dir": str(clear_results_dir),
            "agent_count": agent_count,
            "trajectory_count": traj_count,
            "agents": [d.name for d in agent_dirs if d.is_dir()],
            "structure": {
                "agent_results/": "CLEAR analysis results (model_input & response removed)",
                "trajectory_data.zip": "Trajectory data as Parquet (keeps model_input & response)",
                "metadata.json": "This file - information about the zip contents"
            },
            "optimization": {
                "enabled": True,
                "description": "model_input & response stored only in trajectory_data.zip",
                "join_key": "(task_id, step_in_trace_general)",
                "trajectory_format": "Parquet (compressed)",
                "clear_results_format": "CSV/Parquet (deduplicated)"
            },
            "format_version": "4.0"
        }

        metadata_json = json.dumps(metadata, indent=2)
        zf.writestr("metadata.json", metadata_json)
        logger.info("Added metadata.json")

    final_size = result_zip.stat().st_size
    total_original_size += total_agent_stats['original_size']
    total_compressed_size += total_agent_stats['deduplicated_size']

    logger.info("=" * 80)
    logger.info(f"Successfully created: {result_zip}")
    logger.info(f"Final zip size: {final_size / (1024*1024):.2f} MB")
    if total_original_size > 0:
        reduction = ((total_original_size - final_size) / total_original_size * 100)
        logger.info(f"Total reduction: {total_original_size/(1024*1024):.2f}MB -> {final_size/(1024*1024):.2f}MB ({reduction:.1f}%)")
    logger.info("=" * 80)

    return result_zip


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
    
    logger.info("=" * 80)
    logger.info("CREATING UNIFIED UI ZIP")
    logger.info("=" * 80)
    logger.info(f"Output: {result_zip}")
    
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Add trajectory data if provided
        if traces_data_dir and Path(traces_data_dir).exists():
            traj_data_dir = Path(traces_data_dir)
            logger.info(f"Processing trajectory data from: {traj_data_dir}")
            csv_files = list(traj_data_dir.glob("*.csv"))
            
            if csv_files:
                with tempfile.TemporaryDirectory() as temp_dir_str:
                    temp_dir = Path(temp_dir_str)
                    
                    # Convert CSVs to Parquet
                    parquet_files = convert_traj_csvs_to_parquet(csv_files, temp_dir)
                    
                    # Create trajectory data zip with Parquet files
                    traj_zip_buffer = BytesIO()
                    with zipfile.ZipFile(traj_zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as traj_zf:
                        for parquet_file in sorted(parquet_files):
                            traj_zf.write(parquet_file, arcname=parquet_file.name)
                    
                    traj_zip_buffer.seek(0)
                    zf.writestr("trajectory_data.zip", traj_zip_buffer.read())
                    logger.info(f"Added trajectory_data.zip with {len(parquet_files)} Parquet files")
                    
                    # Also add original CSV files
                    for csv_file in sorted(csv_files):
                        arcname = f"traces_data/{csv_file.name}"
                        zf.write(csv_file, arcname=arcname)
                    logger.info(f"Added {len(csv_files)} CSV files to traces_data/")
        
        # 2. Add step-by-step CLEAR results if provided
        if step_by_step_clear_results_dir and Path(step_by_step_clear_results_dir).exists():
            clear_dir = Path(step_by_step_clear_results_dir)
            logger.info(f"Processing step-by-step CLEAR results from: {clear_dir}")
            
            agent_dirs = [d for d in clear_dir.iterdir() if d.is_dir()]
            agent_count = 0
            
            for agent_dir in sorted(agent_dirs):
                agent_name = agent_dir.name
                zip_files = list(agent_dir.glob("analysis_results_*.zip"))
                
                if zip_files:
                    for zip_file in zip_files:
                        arcname = f"agent_results/{agent_name}.zip"
                        dedup_buffer = BytesIO()
                        try:
                            deduplicate_agent_result_zip(zip_file, dedup_buffer)
                            dedup_buffer.seek(0)
                            zf.writestr(arcname, dedup_buffer.read())
                            agent_count += 1
                        except Exception as e:
                            logger.warning(f"Deduplication failed for {agent_name}, using original: {e}")
                            zf.write(zip_file, arcname=arcname)
                            agent_count += 1
            
            logger.info(f"Added {agent_count} step-by-step agent results")
        
        # 3. Add full trajectory results if provided
        if full_trajectory_results_dir and Path(full_trajectory_results_dir).exists():
            full_traj_dir = Path(full_trajectory_results_dir)
            logger.info(f"Processing full trajectory results from: {full_traj_dir}")
            
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
                "traces_data/": "Original trajectory CSV files",
                "agent_results/": "Step-by-step CLEAR analysis results",
                "full_traj_results/": "Full trajectory evaluation results",
                "metadata.json": "This file - information about the zip contents"
            },
            "format_version": "5.0"
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        logger.info("Added metadata.json")
    
    final_size = result_zip.stat().st_size
    logger.info("=" * 80)
    logger.info(f"Successfully created unified zip: {result_zip}")
    logger.info(f"Final zip size: {final_size / (1024*1024):.2f} MB")
    logger.info("=" * 80)
    
    return result_zip


def main():
    parser = argparse.ArgumentParser(
        description="Create optimized UI input zip from CLEAR results and trajectory data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Root directory containing traces_data/ and clear_results/ subdirectories",
        default="/Users/lilache/PycharmProjects/CLEAR/src/clear_eval/agentic/output/paper_experiments/clear_step/TRAIL/GAIA/clear_result/gpt-oss-120b"
    )
    parser.add_argument(
        "--traces-data",
        type=str,
        default="/Users/lilache/PycharmProjects/CLEAR/src/clear_eval/agentic/data/paper_experiments/TRAIL/GAIA/csvs/",
        help="Path to trajectory data directory (default: output_dir/traces_data)"
    )
    parser.add_argument(
        "--clear-results",
        type=str,
        default="/Users/lilache/PycharmProjects/CLEAR/src/clear_eval/agentic/output/paper_experiments/clear_step/TRAIL/GAIA/clear_result/gpt-oss-120b",
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
