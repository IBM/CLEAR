#!/usr/bin/env python3
"""
Base CLEAR Runner for Evaluation Results Analysis
==================================================

Provides centralized logic for running CLEAR analysis on evaluation results
from trajectory evaluators. Subclasses implement source-specific extraction logic.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from clear_eval.analysis_runner import run_clear_eval_aggregation
from clear_eval.agentic.pipeline.utils import InferenceConfig
from clear_eval.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class BaseClearRunner(ABC):
    """
    Abstract base class for running CLEAR analysis on evaluation results.

    Centralizes common logic:
        - Result file discovery
        - DataFrame creation and grouping
        - CLEAR analysis execution
        - Output organization

    Subclasses implement source-specific extraction:
        - extract_records_from_file: Extract relevant data from eval JSON
        - get_source_name: Return source identifier
        - get_input_columns: Return columns to include in CLEAR input
    """

    def __init__(
        self,
        eval_results_dir: Path,
        output_dir: Path,
        inference_config: InferenceConfig,
        overwrite: bool = False,
    ):
        """
        Initialize CLEAR runner.

        Args:
            eval_results_dir: Directory containing evaluation results
            output_dir: Base directory for CLEAR analysis outputs
            inference_config: LLM inference configuration
            overwrite: Whether to overwrite existing CLEAR results
        """
        self.eval_results_dir = Path(eval_results_dir)
        self.output_dir = Path(output_dir)
        self.inference_config = inference_config
        self.overwrite = overwrite

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Return the source identifier for this runner.
        
        Returns:
            Source name (e.g., "issues", "root_cause")
        """
        pass

    @abstractmethod
    def get_evaluation_type_dir(self) -> str:
        """
        Return the evaluation type directory name to look for results.
        
        Returns:
            Directory name (e.g., "full_trajectory_evaluation", "task_success_evaluation")
        """
        pass

    @abstractmethod
    def get_result_file_suffix(self) -> str:
        """
        Return the file suffix for result files.
        
        Returns:
            File suffix (e.g., "_eval.json", "_success.json")
        """
        pass

    @abstractmethod
    def extract_records_from_file(self, file_path: Path) -> list[dict]:
        """
        Extract records from a single evaluation result file.
        
        Args:
            file_path: Path to evaluation result JSON file
            
        Returns:
            List of record dicts with keys:
                - id: Unique identifier
                - model_input: Context for CLEAR
                - evaluation_summary: Text to analyze
                - score: Numeric score (if applicable)
                - Additional source-specific fields
        """
        pass

    @abstractmethod
    def get_input_columns(self) -> list[str]:
        """
        Return the list of columns to include in CLEAR input CSV.
        
        Returns:
            List of column names
        """
        pass

    def discover_result_files(self) -> list[Path]:
        """
        Discover all evaluation result files in the eval_results_dir.
        
        Returns:
            List of paths to result JSON files
        """
        eval_type_dir = self.eval_results_dir / self.get_evaluation_type_dir()
        
        if not eval_type_dir.exists():
            logger.warning(f"Evaluation results directory not found: {eval_type_dir}")
            return []

        suffix = self.get_result_file_suffix()
        result_files = list(eval_type_dir.glob(f"**/*{suffix}"))
        
        logger.info(
            f"Discovered {len(result_files)} result files in {eval_type_dir}"
        )
        return result_files

    def extract_all_records(self) -> pd.DataFrame:
        """
        Extract records from all discovered result files.
        
        Returns:
            DataFrame with all extracted records
        """
        result_files = self.discover_result_files()
        
        if not result_files:
            logger.warning("No result files found")
            return pd.DataFrame()

        all_records = []
        for file_path in result_files:
            try:
                records = self.extract_records_from_file(file_path)
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Failed to extract from {file_path}: {e}")
                continue

        df = pd.DataFrame(all_records)
        logger.info(f"Extracted {len(df)} records from {len(result_files)} files")
        
        return df

    def get_clear_output_dir(self, group_key: str) -> Path:
        """
        Get output directory for a specific group's CLEAR analysis.
        
        Args:
            group_key: Group identifier (e.g., "dataset_model")
            
        Returns:
            Path to output directory
        """
        source_name = self.get_source_name()
        return self.output_dir / f"clear_analysis_{source_name}" / group_key

    def run_clear_on_group(
        self,
        group_df: pd.DataFrame,
        group_key: str,
    ) -> bool:
        """
        Run CLEAR analysis on a single group of records.
        
        Args:
            group_df: DataFrame with records for this group
            group_key: Group identifier
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = self.get_clear_output_dir(group_key)
        
        if output_dir.exists() and not self.overwrite:
            logger.info(f"Skipping (exists): {group_key}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save input CSV
        csv_path = output_dir / f"input_{self.get_source_name()}.csv"
        group_df.to_csv(csv_path, index=False)

        logger.info(f"Running CLEAR on {group_key} ({len(group_df)} records)...")

        try:
            analysis_kwargs = {
                "provider": self.inference_config.provider,
                "data_path": str(csv_path),
                "gen_model_name": None,
                "eval_model_name": self.inference_config.model_id,
                "output_dir": str(output_dir),
                "perform_generation": False,
                "input_columns": self.get_input_columns(),
                "agent_mode": True,
                "eval_model_params": self.inference_config.model_params,
                "resume_enabled": not self.overwrite,
                "inference_backend": self.inference_config.inference_backend,
                "endpoint_url": self.inference_config.endpoint_url,
            }

            run_clear_eval_aggregation(**analysis_kwargs)
            logger.info(f"Completed: {group_key}")
            return True

        except Exception as e:
            logger.error(f"CLEAR failed for {group_key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_analysis(self) -> dict:
        """
        Run CLEAR analysis on all extracted records.
        
        Returns:
            Summary dict with statistics
        """
        # Extract all records
        df = self.extract_all_records()
        
        if df.empty:
            logger.warning("No records to analyze")
            return {
                "total_records": 0,
                "groups_processed": 0,
                "groups_skipped": 0,
                "groups_failed": 0,
            }

        # Group by trajectory name (or other grouping key if present)
        # For now, treat all records as one group
        group_key = "all_trajectories"
        
        logger.info(f"Running CLEAR analysis on {len(df)} records")
        
        success = self.run_clear_on_group(df, group_key)
        
        return {
            "total_records": len(df),
            "groups_processed": 1 if success else 0,
            "groups_skipped": 0 if success else 0,
            "groups_failed": 0 if success else 1,
            "output_dir": str(self.get_clear_output_dir(group_key)),
        }

    def print_summary(self, summary: dict):
        """Print analysis summary."""
        logger.info("\n" + "=" * 70)
        logger.info("CLEAR Analysis Summary")
        logger.info("=" * 70)
        logger.info(f"  Source:         {self.get_source_name()}")
        logger.info(f"  CLEAR Model:    {self.inference_config.model_id}")
        logger.info(f"  Total Records:  {summary['total_records']}")
        logger.info(f"  Processed:      {summary['groups_processed']}")
        logger.info(f"  Skipped:        {summary['groups_skipped']}")
        logger.info(f"  Failed:         {summary['groups_failed']}")
        if summary.get('output_dir'):
            logger.info(f"  Output:         {summary['output_dir']}")
        logger.info("=" * 70)
