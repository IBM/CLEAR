#!/usr/bin/env python3
"""
Issues CLEAR Runner
===================

Runs CLEAR aggregation on detailed_feedback (issues) from full trajectory
evaluation results.

Extracts issues from full_trajectory results and analyzes them
to discover common themes and patterns.
"""

import json
import logging
from pathlib import Path

from agentic.pipeline.full_traces_evaluation.clear_analysis.base_clear_runner import BaseClearRunner

logger = logging.getLogger(__name__)


class IssuesClearRunner(BaseClearRunner):
    """
    CLEAR runner for analyzing issues from full trajectory evaluations.
    
    Extracts detailed_feedback from full_trajectory_evaluation results
    and runs CLEAR analysis to identify common shortcomings and patterns.
    """

    def get_source_name(self) -> str:
        """Return source identifier."""
        return "issues"

    def get_evaluation_type_dir(self) -> str:
        """Return evaluation type directory name."""
        return "full_trajectory"

    def get_result_file_suffix(self) -> str:
        """Return result file suffix."""
        return "_eval.json"

    def extract_records_from_file(self, file_path: Path) -> list[dict]:
        """
        Extract issue records from a full trajectory evaluation result file.
        
        Args:
            file_path: Path to _eval.json file
            
        Returns:
            List of record dicts with extracted issues
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

        # Extract detailed_feedback (issues)
        feedback = data.get("detailed_feedback")
        if not feedback or not isinstance(feedback, str) or not feedback.strip():
            return []

        # Extract metadata
        traj_name = data.get("trajectory_name", file_path.stem.replace("_eval", ""))
        overall_score = data.get("overall_score")

        # Build context
        model_input = f"Trajectory: {traj_name}"

        record = {
            "id": traj_name,
            "trajectory_name": traj_name,
            "overall_score": overall_score,
            "model_input": model_input,
            "response": "",
            "evaluation_summary": feedback,
            "score": overall_score,
        }

        return [record]

    def get_input_columns(self) -> list[str]:
        """Return columns to include in CLEAR input."""
        return [
            "id",
            "trajectory_name",
            "overall_score",
        ]
