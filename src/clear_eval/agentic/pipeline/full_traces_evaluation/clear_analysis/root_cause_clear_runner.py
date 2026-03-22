#!/usr/bin/env python3
"""
Root Cause CLEAR Runner
========================

Runs CLEAR aggregation on failure_root_cause from task success evaluation results.

Extracts root causes from failed trajectories in task_success_evaluation
results and analyzes them to discover common failure patterns.
"""

import json
import logging
from pathlib import Path

from clear_eval.agentic.pipeline.full_traces_evaluation.clear_analysis.base_clear_runner import BaseClearRunner

logger = logging.getLogger(__name__)


class RootCauseClearRunner(BaseClearRunner):
    """
    CLEAR runner for analyzing root causes from task success evaluations.
    
    Extracts failure_root_cause from task_success_evaluation results
    (only failed trajectories) and runs CLEAR analysis to identify
    common failure patterns.
    """

    def get_source_name(self) -> str:
        """Return source identifier."""
        return "root_cause"

    def get_evaluation_type_dir(self) -> str:
        """Return evaluation type directory name."""
        return "task_success"

    def get_result_file_suffix(self) -> str:
        """Return result file suffix."""
        return "_success.json"

    def extract_records_from_file(self, file_path: Path) -> list[dict]:
        """
        Extract root cause records from a task success evaluation result file.
        
        Only extracts from failed trajectories (success == 0).
        
        Args:
            file_path: Path to _success.json file
            
        Returns:
            List of record dicts with extracted root causes (empty if success == 1)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

        # Only process failed trajectories
        success = data.get("success")
        if success != 0:
            return []

        # Extract failure_root_cause
        root_cause = data.get("failure_root_cause")
        if not root_cause or not isinstance(root_cause, str) or not root_cause.strip():
            return []

        # Extract metadata
        traj_name = data.get("trajectory_name", file_path.stem.replace("_success", ""))
        task_objective = data.get("task_objective", "")
        consideration = data.get("consideration", "")

        # Build feedback text
        feedback = root_cause
        if consideration:
            feedback = (
                f"Consideration: {consideration}\n\n"
                f"Root Cause: {root_cause}"
            )

        # Build context
        model_input = f"Trajectory: {traj_name}"
        if task_objective:
            model_input = f"Task: {task_objective}\n{model_input}"

        record = {
            "id": traj_name,
            "trajectory_name": traj_name,
            "task_objective": task_objective,
            "consideration": consideration,
            "model_input": model_input,
            "response": "",
            "evaluation_summary": feedback,
            "score": 0,  # Failed trajectory
        }

        return [record]

    def get_input_columns(self) -> list[str]:
        """Return columns to include in CLEAR input."""
        return [
            "id",
            "trajectory_name",
            "task_objective",
            "consideration",
        ]
