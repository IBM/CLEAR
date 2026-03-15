#!/usr/bin/env python3
"""
Base Evaluator for Trajectory Evaluation
=========================================

Provides centralized common logic for trajectory evaluation with pluggable
evaluation-specific components via abstract methods.

Design Pattern:
    - Common logic (file I/O, LLM calls, parallel execution) in base class
    - Evaluation-specific logic (prompts, parsing, results) in subclasses
    - Hook methods for customization at key points
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import logging_config
from clear_eval.agentic.full_traj_evaluation.full_traj_utils import (
    get_max_trajectory_chars,
)
from clear_eval.pipeline.llm_client import get_llm_client, run_parallel
from logging_config import setup_logging

setup_logging()


# Trajectory capping utilities
def middle_out(text: str, limit: int) -> str:
    """Truncate text keeping beginning and end, removing middle."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n... [TRUNCATED] ...\n\n{text[-half:]}"


def cap_trajectory(trajectory_text: str, max_len: int) -> str:
    """
    Truncate trajectory text if it exceeds max_len characters.
    
    Uses middle-out truncation to preserve both beginning and end of trajectory.
    
    Args:
        trajectory_text: Full trajectory text
        max_len: Maximum allowed length (from get_max_trajectory_chars)
    
    Returns:
        Truncated trajectory text if needed, otherwise original text
    """
    if len(trajectory_text) > max_len:
        logger.info(
            "Trajectory too long (%d chars), truncating to %d chars",
            len(trajectory_text), max_len
        )
        return middle_out(trajectory_text, max_len)
    return trajectory_text


class TrajectoryEvaluator(ABC):
    """
    Abstract base class for trajectory evaluation.
    
    Centralizes common logic:
        - File loading and saving
        - Trajectory formatting
        - LLM client management
        - Parallel execution
        - Error handling and logging
    
    Subclasses implement evaluation-specific logic via abstract methods:
        - prepare_evaluation_data: Load/prepare evaluation-specific data
        - prepare_context: Build context dict for prompt
        - build_prompt: Generate evaluation prompt
        - get_system_message: Return system message
        - parse_response: Parse LLM response
        - extract_results: Extract evaluation-specific results
        - get_output_suffix: Return output filename suffix
    """

    def __init__(
        self,
        judge_model_id: str,
        provider: str,
        traj_input_dir: Path,
        output_dir: Path,
        context_tokens: int = 128_000,
        overwrite: bool = False,
        concurrency: int = 7,
        eval_model_params: dict | None = None,
        max_files: int | None = None,
    ):
        """
        Initialize evaluator with common configuration.
        
        Args:
            judge_model_id: Model identifier for the judge LLM
            provider: LLM provider (e.g., 'rits', 'openai')
            traj_input_dir: Directory containing trajectory JSON files
            output_dir: Base directory for saving evaluation results
            context_tokens: Context window size for the judge model
            overwrite: Whether to overwrite existing evaluation results
            concurrency: Number of parallel workers
            eval_model_params: Additional parameters for LLM client
            max_files: Maximum number of files to process (for testing)
        """
        self.judge_model_id = judge_model_id
        self.provider = provider
        self.traj_input_dir = Path(traj_input_dir)
        self.output_dir = Path(output_dir)
        self.context_tokens = context_tokens
        self.overwrite = overwrite
        self.concurrency = concurrency
        self.eval_model_params = eval_model_params or {}
        self.max_files = max_files
        
        # Create results directory: output_dir/evaluation_type
        eval_type = self.get_evaluation_type().replace(" ", "_").replace("/", "_")
        self.results_dir = self.output_dir / eval_type
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_evaluation_type(self) -> str:
        """
        Return human-readable evaluation type name for display.
        
        Returns:
            Evaluation type string (e.g., "Task Success Evaluation", "Rubric Evaluation")
        """
        pass

    @abstractmethod
    def get_extra_info(self) -> dict:
        """
        Return evaluation-specific extra information for display in plan.
        
        Returns:
            Dict with extra info to display (e.g., {"Decision": "Binary (0/1)"})
        """
        pass

    def print_evaluation_plan(
        self,
        entries: list[dict],
        data_dir: Path,
        overwrite: bool,
        concurrency: int,
        extra_info: dict | None = None,
    ):
        """
        Print evaluation plan with configuration details.
        
        Args:
            entries: List of trajectory entries to evaluate
            data_dir: Data directory path
            overwrite: Overwrite setting
            concurrency: Concurrency level
            extra_info: Optional dict with additional info to display
        """
        max_traj_chars = self.context_tokens * 4 * 0.9  # Approximate
        
        logger.info("=" * 70)
        logger.info(self.get_evaluation_type())
        logger.info("=" * 70)
        logger.info(f"Provider:     {self.provider}")
        logger.info(f"Judge Model:  {self.judge_model_id}")
        logger.info(f"Context Win:  {self.context_tokens:,} tokens → "
                   f"max trajectory ~{int(max_traj_chars):,} chars")
        logger.info(f"Data Dir:     {data_dir}")
        logger.info(f"Results Dir:  {self.results_dir}")
        
        # Log extra info if provided
        if extra_info:
            for key, value in extra_info.items():
                logger.info(f"{key}:  {value}")
        
        logger.info(f"Overwrite:    {overwrite}")
        logger.info(f"Concurrency:  {concurrency}")
        logger.info(f"Total files:  {len(entries)}")
        logger.info("=" * 70)


    # -------------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def prepare_evaluation_data(
        self, entry: dict, traj_data: dict
    ) -> dict | None:
        """
        Prepare evaluation-specific data before building prompt.
        
        This hook allows subclasses to:
        - Load additional files (e.g., rubrics)
        - Extract specific information from trajectory
        - Perform preprocessing
        
        Args:
            entry: Entry dict with file_path, traj_name
            traj_data: Loaded trajectory data (JSON)
        
        Returns:
            Dict with evaluation-specific data to be used in prepare_context()
            Return None to skip this evaluation (e.g., missing rubrics)
        
        Example returns:
            - Task success: {"task_objective": "..."}
            - Rubric eval: {"rubrics": [...], "task_objective": "..."}
            - Full traj: {} (no additional data needed)
        """
        pass

    @abstractmethod
    def prepare_context(self, trajectory_text: str, eval_data: dict) -> dict:
        """
        Prepare context dictionary for prompt building.
        
        Args:
            trajectory_text: Formatted trajectory text
            eval_data: Data returned from prepare_evaluation_data()
        
        Returns:
            Dict with all data needed for build_prompt()
        
        Example:
            {
                "trajectory_text": trajectory_text,
                "task_objective": eval_data["task_objective"],
                "max_len": get_max_trajectory_chars(self.context_tokens)
            }
        """
        pass

    @abstractmethod
    def build_prompt(self, context: dict) -> str:
        """
        Build evaluation prompt from context.
        
        Args:
            context: Context dict from prepare_context()
        
        Returns:
            Prompt string to send to LLM
        """
        pass

    @abstractmethod
    def get_system_message(self) -> str:
        """
        Return system message for this evaluation type.
        
        Returns:
            System message string
        """
        pass

    @abstractmethod
    def parse_response(self, response_text: str) -> dict | None:
        """
        Parse LLM response into structured data.
        
        Args:
            response_text: Raw response from LLM
        
        Returns:
            Parsed evaluation dict, or None if parsing failed
        """
        pass

    @abstractmethod
    def extract_results(self, evaluation: dict, eval_data: dict) -> dict:
        """
        Extract evaluation-specific results from parsed response.
        
        This dict will be merged with common fields (trajectory_name, dataset,
        model_name, source_file, judge_model, timestamp, raw_response, etc.)
        
        Args:
            evaluation: Parsed evaluation from parse_response()
            eval_data: Data from prepare_evaluation_data()
        
        Returns:
            Dict with evaluation-specific result fields
        
        Example returns:
            - Task success: {"success": 1, "consideration": "...", ...}
            - Rubric eval: {"score": 0.8, "rubric_results": {...}, ...}
            - Full traj: {"overall_score": 0.75, "dimension_scores": {...}, ...}
        """
        pass

    @abstractmethod
    def get_output_suffix(self) -> str:
        """
        Return output filename suffix for this evaluation type.
        
        Returns:
            Suffix string (e.g., '_eval.json', '_rubric_eval.json', '_success.json')
        """
        pass

    # -------------------------------------------------------------------------
    # Common logic - implemented in base class
    # -------------------------------------------------------------------------

    def evaluate_single(
        self,
        entry: dict,
        llm_client: Any,
        overwrite: bool | None = None,
    ) -> dict | None:
        """
        Evaluate a single trajectory - handles all common logic.
        
        Workflow:
            1. Check if output exists (skip if not overwriting)
            2. Load trajectory file
            3. Format trajectory via dataset_obj
            4. Call prepare_evaluation_data() hook
            5. Call prepare_context() hook
            6. Call build_prompt() hook
            7. Call LLM with get_system_message()
            8. Call parse_response() hook
            9. Call extract_results() hook
            10. Build final result dict (common + specific fields)
            11. Save to file
            12. Return result
        
        Args:
            entry: Entry dict with file_path, traj_name
            llm_client: LLM client instance
            overwrite: Override instance overwrite setting (optional)
        
        Returns:
            Result dict if successful, None if skipped or failed
        """
        file_path = Path(entry["file_path"])
        traj_name = entry["traj_name"]

        # Use provided overwrite or fall back to instance setting
        should_overwrite = overwrite if overwrite is not None else self.overwrite

        # Determine output path
        output_file = self.results_dir / f"{traj_name}{self.get_output_suffix()}"

        # Skip if exists and not overwriting
        if output_file.exists() and not should_overwrite:
            logger.info("Skipping (exists): %s", traj_name)
            return None

        # Load trajectory file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                traj_data = json.load(f)
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)
            return None

        # Format trajectory text (simple JSON string conversion)
        try:
            trajectory_text = json.dumps(traj_data, indent=2)
            # Cap trajectory to fit in context window
            max_chars = get_max_trajectory_chars(self.context_tokens)
            trajectory_text = cap_trajectory(trajectory_text, max_chars)
        except Exception as e:
            logger.error("Failed to format trajectory %s: %s", traj_name, e)
            return None

        # Prepare evaluation-specific data
        try:
            eval_data = self.prepare_evaluation_data(entry, traj_data)
            if eval_data is None:
                # Subclass returned None to skip this evaluation
                logger.info("Skipping %s (prepare_evaluation_data returned None)", traj_name)
                return None
        except Exception as e:
            logger.error("Failed to prepare evaluation data for %s: %s", traj_name, e)
            return None

        # Prepare context for prompt building
        try:
            context = self.prepare_context(trajectory_text, eval_data)
        except Exception as e:
            logger.error("Failed to prepare context for %s: %s", traj_name, e)
            return None

        # Build prompt
        try:
            prompt = self.build_prompt(context)
            system_message = self.get_system_message()
        except Exception as e:
            logger.error("Failed to build prompt for %s: %s", traj_name, e)
            return None

        # Call LLM
        start_time = time.time()
        try:
            response_text = llm_client.call(
                prompt=prompt,
                system_message=system_message
            )
        except Exception as e:
            logger.error("LLM call failed for %s: %s", traj_name, e)
            return None
        elapsed = time.time() - start_time

        if not response_text:
            logger.error("Empty response for %s", traj_name)
            return None

        # Parse response
        try:
            evaluation = self.parse_response(response_text)
        except Exception as e:
            logger.error("Failed to parse response for %s: %s", traj_name, e)
            evaluation = None

        # Extract evaluation-specific results
        try:
            specific_results = self.extract_results(evaluation or {}, eval_data)
        except Exception as e:
            logger.error("Failed to extract results for %s: %s", traj_name, e)
            specific_results = {}

        # Build final result dict (common fields + specific fields)
        result = {
            # Common fields
            "trajectory_name": traj_name,
            "source_file": str(file_path),
            "judge_model": self.judge_model_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_time_seconds": round(elapsed, 2),
            "raw_response": response_text,
            "parsed_evaluation": evaluation,
            # Merge evaluation-specific fields
            **specific_results,
        }

        # Save result to file
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save result for %s: %s", traj_name, e)
            return None

        return result

    def run_batch(
        self,
        entries: list[dict],
        concurrency: int = 2,
        eval_model_params: dict | None = None,
    ) -> list:
        """
        Run batch evaluation using pipeline's run_parallel.
        
        Args:
            entries: List of entry dicts (from discover_trajectories)
            concurrency: Number of parallel workers
            eval_model_params: Additional parameters for LLM client
        
        Returns:
            List of ParallelResult objects with evaluation results and status
        """
        # Get LLM client once (will be reused for all evaluations)
        llm_client = get_llm_client(
            provider=self.provider,
            model=self.judge_model_id,
            use_litellm=True,
            eval_mode=True,
            parameters=eval_model_params or {},
        )

        # Prepare inputs as tuples for each entry
        inputs = [
            (entry, llm_client, self.overwrite)
            for entry in entries
        ]

        # Use pipeline's parallel execution with progress bar
        start = time.time()
        parallel_results = run_parallel(
            func=self.evaluate_single,
            inputs=inputs,
            use_async=True,
            max_workers=concurrency,
            progress_desc=f"Evaluating trajectories ({self.__class__.__name__})"
        )
        elapsed = time.time() - start

        # Count successful evaluations
        successful = sum(1 for pr in parallel_results if pr.is_success and pr.result is not None)
        total = len(parallel_results)

        logger.info(f"Evaluation complete: {successful}/{total} trajectories succeeded in {elapsed:.1f}s")

    # TODO: Re-enable summary functionality when needed
    # def save_summary(self, print_func=None) -> dict:
    #     """
    #     Generate, save, and optionally print evaluation summary.
    #
    #     This method:
    #     1. Calls generate_summary() to aggregate results
    #     2. Saves summary to summary.json in results directory
    #     3. Optionally prints summary using provided print function
    #
    #     Args:
    #         print_func: Optional function to print summary (receives summary dict and judge_model_id)
    #
    #     Returns:
    #         Summary dict with aggregated statistics
    #     """
    #     if not self.results_dir.exists():
    #         logger.warning("Results directory does not exist: %s", self.results_dir)
    #         return {}
    #
    #     # Generate summary (calls subclass implementation)
    #     summary = self.generate_summary()
    #
    #     # Save to file
    #     summary_file = self.results_dir / "summary.json"
    #     with open(summary_file, "w") as f:
    #         json.dump(summary, f, indent=2)
    #     print(f"\nSummary saved to: {summary_file}")
    #
    #     # Print if function provided
    #     if print_func:
    #         print_func(summary, self.judge_model_id)
    #
    #     return summary

    # @abstractmethod
    # def generate_summary(self) -> dict:
    #     """
    #     Generate summary statistics from evaluation results.
    #
    #     Subclasses should scan the results directory and aggregate statistics
    #     specific to their evaluation type.
    #
    #     Returns:
    #         Dict with summary statistics (structure varies by evaluator type)
    #     """
    #     pass


    @staticmethod
    def discover_trajectories(traj_input_dir: Path) -> list[dict]:
        """
        Discover all trajectory JSON files in a directory.
        
        Args:
            traj_input_dir: Directory containing trajectory JSON files
        
        Returns:
            List of dicts with keys: file_path, traj_name
        """
        traj_input_dir = Path(traj_input_dir)
        entries = []
        
        for json_file in traj_input_dir.glob("*.json"):
            entries.append({
                "file_path": str(json_file),
                "traj_name": json_file.stem,
            })
        
        logger.info("Discovered %d trajectory files in %s", len(entries), traj_input_dir)
        return entries

    def run_pipeline(self):
        """Run the full evaluation pipeline."""
        # Discover trajectories
        entries = self.discover_trajectories(self.traj_input_dir)

        if not entries:
            logger.warning(f"No trajectory files found in {self.traj_input_dir}")
            return

        # Apply max-files limit
        if self.max_files:
            entries = entries[:self.max_files]

        # Print evaluation plan (uses get_extra_info() from subclass)
        self.print_evaluation_plan(
            entries=entries,
            data_dir=self.traj_input_dir,
            overwrite=self.overwrite,
            concurrency=self.concurrency,
            extra_info=self.get_extra_info(),
        )

        # Run evaluation
        parallel_results = self.run_batch(
            entries=entries,
            concurrency=self.concurrency,
            eval_model_params=self.eval_model_params,
        )

        # TODO: Uncomment when summary is ready
        # self.save_summary()
        
        return parallel_results
