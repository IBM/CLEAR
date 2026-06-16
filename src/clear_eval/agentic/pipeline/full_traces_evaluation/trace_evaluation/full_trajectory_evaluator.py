#!/usr/bin/env python3
"""
Full Trajectory Evaluator
==========================

Concrete evaluator for comprehensive trajectory evaluation using CLEAR framework.

Default mode: Evaluates agent trajectories across 14 dimensions:
- 9 step-level quality dimensions
- 5 trajectory-level holistic dimensions

Custom mode: When full_trace_evaluation_criteria is provided, evaluates
trajectories against a user-defined flat set of dimensions.

Produces:
- Individual dimension scores (0.0-1.0)
- Detailed feedback paragraph
- Overall score (0.0-1.0)
"""

import json
import logging
import re

from clear_eval.agentic.pipeline.full_traces_evaluation.trace_evaluation.base_evaluator import TrajectoryEvaluator
from clear_eval.agentic.pipeline.full_traces_evaluation.trace_evaluation.full_trajectory_prompts import (
    STEP_QUALITY_CRITERIA,
    TRAJECTORY_CRITERIA,
    ALL_CRITERIA,
    SCORING_SCALE,
    build_default_prompt,
    build_custom_criteria_prompt,
)

logger = logging.getLogger(__name__)


# System message for full trajectory evaluation
SYSTEM_MESSAGE_FULL_TRAJ = f"""\
You are an expert AI agent evaluator following the CLEAR evaluation framework. \
You evaluate full agent execution trajectories — the complete sequence of steps \
an AI agent took to accomplish a task, including planning, tool calls, \
observations, and final answers.

Your evaluation must be thorough, fair, and grounded solely in the trajectory \
content. Do NOT let any external metadata influence your judgment.

{SCORING_SCALE}

For each dimension you must provide:
  1. A score (0.0 to 1.0, rounded to 2 decimal places)
  2. A brief justification (1-3 sentences)

Then provide:
  - detailed_feedback: a 4-8 sentence paragraph covering strengths, weaknesses, \
and recommendations. Write this BEFORE deciding the final score so it acts as \
your chain-of-thought reasoning.
  - overall_score: a single 0.0-1.0 number reflecting the weighted average of \
all dimension scores.

IMPORTANT: Return ONLY valid JSON matching the schema in the user prompt. \
No text outside the JSON."""


class FullTrajectoryEvaluator(TrajectoryEvaluator):
    """
    Evaluator for comprehensive trajectory evaluation using CLEAR framework.

    Output fields (default mode):
        - step_quality_dimensions: Dict of dimension scores with justifications
        - trajectory_dimensions: Dict of dimension scores with justifications
        - detailed_feedback: 4-8 sentence analysis paragraph
        - overall_score: Float 0.0-1.0
        - dimension_scores: Flat dict of all dimension scores (for convenience)

    Output fields (custom criteria mode):
        - dimensions: Flat dict of dimension scores with justifications
        - detailed_feedback: 4-8 sentence analysis paragraph
        - overall_score: Float 0.0-1.0
        - dimension_scores: Flat dict of all dimension scores (for convenience)
    """

    @property
    def uses_custom_criteria(self) -> bool:
        """Return True if custom flat criteria are configured."""
        return self.full_trace_evaluation_criteria is not None

    def prepare_evaluation_data(
        self, entry: dict, intent: str
    ) -> dict:
        """
        No additional data needed for full trajectory evaluation.
        
        Args:
            entry: Entry dict with file_path, traj_name
            intent: Task intent/objective (not used in full trajectory eval)
        
        Returns:
            Empty dict (no extra data needed)
        """
        # Full trajectory evaluation doesn't need extra data extraction
        return {}

    def prepare_context(self, trajectory_text: str, eval_data: dict) -> dict:
        """
        Prepare context for full trajectory prompt.
        
        Args:
            trajectory_text: Formatted and capped trajectory
            eval_data: Empty dict (not used)
        
        Returns:
            Context dict with trajectory_text only
        """
        return {"trajectory_text": trajectory_text}

    def build_prompt(self, context: dict) -> str:
        """
        Build full trajectory evaluation prompt.

        Uses custom flat criteria if full_trace_evaluation_criteria is set,
        otherwise uses default nested step/trajectory structure.

        Args:
            context: Dict with trajectory_text (already capped)

        Returns:
            Formatted prompt string
        """
        trajectory_text = context["trajectory_text"]

        if self.uses_custom_criteria:
            return build_custom_criteria_prompt(
                trajectory_text, self.full_trace_evaluation_criteria
            )
        else:
            return build_default_prompt(trajectory_text)

    def get_system_message(self) -> str:
        """Return system message for full trajectory evaluation."""
        return SYSTEM_MESSAGE_FULL_TRAJ

    def parse_response(self, response_text: str) -> dict | None:
        """
        Parse JSON response from LLM.
        
        Tries multiple parsing strategies:
        1. Direct JSON parse
        2. Extract from ```json code block
        3. Find first {...} block
        
        Args:
            response_text: Raw LLM response
        
        Returns:
            Parsed dict or None if parsing failed
        """
        if not response_text:
            return None

        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        json_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first {...} block
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse evaluation response as JSON")
        return None

    def extract_results(self, evaluation: dict | None, eval_data: dict) -> dict:
        """
        Extract full trajectory evaluation results from parsed evaluation.

        Handles both default (nested step/trajectory) and custom (flat dimensions) formats.

        Args:
            evaluation: Parsed evaluation dict (or None if parsing failed)
            eval_data: Empty dict (not used)

        Returns:
            Dict with dimension scores, detailed_feedback, overall_score
        """
        step_quality_dimensions = {}
        trajectory_dimensions = {}
        dimensions = {}
        detailed_feedback = None
        overall_score = None
        dimension_scores = {}

        if evaluation:
            # Extract detailed feedback
            detailed_feedback = evaluation.get("detailed_feedback")

            # Extract overall score
            raw_overall = evaluation.get("overall_score")
            try:
                overall_score = float(raw_overall)
                if overall_score < 0.0:
                    overall_score = 0.0
                elif overall_score > 1.0:
                    overall_score = 1.0
            except (ValueError, TypeError):
                logger.warning("Could not parse overall_score: %s", raw_overall)

            if self.uses_custom_criteria:
                # Custom flat format: {"dimensions": {name: {score, justification}}}
                dims = evaluation.get("dimensions", {})
                if isinstance(dims, dict):
                    for dim_name, dim_data in dims.items():
                        if isinstance(dim_data, dict):
                            dimensions[dim_name] = dim_data
                            if "score" in dim_data:
                                try:
                                    dimension_scores[dim_name] = float(dim_data["score"])
                                except (ValueError, TypeError):
                                    pass
            else:
                # Default nested format
                step_quality = evaluation.get("step_quality_dimensions", {})
                if isinstance(step_quality, dict):
                    for dim_name, dim_data in step_quality.items():
                        if isinstance(dim_data, dict):
                            step_quality_dimensions[dim_name] = dim_data
                            if "score" in dim_data:
                                try:
                                    dimension_scores[dim_name] = float(dim_data["score"])
                                except (ValueError, TypeError):
                                    pass

                traj_dims = evaluation.get("trajectory_dimensions", {})
                if isinstance(traj_dims, dict):
                    for dim_name, dim_data in traj_dims.items():
                        if isinstance(dim_data, dict):
                            trajectory_dimensions[dim_name] = dim_data
                            if "score" in dim_data:
                                try:
                                    dimension_scores[dim_name] = float(dim_data["score"])
                                except (ValueError, TypeError):
                                    pass

            # Fallback: compute overall from dimension average if not provided
            if overall_score is None and dimension_scores:
                overall_score = round(
                    sum(dimension_scores.values()) / len(dimension_scores), 2
                )

        result = {
            "step_quality_dimensions": step_quality_dimensions,
            "trajectory_dimensions": trajectory_dimensions,
            "detailed_feedback": detailed_feedback,
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
        }
        if self.uses_custom_criteria:
            result["dimensions"] = dimensions
        return result

    def get_evaluation_type(self) -> str:
        """Return evaluation type name for display."""
        return "full_trajectory"

    def get_extra_info(self) -> dict:
        """Return extra information for display in evaluation plan."""
        if self.uses_custom_criteria:
            return {
                "Scoring": "0.0–1.0 (CLEAR-style)",
                "Dimensions": f"{len(self.full_trace_evaluation_criteria)} custom",
            }
        return {
            "Scoring": "0.0–1.0 (CLEAR-style)",
            "Dimensions": f"{len(STEP_QUALITY_CRITERIA)} step-quality + "
                         f"{len(TRAJECTORY_CRITERIA)} trajectory = "
                         f"{len(ALL_CRITERIA)} total",
        }

    def get_output_suffix(self) -> str:
        """Return output filename suffix for full trajectory evaluation."""
        return "_eval.json"

    def generate_summary(self) -> dict:
        """
        Generate a summary report from full trajectory evaluation results.

        Scans the output directory and aggregates statistics.
        Handles both default (nested step/trajectory) and custom (flat) output formats.

        Returns:
            Dict with aggregated statistics for the current run
        """
        from collections import defaultdict

        eval_files = list(self.results_dir.glob(f"*{self.get_output_suffix()}"))

        total_evaluations = 0
        overall_scores = []
        step_quality_scores = defaultdict(list)
        trajectory_scores = defaultdict(list)
        custom_dimension_scores = defaultdict(list)

        for ef in eval_files:
            try:
                with open(ef, "r") as f:
                    data = json.load(f)

                total_evaluations += 1

                # Collect overall score
                overall_score = data.get("overall_score")
                if isinstance(overall_score, (int, float)):
                    overall_scores.append(float(overall_score))

                # Collect dimension scores - handle both flat and nested formats
                dimension_scores = data.get("dimension_scores", {})
                for dim_name, score in dimension_scores.items():
                    if isinstance(score, (int, float)):
                        if self.uses_custom_criteria:
                            custom_dimension_scores[dim_name].append(float(score))
                        elif dim_name in STEP_QUALITY_CRITERIA:
                            step_quality_scores[dim_name].append(float(score))
                        elif dim_name in TRAJECTORY_CRITERIA:
                            trajectory_scores[dim_name].append(float(score))

            except Exception as e:
                logger.warning("Failed to read %s: %s", ef, e)
                continue

        # Calculate averages
        overall_score_average = (
            round(sum(overall_scores) / len(overall_scores), 3)
            if overall_scores else None
        )

        if self.uses_custom_criteria:
            dimension_averages = {
                dim: round(sum(scores) / len(scores), 3)
                for dim, scores in custom_dimension_scores.items()
                if scores
            }
            summary = {
                "total_evaluations": total_evaluations,
                "overall_score_average": overall_score_average,
                "dimension_averages": dimension_averages,
            }
        else:
            step_quality_averages = {
                dim: round(sum(scores) / len(scores), 3)
                for dim, scores in step_quality_scores.items()
                if scores
            }
            trajectory_averages = {
                dim: round(sum(scores) / len(scores), 3)
                for dim, scores in trajectory_scores.items()
                if scores
            }
            summary = {
                "total_evaluations": total_evaluations,
                "overall_score_average": overall_score_average,
                "step_quality_averages": step_quality_averages,
                "trajectory_averages": trajectory_averages,
            }

        return summary
