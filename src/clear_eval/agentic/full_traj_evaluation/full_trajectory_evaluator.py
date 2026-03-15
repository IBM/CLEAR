#!/usr/bin/env python3
"""
Full Trajectory Evaluator
==========================

Concrete evaluator for comprehensive trajectory evaluation using CLEAR framework.

Evaluates agent trajectories across 14 dimensions:
- 9 step-level quality dimensions
- 5 trajectory-level holistic dimensions

Produces:
- Individual dimension scores (0.0-1.0)
- Detailed feedback paragraph
- Overall score (0.0-1.0)
"""

import json
import logging
import re
from typing import Any

from clear_eval.agentic.full_traj_evaluation.base_evaluator import TrajectoryEvaluator

logger = logging.getLogger(__name__)


# Step-level quality dimensions (from CLEAR)
STEP_QUALITY_CRITERIA = {
    "Correctness": (
        "The responses and actions produce accurate, logically sound results "
        "for the given task or query."
    ),
    "Completeness": (
        "The responses fully address the user's request. If a response appears "
        "incomplete but is followed by a tool call or action, this is acceptable."
    ),
    "Clarity": (
        "Explanations, reasoning, and any generated code or actions are easy "
        "to read, well-structured, and unambiguous."
    ),
    "Relevance": (
        "Responses stay focused on the task at hand without unnecessary or "
        "off-topic content."
    ),
    "Efficiency": (
        "The solution or action plan is optimized for performance, avoiding "
        "unnecessary complexity or redundant steps."
    ),
    "Robustness": (
        "The solution handles edge cases, potential errors, and unexpected "
        "inputs gracefully."
    ),
    "Best_Practices": (
        "The solution follows accepted conventions, style guidelines, and "
        "maintainable coding / reasoning standards."
    ),
    "Actionability": (
        "Responses provide directly usable steps, code, or API calls without "
        "requiring significant rework."
    ),
    "Transparency": (
        "Reasoning, assumptions, decisions, and intermediate steps are clearly "
        "explained and justified."
    ),
}

# Trajectory-level (holistic) dimensions
TRAJECTORY_CRITERIA = {
    "Objective_Understanding": (
        "How well the agent understood the user's high-level goal from the "
        "start and maintained alignment throughout the trajectory."
    ),
    "Information_Completeness": (
        "Whether the agent gathered all necessary information (via tools, "
        "queries, observations) before acting, and did not leave critical "
        "gaps in its knowledge."
    ),
    "Execution_Quality": (
        "The overall quality of the agent's execution plan — were the right "
        "tools chosen, called in the right order, with correct parameters, "
        "and did the agent recover from errors effectively?"
    ),
    "User_Experience": (
        "How well the trajectory would serve the end-user: clear "
        "communication, appropriate level of detail, no confusing detours, "
        "and timely progress updates."
    ),
    "Final_Deliverable": (
        "The quality and correctness of the agent's final output or answer "
        "relative to the original objective."
    ),
}

# All dimensions combined
ALL_CRITERIA = {**STEP_QUALITY_CRITERIA, **TRAJECTORY_CRITERIA}

# Scoring scale anchors
SCORING_SCALE = """\
Scoring uses a continuous 0 – 1 scale with the following anchors:
  0.00 = completely failed / absent
  0.25 = poor quality, major issues
  0.50 = acceptable but with notable gaps
  0.75 = good quality, minor issues only
  1.00 = excellent, no meaningful issues"""


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
    
    Output fields:
        - step_quality_dimensions: Dict of 9 dimension scores with justifications
        - trajectory_dimensions: Dict of 5 dimension scores with justifications
        - detailed_feedback: 4-8 sentence analysis paragraph
        - overall_score: Float 0.0-1.0
        - dimension_scores: Flat dict of all dimension scores (for convenience)
    """

    def prepare_evaluation_data(
        self, entry: dict, traj_data: dict
    ) -> dict:
        """
        No additional data needed for full trajectory evaluation.
        
        Args:
            entry: Entry dict with file_path, traj_name
            traj_data: Loaded trajectory data
        
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
        Build full trajectory evaluation prompt with all 14 dimensions.
        
        Args:
            context: Dict with trajectory_text (already capped)
        
        Returns:
            Formatted prompt string
        """
        trajectory_text = context["trajectory_text"]

        # Build dimension descriptions
        step_block = "\n".join(
            f"  - **{name}**: {desc}"
            for name, desc in STEP_QUALITY_CRITERIA.items()
        )
        traj_block = "\n".join(
            f"  - **{name}**: {desc}"
            for name, desc in TRAJECTORY_CRITERIA.items()
        )

        # Build JSON schema for dimensions
        step_dims_json = ",\n".join(
            f'    "{dim}": {{"score": "<0.0-1.0>", "justification": "<text>"}}'
            for dim in STEP_QUALITY_CRITERIA
        )
        traj_dims_json = ",\n".join(
            f'    "{dim}": {{"score": "<0.0-1.0>", "justification": "<text>"}}'
            for dim in TRAJECTORY_CRITERIA
        )

        prompt = f"""\
## Trajectory Evaluation Task (CLEAR Framework — Dimension Scoring)

### Scoring Scale

{SCORING_SCALE}

### Step-Level Quality Dimensions (score each 0.0 – 1.0)

{step_block}

### Trajectory-Level Holistic Dimensions (score each 0.0 – 1.0)

{traj_block}

### Full Agent Trajectory

```
{trajectory_text}
```

### Instructions

1. Carefully read the full trajectory above.
2. Score each of the 14 dimensions on a 0.0–1.0 scale with a brief justification.
3. Write your **detailed_feedback** first (4-8 sentences) — this is your \
chain-of-thought reasoning about strengths, weaknesses, and suggestions.
4. Then decide the **overall_score** (0.0–1.0) for the entire trajectory.

### Required Output (valid JSON only, no extra text)

```json
{{
  "step_quality_dimensions": {{
{step_dims_json}
  }},
  "trajectory_dimensions": {{
{traj_dims_json}
  }},
  "detailed_feedback": "<4-8 sentence paragraph>",
  "overall_score": "<0.0-1.0>"
}}
```
"""
        return prompt

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
        
        Args:
            evaluation: Parsed evaluation dict (or None if parsing failed)
            eval_data: Empty dict (not used)
        
        Returns:
            Dict with dimension scores, detailed_feedback, overall_score
        """
        # Initialize result fields
        step_quality_dimensions = {}
        trajectory_dimensions = {}
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
                # Clamp to 0.0-1.0
                if overall_score < 0.0:
                    overall_score = 0.0
                elif overall_score > 1.0:
                    overall_score = 1.0
            except (ValueError, TypeError):
                logger.warning("Could not parse overall_score: %s", raw_overall)

            # Extract step quality dimensions
            step_quality = evaluation.get("step_quality_dimensions", {})
            if isinstance(step_quality, dict):
                for dim_name, dim_data in step_quality.items():
                    if isinstance(dim_data, dict):
                        step_quality_dimensions[dim_name] = dim_data
                        # Also extract score for flat dict
                        if "score" in dim_data:
                            try:
                                dimension_scores[dim_name] = float(dim_data["score"])
                            except (ValueError, TypeError):
                                pass

            # Extract trajectory dimensions
            traj_dims = evaluation.get("trajectory_dimensions", {})
            if isinstance(traj_dims, dict):
                for dim_name, dim_data in traj_dims.items():
                    if isinstance(dim_data, dict):
                        trajectory_dimensions[dim_name] = dim_data
                        # Also extract score for flat dict
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

        return {
            "step_quality_dimensions": step_quality_dimensions,
            "trajectory_dimensions": trajectory_dimensions,
            "detailed_feedback": detailed_feedback,
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,  # Flat dict for convenience
        }

    def get_evaluation_type(self) -> str:
        """Return evaluation type name for display."""
        return "full_trajectory"

    def get_extra_info(self) -> dict:
        """Return extra information for display in evaluation plan."""
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

        Scans the output directory and aggregates statistics:
        - Total evaluations
        - Overall score average
        - Step quality dimension averages
        - Trajectory dimension averages

        Returns:
            Dict with aggregated statistics for the current run
        """
        from collections import defaultdict
        
        eval_files = list(self.output_dir.glob(f"*{self.get_output_suffix()}"))
        
        total_evaluations = 0
        overall_scores = []
        step_quality_scores = defaultdict(list)
        trajectory_scores = defaultdict(list)

        for ef in eval_files:
            try:
                with open(ef, "r") as f:
                    data = json.load(f)
                
                total_evaluations += 1
                
                # Collect overall score
                overall_score = data.get("overall_score")
                if isinstance(overall_score, (int, float)):
                    overall_scores.append(float(overall_score))
                
                # Collect dimension scores
                dimension_scores = data.get("dimension_scores", {})
                for dim_name, score in dimension_scores.items():
                    if isinstance(score, (int, float)):
                        if dim_name in STEP_QUALITY_CRITERIA:
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
