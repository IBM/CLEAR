#!/usr/bin/env python3
"""
Task Success Evaluator
======================

Concrete evaluator for binary task success evaluation.

Evaluates whether an agent successfully completed its assigned task by:
1. Extracting the task objective from the trajectory
2. Analyzing the full execution trace
3. Determining success (1) or failure (0)
4. Providing reasoning and root cause analysis for failures
"""

import json
import logging
import re

from agentic.pipeline.full_traces_evaluation.trace_evaluation.base_evaluator import TrajectoryEvaluator

logger = logging.getLogger(__name__)


# System message for task success evaluation
SYSTEM_MESSAGE_TASK_SUCCESS = """You are an expert evaluator assessing whether an AI agent successfully completed its assigned task.

Your role:
- Carefully read the task objective and the full agent trajectory
- Determine if the agent achieved the stated goal
- Provide clear reasoning for your decision
- If the task failed, identify the root cause

Be objective and thorough in your analysis."""


class TaskSuccessEvaluator(TrajectoryEvaluator):
    """
    Evaluator for binary task success assessment.
    
    Output fields:
        - success: 1 (success) or 0 (failure)
        - consideration: Reasoning for the decision
        - failure_root_cause: Root cause analysis (only if success=0)
        - task_objective: The extracted task objective
    """

    def prepare_evaluation_data(
        self, entry: dict, intent: str
    ) -> dict | None:
        """
        Prepare task objective for evaluation.
        
        Args:
            entry: Entry dict with file_path, traj_name
            intent: Task intent/objective extracted from first row
        
        Returns:
            Dict with task_objective, or None if intent is missing
        """
        # Use the intent passed from base evaluator
        if not intent:
            logger.warning(
                "No intent provided for %s",
                entry["traj_name"]
            )
            return None
        
        return {"task_objective": intent}

    def prepare_context(self, trajectory_text: str, eval_data: dict) -> dict:
        """
        Prepare context for task success prompt.
        
        Note: trajectory_text is already capped by base class.
        
        Args:
            trajectory_text: Formatted and capped trajectory
            eval_data: Dict with task_objective
        
        Returns:
            Context dict with trajectory_text and task_objective
        """
        return {
            "task_objective": eval_data["task_objective"],
            "trajectory_text": trajectory_text,
        }

    def build_prompt(self, context: dict) -> str:
        """
        Build task success evaluation prompt.
        
        Args:
            context: Dict with task_objective and trajectory_text (already capped)
        
        Returns:
            Formatted prompt string
        """
        task_objective = context["task_objective"]
        trajectory_text = context["trajectory_text"]

        prompt = f"""# Task Success Evaluation

## Task Objective

{task_objective}

## Agent Execution Trace

```
{trajectory_text}
```

## Instructions

1. Carefully read the task objective and the full execution trace above.
2. Determine whether the agent successfully completed the task.
3. Provide your reasoning in the "consideration" field (2-4 sentences).
4. If the task failed (success=0), identify the root cause in "failure_root_cause".

## Required Output (valid JSON only, no extra text)

```json
{{
  "consideration": "<2-4 sentence analysis of whether the task was completed>",
  "success": <0 or 1>,
  "failure_root_cause": "<brief root cause if success=0, otherwise null>"
}}
```

Important:
- success must be exactly 0 (failure) or 1 (success)
- consideration is required for both success and failure
- failure_root_cause is only required when success=0
"""
        return prompt

    def get_system_message(self) -> str:
        """Return system message for task success evaluation."""
        return SYSTEM_MESSAGE_TASK_SUCCESS

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
        Extract task success results from parsed evaluation.
        
        Args:
            evaluation: Parsed evaluation dict (or None if parsing failed)
            eval_data: Dict with task_objective
        
        Returns:
            Dict with success, consideration, failure_root_cause, task_objective
        """
        # Initialize result fields
        consideration = None
        success = None
        failure_root_cause = None

        if evaluation:
            # Extract consideration
            consideration = evaluation.get("consideration")

            # Extract and validate success value
            raw_success = evaluation.get("success")
            if raw_success is not None:
                try:
                    success = int(raw_success)
                    # Clamp to 0 or 1
                    if success not in (0, 1):
                        logger.warning(
                            "Unexpected success value %s — clamping to 0 or 1",
                            raw_success
                        )
                        success = 1 if success > 0 else 0
                except (ValueError, TypeError):
                    logger.warning("Could not parse success value: %s", raw_success)
                    pass

            # Extract failure root cause (only relevant if success=0)
            if success == 0:
                failure_root_cause = evaluation.get("failure_root_cause")

        return {
            "task_objective": eval_data.get("task_objective"),
            "consideration": consideration,
            "success": success,
            "failure_root_cause": failure_root_cause,
        }

    def get_evaluation_type(self) -> str:
        """Return evaluation type name for display."""
        return "task_success"

    def get_extra_info(self) -> dict:
        """Return extra information for display in evaluation plan."""
        return {"Decision": "Binary (0 = failure, 1 = success)"}

    def get_output_suffix(self) -> str:
        """Return output filename suffix for task success evaluation."""
        return "_success.json"

    def generate_summary(self) -> dict:
        """
        Generate a summary report from task-success evaluation results.

        Scans the output directory and aggregates statistics:
        - Total evaluations
        - Success count (success=1)
        - Failure count (success=0)
        - Average score (success rate)

        Returns:
            Dict with aggregated statistics for the current run
        """
        success_files = list(self.results_dir.glob(f"*{self.get_output_suffix()}"))
        
        total_evaluations = 0
        success_count = 0
        failure_count = 0
        no_decision_count = 0

        for sf in success_files:
            try:
                with open(sf, "r") as f:
                    data = json.load(f)
                total_evaluations += 1
                s = data.get("success")
                if s == 1:
                    success_count += 1
                elif s == 0:
                    failure_count += 1
                else:
                    no_decision_count += 1
            except Exception as e:
                logger.warning("Failed to read %s: %s", sf, e)
                continue

        # Calculate average score (success rate)
        average_score = (
            round(success_count / total_evaluations, 3) if total_evaluations > 0 else 0.0
        )

        summary = {
            "total_evaluations": total_evaluations,
            "success_count": success_count,
            "failure_count": failure_count,
            "no_decision_count": no_decision_count,
            "average_score": average_score,
        }

        return summary



