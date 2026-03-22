#!/usr/bin/env python3
"""
Rubric Generator for Trajectory Evaluation
===========================================

Generates task-specific evaluation rubrics (3-5 concrete requirements) for each
trajectory based on the task objective. The number of rubrics is determined by
task complexity:
    - Simple tasks  → 3 rubrics
    - Medium tasks  → 4 rubrics
    - Complex tasks → 5 rubrics

Each rubric is a testable assertion that can be independently verified against
the agent's execution trace.
"""

import json
import re
import logging
from typing import Any
from collections import defaultdict

import pandas as pd
from clear_eval.agentic.pipeline.full_traces_evaluation.trace_evaluation.base_evaluator import TrajectoryEvaluator

logger = logging.getLogger(__name__)


class RubricGenerator(TrajectoryEvaluator):
    """
    Generates task-specific rubrics for trajectory evaluation.
    
    For each trajectory, generates 3-5 concrete, testable requirements based on
    task complexity. Rubrics cover aspects like tool usage, parameter handling,
    action sequencing, output correctness, and edge case handling.
    """

    def get_evaluation_type(self) -> str:
        """Return the evaluation type identifier."""
        return "rubric_generation"

    def get_model_subdirectory(self) -> str | None:
        """
        Return model-specific subdirectory for organizing rubrics.
        
        Rubrics are model-specific because different judge models may generate
        different rubrics for the same task.
        """
        # Clean model ID for use as directory name
        model_dir = self.judge_model_id.replace("/", "_").replace(":", "_")
        return model_dir

    def get_output_suffix(self) -> str:
        """Return the suffix for output files."""
        return "_rubrics.json"

    def get_system_message(self) -> str:
        """Return the system message for rubric generation."""
        return """\
You are an expert AI evaluation designer. Your job is to create clear, \
concrete, and testable evaluation rubrics for AI agent tasks.

Given a task description / user intent, you must generate between 3 and 5 \
rubrics (requirements) that an AI agent must fulfil to successfully \
accomplish the task. The number of rubrics should reflect the task's \
complexity:
  - Simple tasks (single action, clear goal) → 3 rubrics
  - Medium tasks (multiple steps, some conditions) → 4 rubrics
  - Complex tasks (multi-step, multi-app, conditional logic) → 5 rubrics

Each rubric must be:
  - **Specific**: Clearly describes a single, verifiable requirement.
  - **Testable**: Can be independently assessed as fulfilled or not \
fulfilled by examining the agent's execution trace.
  - **Non-overlapping**: Each rubric covers a distinct aspect of the task.
  - **Essential**: The rubric reflects a core requirement, not a nice-to-have.

Rubrics should cover aspects such as:
  - Correct identification and use of the right tools / APIs / applications.
  - Proper handling of task-specific parameters and conditions.
  - Correct sequencing of actions when order matters.
  - Producing the expected output or achieving the desired end-state.
  - Handling edge cases or constraints mentioned in the task.

IMPORTANT: Return ONLY valid JSON matching the schema in the user prompt. \
No text outside the JSON."""

    def prepare_evaluation_data(
        self, entry: dict, intent: str
    ) -> dict[str, Any] | None:
        """
        Prepare task objective for rubric generation.
        
        Args:
            entry: Entry dict with file_path, traj_name
            intent: Task intent/objective extracted from first row
            
        Returns:
            Dictionary with 'task_objective' key, or None if intent is missing
        """
        # Use the intent passed from base evaluator
        if not intent:
            logger.warning(f"No intent provided for {entry['traj_name']}")
            return None
            
        return {"task_objective": intent}

    def prepare_context(
        self, trajectory_text: str, eval_data: dict
    ) -> dict[str, Any]:
        """
        Prepare context for rubric generation (no trajectory text needed).
        
        Args:
            trajectory_text: Formatted trajectory text (not used for rubric generation)
            eval_data: Evaluation data from prepare_evaluation_data
            
        Returns:
            Dictionary with task_objective for prompt building
        """
        return {"task_objective": eval_data["task_objective"]}

    def build_prompt(self, context: dict[str, Any]) -> str:
        """
        Build the prompt for rubric generation.
        
        Args:
            context: Dictionary with 'task_objective' key
            
        Returns:
            Formatted prompt string
        """
        task_objective = context["task_objective"]
        task_block = task_objective if task_objective else (
            "[Task objective could not be extracted.]"
        )

        prompt = f"""\
## Task Rubric Generation

### Task / User Intent

```
{task_block}
```

### Instructions

1. Carefully read the task / user intent above.
2. Determine the complexity of the task (simple, medium, or complex).
3. Generate between 3 and 5 rubrics — concrete requirements that the \
agent must fulfil to accomplish this task.
4. Each rubric should have:
   - A short **id** (e.g., "R1", "R2", ...).
   - A concise **description** (1-2 sentences) of the requirement.
   - A **criterion** — a clear, testable statement that can be checked \
against the execution trace (e.g., "The agent called the X API with \
parameter Y" or "The agent's final output contains Z").

### Required Output (valid JSON only, no extra text)

```json
{{
  "task_summary": "<1 sentence summary of the task>",
  "complexity": "<simple | medium | complex>",
  "rubrics": [
    {{
      "id": "R1",
      "description": "<what this requirement is about>",
      "criterion": "<testable assertion to check against the trace>"
    }},
    {{
      "id": "R2",
      "description": "<what this requirement is about>",
      "criterion": "<testable assertion to check against the trace>"
    }}
  ]
}}
```
"""
        return prompt

    def parse_response(self, response_text: str) -> dict | None:
        """
        Parse the JSON rubrics response from the judge model.
        
        Args:
            response_text: Raw response from the LLM
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
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

        # Try finding JSON object boundaries
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse rubrics response as JSON")
        return None

    def extract_results(
        self,
        evaluation: dict,
        eval_data: dict,
    ) -> dict:
        """
        Extract and structure rubric generation results.
        
        Args:
            evaluation: Parsed evaluation from parse_response()
            eval_data: Data from prepare_evaluation_data()
            
        Returns:
            Dictionary with structured rubric generation results
        """
        rubrics = evaluation.get("rubrics", []) if evaluation else []
        task_summary = evaluation.get("task_summary") if evaluation else None
        complexity = evaluation.get("complexity") if evaluation else None

        return {
            "task_objective": eval_data["task_objective"],
            "task_summary": task_summary,
            "complexity": complexity,
            "rubrics": rubrics,
            "num_rubrics": len(rubrics),
        }

    def get_extra_info(self) -> dict:
        """
        Get extra information to display during evaluation planning.
        
        Returns:
            Dict with extra info about this evaluation type
        """
        return {
            "Rubrics": "3-5 per task (based on complexity)",
            "Output": "Task summary, complexity, rubric list"
        }

    def generate_summary(self) -> dict:
        """
        Generate summary statistics for rubric generation.
        
        Returns:
            Dictionary with summary statistics
        """
        rubric_files = list(self.results_dir.glob("*_rubrics.json"))
        
        if not rubric_files:
            return {
                "total_evaluations": 0,
                "total_rubrics": 0,
                "avg_rubrics_per_task": 0.0,
                "complexity_distribution": {},
            }

        total_rubrics = 0
        complexity_counts = defaultdict(int)
        successful_count = 0

        for rubric_file in rubric_files:
            try:
                with open(rubric_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                num_rubrics = data.get("num_rubrics", 0)
                total_rubrics += num_rubrics
                
                complexity = data.get("complexity", "unknown")
                complexity_counts[complexity] += 1
                
                successful_count += 1
            except Exception as e:
                logger.warning(f"Failed to read {rubric_file}: {e}")
                continue

        avg_rubrics = (
            round(total_rubrics / successful_count, 2) if successful_count > 0 else 0.0
        )

        return {
            "total_evaluations": successful_count,
            "total_rubrics": total_rubrics,
            "avg_rubrics_per_task": avg_rubrics,
            "complexity_distribution": dict(complexity_counts),
        }

