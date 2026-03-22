#!/usr/bin/env python3
"""
Rubric Evaluator for Trajectory Evaluation
===========================================

Evaluates agent trajectories against pre-generated task-specific rubrics.
For each rubric, determines if it was fulfilled (1) or not (0) based on
evidence in the execution trace.

Requires rubrics to be generated first using RubricGenerator.
"""

import json
import re
import logging
from pathlib import Path
from typing import Any
from agentic.pipeline.full_traces_evaluation.trace_evaluation.base_evaluator import TrajectoryEvaluator

logger = logging.getLogger(__name__)


class RubricEvaluator(TrajectoryEvaluator):
    """
    Evaluates trajectories against pre-generated rubrics.
    
    For each trajectory, loads its rubrics and evaluates whether each rubric
    was fulfilled based on the execution trace. Produces per-rubric assessments
    with reasoning, plus an overall score (fraction of rubrics fulfilled).
    """

    def __init__(
        self,
        judge_model_id: str,
        provider: str,
        traj_input_dir: Path,
        output_dir: Path,
        rubrics_dir: Path,
        context_tokens: int = None,
        overwrite: bool = False,
        concurrency: int = 7,
        eval_model_params: dict | None = None,
        max_files: int | None = None,
    ):
        """
        Initialize rubric evaluator.
        
        Args:
            judge_model_id: Model identifier for the judge LLM
            provider: LLM provider (e.g., 'rits', 'openai')
            traj_input_dir: Directory containing trajectory JSON files
            output_dir: Base directory for saving evaluation results
            rubrics_dir: Directory containing pre-generated rubric files
            context_tokens: Context window size for the judge model
            overwrite: Whether to overwrite existing evaluation results
            concurrency: Number of parallel workers
            eval_model_params: Additional parameters for LLM client
            max_files: Maximum number of files to process (for testing)
        """
        super().__init__(
            judge_model_id=judge_model_id,
            provider=provider,
            traj_input_dir=traj_input_dir,
            output_dir=output_dir,
            context_tokens=context_tokens,
            overwrite=overwrite,
            concurrency=concurrency,
            eval_model_params=eval_model_params,
            max_files=max_files,
        )
        self.rubrics_dir = Path(rubrics_dir)

    def get_evaluation_type(self) -> str:
        """Return the evaluation type identifier."""
        return "rubric_evaluation"

    def get_output_suffix(self) -> str:
        """Return the suffix for output files."""
        return "_rubric_eval.json"

    def get_system_message(self) -> str:
        """Return the system message for rubric evaluation."""
        return """\
You are an expert AI agent evaluator. Your job is to assess whether an \
AI agent's execution trace fulfils a set of pre-defined rubrics \
(requirements) for a given task.

You will be given:
  1. The task / objective the agent was supposed to accomplish.
  2. A list of rubrics — concrete requirements the agent must fulfil.
  3. The full execution trajectory — the sequence of steps the agent took.

For EACH rubric you must determine:
  - **fulfilled**: 1 if the rubric is met based on the trace evidence, \
0 if it is not.
  - **reasoning**: 1-3 sentences explaining why the rubric is or is not \
fulfilled. Reference specific steps or outputs from the trace.

Your evaluation must be grounded exclusively in the trajectory content. \
Do NOT assume actions that are not evidenced in the trace.

After evaluating all rubrics, provide:
  - **summary**: A 2-4 sentence overall assessment.
  - **score**: The fraction of rubrics fulfilled (e.g., if 3 out of 4 \
rubrics are met, score = 0.75). Round to 2 decimal places.

IMPORTANT: Return ONLY valid JSON matching the schema in the user prompt. \
No text outside the JSON."""

    def prepare_evaluation_data(
        self, entry: dict, intent: str
    ) -> dict[str, Any] | None:
        """
        Load rubrics for this trajectory.
        
        Args:
            entry: Entry dict with file_path, traj_name
            intent: Task intent/objective extracted from first row
            
        Returns:
            Dictionary with 'rubrics' and 'task_objective', or None if rubrics not found
        """
        traj_name = entry["traj_name"]
        
        # Look for rubrics file
        rubric_file = self.rubrics_dir / f"{traj_name}_rubrics.json"
        
        if not rubric_file.exists():
            logger.warning(
                f"No rubrics found for {traj_name} at {rubric_file}. "
                "Run rubric generation first."
            )
            return None

        try:
            with open(rubric_file, "r", encoding="utf-8") as f:
                rubric_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rubrics from {rubric_file}: {e}")
            return None

        rubrics = rubric_data.get("rubrics", [])
        # Use task_objective from rubric file, fallback to intent parameter
        task_objective = rubric_data.get("task_objective", intent)

        if not rubrics:
            logger.warning(f"Empty rubrics for {traj_name}")
            return None

        return {
            "rubrics": rubrics,
            "task_objective": task_objective,
            "rubric_file": str(rubric_file),
        }

    def prepare_context(
        self, trajectory_text: str, eval_data: dict
    ) -> dict[str, Any]:
        """
        Prepare context for rubric evaluation prompt.
        
        Args:
            trajectory_text: Formatted trajectory text
            eval_data: Evaluation data from prepare_evaluation_data
            
        Returns:
            Dictionary with all data needed for prompt building
        """
        return {
            "task_objective": eval_data["task_objective"],
            "rubrics": eval_data["rubrics"],
            "trajectory_text": trajectory_text,
        }

    def build_prompt(self, context: dict[str, Any]) -> str:
        """
        Build the prompt for rubric-based evaluation.
        
        Args:
            context: Dictionary with task_objective, rubrics, trajectory_text
            
        Returns:
            Formatted prompt string
        """
        task_objective = context["task_objective"]
        rubrics = context["rubrics"]
        trajectory_text = context["trajectory_text"]

        task_block = task_objective if task_objective else (
            "[Task objective could not be extracted — infer from the trace.]"
        )

        # Format rubrics
        rubrics_block = ""
        for r in rubrics:
            rid = r.get("id", "?")
            desc = r.get("description", "")
            criterion = r.get("criterion", "")
            rubrics_block += f"  - **{rid}**: {desc}\n"
            rubrics_block += f"    Criterion: {criterion}\n"

        # Build JSON template for response
        rubric_ids_json = ",\n".join(
            f'    "{r.get("id", f"R{i+1}")}": '
            f'{{"fulfilled": "<0 or 1>", "reasoning": "<1-3 sentences>"}}'
            for i, r in enumerate(rubrics)
        )

        prompt = f"""\
## Rubric-Based Trajectory Evaluation

### Task / Objective

```
{task_block}
```

### Rubrics to Evaluate

{rubrics_block}

### Full Agent Trajectory

```
{trajectory_text}
```

### Instructions

1. Carefully read the task objective, all rubrics, and the full trajectory.
2. For EACH rubric, determine whether it is fulfilled (1) or not (0) \
based on evidence in the trajectory. Provide 1-3 sentences of reasoning \
referencing specific steps or outputs.
3. After evaluating all rubrics, write a brief **summary** (2-4 sentences) \
of your overall assessment.
4. Compute the **score** as the fraction of rubrics fulfilled (0.0 – 1.0, \
rounded to 2 decimal places).

### Required Output (valid JSON only, no extra text)

```json
{{
  "rubric_results": {{
{rubric_ids_json}
  }},
  "summary": "<2-4 sentence overall assessment>",
  "score": "<0.0-1.0>"
}}
```
"""
        return prompt

    def parse_response(self, response_text: str) -> dict | None:
        """
        Parse the JSON evaluation response from the judge model.
        
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

        logger.warning("Could not parse evaluation response as JSON")
        return None

    def extract_results(
        self,
        evaluation: dict,
        eval_data: dict,
    ) -> dict:
        """
        Extract rubric evaluation results.
        
        Args:
            evaluation: Parsed evaluation from parse_response()
            eval_data: Data from prepare_evaluation_data()
            
        Returns:
            Dictionary with rubric evaluation results
        """
        rubric_results = evaluation.get("rubric_results", {}) if evaluation else {}
        summary = evaluation.get("summary") if evaluation else None
        raw_score = evaluation.get("score") if evaluation else None

        # Parse score
        score = None
        try:
            score = float(raw_score) if raw_score is not None else None
        except (ValueError, TypeError):
            pass

        # Count fulfilled rubrics
        fulfilled_count = 0
        total_count = len(eval_data["rubrics"])

        for rid, rdata in rubric_results.items():
            if isinstance(rdata, dict):
                raw_f = rdata.get("fulfilled")
                if raw_f is not None:
                    try:
                        if int(raw_f) == 1:
                            fulfilled_count += 1
                    except (ValueError, TypeError):
                        pass

        # Compute score if not provided
        if score is None and total_count > 0:
            score = round(fulfilled_count / total_count, 2)

        return {
            "task_objective": eval_data["task_objective"],
            "rubrics": eval_data["rubrics"],
            "rubrics_file": eval_data["rubric_file"],
            "num_rubrics": total_count,
            "rubric_results": rubric_results,
            "fulfilled_count": fulfilled_count,
            "summary": summary,
            "score": score,
        }

    def get_extra_info(self) -> dict:
        """
        Get extra information to display during evaluation planning.
        
        Returns:
            Dict with extra info about this evaluation type
        """
        return {
            "Rubrics Source": str(self.rubrics_dir),
            "Evaluation": "Per-rubric fulfillment (0/1) + overall score",
        }

    def generate_summary(self) -> dict:
        """
        Generate summary statistics for rubric evaluation.
        
        Returns:
            Dictionary with summary statistics
        """
        eval_files = list(self.results_dir.glob("*_rubric_eval.json"))
        
        if not eval_files:
            return {
                "total_evaluations": 0,
                "average_score": 0.0,
                "total_rubrics_evaluated": 0,
                "total_rubrics_fulfilled": 0,
                "fulfillment_rate": 0.0,
            }

        total_score = 0.0
        total_rubrics = 0
        total_fulfilled = 0
        successful_count = 0

        for eval_file in eval_files:
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                score = data.get("score")
                if score is not None:
                    total_score += float(score)
                
                num_rubrics = data.get("num_rubrics", 0)
                fulfilled_count = data.get("fulfilled_count", 0)
                
                total_rubrics += num_rubrics
                total_fulfilled += fulfilled_count
                successful_count += 1
            except Exception as e:
                logger.warning(f"Failed to read {eval_file}: {e}")
                continue

        avg_score = (
            round(total_score / successful_count, 3) if successful_count > 0 else 0.0
        )
        fulfillment_rate = (
            round(total_fulfilled / total_rubrics, 3) if total_rubrics > 0 else 0.0
        )

        return {
            "total_evaluations": successful_count,
            "average_score": avg_score,
            "total_rubrics_evaluated": total_rubrics,
            "total_rubrics_fulfilled": total_fulfilled,
            "fulfillment_rate": fulfillment_rate,
        }
