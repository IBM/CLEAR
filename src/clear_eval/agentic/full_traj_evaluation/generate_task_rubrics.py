#!/usr/bin/env python3
"""
Generate Task Rubrics via RITS  (Step 1 of Rubric-Based Evaluation)
=====================================================================

Given a task / intent extracted from each trajectory file, this script
prompts a judge model to generate **3-5 concrete, testable requirements**
(rubrics) that an agent must fulfil to accomplish the task.

The number of rubrics is determined by the model based on task complexity:
    - Simple tasks  → 3 rubrics
    - Medium tasks  → 4 rubrics
    - Complex tasks → 5 rubrics

Each rubric is a short assertion that can be independently verified against
the agent's execution trace (Step 2 — see ``run_rubric_evaluation.py``).

Results are saved to:
    full_traj_evaluation/rubrics_{judge}/{dataset}/{model}/{traj}_rubrics.json

Supported judge models (--judge flag):
    oss20b   - openai/gpt-oss-20b   (default, fast for debugging)
    oss120b  - openai/gpt-oss-120b
    deepseek - deepseek-ai/DeepSeek-V3.2

Setup:
    1. Create a ``.env`` file (or export) with:
         RITS_API_KEY=<your-rits-api-key>
    2. Install dependencies:
         pip install aiohttp python-dotenv tqdm
    3. Run:
         python generate_task_rubrics.py [OPTIONS]

Usage examples:
    python generate_task_rubrics.py
    python generate_task_rubrics.py --judge oss120b
    python generate_task_rubrics.py --dataset CUGA --model full --max-files 5
"""

import os
import re
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

from dotenv import load_dotenv

from agentic.full_traj_evaluation.full_traj_utils import discover_trajectories
from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from clear_eval.pipeline.llm_client import get_llm_client, run_parallel
# Import centralized modules
from clear_eval.agentic.full_traj_evaluation.dataset_base import (
    get_dataset_obj,
    TRAJ_DATA_DIR,
    RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# 4. Rubric generation prompt
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE_RUBRICS = """\
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


def build_rubrics_prompt(task_objective: str) -> str:
    """Build the prompt to generate rubrics for a given task."""
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


# ---------------------------------------------------------------------------
# 5. Response parsing
# ---------------------------------------------------------------------------


def parse_rubrics_response(response_text: str) -> dict | None:
    """Parse the JSON rubrics response from the judge model."""
    if not response_text:
        return None

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL
    )
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse rubrics response as JSON")
    return None


# ---------------------------------------------------------------------------
# 6. Single trajectory rubric generation
# ---------------------------------------------------------------------------


def generate_rubrics_single(
    entry: dict,
    rubrics_dir: Path,
    judge_model_id: str,
    llm_client,
    provider = "rits",
    overwrite: bool = False,
) -> dict | None:
    """Generate rubrics for a single trajectory and save them."""
    dataset = entry["dataset"]
    model_name = entry["model_name"]
    file_path = entry["file_path"]
    traj_name = entry["traj_name"]

    output_dir = rubrics_dir / dataset / model_name
    output_file = output_dir / f"{traj_name}_rubrics.json"

    if output_file.exists() and not overwrite:
        logger.info("Skipping (exists): %s/%s/%s", dataset, model_name, traj_name)
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None

    dataset_obj = get_dataset_obj(dataset)
    task_objective = dataset_obj.extract_user_request(traj_data)

    if not task_objective:
        logger.warning(
            "Empty task objective for %s/%s/%s — skipping",
            dataset, model_name, traj_name,
        )
        return None

    prompt = build_rubrics_prompt(task_objective)

    start_time = time.time()

    response_text = llm_client.call(
        prompt=prompt,
        system_message=SYSTEM_MESSAGE_RUBRICS,
    )

    elapsed = time.time() - start_time

    if not response_text:
        logger.error(
            "No response for %s/%s/%s", dataset, model_name, traj_name,
        )
        return None

    parsed = parse_rubrics_response(response_text)

    rubrics = []
    task_summary = None
    complexity = None

    if parsed:
        rubrics = parsed.get("rubrics", [])
        task_summary = parsed.get("task_summary")
        complexity = parsed.get("complexity")

    result = {
        "trajectory_name": traj_name,
        "dataset": dataset,
        "model_name": model_name,
        "source_file": str(file_path),
        "judge_model": judge_model_id,
        "generation_timestamp": datetime.now().isoformat(),
        "generation_time_seconds": round(elapsed, 2),
        "task_objective": task_objective,
        "task_summary": task_summary,
        "complexity": complexity,
        "rubrics": rubrics,
        "num_rubrics": len(rubrics),
        "raw_response": response_text,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ---------------------------------------------------------------------------
# 7. Batch generation with concurrency control
# ---------------------------------------------------------------------------


def run_generation_batch(
    entries: list[dict],
    rubrics_dir: Path,
    judge_model_id: str,
    overwrite: bool = False,
    concurrency: int = 4,
    provider: str = "rits",
    eval_model_params: dict | None = None,
) -> list:
    """Generate rubrics for a batch of trajectories with concurrency.
    
    Returns:
        List of ParallelResult objects containing generation results and status.
    """
    llm_client = get_llm_client(
        provider=provider,
        model=judge_model_id,
        use_litellm=True,
        eval_mode=True,
        parameters=eval_model_params or {},
    )

    inputs = [
        (entry, rubrics_dir, judge_model_id, llm_client, provider, overwrite)
        for entry in entries
    ]

    parallel_results = run_parallel(
        func=generate_rubrics_single,
        inputs=inputs,
        use_async=True,
        max_workers=concurrency,
        progress_desc="Generating rubrics",
    )

    return parallel_results


# ---------------------------------------------------------------------------
# 8. Summary report
# ---------------------------------------------------------------------------


def generate_summary(rubrics_dir: Path) -> dict:
    """Generate a summary of rubric generation results."""
    summary = {}

    for dataset_dir in sorted(rubrics_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        dataset = dataset_dir.name
        summary[dataset] = {}

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            model_name = model_dir.name

            rubric_files = list(model_dir.glob("*_rubrics.json"))
            if not rubric_files:
                continue

            count = 0
            total_rubrics = 0
            complexity_counts = defaultdict(int)

            for rf in rubric_files:
                try:
                    with open(rf, "r") as f:
                        data = json.load(f)
                    n = data.get("num_rubrics", 0)
                    total_rubrics += n
                    c = data.get("complexity", "unknown")
                    complexity_counts[c] += 1
                    count += 1
                except Exception:
                    continue

            summary[dataset][model_name] = {
                "total_tasks": count,
                "total_rubrics": total_rubrics,
                "avg_rubrics_per_task": (
                    round(total_rubrics / count, 1) if count else 0
                ),
                "complexity_distribution": dict(complexity_counts),
            }

    return summary


def print_summary(summary: dict, judge_model_id: str):
    """Pretty-print the rubric generation summary."""
    print("\n" + "=" * 80)
    print("TASK RUBRIC GENERATION SUMMARY")
    print(f"Judge Model: {judge_model_id}")
    print("=" * 80)

    for dataset, models in summary.items():
        print(f"\n{'#' * 70}")
        print(f"# Dataset: {dataset}")
        print(f"{'#' * 70}")

        for model_name, stats in models.items():
            print(f"\n  Model: {model_name}")
            print(f"  Tasks:                {stats['total_tasks']}")
            print(f"  Total rubrics:        {stats['total_rubrics']}")
            print(f"  Avg rubrics/task:     {stats['avg_rubrics_per_task']}")
            dist = stats.get("complexity_distribution", {})
            if dist:
                print(f"  Complexity dist:      {dict(dist)}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = create_base_parser(
        description=
            "Generate task rubrics"
            "(Step 1 of rubric-based evaluation)"
    )
    return parser.parse_args()


def get_rubrics_dir(judge_key: str) -> Path:
    """Return the rubrics directory: rubrics_{judge}/"""
    judge_short_name = judge_key.split("/")[-1].replace("-", "_")
    return RESULTS_DIR / f"rubrics_{judge_short_name}"


def run_main(args):
    """Main entry point."""
    judge_model_id = args.model_id
    rubrics_dir = get_rubrics_dir(judge_model_id)

    if args.summary_only:
        if rubrics_dir.exists():
            summary = generate_summary(rubrics_dir)
            print_summary(summary, judge_model_id)
            summary_file = rubrics_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_file}")
        else:
            print(f"No rubrics found at {rubrics_dir}. Run generation first.")
        return

    entries = discover_trajectories(
        TRAJ_DATA_DIR,
        filter_dataset=args.dataset,
        filter_model=args.model,
    )

    if not entries:
        print("No trajectory files found. Check paths and filters.")
        return

    if args.max_files:
        entries = entries[:args.max_files]

    counts = Counter((e["dataset"], e["model_name"]) for e in entries)

    print("=" * 70)
    print("Task Rubric Generation Plan")
    print("=" * 70)
    print(f"Judge Model:  {judge_model_id} ({args.model_id})")
    print(f"Data Dir:     {TRAJ_DATA_DIR}")
    print(f"Rubrics Dir:  {rubrics_dir}")
    print(f"Overwrite:    {args.overwrite}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Total files:  {len(entries)}")
    print()
    for (ds, model), cnt in sorted(counts.items()):
        print(f"  {ds}/{model}: {cnt} files")
    print("=" * 70)

    start = time.time()
    parallel_results = run_generation_batch(
        entries,
        rubrics_dir=rubrics_dir,
        judge_model_id=judge_model_id,
        overwrite=args.overwrite,
        concurrency=args.concurrency,
        provider=args.provider,
        eval_model_params=args.eval_model_params,
    )
    elapsed = time.time() - start

    # Count successful generations
    successful = sum(1 for pr in parallel_results if pr.is_success and pr.result is not None)
    total = len(parallel_results)
    
    print(f"\nGeneration complete: {successful}/{total} tasks succeeded in {elapsed:.1f}s")

    if rubrics_dir.exists():
        summary = generate_summary(rubrics_dir)
        print_summary(summary, judge_model_id)

        summary_file = rubrics_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")


def main():
    args = parse_args()
    run_main(args)


if __name__ == "__main__":
    main()
