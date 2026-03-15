#!/usr/bin/env python3
"""
Task Success Evaluation - Multi-Provider Support
=================================================

Evaluates whether an AI agent trajectory successfully accomplished its
assigned task.  Produces a binary success label (0 or 1) together with
textual reasoning that explains the decision.

Supports multiple LLM providers:
    - RITS (openai/gpt-oss-20b, openai/gpt-oss-120b, deepseek-ai/DeepSeek-V3.2)
    - watsonx (ibm-granite/granite-3.3-8b-instruct, etc.)
    - OpenAI (gpt-4o, gpt-4o-mini, etc.)

Results are saved to:
    full_traj_evaluation/success_results_{provider}_{model}/

Setup:
    1. Set environment variables for your provider:
         RITS: RITS_API_KEY
         watsonx: WATSONX_URL, WATSONX_APIKEY, WATSONX_PROJECT_ID
         OpenAI: OPENAI_API_KEY
    2. Install dependencies:
         pip install aiohttp python-dotenv tqdm
    3. Run:
         python run_task_success_evaluation.py --provider <provider> --model-id <model> --dataset <dataset>

Usage examples:
    # RITS
    python run_task_success_evaluation.py --provider rits --model-id openai/gpt-oss-120b --dataset CUGA
    
    # watsonx
    python run_task_success_evaluation.py --provider watsonx --model-id ibm-granite/granite-3.3-8b-instruct --dataset FinOps
    
    # OpenAI
    python run_task_success_evaluation.py --provider openai --model-id gpt-4o --dataset WXO
    
    # With filters
    python run_task_success_evaluation.py --provider rits --model-id openai/gpt-oss-20b --dataset CUGA --model full --max-files 5
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

from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from clear_eval.agentic.full_traj_evaluation.full_traj_utils import discover_trajectories, _cap_trajectory, \
    get_max_trajectory_chars
from clear_eval.agentic.full_traj_evaluation.pipeline_inference_adapter import (
    get_llm_client_adapter,
    evaluate_batch_parallel,
)
# Import centralized modules
from clear_eval.agentic.full_traj_evaluation.dataset_base import (
    get_available_datasets,
    TRAJ_DATA_DIR,
    RESULTS_DIR,
    get_dataset_obj,
    get_results_dir,
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

# ---------------------------------------------------------------------------
# 6. Prompt for task-success assessment
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE_TASK_SUCCESS = """\
You are an expert AI agent evaluator. Your job is to determine whether an \
AI agent successfully completed the task it was assigned, based solely on \
the agent's execution trajectory.

You will be given:
  1. The task / objective the agent was supposed to accomplish.
  2. The full execution trajectory — the sequence of steps the agent took, \
including planning, tool calls, observations, and final outputs.

Your evaluation must be grounded exclusively in the trajectory content. \
Do NOT assume success or failure based on external knowledge — judge only \
by what the trajectory shows.

Your output has exactly two or three fields:
  - consideration: A thorough 4-8 sentence analysis explaining your \
reasoning. Discuss what the agent was asked to do, what it actually did, \
whether it reached the desired outcome, and any issues or partial \
completions you observed. Write this BEFORE making your final decision \
so that it acts as chain-of-thought reasoning.
  - success: An integer, either 1 (the agent successfully completed the \
task) or 0 (the agent did NOT successfully complete the task).
  - failure_root_cause (ONLY when success is 0): A concise 1-3 sentence \
description of the most probable root cause that led to the agent's \
failure. Focus on the primary reason (e.g. wrong tool usage, \
hallucinated data, infinite loop, misunderstood objective, API error, \
incomplete execution). Do NOT repeat the full consideration — be specific \
and direct about the root cause.

Guidelines for deciding success:
  - 1 (Success): The agent demonstrably accomplished the core objective. \
Minor imperfections (e.g. slightly verbose output, a non-critical extra \
step) do NOT disqualify success as long as the primary goal was met.
  - 0 (Failure): The agent failed to accomplish the core objective. This \
includes cases where the agent: produced incorrect results, got stuck in \
a loop, crashed or errored out, only partially completed the task with \
critical parts missing, or hallucinated results without actually performing \
the required actions.

IMPORTANT: Return ONLY valid JSON matching the schema below. No text \
outside the JSON."""


def build_task_success_prompt(
    task_objective: str,
    trajectory_text: str,
    *,
    max_len: int,
) -> str:
    """Build the task-success evaluation prompt."""
    trajectory_text = _cap_trajectory(trajectory_text, max_len)

    task_block = task_objective if task_objective else (
        "[Task objective could not be extracted — infer the agent's goal "
        "from the trajectory itself.]"
    )

    prompt = f"""\
## Task Success Evaluation

### Task / Objective

{task_block}

### Full Agent Trajectory

```
{trajectory_text}
```

### Instructions

1. Carefully read the task objective and the full trajectory above.
2. Determine whether the agent accomplished the stated task based on the \
evidence in the trajectory.
3. Write your **consideration** first (4-8 sentences) — this is your \
chain-of-thought reasoning about what the agent did, whether it achieved \
the objective, and any issues you identified.
4. Then provide your **success** decision: 1 if the task was completed \
successfully, 0 if it was not.
5. If your decision is **0 (failure)**, also provide a \
**failure_root_cause** — a concise 1-3 sentence description of the \
primary reason the agent failed. Omit this field when success is 1.

### Required Output (valid JSON only, no extra text)

If the agent SUCCEEDED:
```json
{{
  "consideration": "<4-8 sentence analysis>",
  "success": 1
}}
```

If the agent FAILED:
```json
{{
  "consideration": "<4-8 sentence analysis>",
  "success": 0,
  "failure_root_cause": "<1-3 sentence root cause>"
}}
```
"""
    return prompt


# ---------------------------------------------------------------------------
# 7. Response parsing
# ---------------------------------------------------------------------------


def parse_evaluation_response(response_text: str) -> dict | None:
    """Parse the JSON evaluation response from the judge model."""
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

    logger.warning("Could not parse evaluation response as JSON")
    return None


# ---------------------------------------------------------------------------
# 8. Discovery: find all trajectory files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. Single trajectory evaluation
# ---------------------------------------------------------------------------


def evaluate_single_trajectory(
    entry: dict,
    results_dir: Path,
    judge_model_id: str,
    judge_short_name: str,
    judge_key: str,
    llm_client,
    provider: str = "rits",
    context_tokens: int = 128_000,
    overwrite: bool = False,
) -> dict | None:
    """Evaluate a single trajectory for task success and save results."""
    dataset = entry["dataset"]
    model_name = entry["model_name"]
    file_path = entry["file_path"]
    traj_name = entry["traj_name"]

    output_dir = results_dir / dataset / model_name
    output_file = output_dir / f"{traj_name}_success.json"

    if output_file.exists() and not overwrite:
        logger.info("Skipping (exists): %s", output_file)
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None

    dataset_obj = get_dataset_obj(dataset_name=dataset, data_dir="")

    task_objective = dataset_obj.extract_user_request(traj_data)
    trajectory_text = dataset_obj.format_trajectory(traj_data)
    max_traj_chars = get_max_trajectory_chars(context_tokens)

    prompt = build_task_success_prompt(
        task_objective, trajectory_text, max_len=max_traj_chars,
    )
    system_message = SYSTEM_MESSAGE_TASK_SUCCESS

    start_time = time.time()

    response_text = llm_client.call(
        prompt=prompt,
        system_message=system_message,
    )

    elapsed = time.time() - start_time

    if not response_text:
        logger.error("No response for %s/%s/%s", dataset, model_name, traj_name)
        return None

    evaluation = parse_evaluation_response(response_text)

    consideration = None
    success = None
    failure_root_cause = None

    if evaluation:
        consideration = evaluation.get("consideration")
        raw_success = evaluation.get("success")
        if raw_success is not None:
            try:
                success = int(raw_success)
                if success not in (0, 1):
                    logger.warning(
                        "Unexpected success value %s for %s — clamping",
                        raw_success, traj_name,
                    )
                    success = 1 if success > 0 else 0
            except (ValueError, TypeError):
                pass

        if success == 0:
            failure_root_cause = evaluation.get("failure_root_cause")

    result = {
        "trajectory_name": traj_name,
        "dataset": dataset,
        "model_name": model_name,
        "source_file": str(file_path),
        "judge_model": judge_model_id,
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_time_seconds": round(elapsed, 2),
        "task_objective": task_objective or None,
        "consideration": consideration,
        "success": success,
        "failure_root_cause": failure_root_cause,
        "raw_response": response_text,
        "parsed_evaluation": evaluation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ---------------------------------------------------------------------------
# 10. Batch evaluation with concurrency control
# ---------------------------------------------------------------------------


def run_evaluation_batch(
    entries: list[dict],
    results_dir: Path,
    judge_model_id: str,
    judge_short_name: str,
    judge_key: str,
    provider: str = "rits",
    context_tokens: int = 128_000,
    overwrite: bool = False,
    concurrency: int = 2,
) -> list[dict]:
    """Evaluate a batch of trajectories with concurrency control."""
    llm_client = get_llm_client_adapter(
        provider=provider,
        model_id=judge_model_id,
    )

    inputs = [
        (entry, results_dir, judge_model_id, judge_short_name,
         judge_key, llm_client, provider, context_tokens, overwrite)
        for entry in entries
    ]

    results = evaluate_batch_parallel(
        evaluate_func=evaluate_single_trajectory,
        entries=inputs,
        max_workers=concurrency,
        use_async=False,
        progress_desc="Evaluating task success",
    )

    return results


# ---------------------------------------------------------------------------
# 11. Summary report generation
# ---------------------------------------------------------------------------


def generate_summary_report(results_dir: Path) -> dict:
    """Generate a summary report from task-success evaluation results."""
    summary = {}

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        dataset = dataset_dir.name
        summary[dataset] = {}

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            model_name = model_dir.name

            eval_files = list(model_dir.glob("*_success.json"))
            if not eval_files:
                continue

            successes = 0
            failures = 0
            unknown = 0
            count = 0

            for ef in eval_files:
                try:
                    with open(ef, "r") as f:
                        data = json.load(f)
                    s_val = data.get("success")
                    if s_val == 1:
                        successes += 1
                    elif s_val == 0:
                        failures += 1
                    else:
                        unknown += 1
                    count += 1
                except Exception:
                    continue

            success_rate = None
            if (successes + failures) > 0:
                success_rate = round(successes / (successes + failures), 3)

            summary[dataset][model_name] = {
                "total_evaluations": count,
                "successes": successes,
                "failures": failures,
                "unknown": unknown,
                "success_rate": success_rate,
            }

    return summary


def print_summary(summary: dict, judge_model_id: str):
    """Pretty-print the summary report."""
    print("\n" + "=" * 80)
    print("TASK SUCCESS EVALUATION SUMMARY")
    print(f"Judge Model: {judge_model_id}")
    print("=" * 80)

    for dataset, models in summary.items():
        print(f"\n{'#' * 70}")
        print(f"# Dataset: {dataset}")
        print(f"{'#' * 70}")

        for model_name, stats in models.items():
            rate = stats.get("success_rate")
            rate_str = f"{rate:.1%}" if rate is not None else "N/A"
            print(f"\n  Model: {model_name}")
            print(f"  Evaluations: {stats['total_evaluations']}")
            print(f"  Successes:   {stats['successes']}")
            print(f"  Failures:    {stats['failures']}")
            if stats["unknown"]:
                print(f"  Unknown:     {stats['unknown']}")
            print(f"  Success Rate: {rate_str}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = create_base_parser(description="Task Success Evaluation (binary success label)")
    return parser.parse_args()

def run_main(args):
    """Main entry point."""
    judge_model_id = args.model_id
    results_dir = get_results_dir("success_results", args.model_id)
    context_tokens = args.context_tokens

    if args.summary_only:
        if results_dir.exists():
            summary = generate_summary_report(results_dir)
            print_summary(summary, judge_model_id)
            summary_file = results_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_file}")
        else:
            print(f"No results found at {results_dir}. Run evaluation first.")
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

    max_traj_chars = get_max_trajectory_chars(context_tokens)

    print("=" * 70)
    print("Task Success Evaluation Plan")
    print("=" * 70)
    print(f"Provider:     {args.provider}")
    print(f"Judge Model:  {judge_model_id}")
    print(f"Context Win:  {context_tokens:,} tokens → "
          f"max trajectory {max_traj_chars:,} chars")
    print(f"Data Dir:     {TRAJ_DATA_DIR}")
    print(f"Results Dir:  {results_dir}")
    print(f"Decision:     Binary (0 = failure, 1 = success)")
    print(f"Overwrite:    {args.overwrite}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Total files:  {len(entries)}")
    print()
    for (ds, model), cnt in sorted(counts.items()):
        print(f"  {ds}/{model}: {cnt} files")
    print("=" * 70)

    start = time.time()
    results = run_evaluation_batch(
        entries,
        results_dir=results_dir,
        judge_model_id=judge_model_id,
        judge_short_name="",
        judge_key=args.model_id,
        provider=args.provider,
        context_tokens=context_tokens,
        overwrite=args.overwrite,
        concurrency=args.concurrency,
    )
    elapsed = time.time() - start

    print(f"\nEvaluation complete: {len(results)} trajectories in {elapsed:.1f}s")

    if results_dir.exists():
        summary = generate_summary_report(results_dir)
        print_summary(summary, judge_model_id)

        summary_file = results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")


def main():
    args = parse_args()

    run_main(args)


if __name__ == "__main__":
    main()
