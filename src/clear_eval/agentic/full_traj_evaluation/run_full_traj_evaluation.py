#!/usr/bin/env python3
"""
Full Trajectory Evaluation

Evaluates full agent trajectories from traj_full_data/ using a RITS-hosted
judge model.  Produces a CLEAR-style overall score (0-1) and detailed textual
feedback.

Four evaluation methods (--method flag):
    dimensions_prompt           - Score each of 14 dimensions individually,
                                  then produce detailed feedback + overall score.
    full_trace_prompt           - Dimensions listed as guidance only; model
                                  produces only detailed feedback + overall
                                  score (lighter, more holistic).
    full_trace_prompt_with_step - Like full_trace_prompt but step-level and
                                  trajectory-level dimensions are presented
                                  separately and both are asked to be taken
                                  into account.
    full_trace_prompt_issue     - Like _with_step but the detailed_feedback
                                  focuses exclusively on issues, problems,
                                  and weaknesses (no strengths).

    full_traj_evaluation/results_{judge}/{method}/
"""

import re
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from clear_eval.agentic.full_traj_evaluation.full_traj_utils import _cap_trajectory, discover_trajectories, \
    get_max_trajectory_chars
from clear_eval.pipeline.llm_client import get_llm_client, run_parallel

# Import centralized modules
from clear_eval.agentic.full_traj_evaluation.dataset_base import (
    get_dataset_obj,
    TRAJ_DATA_DIR,
    get_results_dir,
)

# ---------------------------------------------------------------------------
# Evaluation dimensions
# ---------------------------------------------------------------------------

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

# Scoring scale anchors (used in both prompts)
SCORING_SCALE = """\
Scoring uses a continuous 0 – 1 scale with the following anchors:
  0.00 = completely failed / absent
  0.25 = poor quality, major issues
  0.50 = acceptable but with notable gaps
  0.75 = good quality, minor issues only
  1.00 = excellent, no meaningful issues"""

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
# 5. Prompt builders — two methods
# ---------------------------------------------------------------------------

# ===================== METHOD 1: dimensions_prompt =========================

SYSTEM_MESSAGE_DIMENSIONS = f"""\
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


def build_dimensions_prompt(trajectory_text: str, *, max_len: int) -> str:
    """Build the per-dimension scoring prompt (method: dimensions_prompt)."""
    trajectory_text = _cap_trajectory(trajectory_text, max_len)

    step_block = "\n".join(
        f"  - **{name}**: {desc}"
        for name, desc in STEP_QUALITY_CRITERIA.items()
    )
    traj_block = "\n".join(
        f"  - **{name}**: {desc}"
        for name, desc in TRAJECTORY_CRITERIA.items()
    )

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


# ---------------------------------------------------------------------------
# 6. Response parsing
# ---------------------------------------------------------------------------


def parse_evaluation_response(response_text: str) -> dict | None:
    """Parse the JSON evaluation response from the judge model."""
    if not response_text:
        return None

    # Try direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
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


def extract_scores_from_evaluation(evaluation: dict) -> dict:
    """
    Pull all dimension scores out of a parsed evaluation dict.
    Returns {dim_name: float_score} for every dimension found.
    """
    scores = {}
    for section_key in ("step_quality_dimensions", "trajectory_dimensions",
                        "dimensions"):
        section = evaluation.get(section_key, {})
        if isinstance(section, dict):
            for dim_name, dim_data in section.items():
                if isinstance(dim_data, dict) and "score" in dim_data:
                    try:
                        scores[dim_name] = float(dim_data["score"])
                    except (ValueError, TypeError):
                        pass
    return scores

# ---------------------------------------------------------------------------
#  Single trajectory evaluation
# ---------------------------------------------------------------------------


def evaluate_single_trajectory(
    entry: dict,
    results_dir: Path,
    judge_model_id: str,
    llm_client,
    overwrite: bool = False,
    context_tokens: int = 128_000,
) -> dict | None:
    """Evaluate a single trajectory file and save results."""
    dataset = entry["dataset"]
    model_name = entry["model_name"]
    file_path = entry["file_path"]
    traj_name = entry["traj_name"]

    # Output path: results_{judge}/{method}/{dataset}/{model}/{traj}_eval.json
    output_dir = results_dir / dataset / model_name
    output_file = output_dir / f"{traj_name}_eval.json"

    if output_file.exists() and not overwrite:
        logger.info("Skipping (exists): %s/%s/%s", dataset, model_name, traj_name)
        return None

    # Load trajectory
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None

    # Format trajectory text (no dataset/model/name metadata — blind evaluation)
    dataset_obj = get_dataset_obj(dataset_name=dataset, data_dir="")
    trajectory_text = dataset_obj.format_trajectory(traj_data)

    # Compute model-specific trajectory cap
    max_traj_chars = get_max_trajectory_chars(context_tokens)

    prompt = build_dimensions_prompt(trajectory_text, max_len=max_traj_chars)
    system_message = SYSTEM_MESSAGE_DIMENSIONS

    # Call judge using the provided client
    response_text = llm_client.call(
        prompt=prompt,
        system_message=system_message
    )

    # Parse response
    evaluation = parse_evaluation_response(response_text)

    # Extract fields
    dimension_scores = {}
    overall_score = None
    detailed_feedback = None

    if evaluation:
        # detailed_feedback (present in both methods)
        detailed_feedback = evaluation.get("detailed_feedback")

        # overall_score
        raw_overall = evaluation.get("overall_score")
        try:
            overall_score = float(raw_overall)
        except (ValueError, TypeError):
            pass

        # Dimension scores (only in dimensions_prompt method)

        dimension_scores = extract_scores_from_evaluation(evaluation)
        # Fallback overall from dimension average
        if overall_score is None and dimension_scores:
            overall_score = round(
                sum(dimension_scores.values()) / len(dimension_scores), 2
            )

    # Build result
    result = {
        "trajectory_name": traj_name,
        "dataset": dataset,
        "model_name": model_name,
        "source_file": str(file_path),
        "judge_model": judge_model_id,
        "method": "dimensions_prompt",
        "detailed_feedback": detailed_feedback,
        "overall_score": overall_score,
        "dimension_scores": dimension_scores,
        "raw_response": response_text,
        "parsed_evaluation": evaluation,
    }

    # Save result
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ---------------------------------------------------------------------------
# 9. Batch evaluation using pipeline infrastructure
# ---------------------------------------------------------------------------


def run_evaluation_batch(
    entries: list[dict],
    results_dir: Path,
    judge_model_id: str,
    provider: str,
    overwrite: bool,
    concurrency: int,
    context_tokens: int,
    eval_model_params: dict | None = None,
) -> list:
    """Evaluate a batch of trajectories using pipeline's run_parallel.
    
    Returns:
        List of ParallelResult objects containing evaluation results and status.
    """
    
    # Get LLM client once (will be reused for all evaluations)
    llm_client = get_llm_client(
        provider=provider,
        model=judge_model_id,
        use_litellm=True,
        eval_mode=True,
        parameters=eval_model_params or {},
    )
    
    # Prepare inputs as tuples of all parameters for each entry
    inputs = [
        (
            entry,
            results_dir,
            judge_model_id,
            llm_client,
            overwrite,
            context_tokens
        )
        for entry in entries
    ]
    
    # Use pipeline's parallel execution with progress bar
    parallel_results = run_parallel(
        func=evaluate_single_trajectory,
        inputs=inputs,
        use_async=True,
        max_workers=concurrency,
        progress_desc="Evaluating trajectories"
    )

    return parallel_results


# ---------------------------------------------------------------------------
# Summary report generation
# ---------------------------------------------------------------------------


def generate_summary_report(results_dir: Path) -> dict:
    """Generate a summary report from evaluation results under results_dir."""
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

            eval_files = list(model_dir.glob("*_eval.json"))
            if not eval_files:
                continue

            all_dim_scores: dict[str, list[float]] = defaultdict(list)
            overall_scores: list[float] = []
            count = 0

            for ef in eval_files:
                try:
                    with open(ef, "r") as f:
                        data = json.load(f)
                    dim_scores = data.get("dimension_scores", {})
                    for dim, score in dim_scores.items():
                        if isinstance(score, (int, float)):
                            all_dim_scores[dim].append(float(score))
                    os_val = data.get("overall_score")
                    if isinstance(os_val, (int, float)):
                        overall_scores.append(float(os_val))
                    count += 1
                except Exception:
                    continue

            model_summary: dict = {
                "total_evaluations": count,
                "overall_score_average": None,
            }

            # Dimension averages only for dimensions_prompt
            if all_dim_scores:
                sq = {}
                ta = {}
                for dim, scores in all_dim_scores.items():
                    if not scores:
                        continue
                    avg = round(sum(scores) / len(scores), 3)
                    if dim in STEP_QUALITY_CRITERIA:
                        sq[dim] = avg
                    elif dim in TRAJECTORY_CRITERIA:
                        ta[dim] = avg
                    else:
                        sq[dim] = avg
                model_summary["step_quality_averages"] = sq
                model_summary["trajectory_averages"] = ta

            if overall_scores:
                model_summary["overall_score_average"] = round(
                    sum(overall_scores) / len(overall_scores), 3
                )

            summary[dataset][model_name] = model_summary

    return summary


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = create_base_parser("Full traj Evaluation")
    return parser.parse_args()

def run_main(args):
    """Main entry point for evaluation."""
    judge_model_id = args.model_id
    results_dir = get_results_dir("dimensions_prompt", judge_model_id)

    # Discover trajectories
    entries = discover_trajectories(
        TRAJ_DATA_DIR,
        filter_dataset=args.dataset,
        filter_model=args.model,
    )

    if not entries:
        print("No trajectory files found. Check paths and filters.")
        return

    # Apply max-files limit per dataset/model
    if args.max_files:
        entries = entries[:args.max_files]

    # Print plan
    counts = Counter((e["dataset"], e["model_name"]) for e in entries)

    max_traj_chars = get_max_trajectory_chars(args.context_tokens)
    context_tokens = args.context_tokens

    print("=" * 70)
    print("Full Trajectory Evaluation Plan")
    print("=" * 70)
    print(f"Judge Model:  {judge_model_id} ({args.model_id})")
    print(f"Context Win:  {context_tokens:,} tokens → "
          f"max trajectory {max_traj_chars:,} chars")
    print(f"Data Dir:     {TRAJ_DATA_DIR}")
    print(f"Results Dir:  {results_dir}")
    print(f"Scoring:      0.0–1.0 (CLEAR-style)")
    print(f"Dimensions:   {len(STEP_QUALITY_CRITERIA)} step-quality + "
              f"{len(TRAJECTORY_CRITERIA)} trajectory = "
              f"{len(ALL_CRITERIA)} total (scored individually)")
    print(f"Overwrite:    {args.overwrite}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Total files:  {len(entries)}")
    print()
    for (ds, model), cnt in sorted(counts.items()):
        print(f"  {ds}/{model}: {cnt} files")
    print("=" * 70)

    # Run evaluation using pipeline infrastructure
    start = time.time()
    parallel_results = run_evaluation_batch(
        entries=entries,
        results_dir=results_dir,
        judge_model_id=judge_model_id,
        provider=args.provider,
        overwrite=args.overwrite,
        concurrency=args.concurrency,
        context_tokens=context_tokens,
        eval_model_params=args.eval_model_params,
    )
    elapsed = time.time() - start

    # Count successful evaluations
    successful = sum(1 for pr in parallel_results if pr.is_success and pr.result is not None)
    total = len(parallel_results)
    
    print(f"\nEvaluation complete: {successful}/{total} trajectories succeeded in {elapsed:.1f}s")

    # Generate and print summary
    if results_dir.exists():
        summary = generate_summary_report(results_dir)

        summary_file = results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")


def main():
    args = parse_args()
    run_main(args)


if __name__ == "__main__":
    main()
