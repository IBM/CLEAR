#!/usr/bin/env python3
"""
Run CLEAR Analysis on Full-Trajectory Evaluation Results
==========================================================

Takes the evaluation results produced by ``run_full_traj_evaluation.py``
(issues / detailed_feedback) or ``run_task_success_evaluation.py``
(failure_root_cause) and feeds them through the CLEAR analysis pipeline
to discover common themes, shortcomings, and patterns.

Two analysis sources (--source flag):
    issues      – Uses ``detailed_feedback`` from full-traj eval results
                  (results_{judge}/{method}/) as the text to analyse.
    root_cause  – Uses ``failure_root_cause`` from task-success eval
                  results (success_results_{judge}/) for failed
                  trajectories only.

Supported judge models for the *source* results (--eval-judge):
    oss20b   - openai/gpt-oss-20b
    oss120b  - openai/gpt-oss-120b
    deepseek - deepseek-ai/DeepSeek-V3.2

CLEAR configuration:
    The CLEAR pipeline needs its own eval model to synthesise shortcomings.
    Configure via CLEAR_CONFIG at the top of this file or via CLI flags.

Results are saved to:
    full_traj_evaluation/clear_analysis_{source}_{eval_judge}/
        {dataset}/{model}/

Usage examples:
    python run_clear_on_eval_results.py --source issues
    python run_clear_on_eval_results.py --source root_cause
    python run_clear_on_eval_results.py --source issues --eval-judge oss120b --method dimensions_prompt
    python run_clear_on_eval_results.py --source root_cause --dataset CUGA --model full
    python run_clear_on_eval_results.py --source issues --overwrite
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from clear_eval.agentic.full_traj_evaluation.dataset_base import RESULTS_DIR, get_results_dir
from clear_eval.analysis_runner import run_clear_eval_aggregation

# Add project src to path so we can import clear_eval
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clear_eval.analysis_runner import run_clear_eval_analysis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# # Judge models used when *producing* the evaluation results
# EVAL_JUDGE_MODELS = {
#     "oss20b": ("openai/gpt-oss-20b", "gpt-oss-20b"),
#     "oss120b": ("openai/gpt-oss-120b", "gpt-oss-120b"),
#     "deepseek": ("deepseek-ai/DeepSeek-V3.2", "deepseek-v3-2"),
# }
# DEFAULT_EVAL_JUDGE = "oss120b"

# Evaluation methods from run_full_traj_evaluation.py
METHODS = [
    "dimensions_prompt",
    "full_trace_prompt",
    "full_trace_prompt_with_step",
    "full_trace_prompt_issue",
]
DEFAULT_METHOD = "dimensions_prompt"

# CLEAR model configuration (used for the CLEAR analysis itself)
CLEAR_CONFIG_DICT = {
    "openai/gpt-oss-120b": {
    "gen_model_name": None,#"ibm-granite/granite-3.3-8b-instruct",
    "eval_model_name": "openai/gpt-oss-120b",
    "provider": "watsonx",
    "eval_model_params": {"include_reasoning": True, "reasoning_effort": "medium", "max_tokens":32378}
    },
"Azure/gpt-5-2025-08-07": {
    "gen_model_name": None,#"ibm-granite/granite-3.3-8b-instruct",
    "eval_model_name": "Azure/gpt-5-2025-08-07",
    "provider": "openai",
    "eval_model_params": {"max_tokens":32378}
}

#    "api_key": "11d4b77f45336d366ca592d6ba1edfff",
#    "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b/v1",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result discovery helpers
# ---------------------------------------------------------------------------


def get_full_traj_results_dir(eval_judge_key: str, method: str) -> Path:
    """Return the results directory for full-traj evaluation."""#
#    return SCRIPT_DIR / f"results_{eval_judge_key}" / method
    return get_results_dir(method, eval_judge_key)

def get_success_results_dir(eval_judge_key: str) -> Path:
    """Return the results directory for task-success evaluation."""
    return get_results_dir("success_results", eval_judge_key)


def discover_eval_jsons(base_dir: Path, suffix: str = "_eval.json",
                        filter_dataset: str | None = None,
                        filter_model: str | None = None) -> list[Path]:
    """Discover all evaluation JSON files under base_dir/{dataset}/{model}/."""
    results = []
    if not base_dir.exists():
        return results
    for dataset_dir in sorted(base_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if filter_dataset and dataset_dir.name != filter_dataset:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            if filter_model and model_dir.name != filter_model:
                continue
            for json_file in sorted(model_dir.glob(f"*{suffix}")):
                results.append(json_file)
    return results


# ---------------------------------------------------------------------------
# Extraction: issues from full_traj_eval
# ---------------------------------------------------------------------------


def extract_issues_to_dataframe(
    eval_judge_key: str,
    method: str,
    filter_dataset: str | None = None,
    filter_model: str | None = None,
) -> pd.DataFrame:
    """Extract detailed_feedback (issues) from full-traj eval results into a
    DataFrame suitable for CLEAR analysis.

    Columns produced:
        id, dataset, model_name, trajectory_name, overall_score,
        model_input, response
    """
    results_dir = get_full_traj_results_dir(eval_judge_key, method)
    json_files = discover_eval_jsons(
        results_dir, suffix="_eval.json",
        filter_dataset=filter_dataset, filter_model=filter_model,
    )

    if not json_files:
        logger.warning("No eval JSON files found in %s", results_dir)
        return pd.DataFrame()

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load %s: %s", jf, e)
            continue

        feedback = data.get("detailed_feedback")
        if not feedback or not isinstance(feedback, str) or not feedback.strip():
            continue

        traj_name = data.get("trajectory_name", jf.stem)
        dataset = data.get("dataset", jf.parent.parent.name)
        model_name = data.get("model_name", jf.parent.name)
        overall_score = data.get("overall_score")

        model_input = (
            f"Trajectory: {traj_name}\n"
            f"Dataset: {dataset}\n"
            f"Model: {model_name}\n"
            #f"Overall Score: {overall_score}"
        )
        rows.append({
            "id": f"{dataset}__{model_name}__{traj_name}",
            "dataset": dataset,
            "model_name": model_name,
            "trajectory_name": traj_name,
            "overall_score": overall_score,
            "model_input": model_input,
            "response": "",
            "evaluation_summary": feedback,
            "score": overall_score,
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Extracted %d issue records from %s (%s)",
        len(df), results_dir, method,
    )
    return df


# ---------------------------------------------------------------------------
# Extraction: root_cause from success_eval
# ---------------------------------------------------------------------------


def extract_root_causes_to_dataframe(
    eval_judge_key: str,
    filter_dataset: str | None = None,
    filter_model: str | None = None,
) -> pd.DataFrame:
    """Extract failure_root_cause from task-success eval results into a
    DataFrame suitable for CLEAR analysis (only failed trajectories).

    Columns produced:
        id, dataset, model_name, trajectory_name, task_objective,
        consideration, model_input, response
    """
    results_dir = get_success_results_dir(eval_judge_key)
    json_files = discover_eval_jsons(
        results_dir, suffix="_success.json",
        filter_dataset=filter_dataset, filter_model=filter_model,
    )

    if not json_files:
        logger.warning("No success JSON files found in %s", results_dir)
        return pd.DataFrame()

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load %s: %s", jf, e)
            continue

        score = data.get("success")
        if score == 0:
            root_cause = data.get("failure_root_cause")
            if not root_cause or not isinstance(root_cause, str) or not root_cause.strip():
                continue
            consideration = data.get("consideration", "")
            feedback = root_cause
            if consideration:
                feedback = (
                    f"Consideration: {consideration}\n\n"
                    f"Root Cause: {root_cause}"
                )
        else:
            feedback = ""
            consideration = ""

        traj_name = data.get("trajectory_name", jf.stem)
        dataset = data.get("dataset", jf.parent.parent.name)
        model_name = data.get("model_name", jf.parent.name)
        task_objective = data.get("task_objective", "")

        model_input = (
            f"Task Objective: {task_objective}\n"
            f"Trajectory: {traj_name}\n"
            f"Dataset: {dataset}\n"
            f"Model: {model_name}"
        )

        rows.append({
            "id": f"{dataset}__{model_name}__{traj_name}",
            "dataset": dataset,
            "model_name": model_name,
            "trajectory_name": traj_name,
            "task_objective": task_objective or "",
            "consideration": consideration or "",
            "model_input": model_input,
            "response": "",
            "evaluation_summary": feedback,
            "score": score
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Extracted %d root-cause records (failures only) from %s",
        len(df), results_dir,
    )
    return df


# ---------------------------------------------------------------------------
# CLEAR runner
# ---------------------------------------------------------------------------


def get_judge_model_folder_name(eval_model_name: str) -> str:
    """Convert eval model name to a clean folder name."""
    if "/" in eval_model_name:
        judge_name = eval_model_name.split("/")[-1]
    else:
        judge_name = eval_model_name
    return judge_name.replace(".", "-").replace("_", "-").lower()


def run_clear_analysis_on_df(
    df: pd.DataFrame,
    eval_model_name: str,
    source: str,
    eval_judge_key: str,
    method: str | None,
    overwrite: bool = False,
    clear_config: dict = None):
    """Save the DataFrame to a temp CSV, then run CLEAR analysis on it.

    Groups by (dataset, model_name) so that each group gets its own CLEAR
    analysis run, mirroring the structure of the existing run_clear scripts.
    """
    if df.empty:
        print("No records to analyse. Check filters and result directories.")
        return

    clear_judge_folder = get_judge_model_folder_name(
        eval_model_name=eval_model_name
    )

    method_suffix = f"_{method}" if method else ""
    output_base = (
        RESULTS_DIR
        / f"clear_analysis_{source}/{eval_judge_key}/{method_suffix}"
    )

    grouped = df.groupby(["dataset", "model_name"])

    total_groups = len(grouped)
    processed = 0
    skipped = 0
    errors = 0

    print("=" * 70)
    print(f"CLEAR Analysis Plan — source={source}")
    print("=" * 70)
    print(f"  Source judge:     {eval_judge_key}")
    if method:
        print(f"  Eval method:      {method}")
    print(f"  CLEAR eval model: {eval_model_name}")
    print(f"  Total records:    {len(df)}")
    print(f"  Groups (dataset/model): {total_groups}")
    print(f"  Output base:      {output_base}")
    print(f"  Overwrite:        {overwrite}")
    for (ds, mdl), grp in grouped:
        print(f"    {ds}/{mdl}: {len(grp)} records")
    print("=" * 70)

    # evaluation_criteria_dict = {
    #     "Specificity": (
    #         "The identified issue or root cause is specific and clearly "
    #         "describes a concrete problem rather than a vague complaint."
    #     ),
    #     "Actionability": (
    #         "The feedback provides actionable insights that could guide "
    #         "improvements in the agent's behaviour."
    #     ),
    #     "Accuracy": (
    #         "The issue or root cause accurately reflects a real problem "
    #         "in the agent's trajectory, not a hallucinated or fabricated one."
    #     ),
    #     "Completeness": (
    #         "The feedback captures the full scope of the problem without "
    #         "omitting critical aspects."
    #     ),
    #     "Relevance": (
    #         "The issue is relevant to the agent's task and execution, "
    #         "not tangential or irrelevant."
    #     ),
    # }

    for (dataset, model_name), group_df in grouped:
        clear_dir = output_base / dataset / model_name / clear_judge_folder
        output_dir = clear_dir / "output"
        group_label = f"{dataset}/{model_name}"

        if output_dir.exists() and not overwrite:
            print(f"  Skipping (exists): {group_label} → {output_dir}")
            skipped += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = clear_dir / f"_input_{source}.csv"
        group_df.to_csv(csv_path, index=False)

        print(f"\n  Running CLEAR on {group_label} ({len(group_df)} records)...")

        input_columns = ["id", "dataset", "model_name", "trajectory_name", "traj_score"]
        # if source == "issues":
        #     input_columns.append("overall_score")
        if source == "root_cause":
            input_columns.extend(["task_objective", "consideration"])

        try:
            analysis_kwargs = {
                "provider": clear_config["provider"],
                "data_path": str(csv_path),
                "gen_model_name": clear_config["gen_model_name"],
                "eval_model_name": eval_model_name,
                "output_dir": str(output_dir),
                "perform_generation": False,
                "input_columns": input_columns,
                #"evaluation_criteria": evaluation_criteria_dict,
                "agent_mode":True,
                "eval_model_params": clear_config["eval_model_params"],
            }

            run_clear_eval_aggregation(**analysis_kwargs)

            print(f"  Completed: {group_label}")
            processed += 1

        except Exception as exc:
            logger.error("CLEAR failed for %s: %s", group_label, exc)
            import traceback
            traceback.print_exc()
            errors += 1

    print("\n" + "=" * 70)
    print("CLEAR Analysis Summary")
    print("=" * 70)
    print(f"  Groups total:   {total_groups}")
    print(f"  Processed:      {processed}")
    print(f"  Skipped:        {skipped}")
    if errors:
        print(f"  Errors:         {errors}")
    print(f"  Results saved to: {output_base}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = create_base_parser(
        "Run CLEAR analysis on evaluation results from "
        "full-traj or task-success evaluations"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="issues",
        choices=["issues", "root_cause"],
        help=(
            "What to apply CLEAR to. "
            "'issues' → detailed_feedback from full-traj eval results. "
            "'root_cause' → failure_root_cause from task-success eval results."
        ),
    )
    # parser.add_argument(
    #     "--eval-judge",
    #     type=str,
    #     default="openai/gpt-oss-120b",
    #     #choices=list(EVAL_JUDGE_MODELS.keys()),
    #     help=(
    #         "Judge model that produced the evaluation results "
    #         f"(default: openai/gpt-oss-120b)"
    #     ),
    # )
    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        choices=METHODS,
        help=(
            f"Evaluation method for issues source (default: {DEFAULT_METHOD}). "
            "Ignored when --source is root_cause."
        ),
    )
    # parser.add_argument(
    #     "--overwrite",
    #     action="store_true",
    #     help="Re-run CLEAR analysis even if output directory exists",
    # )
    return parser.parse_args()


def main():
    args = parse_args()
    CLEAR_CONFIG = CLEAR_CONFIG_DICT[args.model_id]
    print("=" * 70)
    print("CLEAR on Evaluation Results")
    print("=" * 70)
    print(f"  Source:       {args.source}")
    #print(f"  Eval Judge:   {args.eval_judge}")
    if args.source == "issues":
        print(f"  Method:       {args.method}")
    print(f"  Dataset:      {args.dataset or 'all'}")
    print(f"  Model:        {args.model or 'all'}")
    print(f"  CLEAR model:  {CLEAR_CONFIG['eval_model_name']}")
    print("=" * 70)
    eval_judge_key = args.model_id.replace("/", "_").replace(":", "_").replace("-", "_")
    if args.source == "issues":
        df = extract_issues_to_dataframe(
            eval_judge_key=eval_judge_key,
            method=args.method,
            filter_dataset=args.dataset,
            filter_model=args.model,
        )
        run_clear_analysis_on_df(
            df,
            eval_model_name=args.model_id,
            source="issues",
            eval_judge_key=eval_judge_key,
            method=args.method,
            overwrite=args.overwrite,
            clear_config=CLEAR_CONFIG,
        )
    elif args.source == "root_cause":
        df = extract_root_causes_to_dataframe(
            eval_judge_key=eval_judge_key,
            filter_dataset=args.dataset,
            filter_model=args.model,
        )
        run_clear_analysis_on_df(
            df,
            eval_model_name = args.model_id,
            source="root_cause",
            eval_judge_key=eval_judge_key,
            method=None,
            overwrite=args.overwrite,
            clear_config=CLEAR_CONFIG,
        )


if __name__ == "__main__":
    main()
