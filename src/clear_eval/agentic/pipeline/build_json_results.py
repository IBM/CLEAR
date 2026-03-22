"""
Build comprehensive JSON results from CLEAR pipeline output.

Creates a structured JSON with all issues mapped to spans, including:
- Per-agent issues with span mappings (issues are unique per agent)
- Input/output pairs with scores and evaluations
- Spans with no issues listed separately
- All span metadata from the preprocessed CSV
"""

import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


def _calc_std(values: List[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _safe_float(val, default=0.0):
    """Safely convert value to float."""
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val, max_len=None):
    """Safely convert value to string with optional truncation."""
    if pd.isna(val):
        return ""
    s = str(val)
    if max_len and len(s) > max_len:
        return s[:max_len]
    return s


def _parse_meta_data(meta_data_str):
    """Parse meta_data JSON string from CSV."""
    if pd.isna(meta_data_str) or not meta_data_str:
        return {}
    try:
        if isinstance(meta_data_str, dict):
            return meta_data_str
        return json.loads(meta_data_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_issues_list(recurring_issues_str):
    """Parse recurring issues string into a list."""
    if pd.isna(recurring_issues_str) or not recurring_issues_str:
        return []

    if isinstance(recurring_issues_str, list):
        return recurring_issues_str

    if isinstance(recurring_issues_str, str) and recurring_issues_str.strip():
        # Try to parse as Python list first
        if recurring_issues_str.startswith('['):
            try:
                return eval(recurring_issues_str)
            except:
                pass
        # Fall back to semicolon-separated
        return [x.strip() for x in recurring_issues_str.split(';') if x.strip()]

    return []


def build_comprehensive_json_results(
    judge_results_dir: str | Path,
    traces_data_dir: str | Path,
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build comprehensive JSON results with all issues mapped to spans.

    Structure:
    - metadata: pipeline info and statistics
    - agents: per-agent data with:
      - issues_catalog: issues discovered for this agent
      - issues: list of issues with their occurrences
      - no_issues: spans that had no issues mapped
    - pass_fail_summary: pass/fail metrics per trace and agent

    Args:
        judge_results_dir: Directory containing agent CLEAR result subdirectories
        traces_data_dir: Directory containing trajectory CSV files
        config_dict: Pipeline configuration containing:
            - success_threshold: Threshold for pass/fail (default: 0.7)
            - pass_criteria: "avg" or "min" (default: "avg")

    Returns:
        Comprehensive results dictionary
    """
    # Extract pass/fail config from config_dict
    success_threshold = config_dict.get("success_threshold", 0.7)
    pass_criteria = config_dict.get("pass_criteria", "avg")
    judge_results_path = Path(judge_results_dir)
    traces_data_path = Path(traces_data_dir)

    # Filter config for JSON output (exclude internal params)
    filtered_config = {k: v for k, v in config_dict.items() if k != "provider_defaults"}

    # Initialize result structure
    results = {
        "metadata": {
            "pipeline_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "config": filtered_config,
            "statistics": {
                "total_traces": 0,
                "total_agents": 0,
                "total_issues_discovered": 0,
                "total_interactions_analyzed": 0,
                "total_interactions_with_issues": 0,
                "total_interactions_no_issues": 0,
            }
        },
        "agents": {},
    }

    # Load trajectory data for additional context (keyed by task_id_step)
    traj_data = {}
    for csv_file in traces_data_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'trace_id' in df.columns and 'task_id' not in df.columns:
                df = df.rename(columns={"trace_id": "task_id"})

            for _, row in df.iterrows():
                task_id = row.get('task_id', row.get('trace_id', ''))
                step = row.get('step_in_trace_general', 0)
                key = f"{task_id}_{step}"
                traj_data[key] = row.to_dict()
        except Exception as e:
            logger.warning(f"Could not load trajectory data {csv_file}: {e}")

    all_traces = set()
    total_with_issues = 0
    total_no_issues = 0
    total_issues_discovered = 0

    # Track per-trace metrics: {trace_id: {"scores": [], "has_issues": []}}
    trace_metrics = defaultdict(lambda: {"scores": [], "has_issues": []})

    # Track per-agent metrics for pass/fail
    agent_all_scores = {}  # {agent_name: [scores]}
    agent_weighted_severity = {}  # {agent_name: sum(freq * severity)}

    # Process each agent's results
    agent_dirs = [d for d in judge_results_path.iterdir() if d.is_dir()]

    for agent_dir in sorted(agent_dirs):
        agent_name = agent_dir.name

        # Find the analysis results CSV
        csv_files = list(agent_dir.glob("analysis_results_*.csv"))
        if not csv_files:
            logger.warning(f"No analysis_results CSV found for agent {agent_name}")
            continue

        # Load shortcoming list from dedup.json files
        shortcoming_files = list(agent_dir.glob("*_dedup.json"))
        agent_shortcomings = []
        if shortcoming_files:
            try:
                with open(shortcoming_files[0], 'r', encoding='utf-8') as f:
                    agent_shortcomings = json.load(f)
                logger.info(f"Loaded {len(agent_shortcomings)} shortcomings for {agent_name}")
            except Exception as e:
                logger.warning(f"Could not load shortcomings for {agent_name}: {e}")

        # Load results from CSV
        results_df = None
        for csv_file in csv_files:
            try:
                results_df = pd.read_csv(csv_file)
                logger.info(f"Loaded {len(results_df)} rows from {csv_file.name}")
                break
            except Exception as e:
                logger.warning(f"Could not load results from {csv_file}: {e}")
                continue

        if results_df is None:
            continue

        # Build agent-specific issue ID mapping
        issue_text_to_id = {}
        for idx, shortcoming in enumerate(agent_shortcomings):
            issue_id = f"issue_{idx + 1}"
            issue_text_to_id[shortcoming] = issue_id

        # Initialize agent entry with its own issues_catalog
        agent_result = {
            "agent_summary": {
                "total_interactions": len(results_df),
                "avg_score": _safe_float(results_df['score'].mean()) if 'score' in results_df.columns else 0.0,
                "interactions_with_issues": 0,
                "interactions_no_issues": 0,
                "issues_count": {}
            },
            "issues_catalog": {issue_text_to_id[s]: s for s in agent_shortcomings},
            "issues": [],
            "no_issues": []
        }

        total_issues_discovered += len(agent_shortcomings)

        # Group results by issue
        issue_occurrences = defaultdict(list)
        no_issue_spans = []

        for _, row in results_df.iterrows():
            task_id = row.get('task_id', row.get('question_id', ''))
            step = row.get('step_in_trace_general', 0)

            # Track trace
            if task_id:
                all_traces.add(str(task_id))

            # Get recurring issues for this row
            row_issues = _parse_issues_list(row.get('recurring_issues_str', ''))

            # Build span data using CSV columns
            traj_key = f"{task_id}_{step}"
            traj_row = traj_data.get(traj_key, {})

            # Get span metadata - prefer from results, fallback to trajectory data
            meta_data = _parse_meta_data(row.get('meta_data', ''))
            if not meta_data:
                meta_data = _parse_meta_data(traj_row.get('meta_data', ''))

            # Get values - prefer from results_df, fallback to traj_data
            model_input = row.get('model_input', traj_row.get('model_input', ''))
            response = row.get('response', traj_row.get('response', ''))
            eval_text = row.get('evaluation_text', '')
            eval_summary = row.get('evaluation_summary', '')
            score_val = row.get('score', 0)

            # Get tool_or_agent from CSV (indicates if this is a tool call or agent response)
            tool_or_agent = row.get('tool_or_agent', traj_row.get('tool_or_agent', 'agent'))

            span_data = {
                "trace_id": str(task_id),
                "span_reference": {
                    "span_id": meta_data.get('span_id', f"{task_id}_span_{step}"),
                    "span_name": meta_data.get('span_name', traj_row.get('Name', agent_name)),
                    "span_type": meta_data.get('span_type', ''),  # Original span type (CHAT_MODEL, LLM, AGENT, etc.)
                    "tool_or_agent": _safe_str(tool_or_agent),  # "tool" or "agent" from preprocessing
                    "parent_span_id": meta_data.get('parent_span_id'),
                    "step_in_trace": int(step) if not pd.isna(step) else 0
                },
                "input_output_pair": {
                    "id": _safe_str(row.get('id', traj_key)),
                    "model_input": _safe_str(model_input, max_len=10000),
                    "response": _safe_str(response, max_len=10000),
                    "score": _safe_float(score_val)
                },
                "evaluation": {
                    "evaluation_text": _safe_str(eval_text, max_len=5000),
                    "evaluation_summary": _safe_str(eval_summary, max_len=2000)
                },
                "span_metadata": {
                    "duration_ms": meta_data.get('duration_ms'),
                    "status": meta_data.get('status'),
                    "model": meta_data.get('model'),
                    "provider": meta_data.get('provider'),
                    "tokens": meta_data.get('tokens', {}),
                    "latency": meta_data.get('latency'),
                    "cost": meta_data.get('cost'),
                }
            }

            # Track trace-level metrics
            trace_id_str = str(task_id)
            trace_metrics[trace_id_str]["scores"].append(_safe_float(score_val))
            trace_metrics[trace_id_str]["has_issues"].append(len(row_issues) > 0)

            if row_issues:
                # This span has issues - add to each issue's occurrences
                for issue_text in row_issues:
                    if issue_text in issue_text_to_id:
                        issue_id = issue_text_to_id[issue_text]
                        issue_occurrences[issue_id].append(span_data)

                        # Update count
                        agent_result["agent_summary"]["issues_count"][issue_id] = \
                            agent_result["agent_summary"]["issues_count"].get(issue_id, 0) + 1

                agent_result["agent_summary"]["interactions_with_issues"] += 1
                total_with_issues += 1
            else:
                # No issues for this span
                no_issue_spans.append(span_data)
                agent_result["agent_summary"]["interactions_no_issues"] += 1
                total_no_issues += 1

        # Build issues list for this agent
        total_agent_spans = len(results_df)
        for issue_id, occurrences in issue_occurrences.items():
            occurrence_count = len(occurrences)

            # Calculate frequency: num occurrences / total agent spans
            frequency = occurrence_count / total_agent_spans if total_agent_spans > 0 else 0.0

            # Calculate severity: 1 - average score for spans mapped to this issue
            scores = [occ["input_output_pair"]["score"] for occ in occurrences]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            severity = 1.0 - avg_score

            agent_result["issues"].append({
                "issue_id": issue_id,
                "issue_text": agent_result["issues_catalog"].get(issue_id, ""),
                "occurrence_count": occurrence_count,
                "frequency": round(frequency, 4),
                "severity": round(severity, 4),
                "occurrences": occurrences
            })

        # Add spans with no issues
        agent_result["no_issues"] = no_issue_spans

        # Store agent scores for pass/fail calculation
        all_scores = [_safe_float(row.get('score', 0)) for _, row in results_df.iterrows()]
        agent_all_scores[agent_name] = all_scores

        # Calculate weighted severity: sum(frequency * severity) for all issues
        weighted_sev = sum(
            issue.get("frequency", 0) * issue.get("severity", 0)
            for issue in agent_result["issues"]
        )
        agent_weighted_severity[agent_name] = weighted_sev

        results["agents"][agent_name] = agent_result
        results["metadata"]["statistics"]["total_interactions_analyzed"] += len(results_df)

    # Update statistics
    results["metadata"]["statistics"]["total_traces"] = len(all_traces)
    results["metadata"]["statistics"]["total_agents"] = len(results["agents"])
    results["metadata"]["statistics"]["total_issues_discovered"] = total_issues_discovered
    results["metadata"]["statistics"]["total_interactions_with_issues"] = total_with_issues
    results["metadata"]["statistics"]["total_interactions_no_issues"] = total_no_issues

    # Build pass/fail summary
    pass_fail_summary = {
        "threshold": success_threshold,
        "pass_criteria": pass_criteria,
        "traces": {},
        "agents": {}
    }

    # Per-trace metrics
    for trace_id, metrics in trace_metrics.items():
        scores = metrics["scores"]
        has_issues = metrics["has_issues"]

        if not scores:
            continue

        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        issue_free_count = sum(1 for h in has_issues if not h)
        issue_free_ratio = issue_free_count / len(has_issues) if has_issues else 1.0

        # Determine pass based on criteria
        check_score = avg_score if pass_criteria == "avg" else min_score
        passed = check_score >= success_threshold

        pass_fail_summary["traces"][trace_id] = {
            "pass": passed,
            "avg_score": round(avg_score, 4),
            "min_score": round(min_score, 4),
            "issue_free_ratio": round(issue_free_ratio, 4)
        }

    # Per-agent metrics
    for agent_name, scores in agent_all_scores.items():
        if not scores:
            continue

        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        std = _calc_std(scores)
        consistency = max(0.0, 1.0 - std)  # Higher consistency = lower std
        weighted_severity = agent_weighted_severity.get(agent_name, 0.0)

        # Get issue_free_ratio from agent summary
        agent_data = results["agents"].get(agent_name, {})
        summary = agent_data.get("agent_summary", {})
        total = summary.get("total_interactions", 0)
        no_issues = summary.get("interactions_no_issues", 0)
        issue_free_ratio = no_issues / total if total > 0 else 1.0

        # Determine pass based on criteria
        check_score = avg_score if pass_criteria == "avg" else min_score
        passed = check_score >= success_threshold

        pass_fail_summary["agents"][agent_name] = {
            "pass": passed,
            "avg_score": round(avg_score, 4),
            "min_score": round(min_score, 4),
            "issue_free_ratio": round(issue_free_ratio, 4),
            "consistency": round(consistency, 4),
       #     "weighted_severity": round(weighted_severity, 4)
        }

    results["pass_fail_summary"] = pass_fail_summary

    return results


def save_comprehensive_json_results(
    judge_results_dir: str | Path,
    traces_data_dir: str | Path,
    config_dict: Dict[str, Any],
    output_dir: str | Path = None,
    output_filename: str = "clear_results.json"
) -> Path:
    """
    Build and save comprehensive JSON results.

    Args:
        judge_results_dir: Directory containing agent CLEAR result subdirectories
        traces_data_dir: Directory containing trajectory CSV files
        config_dict: Pipeline configuration containing success_threshold and pass_criteria
        output_dir: Directory to save the JSON output (defaults to judge_results_dir)
        output_filename: Name of the output JSON file

    Returns:
        Path to the saved JSON file
    """
    logger.info("=" * 80)
    logger.info("BUILDING COMPREHENSIVE JSON RESULTS")
    logger.info("=" * 80)

    results = build_comprehensive_json_results(
        judge_results_dir=judge_results_dir,
        traces_data_dir=traces_data_dir,
        config_dict=config_dict
    )

    if not output_dir:
        output_dir = judge_results_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_output_path = output_path / output_filename

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    stats = results["metadata"]["statistics"]
    logger.info(f"Saved comprehensive results to: {json_output_path}")
    logger.info(f"  Total agents: {stats['total_agents']}")
    logger.info(f"  Total traces: {stats['total_traces']}")
    logger.info(f"  Total issues discovered: {stats['total_issues_discovered']}")
    logger.info(f"  Total interactions analyzed: {stats['total_interactions_analyzed']}")
    logger.info(f"  Interactions with issues: {stats['total_interactions_with_issues']}")
    logger.info(f"  Interactions with no issues: {stats['total_interactions_no_issues']}")
    logger.info("=" * 80)

    return json_output_path


# Path to agentic pipeline default config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTIC_DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "setup", "default_config.yaml")


def build_cli_overrides(args) -> dict:
    """
    Build CLI overrides dictionary from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of CLI overrides ready for load_config()
    """
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None and key != 'agentic_config_path'
    }


def main():
    """CLI entry point for building JSON results."""
    import argparse
    from clear_eval.pipeline.config_loader import load_config

    parser = argparse.ArgumentParser(
        description="Build comprehensive JSON results from CLEAR pipeline output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Precedence (lowest to highest):
  1. Default config (setup/default_config.yaml)
  2. User config file (--agentic-config-path)
  3. CLI arguments

Examples:
  # Using default config only
  python -m clear_eval.agentic.pipeline.build_json_results \\
      --judge-results-dir /path/to/clear_results/<judge-model> \\
      --traces-data-dir /path/to/traces_data

  # With user config file
  python -m clear_eval.agentic.pipeline.build_json_results \\
      --agentic-config-path my_config.yaml \\
      --judge-results-dir /path/to/clear_results/<judge-model> \\
      --traces-data-dir /path/to/traces_data

  # With CLI overrides
  python -m clear_eval.agentic.pipeline.build_json_results \\
      --judge-results-dir /path/to/clear_results/<judge-model> \\
      --traces-data-dir /path/to/traces_data \\
      --success-threshold 0.8 \\
      --pass-criteria min
        """
    )
    parser.add_argument(
        "--agentic-config-path",
        type=str,
        default=None,
        help="Path to config file (JSON or YAML) that overrides defaults"
    )
    parser.add_argument(
        "--judge-results-dir",
        type=str,
        required=True,
        help="Directory containing agent CLEAR result subdirectories"
    )
    parser.add_argument(
        "--traces-data-dir",
        type=str,
        required=True,
        help="Directory containing trajectory CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the JSON output (defaults to judge-results-dir)"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="clear_results.json",
        help="Name of the output JSON file (default: clear_results.json)"
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Threshold for pass/fail determination (default: 0.7)"
    )
    parser.add_argument(
        "--pass-criteria",
        choices=['avg', 'min'],
        default=None,
        help="Score type for pass/fail: 'avg' or 'min' (default: avg)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Build CLI overrides from non-None args
    cli_overrides = build_cli_overrides(args)

    # Load configuration with precedence: default -> user config -> CLI overrides
    config_dict = load_config(
        AGENTIC_DEFAULT_CONFIG_PATH,
        args.agentic_config_path,
        **cli_overrides
    )

    save_comprehensive_json_results(
        judge_results_dir=args.judge_results_dir,
        traces_data_dir=args.traces_data_dir,
        config_dict=config_dict,
        output_dir=args.output_dir,
        output_filename=args.output_filename
    )


if __name__ == "__main__":
    main()
