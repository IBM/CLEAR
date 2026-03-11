"""
Build comprehensive JSON results from CLEAR pipeline output.

Creates a structured JSON with all issues mapped to spans, including:
- Per-agent issues with span mappings
- Input/output pairs with scores and evaluations
- Trace-level summaries
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


def build_comprehensive_json_results(
    judge_results_dir: str | Path,
    traces_data_dir: str | Path,
    raw_traces_dir: Optional[str | Path] = None,
    config_dict: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Build comprehensive JSON results with all issues mapped to spans.

    Structure:
    - metadata: pipeline info and statistics
    - issues_catalog: all discovered issues across agents
    - agents: per-agent issues with span mappings
    - traces_index: trace-level summary with spans that have issues

    Args:
        judge_results_dir: Directory containing agent CLEAR result subdirectories
        traces_data_dir: Directory containing trajectory CSV files
        raw_traces_dir: Optional directory containing original trace JSON files
        config_dict: Optional pipeline configuration

    Returns:
        Comprehensive results dictionary
    """
    judge_results_path = Path(judge_results_dir)
    traces_data_path = Path(traces_data_dir)
    raw_traces_path = Path(raw_traces_dir) if raw_traces_dir else None

    # Initialize result structure
    results = {
        "metadata": {
            "pipeline_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "config": config_dict or {},
            "statistics": {
                "total_traces": 0,
                "total_agents": 0,
                "total_issues_discovered": 0,
                "total_interactions_analyzed": 0,
            }
        },
        "issues_catalog": {},
        "agents": {},
        "traces_index": {}
    }

    # Load raw traces for span context (if available)
    raw_traces_data = {}
    if raw_traces_path and raw_traces_path.exists():
        for trace_file in raw_traces_path.glob("*.json"):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)
                    trace_id = trace_data.get("trace_id", trace_file.stem)
                    raw_traces_data[trace_id] = trace_data
            except Exception as e:
                logger.warning(f"Could not load raw trace {trace_file}: {e}")

    # Load trajectory data for input/output context
    traj_data = {}
    for csv_file in traces_data_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Rename trace_id to task_id if needed
            if 'trace_id' in df.columns and 'task_id' not in df.columns:
                df = df.rename(columns={"trace_id": "task_id"})

            for _, row in df.iterrows():
                task_id = row.get('task_id', row.get('trace_id', ''))
                step = row.get('step_in_trace_general', 0)
                key = f"{task_id}_{step}"
                traj_data[key] = row.to_dict()
        except Exception as e:
            logger.warning(f"Could not load trajectory data {csv_file}: {e}")

    # Collect all issues and build agent results
    global_issue_id = 0
    issue_text_to_id = {}
    all_traces = set()

    # Process each agent's results
    agent_dirs = [d for d in judge_results_path.iterdir() if d.is_dir()]

    for agent_dir in sorted(agent_dirs):
        agent_name = agent_dir.name

        # Find the analysis results CSV (not zip)
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

        # Initialize agent entry
        agent_result = {
            "agent_summary": {
                "total_interactions": len(results_df),
                "avg_score": float(results_df['score'].mean()) if 'score' in results_df.columns else 0.0,
                "issues_count": {}
            },
            "issues": []
        }

        # Map shortcomings to global issue IDs
        for shortcoming in agent_shortcomings:
            if shortcoming not in issue_text_to_id:
                global_issue_id += 1
                issue_id = f"issue_{global_issue_id}"
                issue_text_to_id[shortcoming] = issue_id
                results["issues_catalog"][issue_id] = shortcoming

        # Group results by issue
        issue_occurrences = defaultdict(list)

        for _, row in results_df.iterrows():
            task_id = row.get('task_id', row.get('question_id', ''))
            step = row.get('step_in_trace_general', 0)

            # Track trace
            if task_id:
                all_traces.add(str(task_id))

            # Get recurring issues for this row
            recurring_issues_str = row.get('recurring_issues_str', '')
            if pd.isna(recurring_issues_str) or not recurring_issues_str:
                recurring_issues_str = ''

            # Parse recurring issues - handle both string and list formats
            row_issues = []
            if isinstance(recurring_issues_str, str) and recurring_issues_str.strip():
                # Try to parse as Python list first
                if recurring_issues_str.startswith('['):
                    try:
                        row_issues = eval(recurring_issues_str)
                    except:
                        row_issues = [x.strip() for x in recurring_issues_str.split(';') if x.strip()]
                else:
                    row_issues = [x.strip() for x in recurring_issues_str.split(';') if x.strip()]
            elif isinstance(recurring_issues_str, list):
                row_issues = recurring_issues_str

            # Build span occurrence data
            traj_key = f"{task_id}_{step}"
            traj_row = traj_data.get(traj_key, {})

            # Get span context from raw trace
            span_context = {}
            str_task_id = str(task_id)
            if str_task_id in raw_traces_data:
                trace = raw_traces_data[str_task_id]
                spans = trace.get('spans', [])
                # Try to find matching span by step index
                step_int = int(step) if not pd.isna(step) else 0
                if step_int < len(spans):
                    span = spans[step_int]
                    span_context = {
                        "span_id": span.get('span_id'),
                        "span_name": span.get('name'),
                        "span_type": span.get('attributes', {}).get('mlflow.spanType',
                                     span.get('span_type', '')),
                        "parent_span_id": span.get('parent_span_id'),
                        "duration_ms": span.get('duration_ms'),
                        "token_usage": span.get('attributes', {}).get('mlflow.chat.tokenUsage', {})
                    }

            # Safely get values with truncation
            model_input = row.get('model_input', traj_row.get('model_input', ''))
            response = row.get('response', traj_row.get('response', ''))
            eval_text = row.get('evaluation_text', '')
            eval_summary = row.get('evaluation_summary', '')
            score_val = row.get('score', 0)

            occurrence_data = {
                "trace_id": str(task_id),
                "span_reference": {
                    "span_id": span_context.get('span_id', f"{task_id}_span_{step}"),
                    "span_name": span_context.get('span_name', traj_row.get('Name', agent_name)),
                    "span_type": span_context.get('span_type', traj_row.get('tool_or_agent', '')),
                    "step_in_trace": int(step) if not pd.isna(step) else 0
                },
                "input_output_pair": {
                    "id": str(row.get('id', traj_key)),
                    "model_input": str(model_input)[:10000] if not pd.isna(model_input) else "",
                    "response": str(response)[:10000] if not pd.isna(response) else "",
                    "score": float(score_val) if not pd.isna(score_val) else 0.0
                },
                "evaluation": {
                    "evaluation_text": str(eval_text)[:5000] if not pd.isna(eval_text) else "",
                    "evaluation_summary": str(eval_summary)[:2000] if not pd.isna(eval_summary) else ""
                },
                "span_context": span_context if span_context else None
            }

            # Add to each issue this row is mapped to
            for issue_text in row_issues:
                if issue_text in issue_text_to_id:
                    issue_id = issue_text_to_id[issue_text]
                    issue_occurrences[issue_id].append(occurrence_data)

                    # Update count
                    agent_result["agent_summary"]["issues_count"][issue_id] = \
                        agent_result["agent_summary"]["issues_count"].get(issue_id, 0) + 1

        # Build issues list for this agent
        for issue_id, occurrences in issue_occurrences.items():
            agent_result["issues"].append({
                "issue_id": issue_id,
                "issue_text": results["issues_catalog"].get(issue_id, ""),
                "occurrences": occurrences
            })

        results["agents"][agent_name] = agent_result
        results["metadata"]["statistics"]["total_interactions_analyzed"] += len(results_df)

    # Build traces index
    for trace_id in all_traces:
        spans_with_issues = set()
        agents_involved = set()

        for agent_name, agent_data in results["agents"].items():
            for issue in agent_data.get("issues", []):
                for occ in issue.get("occurrences", []):
                    if occ["trace_id"] == trace_id:
                        spans_with_issues.add(occ["span_reference"]["span_id"])
                        agents_involved.add(agent_name)

        trace_info = {
            "spans_with_issues": list(spans_with_issues),
            "agents_involved": list(agents_involved),
            "issue_count": len(spans_with_issues)
        }

        # Add extra context from raw trace
        if trace_id in raw_traces_data:
            raw_trace = raw_traces_data[trace_id]
            trace_info["execution_time_ms"] = raw_trace.get("execution_time_ms")
            trace_info["state"] = raw_trace.get("state")
            trace_info["total_spans"] = len(raw_trace.get("spans", []))

        results["traces_index"][trace_id] = trace_info

    # Update statistics
    results["metadata"]["statistics"]["total_traces"] = len(all_traces)
    results["metadata"]["statistics"]["total_agents"] = len(results["agents"])
    results["metadata"]["statistics"]["total_issues_discovered"] = len(results["issues_catalog"])

    return results


def save_comprehensive_json_results(
    judge_results_dir: str | Path,
    traces_data_dir: str | Path,
    output_dir: str | Path = None,
    raw_traces_dir: Optional[str | Path] = None,
    config_dict: Optional[Dict] = None,
    output_filename: str = "clear_results.json"
) -> Path:
    """
    Build and save comprehensive JSON results.

    Args:
        judge_results_dir: Directory containing agent CLEAR result subdirectories
        traces_data_dir: Directory containing trajectory CSV files
        output_dir: Directory to save the JSON output
        raw_traces_dir: Optional directory containing original trace JSON files
        config_dict: Optional pipeline configuration
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
        raw_traces_dir=raw_traces_dir,
        config_dict=config_dict
    )

    if not output_dir:
        output_dir = judge_results_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_output_path = output_path / output_filename

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved comprehensive results to: {json_output_path}")
    logger.info(f"  Total agents: {results['metadata']['statistics']['total_agents']}")
    logger.info(f"  Total traces: {results['metadata']['statistics']['total_traces']}")
    logger.info(f"  Total issues discovered: {results['metadata']['statistics']['total_issues_discovered']}")
    logger.info(f"  Total interactions analyzed: {results['metadata']['statistics']['total_interactions_analyzed']}")
    logger.info("=" * 80)

    return json_output_path


def main():
    """CLI entry point for building JSON results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build comprehensive JSON results from CLEAR pipeline output"
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
        required=True,
        help="Directory to save the JSON output"
    )
    parser.add_argument(
        "--raw-traces-dir",
        type=str,
        default=None,
        help="Optional directory containing original trace JSON files"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="clear_results.json",
        help="Name of the output JSON file (default: clear_results.json)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    save_comprehensive_json_results(
        judge_results_dir=args.judge_results_dir,
        traces_data_dir=args.traces_data_dir,
        output_dir=args.output_dir,
        raw_traces_dir=args.raw_traces_dir,
        output_filename=args.output_filename
    )


if __name__ == "__main__":
    main()
