#!/usr/bin/env python3
"""
Unified trace preprocessing pipeline.

Converts trace JSON files into trajectory CSV files for evaluation.
Supports multiple agent frameworks (langgraph, crewai) and observability platforms (langfuse, mlflow).

Usage:
    python preprocess_traces.py --input_dir /path/to/traces --output_dir /path/to/output \
        --agent_framework langgraph --observability_framework langfuse
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from clear_eval.agentic.pipeline.preprocess_traces.process_langfuse_traces import (
    extract_llm_calls_from_langgraph_trace,
    extract_llm_calls_from_crewai_trace,
)
from clear_eval.agentic.pipeline.preprocess_traces.process_mlflow_traces import (
    extract_llm_calls_from_mlflow_trace,
)

logger = logging.getLogger(__name__)


# CSV fieldnames for output (unified schema)
CSV_FIELDNAMES = [
    'id', 'Name', 'intent', 'task_id', 'step_in_trace_general',
    'llm_call_index', 'model_input', 'response', 'tool_or_agent',
    'api_spec', 'meta_data', 'traj_score'
]


def get_extractor(
    agent_framework: str,
    observability_framework: str,
    separate_tools: bool = True
):
    """
    Get the appropriate extractor function based on agent framework and observability platform.

    Args:
        agent_framework: The agent framework used (langgraph, crewai)
        observability_framework: The observability platform (langfuse, mlflow)
        separate_tools: Whether to emit separate rows for tools vs agent responses

    Returns:
        Extractor function that takes (json_data, file_name) and returns list of row dicts

    Raises:
        ValueError: If the combination is not supported
    """
    key = (agent_framework.lower(), observability_framework.lower())

    if key == ("langgraph", "langfuse"):
        return lambda trace, file_name: extract_llm_calls_from_langgraph_trace(
            trace, file_name, separate_tools=separate_tools
        )
    elif key == ("crewai", "langfuse"):
        return lambda trace, file_name: extract_llm_calls_from_crewai_trace(
            trace, file_name, separate_tools=separate_tools
        )
    elif key == ("langgraph", "mlflow"):
        return lambda trace, file_name: extract_llm_calls_from_mlflow_trace(
            trace, file_name, separate_tools=separate_tools
        )
    else:
        supported = ["langgraph+langfuse", "crewai+langfuse", "langgraph+mlflow"]
        raise ValueError(
            f"Unsupported combination: {agent_framework}+{observability_framework}. "
            f"Supported combinations: {', '.join(supported)}"
        )


def coerce_to_trace_list(json_data: Any) -> List[Any]:
    """
    Coerce input JSON data to a list of individual traces.

    Handles:
        - Single trace dict
        - List of trace dicts
        - Wrapper dict with 'traces' key

    Args:
        json_data: Parsed JSON data from file

    Returns:
        List of individual trace objects
    """
    if isinstance(json_data, list):
        return json_data

    if isinstance(json_data, dict):
        # Check for wrapper format {"traces": [...]}
        if 'traces' in json_data and isinstance(json_data['traces'], list):
            return json_data['traces']
        # Single trace dict
        return [json_data]

    return []


def process_single_trace(
    trace_data: Any,
    extractor,
    file_name: str,
    trace_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Process a single trace using the given extractor.

    Args:
        trace_data: Single trace JSON data
        extractor: Extractor function to use
        file_name: Base filename for ID generation
        trace_index: Index of trace within file (for multi-trace files)

    Returns:
        List of row dicts
    """
    trace_file_name = f"{file_name}_{trace_index}" if trace_index > 0 else file_name
    return extractor(trace_data, trace_file_name)


def process_traces_to_traj_data(
    input_dir: str,
    output_dir: str,
    agent_framework: str = "langgraph",
    observability_framework: str = "langfuse",
    separate_tools: bool = True,
    overwrite: bool = True,
) -> str:
    """
    Process trace JSON files into trajectory CSV files.

    Args:
        input_dir: Directory containing trace JSON files
        output_dir: Directory to save trajectory CSV files
        agent_framework: Agent framework used (langgraph, crewai)
        observability_framework: Observability platform (langfuse, mlflow)
        separate_tools: For mlflow, whether to emit separate rows for tools vs agent responses
        overwrite: Whether to overwrite existing trajectory CSV files

    Returns:
        Path to the trajectory data directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the appropriate extractor
    extractor = get_extractor(agent_framework, observability_framework, separate_tools)

    json_files = list(input_path.glob('*.json'))
    logger.info("=" * 80)
    logger.info("PROCESSING TRACES")
    logger.info("=" * 80)
    logger.info(f"Agent framework: {agent_framework}")
    logger.info(f"Observability framework: {observability_framework}")
    logger.info(f"Separate tools: {separate_tools}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Found {len(json_files)} JSON files")
    logger.info("=" * 80)

    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return str(output_path)

    total_traces = 0
    total_skipped = 0
    traces_with_no_llm_calss  = 0
    processed_traces = 0
    total_llm_calls = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            traces = coerce_to_trace_list(json_data)

            if not traces:
                logger.warning(f"No traces found in {json_file.name}")
                continue

            for trace_idx, trace_data in enumerate(traces):
                total_traces += 1

                llm_calls = process_single_trace(
                    trace_data=trace_data,
                    extractor=extractor,
                    file_name=json_file.stem,
                    trace_index=trace_idx
                )

                if not llm_calls:
                    traces_with_no_llm_calss += 1
                    continue

                # Determine output filename
                trace_id = llm_calls[0].get('task_id', json_file.stem)
                safe_trace_id = "".join(c if c.isalnum() or c in '-_' else '_' for c in str(trace_id))
                csv_filename = f"{safe_trace_id}.csv"
                output_csv_path = output_path / csv_filename

                if output_csv_path.exists() and not overwrite:
                    logger.debug(f"Skipping (exists): {csv_filename}")
                    total_skipped += 1
                    continue

                with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(llm_calls)

                processed_traces += 1
                total_llm_calls += len(llm_calls)

                trace_info = f"trace {trace_idx + 1}" if len(traces) > 1 else ""
                logger.info(f"Processed {json_file.name} {trace_info} -> {csv_filename} ({len(llm_calls)} LLM calls)")

        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 80)
    logger.info("TRACE PROCESSING COMPLETE")
    logger.info(f"Total traces: {total_traces}, skipped: {total_skipped}, no LLM calls: {traces_with_no_llm_calss}, with llm calls: {processed_traces}")
    logger.info(f"Total LLM calls extracted: {total_llm_calls}")
    logger.info("=" * 80)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess agent traces into trajectory CSV files for evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process LangGraph traces from Langfuse
    python preprocess_traces.py --input_dir ./traces --output_dir ./output \\
        --agent_framework langgraph --observability_framework langfuse

    # Process CrewAI traces from Langfuse
    python preprocess_traces.py --input_dir ./traces --output_dir ./output \\
        --agent_framework crewai --observability_framework langfuse

    # Process LangGraph traces from MLflow (separate tool rows)
    python preprocess_traces.py --input_dir ./traces --output_dir ./output \\
        --agent_framework langgraph --observability_framework mlflow --separate_tools \\

    # Process MLflow traces with combined output (single row per model call)
    python preprocess_traces.py --input_dir ./traces --output_dir ./output \\
        --agent_framework langgraph --observability_framework mlflow

Supported combinations:
    - langgraph + langfuse
    - crewai + langfuse
    - langgraph + mlflow
        """
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing trace JSON files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trajectory CSV files"
    )

    parser.add_argument(
        "--agent_framework",
        type=str,
        default="langgraph",
        choices=["langgraph", "crewai"],
        help="Agent framework used to generate the traces"
    )

    parser.add_argument(
        "--observability_framework",
        type=str,
        default="mlflow",
        choices=["langfuse", "mlflow"],
        help="Observability platform that captured the traces (mlflow only supports langgraph)"
    )

    parser.add_argument(
        "--separate-tools",
        dest="separate_tools",
        action="store_true",
        help="Emit separate rows for tool calls vs text responses (default: True)"
    )

    args = parser.parse_args()

    if args.observability_framework == "mlflow" and args.agent_framework != "langgraph":
        parser.error("MLflow observability only supports langgraph agent framework")

    process_traces_to_traj_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        agent_framework=args.agent_framework,
        observability_framework=args.observability_framework,
        separate_tools=args.separate_tools
    )


if __name__ == "__main__":
    main()
