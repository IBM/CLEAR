"""
Shared Argument Parsing
=======================

Common argument parsing utilities for evaluation scripts.
"""

import argparse
import json
from typing import Optional


def parse_dict(arg: str) -> dict:
    """Parse JSON string to dictionary."""
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}")


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create base argument parser with common arguments.
    
    Args:
        description: Script description
        
    Returns:
        ArgumentParser with common arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Input/Output directories
    parser.add_argument(
        "--traj-input-dir",
        type=str,
        required=True,
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory for saving evaluation results",
    )
    
    # Provider and model arguments
    parser.add_argument(
        "--provider",
        type=str,
        default="watsonx",
        choices=["rits", "watsonx", "openai"],
        help="LLM provider to use (default: watsonx)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model ID for the judge (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=128_000,
        help="Model context window size in tokens (default: 128000)",
    )
    parser.add_argument(
        "--eval-model-params",
        type=parse_dict,
        default=None,
        help="JSON dictionary of eval model parameters. Example: --eval-model-params '{\"temperature\": 0.7, \"max_tokens\": 2000}'",
    )
    # Dataset and model filtering
    parser.add_argument(
        "--dataset",
        type=str,
        default="HAL",
        choices=["CUGA","gaia","swebench", "HAL","TRAIL"], #["CUGA", "FinOps", "WXO"],
        help="Filter by dataset name (default: CUGA)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="orig_results",
        help="Filter by model name (e.g., gpt4o, full, granit)",
    )
    
    # Execution control
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate even if results already exist",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max files to evaluate per dataset/model combo",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only generate summary from existing results (no evaluation)",
    )
    
    return parser

