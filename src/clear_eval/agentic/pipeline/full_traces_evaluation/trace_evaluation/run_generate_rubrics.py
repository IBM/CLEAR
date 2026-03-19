#!/usr/bin/env python3
"""
Generate Task Rubrics - Entry Point
====================================

Generates 3-5 task-specific evaluation rubrics for each trajectory based on
task complexity. Uses the RubricGenerator class for centralized logic.

Usage:
    python run_generate_rubrics.py --agentic-input-dir <dir> --agentic-output-dir <dir> \\
        --eval-model-name <model> --provider <provider>

Example:
    python run_generate_rubrics.py \\
        --agentic-input-dir data/trajectories \\
        --agentic-output-dir results/rubrics \\
        --eval-model-name gpt-4o \\
        --provider openai
"""

from pathlib import Path

from agentic.pipeline.full_traces_evaluation.argument_parser import create_base_parser
from agentic.pipeline.full_traces_evaluation.trace_evaluation.rubric_generator import RubricGenerator


def main():
    """Main entry point for rubric generation."""
    parser = create_base_parser(
        description="Generate task-specific rubrics for trajectory evaluation"
    )
    args = parser.parse_args()
    
    # Create evaluator (using unified argument names)
    evaluator = RubricGenerator(
        judge_model_id=args.eval_model_name,
        provider=args.provider,
        traj_input_dir=Path(args.agentic_input_dir),
        output_dir=Path(args.agentic_output_dir),
        context_tokens=args.context_tokens,
        overwrite=args.overwrite,
        concurrency=args.concurrency,
        eval_model_params=args.eval_model_params,
        max_files=args.max_files,
    )
    
    # Run pipeline
    evaluator.run_pipeline()


if __name__ == "__main__":
    main()
