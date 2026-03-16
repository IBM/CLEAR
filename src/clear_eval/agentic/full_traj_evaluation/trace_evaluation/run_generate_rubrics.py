#!/usr/bin/env python3
"""
Generate Task Rubrics - Entry Point
====================================

Generates 3-5 task-specific evaluation rubrics for each trajectory based on
task complexity. Uses the RubricGenerator class for centralized logic.

Usage:
    python run_generate_rubrics.py --traj-input-dir <dir> --output-dir <dir> \\
        --model-id <model> --provider <provider>

Example:
    python run_generate_rubrics.py \\
        --traj-input-dir data/trajectories \\
        --output-dir results/rubrics \\
        --model-id gpt-4 \\
        --provider openai
"""

from pathlib import Path

from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from agentic.full_traj_evaluation.trace_evaluation.rubric_generator import RubricGenerator


def main():
    """Main entry point for rubric generation."""
    parser = create_base_parser(
        description="Generate task-specific rubrics for trajectory evaluation"
    )
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = RubricGenerator(
        judge_model_id=args.model_id,
        provider=args.provider,
        traj_input_dir=Path(args.traj_input_dir),
        output_dir=Path(args.output_dir),
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
