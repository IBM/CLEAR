#!/usr/bin/env python3
"""
Evaluate Trajectories Against Rubrics - Entry Point
====================================================

Evaluates agent trajectories against pre-generated task-specific rubrics.
Uses the RubricEvaluator class for centralized logic.

Requires rubrics to be generated first using run_generate_rubrics.py.

Usage:
    python run_evaluate_rubrics.py --agentic-input-dir <dir> --agentic-output-dir <dir> \\
        --rubrics-dir <dir> --eval-model-name <model> --provider <provider>

Example:
    python run_evaluate_rubrics.py \\
        --agentic-input-dir data/trajectories \\
        --agentic-output-dir results/rubric_eval \\
        --rubrics-dir results/rubrics/rubric_generation/gpt-4o \\
        --eval-model-name gpt-4o \\
        --provider openai
"""

from pathlib import Path

from agentic.pipeline.full_traces_evaluation.argument_parser import create_base_parser
from agentic.pipeline.full_traces_evaluation.trace_evaluation.rubric_evaluator import RubricEvaluator


def main():
    """Main entry point for rubric evaluation."""
    parser = create_base_parser(
        description="Evaluate trajectories against pre-generated rubrics"
    )
    
    # Add rubrics-dir argument (specific to rubric evaluation)
    parser.add_argument(
        "--rubrics-dir",
        type=str,
        required=True,
        help="Directory containing pre-generated rubric files (model-specific subdirectory)",
    )
    
    args = parser.parse_args()
    
    # Create evaluator (using unified argument names)
    evaluator = RubricEvaluator(
        judge_model_id=args.eval_model_name,
        provider=args.provider,
        traj_input_dir=Path(args.agentic_input_dir),
        output_dir=Path(args.agentic_output_dir),
        rubrics_dir=Path(args.rubrics_dir),
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

