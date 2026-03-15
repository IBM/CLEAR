#!/usr/bin/env python3
"""
Task Success Evaluation Script
===============================

Evaluates whether AI agent trajectories successfully completed their assigned tasks.
"""

from pathlib import Path
from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from clear_eval.agentic.full_traj_evaluation.task_success_evaluator import TaskSuccessEvaluator


def parse_args():
    parser = create_base_parser(description="Task Success Evaluation (binary success label)")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create evaluator
    evaluator = TaskSuccessEvaluator(
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

    # Run evaluation pipeline
    evaluator.run_pipeline()


if __name__ == "__main__":
    main()