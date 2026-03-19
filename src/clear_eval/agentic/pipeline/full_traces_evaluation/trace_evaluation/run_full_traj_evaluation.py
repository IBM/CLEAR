#!/usr/bin/env python3
"""
Full Trajectory Evaluation Script
==================================

Evaluates agent trajectories using the FullTrajectoryEvaluator.

Evaluates across 14 CLEAR dimensions:
- 9 step-level quality dimensions
- 5 trajectory-level holistic dimensions

Produces detailed feedback and overall scores (0.0-1.0).
"""

from pathlib import Path
from agentic.pipeline.full_traces_evaluation.argument_parser import create_base_parser
from agentic.pipeline.full_traces_evaluation.trace_evaluation.full_trajectory_evaluator import FullTrajectoryEvaluator


def parse_args():
    parser = create_base_parser(description="Full Trajectory Evaluation (CLEAR framework with 14 dimensions)")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create evaluator (using unified argument names)
    evaluator = FullTrajectoryEvaluator(
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
    evaluator.run_pipeline()


if __name__ == "__main__":
    main()
