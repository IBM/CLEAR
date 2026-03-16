#!/usr/bin/env python3
"""
Run CLEAR Analysis on Evaluation Results - Entry Point
=======================================================

Runs CLEAR aggregation on evaluation results from trajectory evaluators
to discover common themes, shortcomings, and patterns.

Two analysis sources (--source flag):
    issues      – Analyzes detailed_feedback from full trajectory evaluations
    root_cause  – Analyzes failure_root_cause from task success evaluations
                  (failed trajectories only)

Usage:
    python run_clear_analysis.py --source issues \\
        --eval-results-dir results/evals \\
        --output-dir results/clear_analysis \\
        --model-id gpt-4 \\
        --provider openai

    python run_clear_analysis.py --source root_cause \\
        --eval-results-dir results/evals \\
        --output-dir results/clear_analysis \\
        --model-id gpt-4 \\
        --provider openai

Example workflow:
    1. Run trajectory evaluation:
       python run_full_traj_evaluation.py --traj-input-dir data \\
           --output-dir results/evals --model-id gpt-4 --provider openai
    
    2. Run CLEAR analysis on results:
       python run_clear_analysis.py --source issues \\
           --eval-results-dir results/evals --output-dir results/clear \\
           --model-id gpt-4 --provider openai
"""

from pathlib import Path

from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from agentic.full_traj_evaluation.clear_analysis.issues_clear_runner import IssuesClearRunner
from agentic.full_traj_evaluation.clear_analysis.root_cause_clear_runner import RootCauseClearRunner


def main():
    """Main entry point for CLEAR analysis on evaluation results."""
    parser = create_base_parser(
        description="Run CLEAR analysis on evaluation results"
    )
    
    # Add source argument
    parser.add_argument(
        "--source",
        type=str,
        default="issues",
        choices=["issues", "root_cause"],
        help=(
            "Source of data for CLEAR analysis. "
            "'issues' = detailed_feedback from full trajectory evaluations. "
            "'root_cause' = failure_root_cause from task success evaluations."
        ),
    )
    
    # Add eval-results-dir argument
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation results (output_dir from evaluators)",
    )
    
    args = parser.parse_args()
    
    # Create appropriate runner based on source
    if args.source == "issues":
        runner = IssuesClearRunner(
            eval_results_dir=Path(args.eval_results_dir),
            output_dir=Path(args.output_dir),
            clear_model_id=args.model_id,
            provider=args.provider,
            overwrite=args.overwrite,
            eval_model_params=args.eval_model_params,
        )
    elif args.source == "root_cause":
        runner = RootCauseClearRunner(
            eval_results_dir=Path(args.eval_results_dir),
            output_dir=Path(args.output_dir),
            clear_model_id=args.model_id,
            provider=args.provider,
            overwrite=args.overwrite,
            eval_model_params=args.eval_model_params,
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")
    
    # Print configuration
    print("=" * 70)
    print("CLEAR Analysis on Evaluation Results")
    print("=" * 70)
    print(f"  Source:             {args.source}")
    print(f"  Eval Results Dir:   {args.eval_results_dir}")
    print(f"  Output Dir:         {args.output_dir}")
    print(f"  CLEAR Model:        {args.model_id}")
    print(f"  Provider:           {args.provider}")
    print(f"  Overwrite:          {args.overwrite}")
    print("=" * 70)
    
    # Run analysis
    summary = runner.run_analysis()
    
    # Print summary
    runner.print_summary(summary)


if __name__ == "__main__":
    main()

