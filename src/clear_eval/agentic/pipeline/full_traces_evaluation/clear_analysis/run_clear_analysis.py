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
        --agentic-output-dir results/clear_analysis \\
        --eval-model-name gpt-4o \\
        --provider openai

    python run_clear_analysis.py --source root_cause \\
        --eval-results-dir results/evals \\
        --agentic-output-dir results/clear_analysis \\
        --eval-model-name gpt-4o \\
        --provider openai

Example workflow:
    1. Run trajectory evaluation:
       python run_full_traj_evaluation.py --agentic-input-dir data \\
           --agentic-output-dir results/evals --eval-model-name gpt-4o --provider openai
    
    2. Run CLEAR analysis on results:
       python run_clear_analysis.py --source issues \\
           --eval-results-dir results/evals --agentic-output-dir results/clear \\
           --eval-model-name gpt-4o --provider openai
"""
import logging
from pathlib import Path
from clear_eval.agentic.pipeline.full_traces_evaluation.argument_parser import create_base_parser
from clear_eval.agentic.pipeline.full_traces_evaluation.clear_analysis.issues_clear_runner import IssuesClearRunner
from clear_eval.agentic.pipeline.full_traces_evaluation.clear_analysis.root_cause_clear_runner import RootCauseClearRunner
from clear_eval.agentic.pipeline.utils import InferenceConfig
from clear_eval.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

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

    # Create inference config
    inference_config = InferenceConfig(
        model_id=args.eval_model_name,
        provider=args.provider,
        inference_backend=getattr(args, 'inference_backend', None) or 'litellm',
        endpoint_url=getattr(args, 'endpoint_url', None),
        model_params=args.eval_model_params or {},
    )

    # Create appropriate runner based on source
    if args.source == "issues":
        runner = IssuesClearRunner(
            eval_results_dir=Path(args.eval_results_dir),
            output_dir=Path(args.agentic_output_dir),
            inference_config=inference_config,
            overwrite=args.overwrite,
        )
    elif args.source == "root_cause":
        runner = RootCauseClearRunner(
            eval_results_dir=Path(args.eval_results_dir),
            output_dir=Path(args.agentic_output_dir),
            inference_config=inference_config,
            overwrite=args.overwrite,
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")
    
    # Print configuration
    logging.info("=" * 70)
    logging.info("CLEAR Analysis on Evaluation Results")
    logging.info("=" * 70)
    logging.info(f"  Source:             {args.source}")
    logging.info(f"  Eval Results Dir:   {args.eval_results_dir}")
    logging.info(f"  Output Dir:         {args.agentic_output_dir}")
    logging.info(f"  CLEAR Model:        {inference_config.model_id}")
    logging.info(f"  Provider:           {inference_config.provider}")
    logging.info(f"  Inference Backend:  {inference_config.inference_backend}")
    logging.info(f"  Overwrite:          {args.overwrite}")
    logging.info("=" * 70)
    
    # Run analysis
    summary = runner.run_analysis()
    
    # Print summary
    runner.print_summary(summary)


if __name__ == "__main__":
    main()

