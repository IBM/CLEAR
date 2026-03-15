from clear_eval.agentic.full_traj_evaluation.argument_parser import create_base_parser
from clear_eval.agentic.full_traj_evaluation.task_success_evaluator import TaskSuccessEvaluator
from clear_eval.agentic.full_traj_evaluation.dataset_base import (
    TRAJ_DATA_DIR,
    get_results_dir,
)

def parse_args():
    parser = create_base_parser(description="Task Success Evaluation (binary success label)")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    results_dir = get_results_dir("success_results", args.model_id)

    # Create evaluator
    evaluator = TaskSuccessEvaluator(
        judge_model_id=args.model_id,
        provider=args.provider,
        results_dir=results_dir,
        context_tokens=args.context_tokens,
        overwrite=args.overwrite,
        concurrency = args.concurrency,
        eval_model_params = args.eval_model_params,
        max_files = args.max_files,
        dataset = args.dataset,
        model= args.model,
    )

    if args.summary_only:
        evaluator.save_summary()
        return

    evaluator.run_pipeline()


if __name__ == "__main__":
    main()