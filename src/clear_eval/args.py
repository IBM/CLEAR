import argparse
import json


def parse_dict(arg: str) -> dict:
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}")


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_clear_args_to_parser(parser: argparse.ArgumentParser | None = None, group_name: str | None = None) -> argparse.ArgumentParser:
    """
    Add all CLEAR configuration arguments to a parser.

    Args:
        parser: The ArgumentParser to add arguments to. If None, creates a new one.
        group_name: Optional group name. If provided, args are added to a named group.
                   If None, args are added directly to the parser.

    Returns:
        The parser with CLEAR arguments added
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    if group_name:
        target = parser.add_argument_group(group_name)
    else:
        target = parser

    target.add_argument("--data-path", help="Path to the data csv file")
    target.add_argument("--output-dir", default=None, help="Output directory")
    target.add_argument("--provider", help="LLM provider (e.g., openai, watsonx, rits, or any LiteLLM provider with --use-litellm)")
    target.add_argument("--eval-model-name", help="Name of the model used by CLEAR for evaluating and analyzing outputs")
    target.add_argument("--gen-model-name", help="Name of the generator model whose responses are evaluated (e.g. gpt-3.5-turbo)",
                        default=None)

    target.add_argument("--config-path", default=None, help="Optional: path to the config file")
    target.add_argument("--perform-generation", type=str2bool, default=None,
                        help="Whether to perform generations or use existing generations")
    target.add_argument("--is-reference-based", type=str2bool, default=None,
                        help="Whether to use references for the evaluations (if true, references must be stored in the 'reference' column of the input.")
    target.add_argument("--resume-enabled", type=str2bool, default=None,
                        help="Whether to use intermediate results found in the output dir")
    target.add_argument("--run-name", default=None,
                        help="Unique identifier for the run")
    target.add_argument("--task", default=None,
                        help="task to evaluate: general, tool_call, math, rag, external")
    target.add_argument("--evaluation-criteria", type=parse_dict, default=None,
                        help="Json of a dictionary of evaluation criteria for the judge. Example: --evaluation-criteria '{\"correction\": \"Response is factually correct\"}'")
    target.add_argument("--max-examples-to-analyze", type=int, default=None,
                        help="Analyze only the specified number of examples")
    target.add_argument("--input-columns", nargs='+', default=None,
                        help="List of column names to present in the ui")
    target.add_argument("--high-score-threshold", type=float, default=None,
                        help="The minimum judge score required for a single record to be considered successful")
    target.add_argument("--agent-mode", type=str2bool, default=None,
                        help="Whether to use a default evaluation criteria suited for an agentic step and not a single llm response")

    target.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel inferences")
    target.add_argument("--predefined-issues", nargs='+', default=None,
                        help="Predefined issues to use")
    target.add_argument("--issues-format", choices=["shortcomings", "recommendations"], default=None,
                        help="Output format for identified issues: 'shortcomings' (problem-focused, default) or 'recommendations' (solution-focused)")

    # External judge arguments (used when task is 'external')
    target.add_argument("--external-judge-path", default=None,
                        help="Path to Python file containing external judge function (used when task is 'external')")
    target.add_argument("--external-judge-function", default=None,
                        help="Name of the function to call in the external judge file (default: 'evaluate')")
    target.add_argument("--external-judge-config", type=parse_dict, default=None,
                        help="JSON dictionary of additional configuration for the external judge")
    target.add_argument("--eval-model-params", type=parse_dict, default=None,
                        help="JSON dictionary of eval model parameters. Example: --eval-model-params '{\"temperature\": 0.7, \"max_tokens\": 2000}'")
    target.add_argument("--use-litellm", type=str2bool, default=None,
                        help="Whether to use litellm for inference")

    return parser


def get_clear_arg_names(parser: argparse.ArgumentParser, group_name: str | None = None) -> list[str]:
    """
    Extract argument destination names from a parser or argument group.
    
    Args:
        parser: The ArgumentParser to extract from
        group_name: Optional group name. If provided, extracts only from that group.
                   If None, extracts from all actions.
    
    Returns:
        List of argument destination names
    """
    if group_name:
        # Find the argument group by title
        for group in parser._action_groups:
            if group.title == group_name:
                return [action.dest for action in group._group_actions]
        return []
    else:
        # Get all argument destinations (excluding 'help')
        return [action.dest for action in parser._actions if action.dest != 'help']


def create_clear_parser() -> argparse.ArgumentParser:
    """Create a new ArgumentParser with all CLEAR arguments."""
    return add_clear_args_to_parser(parser=None, group_name=None)


def extract_clear_overrides(parser: argparse.ArgumentParser, group_name: str | None = None) -> dict:
    """
    Parse arguments and extract non-None CLEAR config values, filtered by argument group.

    Args:
        parser: ArgumentParser to parse and extract CLEAR argument names from
        group_name: Optional group name to filter arguments by (e.g., "CLEAR Configuration").
                   If None, extracts all arguments from the parser.

    Returns:
        Dictionary of non-None CLEAR argument values from the specified group
    """
    # Parse the arguments
    args = parser.parse_args()

    # Compute clear_arg_names from the parser's argument group
    clear_arg_names = get_clear_arg_names(parser, group_name)

    return {
        k: v for k, v in vars(args).items()
        if v is not None
        and k in clear_arg_names
    }


def parse_args():
    """
    Parse command line arguments for standalone CLEAR usage.

    Returns:
        Dictionary of non-None argument values
    """
    parser = create_clear_parser()
    # When using create_clear_parser, arguments are not in a named group
    # so we pass None for group_name to get all arguments
    return extract_clear_overrides(parser, group_name=None)
