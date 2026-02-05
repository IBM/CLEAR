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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="Path to the data csv file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--provider", choices=["azure", "openai", "watsonx", "rits"])
    parser.add_argument("--eval-model-name", help="Name of the model used by CLEAR for evaluating and analyzing outputs")
    parser.add_argument("--gen-model-name", help="Name of the generator model whose responses are evaluated (e.g. gpt-3.5-turbo)",
                        default=None)

    parser.add_argument("--config-path", default=None, help="Optional: path to the config file")
    parser.add_argument("--perform-generation", type=str2bool, default=None, help="Whether to perform generations or"
                                                                    "use existing generations")
    parser.add_argument("--is-reference-based", type=str2bool, default=None,
                        help="Whether to use use references for the evaluations (if true, references must be stored in the 'reference' column of the input.")
    parser.add_argument("--resume-enabled", type=str2bool, default=None,
                        help="Whether to use use intermediate results found in the output dir")
    parser.add_argument("--run-name", default=None,
                        help="Unique identifier for the run")
    parser.add_argument("--task", default="general",
                        help="task to evaluate: general, tool_call, math, rag")
    parser.add_argument("--evaluation-criteria", type=parse_dict, help="Json of a dictionary of evaluation criteria for"
                                                "the judge. Example: --evaluation-criteria '{\"correction\": \"Response is factually correct\"}'")
    parser.add_argument("--max-examples-to-analyze", type=int, help="Analyze only the specified number of examples")
    parser.add_argument("--input-columns", nargs='+', help="List of column names to present in the ui")
    parser.add_argument("--success-threshold", type=float, help="the minimum judge score required for a single record to be considered successful ")
    parser.add_argument("--agent-mode", type=str2bool, default=None,
                        help="Whether to use a default evaluation criteria suited for an agentic step and not a single llm response")

    parser.add_argument("--max-workers", type=int, default=None,
                        help="number of parallel inferences")
    parser.add_argument("--predefined-issues",nargs='+', help="Predefined issues to use")
    
    # External judge arguments (used when task is 'external')
    parser.add_argument("--external-judge-path", default=None,
                        help="Path to Python file containing external judge function (used when task is 'external')")
    parser.add_argument("--external-judge-function", default=None,
                        help="Name of the function to call in the external judge file (default: 'evaluate')")
    parser.add_argument("--external-judge-config", type=parse_dict, default=None,
                        help="JSON dictionary of additional configuration for the external judge")

    args = parser.parse_args()

    # Only keep explicitly passed args (ignore None)
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None
    }
    return overrides
