import argparse
import json


def parse_dict(arg: str) -> dict:
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}")

def str2bool(v):
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
    parser.add_argument("--perform-generation", type=str2bool, default=True, help="Whether to perform generations or"
                                                                    "use existing generations")
    parser.add_argument("--is-reference-based", type=str2bool, default=False,
                        help="Whether to use use references for the evaluations (if true, references must be stored in the 'reference' column of the input.")
    parser.add_argument("--resume-enabled", type=str2bool, default=True,
                        help="Whether to use use intermediate results found in the output dir")
    parser.add_argument("--run-name", default=None,
                        help="Unique identifier for the run")
    parser.add_argument("--evaluation-criteria", type=parse_dict, help="Json of a dictionary of evaluation criteria for"
                                                "the judge. Example: --evaluation-criteria '{\"correction\": \"Response is factually correct\"}'")
    parser.add_argument("--max-examples-to-analyze", type=int, help="Analyze only the specified number of examples")
    parser.add_argument("--input-columns", nargs='+', help="List of column names to present in the ui")

    args = parser.parse_args()

    # Only keep explicitly passed args (ignore None)
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None
    }
    return overrides
