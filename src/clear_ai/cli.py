import argparse
import json
import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli
from clear_ai.analysis_runner import run_analysis_pipeline, run_generation_pipeline, run_agg_pipeline, run_gen_pipeline


def parse_dict(arg: str) -> dict:
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="Path to the data csv file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--provider", choices=["azure", "openai", "watsonx", "rits"])
    parser.add_argument("--eval-model-name", help="name of the judge model")
    parser.add_argument("--gen-model-name", help="the model whose response are evaluated. Only for running generations.",
                        default=None)

    parser.add_argument("--config_path", default=None, help="Optional: path to the config file")
    parser.add_argument("--perform-generations", default=True, help="Whether to perform generations or"
                                                                    "to use existing generations")
    parser.add_argument("--is-reference-based", action='store_true',
                        help="Whether to use use references for the evaluations (if true, references must be stored in the 'reference' column of the input.")
    parser.add_argument("--resume-enabled", default=True,
                        help="Whether to use use intermediate results found in the output dir")
    parser.add_argument("--run-name", default=None,
                        help="Unique identifier for the run")
    parser.add_argument("--evaluation_criteria", type=parse_dict, help="Json of a dictionary of evaluation criteria for"
                                                "the judge. Example: --evaluation_criteria '{\"correction\": 0.\"Response is factually correct\"}'")
    parser.add_argument("--max_examples_to_analyze", type=int, help="Analyze only the specified number of examples")
    parser.add_argument("--input_columns", nargs='+', help="List of column names to present in the ui")

    args = parser.parse_args()

    # Only keep explicitly passed args (ignore None)
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None
    }
    return overrides

def main():
    overrides = parse_args()
    run_analysis_pipeline(**overrides)

def run_generation():
    overrides = parse_args()
    run_gen_pipeline(**overrides)

def run_aggregation():
    overrides = parse_args()
    run_agg_pipeline(**overrides)

def run_dashboard():
    streamlit_app = Path(__file__).parent / "load_ui.py"
    sys.argv = ["streamlit", "run", str(streamlit_app)]
    stcli.main()

if __name__ == "__main__":
    main()
