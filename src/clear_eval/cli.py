import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit.web.cli as stcli
from clear_eval.analysis_runner import run_clear_eval_analysis, run_clear_eval_generation, run_clear_eval_aggregation
from clear_eval.args import parse_args

# Load environment variables from .env file
load_dotenv()


def main():
    overrides = parse_args()
    run_clear_eval_analysis(**overrides)

def run_generation_cli():
    overrides = parse_args()
    run_clear_eval_generation(**overrides)

def run_evaluation_cli():
    overrides = parse_args()
    overrides["perform_generation"] = False
    run_clear_eval_analysis(**overrides)

def run_aggregation_cli():
    overrides = parse_args()
    run_clear_eval_aggregation(**overrides)

def run_dashboard_cli():
    parser = argparse.ArgumentParser(description="Run the dashboard.")
    parser.add_argument("--port", type=int, help="Optional port to run the dashboard on.")
    args = parser.parse_args()

    streamlit_app = Path(__file__).parent / "load_ui.py"
    sys.argv = ["streamlit", "run", str(streamlit_app)]
    if args.port:
        sys.argv += ["--server.port", str(args.port)]

    stcli.main()

def run_clear_agentic_eval():
    """Run the unified agentic pipeline for step-by-step and/or full trajectory analysis."""
    from clear_eval.agentic.pipeline.run_clear_agentic_eval import main as agentic_main
    agentic_main()

def run_clear_agentic_dashboard():
    """Run the agentic workflow dashboard (NiceGUI)."""
    from clear_eval.agentic.dashboard.launch_dashboard import main as dashboard_main
    dashboard_main()


if __name__ == "__main__":
    main()
