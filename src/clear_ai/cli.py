import argparse
import json
import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli
from clear_ai.analysis_runner import run_analysis, run_generation, run_aggregation
from clear_ai.args import parse_args


def main():
    overrides = parse_args()
    run_analysis(**overrides)

def run_generation_cli():
    overrides = parse_args()
    run_generation(**overrides)

def run_evaluation_cli():
    overrides = parse_args()
    overrides["perform_generation"] = False
    run_analysis(**overrides)

def run_aggregation_cli():
    overrides = parse_args()
    run_aggregation(**overrides)

def run_dashboard_cli():
    streamlit_app = Path(__file__).parent / "load_ui.py"
    sys.argv = ["streamlit", "run", str(streamlit_app)]
    stcli.main()

if __name__ == "__main__":
    main()
