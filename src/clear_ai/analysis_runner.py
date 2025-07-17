import os
from clear_ai.pipeline.full_pipeline import run_eval_pipeline, run_generation_pipeline, run_aggregation_pipeline
from clear_ai.pipeline.config_loader import load_config

script_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(script_dir, "pipeline", "setup", "default_config.yaml")

def run_analysis_pipeline(config_path=None, **overrides):
    config_dict = load_config(DEFAULT_CONFIG_PATH, config_path, **overrides)
    run_eval_pipeline(config_dict)


def run_gen_pipeline(config_path=None, **overrides):
    config_dict = load_config(DEFAULT_CONFIG_PATH, config_path, **overrides)
    run_generation_pipeline(config_dict)

def run_agg_pipeline(config_path=None, **overrides):
    config_dict = load_config(DEFAULT_CONFIG_PATH, config_path, **overrides)
    run_aggregation_pipeline(config_dict)