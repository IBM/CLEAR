import os
from clear_eval.pipeline.full_pipeline import run_eval_pipeline, run_generation_pipeline, run_aggregation_pipeline
from clear_eval.pipeline.config_loader import load_config
from clear_eval.logging_config import setup_logging
setup_logging()

script_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(script_dir, "pipeline", "setup", "default_config.yaml")

def run_clear_eval_analysis(config_path=None, **kwargs):
    config_dict = load_config(DEFAULT_CONFIG_PATH, config_path, **kwargs)
    run_eval_pipeline(config_dict)

def run_analysis_pipeline(config_path=None, **kwargs):
    run_clear_eval_analysis(config_path, **kwargs)

def run_clear_eval_evaluation(config_path=None, **kwargs):
    config_dict = load_config(DEFAULT_CONFIG_PATH, config_path, **kwargs)
    config_dict["perform_generation"] = False
    run_eval_pipeline(config_dict)

def run_clear_eval_generation(config_path=None, **kwargs):
    config_dict = load_config(DEFAULT_CONFIG_PATH, config_path, **kwargs)
    run_generation_pipeline(config_dict)

def run_clear_eval_aggregation(config_path=None, **kwargs):
    config_dict = load_config(DEFAULT_CONFIG_PATH, config_path, **kwargs)
    run_aggregation_pipeline(config_dict)