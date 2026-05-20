import json
import logging
import os
import zipfile
import io
import numpy as np
logger = logging.getLogger(__name__)

import pandas as pd

from clear_eval.pipeline.use_cases.use_case_utils import get_task_data_obj
from clear_eval.pipeline.constants import (GENERATION_FILE_PREFIX, SHORTCOMING_LIST_FILE_PREFIX,
                                           IDENTIFIED_SHORTCOMING_COL, SCORE_COL, EVALUATION_SUMMARY_COL,
                                           DEFAULT_ISSUES_FORMAT_MODE)

from clear_eval.pipeline.caching_utils import load_dataframe_from_cache, save_dataframe_to_cache, save_json_to_cache, \
    ensure_dir, \
    load_json_from_cache, resolve_data_path
from clear_eval.pipeline.eval_utils import map_shortcomings_to_records, get_model_name_for_file, convert_results_to_ui_input, \
    load_inputs, synthesize_shortcomings_from_df, \
    remove_duplicates_shortcomings, run_predictions_generation_save_results, produce_summaries_per_record, \
    generate_model_predictions
from clear_eval.pipeline.inference_utils.llm_client import get_llm_client
from clear_eval.pipeline.config_loader import load_yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PREFIX = "checkpoint"


def get_issues_format(config):
        return config.get('issues_format', DEFAULT_ISSUES_FORMAT_MODE)

def get_run_name(config):
    run_name = config.get("run_name")
    if not run_name:
        data_name = os.path.basename(config.get("data_path"))
        run_name = data_name.replace(".csv", "").split("_")[0]
    config["run_name"] = run_name
    return run_name


def get_run_info(config):
    eval_model_str = get_model_name_for_file(config["eval_model_name"])
    run_name = get_run_name(config)
    return f"{run_name}_eval_{eval_model_str}"


def run_generation_pipeline(config):
    task = config.get("task")
    if not task:
        raise ValueError(f"task config not specified")
    task_data = get_task_data_obj(task)

    data_path = resolve_data_path(config["data_path"])
    data_df = load_inputs(config, data_path, load_predictions=False, task_data=task_data)

    gen_model = config.get('gen_model_name')
    output_dir = config['output_dir']
    ensure_dir(output_dir)
    run_name = get_run_name(config)
    gen_file_name = get_gen_file_name(run_name, gen_model)
    gen_output_path = os.path.join(output_dir, gen_file_name)
    gen_llm = get_llm_from_config(config, eval_mode=False)
    run_predictions_generation_save_results(data_df, gen_llm, config, gen_output_path)

def get_gen_file_name(run_name, gen_model):
    gen_model_str = get_model_name_for_file(gen_model)
    return f"{GENERATION_FILE_PREFIX}_{run_name}_gen_{gen_model_str}.csv"


def get_parquet_bytes(output_df):
    def convert_nested_to_str(x):
        if isinstance(x, (list, dict, tuple, set, np.ndarray)):
            return str(x)
        return x

    for col in output_df.select_dtypes(include="object"):
        if output_df[col].map(lambda x: isinstance(x, (list, dict, set, tuple, np.ndarray))).any():
            output_df[col] = output_df[col].map(convert_nested_to_str)

    parquet_buffer = io.BytesIO()
    output_df.to_parquet(parquet_buffer, compression="brotli", engine="pyarrow", use_dictionary=True, index=False)
    return parquet_buffer.getvalue()

def get_predefined_issues_list(config):
    issues = config.get("predefined_issues")
    if not issues:
        return None
    if isinstance(issues, list):
        return issues
    elif isinstance(issues, str):
        return json.loads(issues)
    return None

def convert_to_ui_format(mapped_data_df, output_dir, config, file_name_info):
    task_data = get_task_data_obj(config.get("task"))
    output_df = convert_results_to_ui_input(mapped_data_df, config, task_data)
    output_path = f"{output_dir}/analysis_results_{file_name_info}.csv"
    logger.info(f"\n--- Saving Custom Formatted Analysis to {output_dir} ---")
    save_dataframe_to_cache(output_df, output_path)
    logger.info(f"Custom formatted analysis results saved to {output_path}")
    save_ui_input_results(output_df, output_path, config)

def save_ui_input_results(output_df, output_path, config):
    parquet_bytes = get_parquet_bytes(output_df)
    json_bytes = json.dumps(config, indent=2).encode()

    zip_output_path = output_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_output_path, mode="w") as zf:
        zf.writestr("results.parquet", parquet_bytes)
        zf.writestr("metadata.json", json_bytes)
    logger.info(f"Results for uploading to ui are saved to {zip_output_path}")


def run_aggregation_pipeline(config):
    logger.info(f"run_aggregation_pipeline received run config: {config}")
    run_info = get_run_info(config)
    eval_dir = config["output_dir"]
    eval_file = os.path.join(eval_dir, f"{CHECKPOINT_FILE_PREFIX}_{run_info}.csv")
    # if evaluation file doesn't exist, fallback to treat data_path as input for aggregation
    if os.path.exists(eval_file):
        eval_df = pd.read_csv(eval_file)
    else:
        eval_file = config.get("data_path")
        if not os.path.exists(eval_file):
            logger.info(f"No evaluation file found at {eval_file}")
            return
        eval_df = pd.read_csv(eval_file)
    run_aggregation_from_df(config, eval_df, run_info)


def get_eval_llm_from_config(config):
    return get_llm_from_config(config, eval_mode=True)

def get_llm_from_config(config, eval_mode=True):
    """
    Get LLM client from configuration.

    Supports three inference backends:
    1. langchain: Use LangChain (default for built-in providers)
    2. litellm: Use LiteLLM (supports many providers)
    3. endpoint: Use direct HTTP endpoint backend

    Backward compatible with use_litellm boolean field.
    """
    if config.get("use_litellm", False):
        inference_backend = "litellm"
    else:
        inference_backend = config.get("inference_backend")

    model_name_field = "eval_model_name" if eval_mode else "gen_model_name"
    model_params_field = "eval_model_params" if eval_mode else "gen_model_params"

    client_args = {
        "provider": config["provider"],
        "model": config[model_name_field],
        "inference_backend": inference_backend,
        "parameters": config.get(model_params_field),
        "eval_mode": eval_mode,
        "endpoint_url": config.get("endpoint_url"),
    }

    return get_llm_client(**client_args)


def run_evaluation_from_df(config, response_df):
    eval_llm = get_eval_llm_from_config(config)
    task_data = get_task_data_obj(config["task"])
    eval_df = task_data.eval_records(response_df, eval_llm, config)
    if not config.get("use_full_text_for_analysis"):
        eval_df = produce_summaries_per_record(eval_df, eval_llm, config)
    return eval_df


def resolve_issues_list(df, config, eval_llm, resume_enabled,
                        shortcoming_list_output_path, deduplicated_shortcomings_list_output_path, format_mode):
    """Resolve the shortcoming/issues list: from predefined, cache, or synthesis. Returns shortcoming_list."""
    shortcoming_list = get_predefined_issues_list(config)
    if shortcoming_list:
        return shortcoming_list

    shortcoming_list = None
    if resume_enabled:
        if config.get("perform_clustering", False):
            shortcoming_list = load_json_from_cache(deduplicated_shortcomings_list_output_path)
        if shortcoming_list is None:
            raw_issues = load_json_from_cache(shortcoming_list_output_path)
            if raw_issues is not None:
                if config.get("perform_clustering", False):
                    shortcoming_list = remove_duplicates_shortcomings(
                        raw_issues, eval_llm, max_shortcomings=config["max_shortcomings"], format_mode=format_mode)
                    save_json_to_cache(shortcoming_list, deduplicated_shortcomings_list_output_path)
                else:
                    shortcoming_list = raw_issues

    if shortcoming_list is None:
        synthesis_template = config.get("synthesis_template")
        shortcoming_list = synthesize_shortcomings_from_df(df, eval_llm, config,
                                                           synthesis_template=synthesis_template,
                                                           format_mode=format_mode)
        save_json_to_cache(shortcoming_list, shortcoming_list_output_path)
        if config.get("perform_clustering", False):
            shortcoming_list = remove_duplicates_shortcomings(
                shortcoming_list, eval_llm, max_shortcomings=config["max_shortcomings"], format_mode=format_mode)
            save_json_to_cache(shortcoming_list, deduplicated_shortcomings_list_output_path)

    return shortcoming_list


def resolve_issues_and_map(df, config, eval_llm, resume_enabled, checkpoint_path,
                           shortcoming_list_output_path, deduplicated_shortcomings_list_output_path, format_mode):
    """Resolve issues list and map shortcomings to records. Returns updated df."""
    if IDENTIFIED_SHORTCOMING_COL in df.columns:
        return df

    if eval_llm is None:
        eval_llm = get_eval_llm_from_config(config)

    shortcoming_list = resolve_issues_list(
        df, config, eval_llm, resume_enabled,
        shortcoming_list_output_path, deduplicated_shortcomings_list_output_path, format_mode)

    use_full_text = config['use_full_text_for_analysis']
    qid_col = config['qid_column']
    max_workers = config['max_workers']
    high_score_threshold = config.get("high_score_threshold", 1)
    df = map_shortcomings_to_records(df, eval_llm, shortcoming_list, use_full_text,
                                     qid_col, max_workers, high_score_threshold, format_mode=format_mode)
    save_dataframe_to_cache(df, checkpoint_path)
    return df


def run_aggregation_from_df(config, df, file_name_info, eval_llm=None):
    task = config.get("task")
    if not task:
        raise ValueError(f"task config not specified")

    output_dir = config['output_dir']
    ensure_dir(output_dir)
    resume_enabled = config['resume_enabled']
    format_mode = get_issues_format(config)
    checkpoint_path = f"{output_dir}/{CHECKPOINT_FILE_PREFIX}_{file_name_info}.csv"
    shortcoming_list_output_path = f"{output_dir}/{SHORTCOMING_LIST_FILE_PREFIX}_{file_name_info}.json"
    deduplicated_shortcomings_list_output_path = f"{output_dir}/{SHORTCOMING_LIST_FILE_PREFIX}_{file_name_info}_dedup.json"
    zip_path = f"{output_dir}/analysis_results_{file_name_info}.zip"

    if resume_enabled and os.path.exists(zip_path):
        return

    df = resolve_issues_and_map(df, config, eval_llm, resume_enabled, checkpoint_path,
                                   shortcoming_list_output_path, deduplicated_shortcomings_list_output_path, format_mode)
    convert_to_ui_format(df, output_dir, config, file_name_info)


def run_eval_pipeline(config):
    logger.info(f"run_eval_pipeline received run config: {config}")
    task = config.get("task")
    if not task:
        raise ValueError(f"task config not specified")

    output_dir = config['output_dir']
    ensure_dir(output_dir)
    resume_enabled = config['resume_enabled']
    perform_generation = config['perform_generation']
    task_data = get_task_data_obj(config["task"])
    run_info = get_run_info(config)
    with open(os.path.join(output_dir, f"config_{run_info}.json"), 'w') as f:
        json.dump(config, f)

    generate_issues = config.get("generate_issues", True)
    checkpoint_path = f"{output_dir}/{CHECKPOINT_FILE_PREFIX}_{run_info}.csv"
    zip_path = f"{output_dir}/analysis_results_{run_info}.zip"

    # If final output exists, nothing to do
    if resume_enabled and os.path.exists(zip_path):
        return

    # Load checkpoint if resuming
    df = None
    eval_llm = None
    if resume_enabled:
        df = load_dataframe_from_cache(checkpoint_path)

    # Determine completed stages by column presence
    has_generation = df is not None and config['model_output_column'] in df.columns
    has_eval = df is not None and SCORE_COL in df.columns
    has_summaries = df is not None and EVALUATION_SUMMARY_COL in df.columns
    has_mapping = df is not None and IDENTIFIED_SHORTCOMING_COL in df.columns

    if not has_mapping:
        eval_llm = get_eval_llm_from_config(config)

    # --- Generation ---
    if not has_generation:
        data_path = resolve_data_path(config["data_path"])
        data_df = load_inputs(config, data_path, load_predictions=not perform_generation, task_data=task_data)
        if perform_generation:
            logger.info(f"Performing generation analysis on {len(data_df)} examples")
            gen_llm = get_llm_from_config(config, eval_mode=False)
            df = generate_model_predictions(data_df, gen_llm, config)
        else:
            df = data_df
            logger.info(f"Using input generation results for {len(data_df)} examples")
        save_dataframe_to_cache(df, checkpoint_path)

    # --- Evaluation ---
    if not has_eval:
        df = task_data.eval_records(df, eval_llm, config)
        save_dataframe_to_cache(df, checkpoint_path)

    # --- Summaries ---
    if not has_summaries:
        df = produce_summaries_per_record(df, eval_llm, config)
        save_dataframe_to_cache(df, checkpoint_path)

    if not generate_issues:
        return

    # --- Aggregation (issues + mapping + UI output) ---
    run_aggregation_from_df(config, df, run_info, eval_llm=eval_llm)


if __name__ == "__main__":
    main_config = load_yaml(os.path.join(SCRIPT_DIR, 'setup', 'default_config.yaml'))
    run_eval_pipeline(main_config)