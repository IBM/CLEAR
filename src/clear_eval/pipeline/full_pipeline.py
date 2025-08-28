import json
import logging
import sys
import os
import zipfile
import io
import numpy as np
logger = logging.getLogger(__name__)

import pandas as pd

from clear_eval.pipeline.EvalUseCase import task_to_use_case_class
from clear_eval.pipeline.constants import (GENERATION_FILE_PREFIX, EVALUATION_FILE_PREFIX_WITH_SUMMARIES,
                                           EVALUATION_FILE_PREFIX_NO_SUMMARIES, SHORTCOMING_LIST_FILE_PREFIX, \
                                           MAPPING_FILE_PREFIX)

from clear_eval.pipeline.caching_utils import load_dataframe_from_cache, save_dataframe_to_cache, save_json_to_cache, \
    ensure_dir, \
    load_json_from_cache, resolve_data_path
from clear_eval.pipeline.eval_utils import map_shortcomings_to_records, get_model_name_for_file, convert_results_to_ui_input, get_llm, \
    load_inputs, generate_model_predictions, evaluate_single_records, synthesize_shortcomings_from_df, \
    remove_duplicates_shortcomings, run_predictions_generation_save_results, produce_summaries_per_record
from clear_eval.pipeline.config_loader import load_yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_run_name(config):
    run_name = config.get("run_name")
    if not run_name:
        data_name = os.path.basename(config.get("data_path"))
        run_name = data_name.replace(".csv", "").split("_")[0]
    config["run_name"] = run_name
    return run_name


def run_generation_pipeline(config):
    task = config.get("task")
    if not task:
        raise ValueError(f"task config not specified")
    task_data = task_to_use_case_class.get(task)

    data_path = resolve_data_path(config["data_path"])
    data_df = load_inputs(config, data_path, load_predictions = False, task_data = task_data)

    gen_model = config.get('gen_model_name')
    output_dir = config['output_dir']
    ensure_dir(output_dir)
    run_name = get_run_name(config)
    gen_file_name = get_gen_file_name(run_name, gen_model)
    gen_output_path = os.path.join(output_dir, gen_file_name)
    run_predictions_generation_save_results(data_df, config, gen_output_path)

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


def aggregate_evaluations(config, output_dir, resume_enabled, eval_df, eval_llm, file_name_info, required_input_fields ):
    # step3: generate shortcomings
    shortcoming_list_output_path = f"{output_dir}/{SHORTCOMING_LIST_FILE_PREFIX}_{file_name_info}.json"
    shortcoming_list = None
    if resume_enabled:
        shortcoming_list = load_json_from_cache(shortcoming_list_output_path)
    if shortcoming_list is None:
        shortcoming_list = synthesize_shortcomings_from_df(eval_df, eval_llm, config)
        save_json_to_cache(shortcoming_list, shortcoming_list_output_path)
        resume_enabled = False

    if config["perform_clustering"]:
        # step3.5: cluster shortcomings
        deduplicated_shortcomings_list_output_path = f"{output_dir}/{SHORTCOMING_LIST_FILE_PREFIX}_{file_name_info}_dedup.json"
        deduplicated_shortcomings_list = None
        if resume_enabled:
            deduplicated_shortcomings_list = load_json_from_cache(deduplicated_shortcomings_list_output_path)
        if deduplicated_shortcomings_list is None:
            deduplicated_shortcomings_list = remove_duplicates_shortcomings(shortcoming_list, eval_llm, config["max_shortcomings"])
            save_json_to_cache(deduplicated_shortcomings_list, deduplicated_shortcomings_list_output_path)
            resume_enabled = False
    else:
        deduplicated_shortcomings_list = shortcoming_list

    # step4: map to records
    mapped_data_df = None
    mapping_data_output_path = f"{output_dir}/{MAPPING_FILE_PREFIX}_{file_name_info}.csv"
    if resume_enabled:
        mapped_data_df = load_dataframe_from_cache(mapping_data_output_path, expected_rows=len(eval_df))
    if mapped_data_df is None:
        use_full_text = config['use_full_text_for_analysis']
        qid_col = config['qid_column']
        max_workers = config['max_workers']
        high_score_threshold = config.get("high_score_threshold", 1)
        mapped_data_df = map_shortcomings_to_records(eval_df, eval_llm, deduplicated_shortcomings_list, use_full_text,
                                                     qid_col, max_workers, high_score_threshold)
        save_dataframe_to_cache(mapped_data_df, mapping_data_output_path)
        resume_enabled = False

    # step5 : convert to ui format and save
    output_df = convert_results_to_ui_input(mapped_data_df, config, required_input_fields)
    output_path = f"{output_dir}/analysis_results_{file_name_info}.csv"
    logger.info(f"\n--- Saving Custom Formatted Analysis to {output_dir} ---")
    save_dataframe_to_cache(output_df, output_path)
    logger.info(f"Custom formatted analysis results saved to {output_path}")

    # save outputs to zip
    parquet_bytes = get_parquet_bytes(output_df)
    #csv_bytes = output_df.to_csv(index=False).encode()
    json_bytes = json.dumps(config, indent=2).encode()

    # 3. Write to .zip
    zip_output_path = output_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_output_path, mode="w") as zf:
        zf.writestr("results.parquet", parquet_bytes)
        #zf.writestr("results.csv", csv_bytes)
        zf.writestr("metadata.json", json_bytes)
    logger.info(f"Results for uploading to ui are saved to {zip_output_path}")

def run_aggregation_pipeline(config):
    logger.info(f"run_aggregation_pipeline received run config: {config}")
    run_info = get_run_info(config)
    eval_dir = config["output_dir"]
    eval_file = os.path.join(eval_dir, f"{EVALUATION_FILE_PREFIX_WITH_SUMMARIES}_{run_info}.csv")
    eval_df = pd.read_csv(eval_file)
    run_aggregation_from_df(config, eval_df, run_info)


def run_aggregation_from_df(config, eval_df, file_name_info):

    task = config.get("task")
    if not task:
        raise ValueError(f"task config not specified")
    task_data = task_to_use_case_class.get(task)
    required_input_fields = task_data.required_input_fields
    if not task_data:
        raise ValueError(f"Invalid task specified: {task}, supported tasks are {list(task_to_use_case_class.keys())}")

    provider = config["provider"]
    eval_llm = get_llm(provider, config["eval_model_name"])
    output_dir = config['output_dir']
    ensure_dir(output_dir)
    resume_enabled = config['resume_enabled']
    aggregate_evaluations(config, output_dir, resume_enabled, eval_df, eval_llm, file_name_info, required_input_fields)


def get_run_info(config):
    reference_str = "based" if config['is_reference_based'] else 'free'
    eval_model_str = get_model_name_for_file(config["eval_model_name"])
    gen_model = config.get('gen_model_name')
    gen_model_str = get_model_name_for_file(gen_model)
    run_info = f"reference_{reference_str}_gen_{gen_model_str}_eval_{eval_model_str}"
    run_name = get_run_name(config)
    return f"{run_name}_{run_info}"


def run_eval_pipeline(config):
    # initialize
    logger.info(f"run_eval_pipeline received run config: {config}")
    task = config.get("task")
    if not task:
        raise ValueError(f"task config not specified")
    task_data = task_to_use_case_class.get(task)
    if not task_data:
        raise ValueError(f"Invalid task specified: {task}, supported tasks are {list(task_to_use_case_class.keys())}")

    provider = config["provider"]
    eval_llm = get_llm(provider, config["eval_model_name"])
    output_dir = config['output_dir']
    ensure_dir(output_dir)
    resume_enabled = config['resume_enabled']
    perform_generation = config['perform_generation']

    run_info = get_run_info(config)
    with open(os.path.join(output_dir, f"config_{run_info}.json"), 'w') as f:
        json.dump(config, f)

    # step0: load input data
    data_path = resolve_data_path(config["data_path"])
    data_df = load_inputs(config, data_path, load_predictions = not perform_generation, task_data = task_data)

    # step 1: perform generation (if needed)
    if perform_generation:
        logger.info(f"Performing generation analysis on {len(data_df)} examples")
        gen_df = None
        gen_model = config.get('gen_model_name')
        run_name = get_run_name(config)
        gen_file_name = get_gen_file_name(run_name, gen_model)
        gen_output_path = f"{output_dir}/{gen_file_name}"
        if resume_enabled:
            gen_df = load_dataframe_from_cache(gen_output_path, expected_rows=len(data_df))
        if gen_df is None:
            gen_df = run_predictions_generation_save_results(data_df, config, gen_output_path)
            # Do not use newer cached results, using these generations
            resume_enabled = False
    else:
        gen_df = data_df
        logger.info(f"Using input generation results for {len(data_df)} examples")

    # step2: generate evaluations + scores per single records
    evaluation_output_path_0 = f"{output_dir}/{EVALUATION_FILE_PREFIX_NO_SUMMARIES}_{run_info}.csv"
    eval_df_0 = None
    if resume_enabled:
        eval_df_0 = load_dataframe_from_cache(evaluation_output_path_0, expected_rows=len(gen_df))
        if eval_df_0 is None: # FOR BACKWARD COMPATIBILITY WHEN NEXT STEP INCLUDES RESULTS FOR THIS STEP
            evaluation_output_path_1 = f"{output_dir}/{EVALUATION_FILE_PREFIX_WITH_SUMMARIES}_{run_info}.csv"
            eval_df_0 = load_dataframe_from_cache(evaluation_output_path_1, expected_rows=len(gen_df))
    if eval_df_0 is None:
        eval_df_0 = evaluate_single_records(gen_df, eval_llm, config, task_data.generate_evaluation_model_prompt)
        save_dataframe_to_cache(eval_df_0, evaluation_output_path_0)
        resume_enabled = False

    # step2.5: SUMMARIES
    evaluation_output_path = f"{output_dir}/{EVALUATION_FILE_PREFIX_WITH_SUMMARIES}_{run_info}.csv"
    eval_df = None
    if resume_enabled:
        eval_df = load_dataframe_from_cache(evaluation_output_path, expected_rows=len(eval_df_0))
    if eval_df is None:
        eval_df = produce_summaries_per_record(eval_df_0, eval_llm, config,)
        save_dataframe_to_cache(eval_df, evaluation_output_path)
        resume_enabled = False

    generate_issues = config.get("generate_issues", True)
    if not generate_issues:
        return

    aggregate_evaluations(config, output_dir, resume_enabled, eval_df, eval_llm ,
                    file_name_info = run_info,
                    required_input_fields = task_data.required_input_fields)


if __name__ == "__main__":
    config = load_yaml(os.path.join(SCRIPT_DIR, 'setup', 'default_config.yaml'))
    run_eval_pipeline(config)
