import logging
import os
import numpy as np
from clear_eval.pipeline.caching_utils import save_dataframe_to_cache
from clear_eval.pipeline.llm_chat_utils import get_chat_llm
import random
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from clear_eval.pipeline.constants import IDENTIFIED_SHORTCOMING_COL, EVALUATION_TEXT_COL, EVALUATION_SUMMARY_COL, \
    SHORTCOMING_PREFIX, SCORE_COL, MAPPING_NO_ISSUES, ANALYSIS_SKIPPED
from clear_eval.pipeline.propmts import get_summarization_prompt, get_shortcomings_synthesis_prompt, \
     get_shortcomings_clustering_prompt, get_shortcomings_mapping_system_prompt, \
    get_shortcomings_mapping_human_prompt, get_shortcomings_synthesis_prompt_cont
import re
from clear_eval.pipeline.threading_utils import run_func_in_threads
logger = logging.getLogger(__name__)

def is_missing_or_error(eval_text):
    if pd.isna(eval_text) or not eval_text or not eval_text.strip() or \
        eval_text.startswith(ANALYSIS_SKIPPED) or eval_text.startswith("Error:"):
        return True
    return False

def get_model_name_for_file(model_name):
    if not model_name:
        return "none"
    return model_name.split("/")[-1].replace("-", "_").lower().replace(".","_").replace("-vision", "")

def evaluate_row(row, config, llm, generate_evaluation_model_prompt_func):
        prompt = generate_evaluation_model_prompt_func(row, config)
        if prompt.startswith(ANALYSIS_SKIPPED):
            return prompt, pd.NA
        try:
            response = llm.invoke(prompt)
            return parse_evaluation_response(response.content)
        except Exception as e:
            return f"Error during evaluation: {str(e)}", pd.NA


def evaluate_single_records(df, llm, config, get_evaluation_prompt_func):
    """Evaluates predictions and adds scores."""
    logger.info(f"\n--- Evaluating Predictions ---")

    if llm is None:
        logger.error("Error: Evaluation LLM not initialized. Skipping evaluation.")
        df[EVALUATION_TEXT_COL] = "Error: LLM not available"
        df[SCORE_COL] = pd.NA
        return df

    df[EVALUATION_TEXT_COL] = ""
    df[SCORE_COL] = pd.NA  # Use Pandas NA for missing scores

    inputs_for_threading = []
    for idx, row in df.iterrows():
        inputs_for_threading.append((
           row, config, llm, get_evaluation_prompt_func
        )
    )

    results = run_func_in_threads(
        evaluate_row,
        inputs_for_threading,
        max_workers=config['max_workers'],  # Or make this configurable
        error_prefix="Error: Evaluation Error for ",
        progress_desc=f"Evaluating predictions "
    )
    for i, result in enumerate(results):
        if result.is_success:
            (eval_text, score) = result.result
            score = score if pd.isna(score) else float(score)
        else:
            eval_text = result.error
            score = None

        df.at[df.index[i], EVALUATION_TEXT_COL] = eval_text
        df.at[df.index[i], SCORE_COL] = score if pd.isna(score) else float(score)

    logger.info("Finished evaluating predictions.")
    # Convert score column to nullable float type
    df[SCORE_COL] = df[SCORE_COL].astype('Float64')
    return df

def produce_summaries_per_record(df, llm, config):
    #### generate evaluation summaries
    inputs_for_summary = []
    for _, row in df.iterrows():
        inputs_for_summary.append((row.get(EVALUATION_TEXT_COL), llm, row.get(config['qid_column'], 'N/A')))

    # Use run_func_in_threads for parallel summary generation
    logger.info(f"Generating evaluation summaries for {len(inputs_for_summary)} items ...")
    thread_results = run_func_in_threads(
        generate_evaluation_summary,
        inputs_for_summary,
        max_workers=config['max_workers'],  # Or make this configurable
        error_prefix="Error: Summary Generation Error for ",
        progress_desc=f"Generating evaluation summaries"
    )
    results = [r.result if r.is_success else r.error for r in thread_results]
    df[EVALUATION_SUMMARY_COL] = results
    return df

def predict_row(llm, model_input, question_id):
    try:
        response = llm.invoke(model_input)
        return response.content
    except Exception as e:
        logger.error(f"Error processing example ({question_id}): {e}")
        return f"Error: {str(e)}"


def parse_evaluation_response(response_content):
    """Parses LLM response for evaluation text and score."""
    text = response_content.strip()
    score = None

    # Attempt to find a score line like "Evaluation score: X.Y"
    # Make regex more robust to handle variations
    score_match = re.search(r"Evaluation score:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if score_match:
        try:
            s = float(score_match.group(1))
            if 0 <= s <= 1:
                score = s
                # Try to remove the score line and preceding/following whitespace
                text = re.sub(r"(\n|^)\s*Evaluation score:\s*\d+(?:\.\d+)?\s*(\n|$)", "\n", text).strip()

        except ValueError:
            pass  # Keep searching if conversion fails

    # Fallback: try to find any float between 0 and 1 if not found above
    if score is None:
        # Look for floats or integers (0 or 1) that could be scores
        potential_scores = re.findall(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', response_content)
        for num_str in reversed(potential_scores):  # Check from end, often score is last
            try:
                s = float(num_str)
                # Check if it's exactly 0, 1, or between 0 and 1
                if 0 <= s <= 1:
                    score = s
                    # Simple removal - might not be perfect but better than nothing
                    text = text.replace(num_str, "").strip()
                    break
            except ValueError:
                continue

    if score is None:
        logger.warning(f"Warning: Could not extract valid score from evaluation response: {response_content}")

    # Clean up common artifacts if needed
    text = text.replace("--- Begin Evaluation ---", "").replace("Textual Evaluation:", "").strip()

    return text, score


def generate_evaluation_summary(evaluation_text, llm, question_id="N/A"):
    """Generates a concise summary of the evaluation text using an LLM."""
    if is_missing_or_error(evaluation_text):
        return "Evaluation text was empty or missing."
    if llm is None:
        logger.warning(f"Skipping summary generation for QID {question_id} as LLM is not available.")
        return "Error: LLM not available for summary."

    prompt = get_summarization_prompt(evaluation_text)
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        summary = response.content.strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating evaluation summary for QID {question_id}: {e}")
        return "Error during summary generation."


def sample_summaries_by_score(df, N):
    scores = df[SCORE_COL]
    score_indices = df.index
    alpha = 4.0
    weights = (1 - scores) ** alpha
    probabilities = weights / weights.sum()
    sampled_indices = list(np.random.choice(score_indices, size=N, replace=False, p=probabilities))
    sampled_df_p = df.loc[sampled_indices]
    return sampled_df_p


def get_evaluation_texts_for_synthesis(df, use_full_text, score_col, score_threshold=1, max_eval_text_for_synthesis=None):
    # Get valid evaluation texts from evaluation texts with score < 1
    evaluation_text_col = EVALUATION_TEXT_COL if use_full_text else EVALUATION_SUMMARY_COL
    valid_df = df[df[score_col] < score_threshold]
    valid_df = valid_df[~valid_df[evaluation_text_col].apply(is_missing_or_error)]

    if max_eval_text_for_synthesis and max_eval_text_for_synthesis < len(valid_df):
        final_df = sample_summaries_by_score(valid_df, max_eval_text_for_synthesis)
    else:
        final_df = valid_df

    valid_eval_texts = final_df[evaluation_text_col].dropna().tolist()
    logger.info(f"returning {len(valid_eval_texts)}/{len(valid_df)} valid evaluation texts ({len(df)} total)")
    return valid_eval_texts

def synthesize_shortcomings_from_df(df, llm, config):
    use_full_text = config['use_full_text_for_analysis']
    max_eval_text_for_synthesis = config['max_eval_text_for_synthesis']
    eval_texts = get_evaluation_texts_for_synthesis(df, use_full_text=use_full_text, score_col=SCORE_COL,
                                                    score_threshold=config.get("high_score_threshold", 1),
                                                    max_eval_text_for_synthesis=max_eval_text_for_synthesis)
    return synthesize_shortcomings(eval_texts, llm,
                                   min_shortcomings=config['min_shortcomings'])

def synthesize_shortcomings(evaluation_text_list, llm, min_shortcomings=None,
                            max_shortcomings = None, batch_size=100, ):
    """Analyzes evaluation texts to identify common shortcomings."""
    logger.info(f"\nSynthesizing Shortcomings List")

    if llm is None:
        logger.error("Error: LLM not initialized. Cannot synthesize shortcomings.")
        return None
    if not evaluation_text_list:
        logger.info("No valid evaluation texts found to analyze for shortcomings.")
        return []

    batches = [evaluation_text_list[x:x+batch_size] for x in range(0, len(evaluation_text_list), batch_size)]
    logger.info(f"Sending {len(evaluation_text_list)} evaluation texts to LLM for shortcoming synthesis in {len(batches)} batches")
    overall_shortcoming_list = []
    for evaluation_text_batch in batches:
        # Concatenate texts with separators
        concatenated_texts = "\n---\n".join(evaluation_text_batch)

        # Create the prompt for synthesis
        if not overall_shortcoming_list:
            synthesis_prompt = get_shortcomings_synthesis_prompt(concatenated_texts, max_shortcomings)
        else:
            shortcoming_list_text = "\n---\n".join(overall_shortcoming_list)
            synthesis_prompt = get_shortcomings_synthesis_prompt_cont(concatenated_texts, shortcoming_list_text, max_shortcomings)

        try:
            messages = [
                SystemMessage(
                    content="You are an analyst synthesizing common shortcomings from evaluation texts. Respond ONLY with a Python list of strings."),
                HumanMessage(content=synthesis_prompt)
            ]
            response = llm.invoke(messages).content.strip()


            synthesized_list = parse_shortcoming_list_response(response)
            if synthesized_list:
                logger.info(f"Received synthesis response of {len(synthesized_list)} shortcomings")
                overall_shortcoming_list.extend(synthesized_list)
            else:
                logger.info(f"Could not identify new shortcomings")
        except Exception as e:
            logger.error(f"key points synthesis failed: {e}")
            continue

    if overall_shortcoming_list:
        if min_shortcomings and len(overall_shortcoming_list) < min_shortcomings:
            logger.warning(
                f"Warning: Only {len(overall_shortcoming_list)} shortcomings identified, below minimum of {min_shortcomings}")

        return overall_shortcoming_list
    else:
        logger.warning("Failed to parse a valid list from the synthesis response.")
        return []


def parse_shortcoming_list_response(response_content):
    """Parses LLM response expected to be a Python list of strings."""
    shortcomings = None
    try:
        # Find the list within the response
        list_match = re.search(r'\[\s*(".*?"(?:\s*,\s*".*?")*)\s*\]', response_content, re.DOTALL)
        if list_match:
            list_content = list_match.group(1)
            # Split by comma, handling potential commas inside quotes carefully
            shortcomings = re.findall(r'"(.*?)"', list_content)
        else:
            #extract list with single quotes:
            list_match = re.search(r"\[\s*('.*?'(?:\s*,\s*'.*?')*)\s*\]", response_content, re.DOTALL)
            if list_match:
                list_content = list_match.group(1)
                # Match quotes, but allow apostrophes inside unless right before , or end of str
                shortcomings = re.findall(r"'(.*?)'(?=\s*(?:,|$))", list_content, re.DOTALL)
            # extract empty list
            elif re.search(r'\[[\s]*\]', response_content, re.DOTALL):
                logger.warning(f"Warning: Parsed an empty list of shortcomings from: {response_content}")
                shortcomings = []
        if shortcomings is not None:
            shortcomings = [s.strip() for s in shortcomings if s.strip()]
            return shortcomings
        else:
            logger.warning(f"Warning: Could not parse list of shortcomings from: {response_content}")
            return None
    except Exception as e:
        logger.error(f"Error parsing shortcoming list response: {e}\nResponse: {response_content}")
        return None


def parse_mapping_response(response, question_id, num_shortcomings, ):
    # Attempt to parse the list
    parsed_list = None
    # Regex to find list like [0, 1, 0] with optional spaces
    list_match = re.search(r'\[\s*([01](?:\s*,\s*[01])*)?\s*\]', response)
    if list_match:
        list_str = list_match.group(1)  # Content within brackets
        if list_str:  # Check if list is not empty like '[]'
            binary_values = [val.strip() for val in list_str.split(',')]
            if len(binary_values) == num_shortcomings:
                try:
                    parsed_list = [int(value) for value in binary_values]
                except ValueError:
                    logger.warning(
                        f"Warning: Could not convert parsed list values to int for {question_id}: {binary_values}")
            else:
                logger.warning(
                    f"Warning: Parsed list length mismatch for {question_id}. Expected {num_shortcomings}, got {len(binary_values)}. Response: {response}")
        else:  # Handle empty list '[]'
            if num_shortcomings == 0:  # If expecting 0 shortcomings, empty list is correct
                parsed_list = []
            else:  # If expecting > 0 shortcomings, empty list is wrong
                logger.warning(
                    f"Warning: Parsed empty list '[]' but expected {num_shortcomings} shortcomings for {question_id}. Response: {response}")

    if parsed_list is not None:
        shortcomings_result = parsed_list
    else:
        # Fallback if parsing fails
        logger.warning(
            f"Could not parse LLM response list format for {question_id}: {response}. Defaulting to zeros.")
        shortcomings_result = [0] * num_shortcomings

    return shortcomings_result


def analyze_shortcoming_row(eval_text, question_id, shortcomings_list, llm, system_prompt ):
        # Skip analysis if eval text is invalid or indicates prior errors
        num_shortcomings = len(shortcomings_list)
        if is_missing_or_error(eval_text):
            shortcomings_result = [0] * num_shortcomings
            identified_shortcomings_names = []
            return shortcomings_result, identified_shortcomings_names
        elif eval_text.startswith(MAPPING_NO_ISSUES):
            shortcomings_result = [0] * num_shortcomings
            identified_shortcomings_names = []
            return shortcomings_result, identified_shortcomings_names
        else:
            human_prompt = get_shortcomings_mapping_human_prompt(eval_text, num_shortcomings)
            try:
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
                response = llm.invoke(messages).content.strip()

                shortcomings_result =  parse_mapping_response(response, question_id, num_shortcomings)
                identified_shortcomings_names = [shortcomings_list[i] for i, present in enumerate(shortcomings_result)
                                                 if present == 1]
                return shortcomings_result, identified_shortcomings_names

            except Exception as e:
                logger.error(f"LLM analysis failed for {question_id}: {e}")
                shortcomings_result = [0] * num_shortcomings
                identified_shortcomings_names = ["Analysis Error"]
                return shortcomings_result, identified_shortcomings_names


def map_shortcomings_to_records(df, llm, shortcomings_list,
                                use_full_text, qid_col, max_workers, high_score_threshold):
    """Analyzes evaluation text for the dynamically generated shortcomings."""
    logger.info(f"\n--- Analyzing Shortcomings based on Synthesized List ---")
    df[IDENTIFIED_SHORTCOMING_COL] = ""
    evaluation_text_col = EVALUATION_TEXT_COL if use_full_text else EVALUATION_SUMMARY_COL
    if shortcomings_list is None:
        logger.error("Error: Shortcomings list was not generated successfully. Skipping analysis.")
        df[IDENTIFIED_SHORTCOMING_COL] = "Error: Synthesis failed"
        return df
    if not shortcomings_list:
        logger.warning("Warning: Synthesized shortcomings list is empty. Skipping analysis.")
        return df

    num_shortcomings = len(shortcomings_list)
    #print(f"Analyzing shortcomings using {llm.model_name} against {num_shortcomings} synthesized criteria...")

    # Prepare components for the analysis prompt
    system_prompt = get_shortcomings_mapping_system_prompt(shortcomings_list)

    # Initialize shortcoming columns in DataFrame
    for i in range(num_shortcomings):
        df[f'{SHORTCOMING_PREFIX}{i + 1}'] = 0  # Initialize with 0
    df[IDENTIFIED_SHORTCOMING_COL] = ""  # Initialize as empty string

    inputs_for_threading = []
    n_records_to_map = 0
    for idx, row in df.iterrows():
        if pd.isna(row[SCORE_COL]):
            inputs_for_threading.append(("", row.get(qid_col, f"row_{idx}"), shortcomings_list, llm, system_prompt))
        elif row[SCORE_COL] >= high_score_threshold:
            inputs_for_threading.append((MAPPING_NO_ISSUES, row.get(qid_col, f"row_{idx}"), shortcomings_list, llm, system_prompt))
        else:
            n_records_to_map += 1
            inputs_for_threading.append((str(row[evaluation_text_col]), row.get(qid_col, f"row_{idx}"), shortcomings_list, llm, system_prompt))
    logger.info(f"Mapping {n_records_to_map}/{len(df)} records to {len(shortcomings_list)} discovered shortcomings.")
    thread_results = run_func_in_threads(
        analyze_shortcoming_row,
        inputs_for_threading,
        max_workers=max_workers,
        error_prefix="Error: Shortcoming Analysis Error for ",
        progress_desc=f"Analyzing shortcomings"
    )

    for i, result in enumerate(thread_results):
        if result.is_success:
            (shortcomings_result, identified_shortcomings_names) = result.result
        else:
            shortcomings_result = [0] * num_shortcomings
            identified_shortcomings_names = [ANALYSIS_SKIPPED]
        # Store results back into DataFrame
        for j in range(num_shortcomings):
            # Use .iloc for setting value by position to avoid index alignment issues if df index isn't standard range
            df.iloc[i, df.columns.get_loc(f'{SHORTCOMING_PREFIX}{j + 1}')] = shortcomings_result[j]
        df.iloc[i, df.columns.get_loc(IDENTIFIED_SHORTCOMING_COL)] = '; '.join(
                identified_shortcomings_names) if identified_shortcomings_names else ''

    return df


def load_inputs(config, data_path, load_predictions, task_data):

    if not data_path or not isinstance(data_path, str):
        raise TypeError("Please provide a valid data path")
    if not data_path.endswith(".csv"):
        raise ValueError("Data path must end with .csv")
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    if not task_data:
        raise ValueError(f"Task not specified: {data_path}")

    data_df = pd.read_csv(data_path)
    if config["reference_column"] not in list(data_df.columns) and config["is_reference_based"]:
        raise ValueError(f"Reference column {config['reference_column']} not found in data and is_reference_based is true")

    if load_predictions and config["model_output_column"] not in list(data_df.columns):
        ValueError(f"Response column {config['model_output_column']} not found in data and perform_generation is False."
                         f"Cannot read previous predictions")

    if not load_predictions and config["model_output_column"] in list(data_df.columns):
        logger.warning("WARNING: Model output column exists in data but perform_generation is True: overriding existing generations.")

    if config['qid_column'] not in list(data_df.columns):
        logger.info(f"question_id column {config['qid_column']} not found in data")
        data_df[config['qid_column']] = list(range(len(data_df)))

    model_input_column = config["model_input_column"]
    if model_input_column not in list(data_df.columns):
        data_df.loc[:, model_input_column] = data_df.apply(lambda row: task_data.get_default_generation_model_inputs(row, config), axis=1)

    for c in task_data.required_input_fields:
        if config[c] not in list(data_df.columns):
              raise ValueError(f"Required column {config[c]} not found in data")

    max_samples = config["max_examples_to_analyze"]
    if max_samples and max_samples < len(data_df):
        logger.info(f"Selecting first {max_samples}/{len(data_df)} examples to analyze")
        data_df = data_df.head(max_samples)

    return data_df

def run_predictions_generation_save_results(data_df, config, output_path):
    provider = config["provider"]
    gen_llm = get_llm(provider=provider, model_name=config["gen_model_name"], eval_mode=False)
    gen_df = generate_model_predictions(data_df, gen_llm, config)
    save_dataframe_to_cache(gen_df, output_path)
    return gen_df

def generate_model_predictions(df, llm, config):
    """Generates model responses for the formatted data."""
    logger.info(f"\n--- Running Predictions ---")
    if llm is None:
        logger.error("Error: Prediction LLM not initialized. Skipping prediction step.")
        df['model_output'] = "Error: LLM not available"
        return df

    logger.info(f"Generating responses for {len(df)} examples")

    inputs_for_threading = []
    for i, row in df.iterrows():
        inputs_for_threading.append((llm, row[config['model_input_column']], row[config['qid_column']]))

    thread_results = run_func_in_threads(
        predict_row,
        inputs_for_threading,
        max_workers=config["max_workers"],  # Or make this configurable
        error_prefix="Error: Prediction Error for ",
        progress_desc=f"Generating predictions"
    )

    for i, result in enumerate(thread_results):
        result = result.result if result.is_success else result.error
        df.at[df.index[i], config["model_output_column"]] = result

    return df

def remove_duplicates_shortcomings(shortcoming_list, llm, max_shortcomings, num_retries=3):
    logger.info(f"Removing duplications from list of {len(shortcoming_list)} shortcomings")
    try:
        clustering_prompt = get_shortcomings_clustering_prompt(shortcoming_list, max_shortcomings)
        new_shortcoming_list = []
        retries = 0
        while retries < num_retries:
            try:
                analysis_result = llm.invoke(clustering_prompt)
                analysis_result = analysis_result.content
                if is_missing_or_error(analysis_result):
                    raise RuntimeError(f"Failed to get shortcomings without duplications response")

                new_shortcoming_list = parse_shortcoming_list_response(analysis_result)
                if new_shortcoming_list is None:
                    logger.error(f"Failed attempt {retries}/{num_retries} to parse shortcomings without duplications from response {analysis_result}")
                    raise RuntimeError(f"Failed to parse new shortcomings without duplications from response")
                break
            except Exception as e:
                retries += 1
                logger.warning(f"Failed to parse new shortcomings without duplications from response")
                if retries < num_retries:
                    continue
                else:
                    raise e


        if new_shortcoming_list and max_shortcomings and len(new_shortcoming_list) > max_shortcomings:
            logger.warning(f"Limiting to top {max_shortcomings}/{len(new_shortcoming_list)} most significant shortcomings")
            return new_shortcoming_list[:max_shortcomings]
        else:
            logger.info(f"Returning list of {len(new_shortcoming_list)} shortcomings")
            return new_shortcoming_list

    except Exception as e:
       raise Exception(f"Failed to get shortcomings without duplications: {e}")

def convert_results_to_ui_input(df, config, required_input_fields):
    try:
        custom_output_df = pd.DataFrame()

        for c in required_input_fields:
            custom_output_df[c.replace("_column", "")] = df.get(config[c], pd.Series(dtype='str'))
        for c in config.get("input_columns", []):
            custom_output_df[c] = df.get(c, pd.Series(dtype='str'))

        custom_output_df[SCORE_COL] = df.get(SCORE_COL, pd.Series(dtype='float'))  # Or 'reference_free_score'
        custom_output_df[EVALUATION_TEXT_COL] = df.get(EVALUATION_TEXT_COL, pd.Series(dtype='str'))
        custom_output_df[EVALUATION_SUMMARY_COL] = df.get(EVALUATION_SUMMARY_COL, pd.Series(dtype='str'))  # From new step
        custom_output_df['question_id'] = df.get(config['qid_column'], pd.Series(dtype='str'))
        custom_output_df['model_input'] = df.get(config['model_input_column'], pd.Series(dtype='str'))
        custom_output_df['response'] = df.get(config['model_output_column'], pd.Series(dtype='str'))
        custom_output_df['ground_truth'] = df.get(config['reference_column'], pd.Series(dtype='str'))


        def get_recurring_issues_indices(r):
            ids_col = [c for c in r.keys() if c and isinstance(c, str) and c.startswith(SHORTCOMING_PREFIX)]
            return [int(c.replace(SHORTCOMING_PREFIX, "")) for c in ids_col if r[c]]

        def get_recurring_issues_list(r, delimiter=";"):
            shortcomings_list = r.get(IDENTIFIED_SHORTCOMING_COL)
            if not shortcomings_list or pd.isna(shortcomings_list):
                return []
            return [x.strip() for x in shortcomings_list.split(delimiter)]

        df.loc[:, "recurring_issues"] = df.apply(lambda r: get_recurring_issues_indices(r), axis=1)
        custom_output_df["recurring_issues"] = df["recurring_issues"]

        df.loc[:, "recurring_issues_str"] = df.apply(lambda r: get_recurring_issues_list(r), axis=1)
        custom_output_df["recurring_issues_str"] = df["recurring_issues_str"]

        required_cols =[config[r] for r in required_input_fields] + config.get("input_columns", []) + \
                         ["question_id", 'model_input', 'response',
                         'score', 'evaluation_text', 'evaluation_summary',
                         'recurring_issues', 'recurring_issues_str', 'ground_truth']
        required_cols = list(dict.fromkeys(required_cols))

        for col in required_cols:
            if col not in custom_output_df.columns:
                custom_output_df[col] = pd.Series(dtype='object')  # Add empty series if any is missing

        custom_output_df = custom_output_df[required_cols]
        return custom_output_df
    except Exception as e:
        logger.error(f"Warning: Error converting custom analysis results to CSV: {e}")
        return None

def get_llm(provider, model_name, eval_mode=True):
    try:
        logger.info(f"Getting llm for model: {model_name}, provider {provider}, eval_mode {eval_mode}")
        llm = get_chat_llm(provider, model_name, eval_mode=eval_mode)
    except Exception as e:
        raise Exception(f"Error initializing LLM {provider}, {model_name}). Details: {e}")
    if llm is None:
        raise ValueError(f"Error initializing LLM ({provider}, {model_name}).")
    return llm

def create_aggregations_from_df(df, eval_llm, use_full_text, max_shortcomings, high_score_threshold, max_workers,
                                score_col = SCORE_COL, qid_col="id", max_eval_text_for_synthesis=None):
    evaluation_text_list = get_evaluation_texts_for_synthesis(df, use_full_text, score_col, high_score_threshold, max_eval_text_for_synthesis)
    shortcoming_list = synthesize_shortcomings(evaluation_text_list, eval_llm, min_shortcomings=None,
                            max_shortcomings = None, batch_size=100, )
    deduplicated_shortcomings_list = remove_duplicates_shortcomings(shortcoming_list, eval_llm, max_shortcomings)
    mapped_data_df = map_shortcomings_to_records(df, eval_llm, deduplicated_shortcomings_list, use_full_text,
                                                 qid_col, max_workers, high_score_threshold)
    qid_to_issues = zip(mapped_data_df[qid_col], mapped_data_df[IDENTIFIED_SHORTCOMING_COL])
    return qid_to_issues


