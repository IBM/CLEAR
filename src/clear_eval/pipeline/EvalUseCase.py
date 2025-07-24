from typing import List

import pandas as pd

from clear_eval.pipeline.evaluation_criteria import EvaluationCriteria, get_default_evaluation_criteria
from clear_eval.pipeline.propmts import get_math_evaluation_prompt_reference_based, get_math_evaluation_prompt_reference_less, \
    get_rag_evaluation_prompt_reference_based, get_rag_evaluation_prompt_reference_free, \
    get_general_evaluation_prompt_reference_less, get_general_evaluation_prompt_reference_based
from clear_eval.pipeline.constants import ANALYSIS_SKIPPED


class EvalUseCase:

    required_input_fields: List[str] = []
    is_reference_based: bool = False

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        pass

    @staticmethod
    def generate_evaluation_model_prompt(row, config):
        pass

    @staticmethod
    def generate_general_evaluation_model_prompt(row, config):
        model_answer = row[config['model_output_column']]
        model_input = row[config['model_input_column']]

        if  pd.isna(model_input) or pd.isna(model_answer):
            return f"{ANALYSIS_SKIPPED} - Missing Input"

        # Check if model output indicates a previous error
        if isinstance(model_answer, str) and model_answer.startswith("Error:"):
            return f"{ANALYSIS_SKIPPED} - Prediction Error"

        evaluation_criteria = config.get('evaluation_criteria')
        if not evaluation_criteria:
            evaluation_criteria = get_default_evaluation_criteria()
        if isinstance(evaluation_criteria, dict):
            evaluation_criteria = EvaluationCriteria.from_dict(evaluation_criteria)
        evaluation_criteria_str = evaluation_criteria.to_str()

        if config["is_reference_based"]:
            reference = row[config['reference_column']]
            if pd.isna(reference):
                return f"{ANALYSIS_SKIPPED} - Missing reference"
            return get_general_evaluation_prompt_reference_based(model_input, model_answer, reference, evaluation_criteria_str)
        else:
            return get_general_evaluation_prompt_reference_less(model_input, model_answer, evaluation_criteria_str)


class GeneralEvalUseCase(EvalUseCase):
    required_input_fields = ['model_input_column']

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        raise ValueError("model inputs should have been specified in the input")

    @staticmethod
    def generate_evaluation_model_prompt(row, config):
        return GeneralEvalUseCase.generate_general_evaluation_model_prompt(row, config)

class MathUseCase(EvalUseCase):
    required_input_fields = ["question_column"]

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        model_input = (f"Answer the following math word problem:\n"
                       f"Question: %s\n"
                       f"Answer:"
                       )
        return model_input % row[config['question_column']]

    @staticmethod
    def generate_evaluation_model_prompt(row, config):
        if config["use_general_prompt"]:
            return GeneralEvalUseCase.generate_general_evaluation_model_prompt(row, config)

        question = row[config['question_column']]
        model_answer = row[config['model_output_column']]

        # Basic check for valid inputs needed for evaluation
        if pd.isna(question) or pd.isna(model_answer):
            return f"{ANALYSIS_SKIPPED} - Missing Input"

        # Check if model output indicates a previous error
        if isinstance(model_answer, str) and model_answer.startswith("Error:"):
            return f"{ANALYSIS_SKIPPED} - Prediction Error"

        if config["is_reference_based"]:
            reference = row[config['reference_column']]
            if pd.isna(reference):
                return f"{ANALYSIS_SKIPPED} - Missing reference"
            return get_math_evaluation_prompt_reference_based(question, model_answer, reference)
        else:
            return get_math_evaluation_prompt_reference_less(question, model_answer)


class RAGUseCase(EvalUseCase):
    required_input_fields = ["question_column", "documents_column"]

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        question = row[config['question_column']]
        contexts = row[config['documents_column']]
        return (f"Answer the following question based on the given context:\n"
                    f"Context: {contexts}\n"
                    f"Question: {question}\n"
                    f"Answer:"
                )

    @staticmethod
    def generate_evaluation_model_prompt(row, config):

        if config["use_general_prompt"]:
            return GeneralEvalUseCase.generate_general_evaluation_model_prompt(row, config)

        question = row[config['question_column']]
        documents = row[config['documents_column']]
        model_answer = row[config['model_output_column']]

        # Basic check for valid inputs needed for evaluation
        if pd.isna(question) or pd.isna(documents) or pd.isna(model_answer):
            return f"{ANALYSIS_SKIPPED} - Missing Input"

        # Check if model output indicates a previous error
        if isinstance(model_answer, str) and model_answer.startswith("Error:"):
            return f"{ANALYSIS_SKIPPED} - Prediction Error"

        if config["is_reference_based"]:
            reference = row[config['reference_column']]
            if pd.isna(reference):
                return f"{ANALYSIS_SKIPPED} - Missing reference"
            return get_rag_evaluation_prompt_reference_based(question, model_answer, reference)
        else:
            return get_rag_evaluation_prompt_reference_free(question, documents, model_answer)


task_to_use_case_class = {
    "math": MathUseCase,
    "rag": RAGUseCase,
    "general": GeneralEvalUseCase
}