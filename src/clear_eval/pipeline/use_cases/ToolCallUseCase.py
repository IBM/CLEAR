from typing import Tuple, Any
import pandas as pd
from clear_eval.pipeline.use_cases.EvalUseCase import EvalUseCase
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL

import logging
logger = logging.getLogger(__name__)

class ToolCallEvalUseCase(EvalUseCase):

    spec_cols = "API_SPEC"
    context_col = "CONVERSATION_HISTORY"


    def eval_records(self, df, llm, config, score_col = SCORE_COL):
        """Evaluates predictions and adds scores."""
        logger.info(f"\n--- Evaluating Tool calls predictionsPredictions ---")
        df[EVALUATION_TEXT_COL] = ""
        df[score_col] = pd.NA  # Use Pandas NA for missing scores

        # TODO
        # spark_llm = get_spark_llm(llm)
        # construct input examples from df (input fields spec_cols and context_col given)
        # construct spark pipeline

        # call spark with pipeline over examples, results store sorted results over the examples
        results = []

        # extract output score and evaluation text from each results (concatenate failing explanations? minimum/average score over metrics?)
        for i, result in enumerate(results):
            (eval_text, score) = self.get_eval_from_results(result)  # TODO extract eval text and score from results
            score = score if pd.isna(score) else float(score)

            df.at[df.index[i], EVALUATION_TEXT_COL] = eval_text
            df.at[df.index[i], score_col] = score if pd.isna(score) else float(score)

        logger.info("Finished evaluating predictions.")
        # Convert score column to nullable float type
        df[score_col] = df[score_col].astype('Float64')
        return df


    @staticmethod
    def generate_evaluation_model_prompt(row, config):
        return None

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        raise NotImplementedError("Tool Call generations must be provided")

    def get_eval_from_results(self, result: Any) -> Tuple[str, float]:
        # TODO: IMPLEMENT
        pass
