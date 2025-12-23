import asyncio
import json
from importlib.resources import files
from typing import Tuple, Any, List, Dict
import numpy as np
import pandas as pd
from clear_eval.pipeline.use_cases.EvalUseCase import EvalUseCase
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL
from altk.pre_tool.sparc import SPARCReflectionComponent
from altk.core.toolkit import AgentPhase, ComponentConfig
from altk.pre_tool.core import SPARCExecutionMode, SPARCReflectionRunInput, Track, SPARCReflectionResult
from altk.core.llm import get_llm, BaseLLMClient

import logging

from clear_eval.pipeline.config_loader import load_config
from clear_eval.pipeline.llm_chat_utils import get_chat_llm

logger = logging.getLogger(__name__)

class ToolCallEvalUseCase(EvalUseCase):

    SPECS_COL = "api_spec"
    CONTEXT_COL = "context"
    RESPONSE_COL = "response"
    ID_COL = "id"

    async def eval_records(self, df, llm, config, score_col = SCORE_COL):
        """Evaluates predictions and adds scores."""
        logger.info(f"\n--- Evaluating Tool calls predictionsPredictions ---")
        df[EVALUATION_TEXT_COL] = ""
        df[score_col] = pd.NA  # Use Pandas NA for missing scores

        # convert CLEAR llm to ALTK llm
        altk_llm_client = self.clear_llm_client_to_altk_llm_client(llm, config.get("provider"), config.get("eval_model_name"))

        # call sparc with pipeline over examples, results store sorted results over the examples
        results = await self.generate_sparc_evaluation_results(df=df, llm_client=altk_llm_client)

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
    

    def clear_llm_client_to_altk_llm_client(self, llm, provider: str, model_name: str) -> BaseLLMClient:
        """Convert CLEAR's LLM object to ALTK's LLM Object."""

        if provider == "watsonx":
            MetricsClientCls = get_llm("watsonx.output_val")
            if llm.space_id:
                llm_client = MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url,
                    space_id=llm.space_id,
                )
            elif llm.project_id:
                llm_client = MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url._secret_value,
                    project_id=llm.project_id
            )
            else:
                raise KeyError("Either space_id or project_id must be specified for watsonx inference.")
            
        elif provider == "azure":
            MetricsClientCls = get_llm("azure_openai.async.output_val")
            llm_client = MetricsClientCls(
                model=model_name,
                api_key=llm.api_key._secret_value,
            )
        elif provider == "openai":
            MetricsClientCls = get_llm("openai.async.output_val")
            llm_client = MetricsClientCls(
                model=llm.model_name,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return llm_client


    @staticmethod
    def generate_evaluation_model_prompt(row, config):
        return None

    @staticmethod
    def get_default_generation_model_inputs(row, config):
        return ""

    def get_eval_from_results(self, result: SPARCReflectionResult) -> Tuple[str, float]:
        """
        Compute grade from LLMEvalKit pipeline result according to the specified logic.

        Args:
            result: PipelineResult from LLMEvalKit

        Returns:
            Grade as float in [0, 1]
        """
        logger.debug("=== DEBUG: Full result structure ===")
        logger.debug(result)
        if result.decision.name == "APPROVE":
            return "Tool call is valid.", 1.0
        else:
            explanation_text = "\n".join([issue.explanation for issue in result.issues])
            logger.debug("=== DEBUG: Explanation Text ===")
            logger.debug(explanation_text)
            return f"Tool call is invalid. Reasons:\n{explanation_text}", 0.0
    

    async def generate_sparc_evaluation_results(self, df: pd.DataFrame, llm_client: BaseLLMClient) -> List[SPARCReflectionResult]:
        """Generates sparc evaluation results."""
        sparc_component = SPARCReflectionComponent(
            config=ComponentConfig(llm_client=llm_client),
            track=Track.SLOW_TRACK,  # Use slow track for performance
        )
        reflection_results = []
        for _, example in df.iterrows():
            run_input = SPARCReflectionRunInput(
                    messages=json.loads(example[self.CONTEXT_COL]),
                    tool_specs=json.loads(example[self.SPECS_COL]),
                    tool_calls=[json.loads(example[self.RESPONSE_COL])],
                )
            reflection_result = await sparc_component.aprocess(run_input, phase=AgentPhase.RUNTIME)
            reflection_result = reflection_result.output.reflection_result
            reflection_results.append(reflection_result)
        
        return reflection_results


if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = str(files("clear_eval.pipeline.setup").joinpath("default_config.yaml"))
    # provider = "openai"
    # model_name = "gpt-4o-mini"
    provider = "watsonx"
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    config = load_config(DEFAULT_CONFIG_PATH, user_config_path=None, provider=provider, eval_model_name=model_name)

    sample_data_file = str(files("clear_eval.sample_data.tool_calls").joinpath("tool_calls_sample_data.csv"))
    df = pd.read_csv(sample_data_file)
    llm = get_chat_llm(config["provider"], config["eval_model_name"], eval_mode=True)

    tool_call_use_case = ToolCallEvalUseCase()
    evaluated_df = asyncio.run(tool_call_use_case.eval_records(df, llm, config))
    evaluated_df.to_csv(sample_data_file.replace(".csv", "_eval.csv"), index=False)