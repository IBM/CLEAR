import json
from tqdm import tqdm
from importlib.resources import files
from typing import Tuple, List
import pandas as pd
from clear_eval.pipeline.use_cases.EvalUseCase import EvalUseCase
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL
from altk.pre_tool.sparc import SPARCReflectionComponent
from altk.core.toolkit import AgentPhase, ComponentConfig
from altk.pre_tool.core import SPARCReflectionRunInput, Track, SPARCReflectionResult, SPARCExecutionMode
from altk.core.llm import get_llm, BaseLLMClient

import logging

from clear_eval.pipeline.config_loader import load_config
from clear_eval.pipeline.inference_utils.llm_client import run_async, LiteLLMClient, LangChainClient
from clear_eval.pipeline.full_pipeline import get_eval_llm_from_config

logger = logging.getLogger(__name__)


class ToolCallEvalUseCase(EvalUseCase):
    SPECS_COL = "api_spec"
    CONTEXT_COL = "context"
    RESPONSE_COL = "response"
    required_input_fields = [CONTEXT_COL, SPECS_COL]

    def eval_records(self, df, llm, config, score_col=SCORE_COL):
        return run_async(self.eval_records_async(df, llm, config, score_col))

    async def eval_records_async(self, df, llm, config, score_col=SCORE_COL):
        """Evaluates predictions and adds scores."""
        logger.info(f"\n--- Evaluating Tool calls predictionsPredictions ---")
        df[EVALUATION_TEXT_COL] = ""
        df[score_col] = pd.NA  # Use Pandas NA for missing scores

        # convert CLEAR llm to ALTK llm
        altk_llm_client = self.clear_llm_client_to_altk_llm_client(llm, config.get("provider"),
                                                                   config.get("eval_model_name"))

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

    def clear_llm_client_to_altk_llm_client(self, llm_client, provider: str, model_name: str) -> BaseLLMClient:
        """Convert CLEAR's LLM object to ALTK's LLM Object."""

        # LiteLLMClient - use ALTK's native litellm support
        if isinstance(llm_client, LiteLLMClient):
            MetricsClientCls = get_llm("litellm.output_val")
            # LiteLLM model format: provider/model_name (consistent with LiteLLMClient)
            litellm_model = f"{provider}/{model_name}"
            return MetricsClientCls(model_name=litellm_model)

        # LangChainClient - extract from underlying LangChain object
        if isinstance(llm_client, LangChainClient):
            llm = llm_client.llm
        else:
            # Fallback for raw LangChain objects
            llm = llm_client

        if provider == "watsonx":
            MetricsClientCls = get_llm("watsonx.output_val")
            if llm.space_id:
                return MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url,
                    space_id=llm.space_id,
                )
            elif llm.project_id:
                return MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url._secret_value,
                    project_id=llm.project_id
                )
            else:
                raise KeyError("Either space_id or project_id must be specified for watsonx inference.")

        elif provider == "azure":
            MetricsClientCls = get_llm("azure_openai.async.output_val")
            return MetricsClientCls(
                model=model_name,
                api_key=llm.api_key._secret_value,
            )
        elif provider == "openai":
            MetricsClientCls = get_llm("openai.async.output_val")
            kwargs = {"model": llm.model_name}

            if hasattr(llm, 'openai_api_base') and llm.openai_api_base:
                kwargs["base_url"] = llm.openai_api_base

            if hasattr(llm, 'openai_api_key') and llm.openai_api_key:
                kwargs["api_key"] = llm.openai_api_key._secret_value

            return MetricsClientCls(**kwargs)
        else:
            raise ValueError(f"Unsupported provider '{provider}' for tool_call task. "
                             f"Supported providers: openai, watsonx, or use_litellm=True.")

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

    async def generate_results(self, df, llm_client, has_spec):
        sparc_component = SPARCReflectionComponent(
            config=ComponentConfig(llm_client=llm_client),
            track=Track.SLOW_TRACK if has_spec else Track.SPEC_FREE,
            execution_mode=SPARCExecutionMode.ASYNC,
        )
        reflection_results = []
        for _, example in tqdm(df.iterrows(), total=len(df), desc="Evaluating tool calls with SPARC"):
            run_input = SPARCReflectionRunInput(
                messages=json.loads(example[self.CONTEXT_COL]),
                tool_specs=json.loads(example[self.SPECS_COL]) if has_spec else [],
                tool_calls=[json.loads(example[self.RESPONSE_COL])],
            )
            reflection_result = await sparc_component.aprocess(run_input, phase=AgentPhase.RUNTIME)
            reflection_result = reflection_result.output.reflection_result
            reflection_results.append(reflection_result)
        return reflection_results

    async def generate_sparc_evaluation_results(self, df: pd.DataFrame, llm_client: BaseLLMClient) -> List[
        SPARCReflectionResult]:
        """Generates sparc evaluation results."""
        # Dictionary to store results with their original indices
        results_dict = {}

        if self.SPECS_COL in df.columns:
            is_truth = lambda x: x is not None and not pd.isna(x) and bool(x)
            mask = df.apply(lambda r:is_truth(r[self.SPECS_COL]),axis=1)
            df_with_spec = df[mask]
            if len(df_with_spec) > 0:
                results_with_spec = await self.generate_results(df_with_spec, llm_client, has_spec=True)
                # Store results with their original indices
                for idx, result in zip(df_with_spec.index, results_with_spec):
                    results_dict[idx] = result
            df_no_spec = df[~mask]
        else:
            df_no_spec = df
        
        if len(df_no_spec) > 0:
            results_no_spec = await self.generate_results(df_no_spec, llm_client, has_spec=False)
            # Store results with their original indices
            for idx, result in zip(df_no_spec.index, results_no_spec):
                results_dict[idx] = result

        # Return results in the original dataframe order
        reflection_results = [results_dict[idx] for idx in df.index]
        return reflection_results


if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = str(files("clear_eval.pipeline.setup").joinpath("default_config.yaml"))
    sample_data_file = str(files("clear_eval.sample_data.tool_calls").joinpath("tool_calls_sample_data.csv"))
    df = pd.read_csv(sample_data_file)

    for provider in ["openai", "watsonx"]:
        for inference_backend in ["langchain", "litellm"]:
            #model_name = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
            model_name = "Azure/gpt-4.1" if provider == "openai" else "openai/gpt-oss-120b"
            config = load_config(DEFAULT_CONFIG_PATH, user_config_path=None, provider=provider , eval_model_name=model_name, inference_backend=inference_backend)

            llm = get_eval_llm_from_config(config)

            tool_call_use_case = ToolCallEvalUseCase()
            evaluated_df = tool_call_use_case.eval_records(df.copy(), llm, config)
            evaluated_df.to_csv(sample_data_file.replace(".csv", f"_eval_{provider}_{inference_backend}.csv"), index=False)