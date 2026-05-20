import json
from tqdm import tqdm
from importlib.resources import files
from typing import Any, Dict, Tuple, List, Optional
import pandas as pd
from clear_eval.pipeline.use_cases.eval_use_case import EvalUseCase
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


# Providers whose LLM endpoints do NOT support response_format with a Pydantic
# model (OpenAI-style structured output). For these we flip ALTK's
# ``prompt_based_validation`` knob — see altk.core.llm.ValidatingLLMClient.
_PROVIDERS_WITHOUT_STRUCTURED_OUTPUT = {"watsonx"}


def _forwardable_generation_kwargs(eval_model_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick only the inference-time knobs SPARC cares about from
    ``eval_model_params`` (CLEAR config) and return a dict suitable for
    ``ValidatingLLMClient.default_generation_kwargs``."""
    if not eval_model_params:
        return {}
    out: Dict[str, Any] = {}
    for k in ("max_tokens", "temperature"):
        if k in eval_model_params:
            out[k] = eval_model_params[k]
    return out


class ToolCallEvalUseCase(EvalUseCase):
    SPECS_COL = "api_spec"
    CONTEXT_COL = "model_input"
    RESPONSE_COL = "response"
    required_input_fields = [CONTEXT_COL, SPECS_COL]

    def eval_records(self, df, llm, config, score_col=SCORE_COL):
        return run_async(self.eval_records_async(df, llm, config, score_col))

    async def eval_records_async(self, df, llm, config, score_col=SCORE_COL):
        """Evaluates predictions and adds scores.

        Output columns (added or overwritten):
          - ``evaluation_text``: human-readable verdict ("Tool call is valid." or
            a concatenation of per-issue explanations).
          - ``score`` (``score_col``): normalized SPARC rubric score in [0, 1],
            derived from the 1-5 mean of every semantic metric's output. Falls
            back to the boolean decision (1.0 APPROVE / 0.0 REJECT) when the
            pipeline didn't produce a numeric rating (e.g. static-only track).
          - ``sparc_decision``: boolean — True iff SPARC decided APPROVE.
          - ``sparc_score_1_to_5``: raw 1-5 rubric mean (None for static-only).
          - ``sparc_recommendations``: JSON array of SPARCRecommendation dicts
            (unified-diff + importance). Empty array ``"[]"`` in runtime mode.
        """
        logger.info(f"\n--- Evaluating Tool calls predictionsPredictions ---")
        df[EVALUATION_TEXT_COL] = ""
        df[score_col] = pd.NA  # Use Pandas NA for missing scores
        df["sparc_decision"] = pd.NA
        df["sparc_score_1_to_5"] = pd.NA
        df["sparc_recommendations"] = "[]"

        # convert CLEAR llm to ALTK llm
        altk_llm_client = self.clear_llm_client_to_altk_llm_client(llm, config.get("provider"),
                                                                   config.get("eval_model_name"),
                                                                   config.get("eval_model_params"))

        # call sparc with pipeline over examples, results store sorted results over the examples
        results = await self.generate_sparc_evaluation_results(
            df=df,
            llm_client=altk_llm_client,
            track_name=config.get("track", "slow_track"),
            runtime_pipeline=bool(config.get("runtime_pipeline", True)),
        )

        for i, result in enumerate(results):
            (eval_text, score, decision_bool, raw_score) = self.get_eval_from_results(result)

            df.at[df.index[i], EVALUATION_TEXT_COL] = eval_text
            df.at[df.index[i], score_col] = float(score) if score is not None else pd.NA
            df.at[df.index[i], "sparc_decision"] = decision_bool
            df.at[df.index[i], "sparc_score_1_to_5"] = (
                float(raw_score) if raw_score is not None else pd.NA
            )
            # Serialize per-row recommendations; always a JSON array (empty
            # in runtime mode, non-empty in evaluation mode when the LLM
            # emitted fixable-artifact suggestions).
            recs = getattr(result, "all_recommendations", None) or []
            df.at[df.index[i], "sparc_recommendations"] = json.dumps(
                [r.model_dump(mode="json") for r in recs]
            )

        logger.info("Finished evaluating predictions.")
        # Convert score column to nullable float type
        df[score_col] = df[score_col].astype('Float64')
        df["sparc_score_1_to_5"] = df["sparc_score_1_to_5"].astype('Float64')
        df["sparc_decision"] = df["sparc_decision"].astype('boolean')
        return df

    def clear_llm_client_to_altk_llm_client(self, llm_client, provider: str, model_name: str,
                                               eval_model_params: Optional[Dict] = None) -> BaseLLMClient:
        """Convert CLEAR's LLM object to ALTK's LLM Object.

        The provider-compatibility knobs previously handled by ad-hoc
        monkey-patches (free-form object types, prompt-based validation,
        default generation kwargs, reasoning-budget retry) now live on
        ``ValidatingLLMClient`` itself — see
        ``altk.core.llm.output_parser.ValidatingLLMClient``.
        """
        default_gen = _forwardable_generation_kwargs(eval_model_params)
        needs_prompt_validation = provider in _PROVIDERS_WITHOUT_STRUCTURED_OUTPUT

        # LiteLLMClient - use ALTK's native litellm support
        if isinstance(llm_client, LiteLLMClient):
            MetricsClientCls = get_llm("litellm.output_val")
            litellm_model = f"{provider}/{model_name}"
            # litellm.completion accepts max_tokens/temperature as top-level
            # kwargs, so keeping them as constructor kwargs continues to work.
            return MetricsClientCls(
                model_name=litellm_model,
                free_form_object_as_str=True,
                prompt_based_validation=needs_prompt_validation,
                default_generation_kwargs=default_gen,
                **default_gen,
            )

        # LangChainClient - extract from underlying LangChain object
        if isinstance(llm_client, LangChainClient):
            llm = llm_client.llm
        else:
            # Fallback for raw LangChain objects
            llm = llm_client

        if provider == "watsonx":
            MetricsClientCls = get_llm("watsonx.output_val")
            watsonx_kwargs: Dict[str, Any] = {
                "model_id": llm.model_id,
                "api_key": llm.api_key._secret_value,
                "free_form_object_as_str": True,
                "prompt_based_validation": True,
                # Watsonx SDK expects generation params inside a ``params``
                # dict, not as top-level kwargs. ALTK's watsonx client
                # already merges default_generation_kwargs into that dict.
                "default_generation_kwargs": {"params": default_gen} if default_gen else {},
            }
            if llm.space_id:
                watsonx_kwargs["url"] = llm.url
                watsonx_kwargs["space_id"] = llm.space_id
            elif llm.project_id:
                watsonx_kwargs["url"] = llm.url._secret_value
                watsonx_kwargs["project_id"] = llm.project_id
            else:
                raise KeyError(
                    "Either space_id or project_id must be specified for watsonx inference."
                )
            return MetricsClientCls(**watsonx_kwargs)

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

    def get_eval_from_results(
        self, result: SPARCReflectionResult
    ) -> Tuple[str, Optional[float], bool, Optional[float]]:
        """Turn a SPARC reflection result into the fields CLEAR writes per row.

        Returns:
            (evaluation_text, normalized_score, decision_bool, raw_score_1_5)
            - ``normalized_score``: SPARC's aggregate rubric mean (1-5) mapped
              into [0, 1]. Falls back to 1.0 / 0.0 from the boolean decision
              when the pipeline didn't produce a numeric rating (static-only
              track, all-error, etc.).
            - ``decision_bool``: True iff SPARC decided APPROVE.
            - ``raw_score_1_5``: the 1-5 rubric mean (None when unavailable).
        """
        logger.debug("=== DEBUG: Full result structure ===")
        logger.debug(result)
        decision_bool = result.decision.name == "APPROVE"
        raw_score = result.score
        normalized = result.normalized_score
        if normalized is None:
            # Static-only track / all-error: fall back to the boolean decision
            # so CLEAR always has a numeric score to aggregate on.
            normalized = 1.0 if decision_bool else 0.0

        if decision_bool:
            return "Tool call is valid.", normalized, True, raw_score
        explanation_text = "\n".join(issue.explanation for issue in result.issues)
        logger.debug("=== DEBUG: Explanation Text ===")
        logger.debug(explanation_text)
        return (
            f"Tool call is invalid. Reasons:\n{explanation_text}",
            normalized,
            False,
            raw_score,
        )

    async def generate_results(
        self,
        df,
        llm_client,
        has_spec,
        track_name: str = "slow_track",
        runtime_pipeline: bool = True,
    ):
        # Pick the SPARC track: the user-selected ``track_name`` is honored
        # when we have tool specs; without specs we must fall back to
        # ``SPEC_FREE`` regardless of the user's choice.
        spec_track = Track(track_name)
        sparc_component = SPARCReflectionComponent(
            config=ComponentConfig(llm_client=llm_client),
            track=spec_track if has_spec else Track.SPEC_FREE,
            execution_mode=SPARCExecutionMode.ASYNC,
            runtime_pipeline=runtime_pipeline,
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

    async def generate_sparc_evaluation_results(
        self,
        df: pd.DataFrame,
        llm_client: BaseLLMClient,
        track_name: str = "slow_track",
        runtime_pipeline: bool = True,
    ) -> List[SPARCReflectionResult]:
        """Generates sparc evaluation results."""
        # Dictionary to store results with their original indices
        results_dict = {}

        if self.SPECS_COL in df.columns:
            is_truth = lambda x: x is not None and not pd.isna(x) and bool(x)
            mask = df.apply(lambda r:is_truth(r[self.SPECS_COL]),axis=1)
            df_with_spec = df[mask]
            if len(df_with_spec) > 0:
                results_with_spec = await self.generate_results(
                    df_with_spec,
                    llm_client,
                    has_spec=True,
                    track_name=track_name,
                    runtime_pipeline=runtime_pipeline,
                )
                # Store results with their original indices
                for idx, result in zip(df_with_spec.index, results_with_spec):
                    results_dict[idx] = result
            df_no_spec = df[~mask]
        else:
            df_no_spec = df

        if len(df_no_spec) > 0:
            results_no_spec = await self.generate_results(
                df_no_spec,
                llm_client,
                has_spec=False,
                track_name=track_name,
                runtime_pipeline=runtime_pipeline,
            )
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

    # for provider in ["watsonx"]:
    for provider in ["openai", "watsonx"]:
        # for inference_backend in ["litellm"]:
        for inference_backend in ["langchain", "litellm"]:
            print(f"=======provider: {provider}, inference_backend: {inference_backend}======")
            # model_name = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
            model_name = "gpt-4.1" if provider == "openai" else "openai/gpt-oss-120b"
            config = load_config(DEFAULT_CONFIG_PATH, user_config_path=None, provider=provider , eval_model_name=model_name, inference_backend=inference_backend)

            llm = get_eval_llm_from_config(config)

            tool_call_use_case = ToolCallEvalUseCase()
            evaluated_df = tool_call_use_case.eval_records(df.copy(), llm, config)
            evaluated_df.to_csv(sample_data_file.replace(".csv", f"_eval_{provider}_{inference_backend}.csv"), index=False)