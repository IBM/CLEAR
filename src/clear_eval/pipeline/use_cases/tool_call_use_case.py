import json
import os
from importlib.resources import files
from typing import Any, Dict, Tuple, List, Optional
import pandas as pd
from clear_eval.pipeline.use_cases.eval_use_case import EvalUseCase
from clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL, DEFAULT_ISSUES_FORMAT_MODE, DEFAULT_SPARC_TRACK
from altk.pre_tool.sparc import SPARCReflectionComponent
from altk.core.toolkit import AgentPhase, ComponentConfig
from altk.pre_tool.core import SPARCReflectionRunInput, Track, SPARCReflectionResult, SPARCExecutionMode
from altk.core.llm import get_llm, BaseLLMClient

import logging

from clear_eval.pipeline.config_loader import load_config
from clear_eval.pipeline.inference_utils.llm_client import run_parallel, LiteLLMClient, LangChainClient
from clear_eval.pipeline.full_pipeline import get_eval_llm_from_config

logger = logging.getLogger(__name__)

# Suppress verbose per-call logs from SPARC/ALTK internals
logging.getLogger("altk").setLevel(logging.WARNING)

# Providers whose LLM endpoints do NOT support response_format with a Pydantic
# model (OpenAI-style structured output). For these we flip ALTK's
# ``prompt_based_validation`` knob — see altk.core.llm.ValidatingLLMClient.
_PROVIDERS_WITHOUT_STRUCTURED_OUTPUT = {"watsonx", "rits", "openai"}


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
        logger.info(f"\n--- Evaluating Tool calls Predictions ---")
        df[EVALUATION_TEXT_COL] = ""
        df[score_col] = pd.NA  # Use Pandas NA for missing scores
        df["sparc_decision"] = pd.NA
        df["sparc_score_1_to_5"] = pd.NA
        df["sparc_recommendations"] = "[]"

        # convert CLEAR llm to ALTK llm
        altk_llm_client = self.clear_llm_client_to_altk_llm_client(llm, config.get("provider"),
                                                                   config.get("eval_model_name"),
                                                                   config.get("eval_model_params"))

        # runtime_pipeline = True: surface issues. False: surface recommendations
        format_mode = config.get('issues_format', DEFAULT_ISSUES_FORMAT_MODE)
        runtime_pipeline = bool(format_mode==DEFAULT_ISSUES_FORMAT_MODE)
        # call sparc with pipeline over examples, results store sorted results over the examples
        results = self.generate_sparc_evaluation_results(
            df=df,
            llm_client=altk_llm_client,
            config=config,
            track_name=config.get("sparc_track", DEFAULT_SPARC_TRACK),
            runtime_pipeline=runtime_pipeline
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

    def _create_rits_altk_client(self, model_name: str, default_gen: Dict[str, Any],
                                  eval_model_params: Optional[Dict] = None) -> BaseLLMClient:
        """Create ALTK OpenAI client configured for RITS."""
        from clear_eval.pipeline.inference_utils.langchain_chat_models import get_rits_base
        import os
        
        rits_api_key = os.getenv("RITS_API_KEY")
        if not rits_api_key:
            raise KeyError("RITS_API_KEY env var required for RITS.")
        
        # Check if api_base is already provided in params
        if eval_model_params and "api_base" in eval_model_params:
            base_url = eval_model_params["api_base"]
        else:
            # Construct URL from model name
            model_base = get_rits_base(model_name)
            base_url = f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_base}/v1"
        
        # Use OpenAI ALTK client with RITS configuration
        MetricsClientCls = get_llm("openai.async.output_val")
        return MetricsClientCls(
            model=model_name,
            api_key="/",  # RITS uses header-based auth, not API key in URL
            base_url=base_url,
            default_headers={"RITS_API_KEY": rits_api_key},
            free_form_object_as_str=True,
            prompt_based_validation=True,
            default_generation_kwargs=default_gen,
        )

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

        # ALTK provider subclasses stash any extra ``**kwargs`` and replay them
        # into every generate call. Passing the validation knobs there leaks
        # ``free_form_object_as_str`` / ``prompt_based_validation`` /
        # ``default_generation_kwargs`` into the underlying SDK (watsonx's
        # ``ModelInference.achat`` rejects them). Use ``configure_validation``
        # — ALTK's documented post-construction API — so the knobs only live
        # on the wrapper.
        def _configure(client: BaseLLMClient) -> BaseLLMClient:
            return client.configure_validation(
                free_form_object_as_str=True,
                prompt_based_validation=needs_prompt_validation,
                default_generation_kwargs=default_gen,
            )
        # RITS uses OpenAI-compatible API - handle it specially
        if provider == "rits":
            return self._create_rits_altk_client(model_name, default_gen, eval_model_params)

        # LiteLLMClient - use ALTK's native litellm support
        if isinstance(llm_client, LiteLLMClient):
            client = get_llm("litellm.output_val")
            litellm_model = f"{provider}/{model_name}"
            # litellm.completion accepts max_tokens/temperature as top-level
            # kwargs, so keeping them as constructor kwargs continues to work.
            return _configure(client(model_name=litellm_model, **default_gen))

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
            # Watsonx SDK expects generation params inside a ``params`` dict,
            # not top-level. configure_validation merges these into every
            # generate call.
            client = MetricsClientCls(**watsonx_kwargs)
            return client.configure_validation(
                free_form_object_as_str=True,
                prompt_based_validation=True,
                default_generation_kwargs={"params": default_gen} if default_gen else {},
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

    def generate_sparc_evaluation_results(
        self,
        df: pd.DataFrame,
        llm_client: BaseLLMClient,
        config: dict,
        track_name: str,
        runtime_pipeline: bool = True,
    ) -> List[SPARCReflectionResult]:
        """Evaluate all rows in a single parallel pass, selecting track per row."""
        spec_track = Track(track_name)
        component_config = ComponentConfig(llm_client=llm_client)

        # Determine per-row whether spec is available
        has_spec_col = self.SPECS_COL in df.columns

        def _row_has_spec(row):
            if not has_spec_col:
                return False
            val = row.get(self.SPECS_COL)
            return val is not None and not pd.isna(val) and bool(val)

        async def _evaluate_single(example_row):
            # Each concurrent call gets its own component instance to avoid
            # shared mutable state (SPARCReflectionComponent._arun sets
            # self._tool_specs per call — not safe under concurrency).
            row_has_spec = _row_has_spec(example_row)
            track = spec_track if row_has_spec else Track.SPEC_FREE
            sparc_component = SPARCReflectionComponent(
                config=component_config,
                track=track,
                execution_mode=SPARCExecutionMode.ASYNC,
                runtime_pipeline=runtime_pipeline,
            )
            run_input = SPARCReflectionRunInput(
                messages=json.loads(example_row[self.CONTEXT_COL]),
                tool_specs=json.loads(example_row[self.SPECS_COL]) if row_has_spec else [],
                tool_calls=[json.loads(example_row[self.RESPONSE_COL])],
            )
            reflection_result = await sparc_component.aprocess(run_input, phase=AgentPhase.RUNTIME)
            result = reflection_result.output.reflection_result
            # Return as dict for JSON-serializable caching
            return result.model_dump(mode="json")

        inputs = [row for _, row in df.iterrows()]

        checkpoint_every = config.get('checkpoint_every', 0)
        qid_col = config.get('qid_column', '')
        item_ids = [str(row.get(qid_col, i)) for i, (_, row) in enumerate(df.iterrows())] if checkpoint_every else None
        cache_path = None
        if checkpoint_every:
            checkpoint_base = config.get('checkpoint_path', '')
            if checkpoint_base:
                cache_path = checkpoint_base.replace('.csv', '_cache_sparc.jsonl')
            else:
                output_dir = config.get('output_dir', '.')
                cache_path = os.path.join(output_dir, "cache_sparc.jsonl")

        parallel_results = run_parallel(
            func=_evaluate_single,
            inputs=inputs,
            max_workers=config.get('max_workers', 10),
            error_prefix="SPARC: ",
            progress_desc="Evaluating tool calls with SPARC",
            checkpoint_every=checkpoint_every,
            checkpoint_path=cache_path,
            item_ids=item_ids,
        )

        # Reconstruct SPARCReflectionResult from dicts
        from altk.pre_tool.core import SPARCReflectionDecision
        reflection_results = []
        for pr in parallel_results:
            if pr.is_success:
                reflection_results.append(SPARCReflectionResult.model_validate(pr.result))
            else:
                logger.error(f"SPARC evaluation failed: {pr.error}")
                reflection_results.append(SPARCReflectionResult(
                    decision=SPARCReflectionDecision.REJECT,
                    issues=[],
                    score=None,
                ))
        return reflection_results


if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = str(files("clear_eval.pipeline.setup").joinpath("default_config.yaml"))
    sample_data_file = str(files("clear_eval.sample_data.tool_calls").joinpath("tool_calls_sample_data.csv"))
    df = pd.read_csv(sample_data_file)

    # for provider in ["watsonx"]:
    for provider in ["rits"]:#, "watsonx"]:
        # for inference_backend in ["litellm"]:
        for inference_backend in ["langchain", "litellm"]:
            print(f"=======provider: {provider}, inference_backend: {inference_backend}======")
            # model_name = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
            model_name = "gpt-4.1" if provider == "openai" else "openai/gpt-oss-120b"
            config = load_config(DEFAULT_CONFIG_PATH, user_config_path=None, provider=provider , eval_model_name=model_name, inference_backend=inference_backend)
            config["checkpoint_every"] = 5
            df = df.rename(columns={"context":"model_input"})
            llm = get_eval_llm_from_config(config)

            tool_call_use_case = ToolCallEvalUseCase()
            evaluated_df = tool_call_use_case.eval_records(df.copy(), llm, config)
            evaluated_df.to_csv(sample_data_file.replace(".csv", f"_eval_{provider}_{inference_backend}.csv"), index=False)