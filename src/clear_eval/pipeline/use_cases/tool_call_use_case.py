import json
from tqdm import tqdm
from importlib.resources import files
from typing import Any, Dict, Tuple, List, Optional, Type, TypeVar, Union
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


def _patch_json_schema_to_pydantic_model():
    """
    Monkey-patch altk's json_schema_to_pydantic_model to fix OpenAI structured
    output compatibility.

    OpenAI's structured output API requires 'additionalProperties: false' on all
    object-type schemas. When the original function maps JSON Schema "object" type
    to Python ``dict``, Pydantic generates a schema with 'additionalProperties: true',
    which OpenAI rejects. This patch maps free-form "object" fields (like the
    "correction" field in SPARC metrics) to ``str`` instead, so the LLM returns
    them as JSON-formatted strings.
    """
    from pydantic import BaseModel, create_model, Field as PydField

    _T = TypeVar("_T")

    def patched_json_schema_to_pydantic_model(
        schema: Dict[str, Any], model_name: str = "AutoModel"
    ) -> Type[BaseModel]:
        fields: Dict[str, Any] = {}
        required_fields = set(schema.get("required", []))

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            # Map "object" to str to avoid OpenAI additionalProperties issue.
            # Free-form object fields (no sub-properties defined) cannot satisfy
            # OpenAI's requirement for additionalProperties: false, so we
            # represent them as JSON strings instead.
            "object": str,
            "null": type(None),
        }

        def parse_type(type_def: Union[str, List[str]]) -> Type[_T]:
            if isinstance(type_def, list):
                python_types = [type_mapping.get(t, Any) for t in type_def]
                if type(None) in python_types:
                    python_types.remove(type(None))
                    if len(python_types) == 1:
                        return Optional[python_types[0]]  # type: ignore
                    else:
                        return Optional[Union[tuple(python_types)]]  # type: ignore
                else:
                    return Union[tuple(python_types)]  # type: ignore
            else:
                return type_mapping.get(type_def, Any)

        for prop_name, prop_schema in schema.get("properties", {}).items():
            field_type: Any = parse_type(prop_schema.get("type"))
            default = ... if prop_name in required_fields else None
            description = prop_schema.get("description", None)
            field_args = {"description": description} if description else {}
            fields[prop_name] = (field_type, PydField(default, **field_args))

        return create_model(model_name, **fields)  # type: ignore

    import altk.core.llm.output_parser as _output_parser_module
    _output_parser_module.json_schema_to_pydantic_model = patched_json_schema_to_pydantic_model


# Apply the patch at import time so all downstream ALTK code uses the fixed version
_patch_json_schema_to_pydantic_model()


def _make_prompt_validated_wrapper(client):
    """
    Wrap an ALTK ValidatingLLMClient so that ``generate`` / ``generate_async``
    use **prompt-based** schema validation instead of the provider-native
    ``response_format`` parameter.

    This is needed for providers (e.g. watsonx) that do not support OpenAI-style
    structured output via ``response_format``.  The wrapper overrides every call
    so that:
      • ``schema_field`` is forced to ``None`` (no ``response_format`` kwarg).
      • ``include_schema_in_system_prompt`` is forced to ``True`` so the schema
        description is injected into the system message and the response is
        validated client-side by ALTK's retry loop.

    Additionally, ``_parse_llm_response`` is patched so that an empty LLM
    response returns ``""`` instead of raising ``ValueError``.  This lets the
    ALTK validation/retry loop treat it as a malformed output and retry,
    rather than bubbling up an unrecoverable error.
    """
    _orig_generate = client.generate
    _orig_generate_async = client.generate_async
    _orig_parse = client._parse_llm_response

    # --- Resilient response parser -----------------------------------
    def _safe_parse_llm_response(raw):
        try:
            return _orig_parse(raw)
        except (ValueError, KeyError):
            # Return empty string so the ValidatingLLMClient retry loop
            # sees it as invalid output and retries instead of crashing.
            logger.debug("LLM returned empty/unparseable response; will retry.")
            return ""

    client._parse_llm_response = _safe_parse_llm_response

    # --- Wrappers that force prompt-based validation -----------------
    def _wrapped_generate(prompt, *, schema, schema_field=None, retries=3, **kw):
        return _orig_generate(
            prompt,
            schema=schema,
            schema_field=None,
            retries=retries,
            include_schema_in_system_prompt=True,
            **kw,
        )

    async def _wrapped_generate_async(prompt, *, schema, schema_field=None, retries=3, **kw):
        return await _orig_generate_async(
            prompt,
            schema=schema,
            schema_field=None,
            retries=retries,
            include_schema_in_system_prompt=True,
            **kw,
        )

    client.generate = _wrapped_generate
    client.generate_async = _wrapped_generate_async
    return client


# Providers whose LLM endpoints do NOT support response_format with a
# Pydantic model (i.e. OpenAI-style structured output).
_PROVIDERS_WITHOUT_STRUCTURED_OUTPUT = {"watsonx"}


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
            altk_client = MetricsClientCls(model_name=litellm_model)
            # Providers that don't support OpenAI-style response_format need
            # prompt-based schema validation instead.
            if provider in _PROVIDERS_WITHOUT_STRUCTURED_OUTPUT:
                altk_client = _make_prompt_validated_wrapper(altk_client)
            return altk_client

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

    # for provider in ["watsonx"]:
    for provider in ["openai", "watsonx"]:
        # for inference_backend in ["litellm"]:
        for inference_backend in ["langchain", "litellm"]:
            # model_name = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
            model_name = "gpt-4.1" if provider == "openai" else "openai/gpt-oss-120b"
            config = load_config(DEFAULT_CONFIG_PATH, user_config_path=None, provider=provider , eval_model_name=model_name, inference_backend=inference_backend)

            llm = get_eval_llm_from_config(config)

            tool_call_use_case = ToolCallEvalUseCase()
            evaluated_df = tool_call_use_case.eval_records(df.copy(), llm, config)
            evaluated_df.to_csv(sample_data_file.replace(".csv", f"_eval_{provider}_{inference_backend}.csv"), index=False)