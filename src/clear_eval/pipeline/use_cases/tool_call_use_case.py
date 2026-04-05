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


def _relax_freeform_object_types(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a *copy* of *schema* where every free-form ``"type": "object"``
    property (i.e. one that declares no ``properties`` of its own) is loosened
    to ``"type": ["object", "string"]``.

    This keeps the validation schema in sync with the Pydantic model produced
    by the patched ``json_schema_to_pydantic_model`` (which maps ``"object"``
    → ``str``).  Without this, OpenAI returns a JSON string for such fields
    but ``jsonschema.validate`` rejects it because the original schema only
    allows ``"object"``.
    """
    import copy
    schema = copy.deepcopy(schema)
    for _prop_name, prop_schema in schema.get("properties", {}).items():
        if (
            prop_schema.get("type") == "object"
            and "properties" not in prop_schema
        ):
            prop_schema["type"] = ["object", "string"]
    return schema


def _patch_validate(client):
    """
    Monkey-patch ``_validate`` on *client* so that dict schemas are relaxed
    (via `_relax_freeform_object_types`) before ``jsonschema.validate`` runs.
    """
    _orig_validate = client._validate

    def _relaxed_validate(raw, schema):
        if isinstance(schema, dict):
            schema = _relax_freeform_object_types(schema)
        return _orig_validate(raw, schema)

    client._validate = _relaxed_validate
    return client


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
            # Check if the response contains reasoning_content but no actual
            # content — this happens when reasoning models (e.g. on watsonx)
            # exhaust the max_tokens budget on "thinking" tokens.
            _choices = getattr(raw, "choices", None) or (raw.get("choices", []) if isinstance(raw, dict) else [])
            if _choices:
                _msg = getattr(_choices[0], "message", None) or (_choices[0].get("message", {}) if isinstance(_choices[0], dict) else {})
                _reasoning = getattr(_msg, "reasoning_content", None) or (_msg.get("reasoning_content") if isinstance(_msg, dict) else None)
                _finish = getattr(_choices[0], "finish_reason", None) or (_choices[0].get("finish_reason") if isinstance(_choices[0], dict) else None)
                if _reasoning and _finish == "length":
                    logger.warning(
                        "LLM reasoning consumed entire token budget (finish_reason='length'). "
                        "Consider increasing 'max_tokens' in eval_model_params. Will retry."
                    )
            else:
                logger.debug("LLM returned empty/unparseable response; will retry.")
            # Return empty string so the ValidatingLLMClient retry loop
            # sees it as invalid output and retries instead of crashing.
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


def _inject_default_generation_args(client, eval_model_params: Dict[str, Any]):
    """
    Monkey-patch ``generate`` and ``generate_async`` on *client* so that
    inference parameters from *eval_model_params* (e.g. ``max_tokens``,
    ``temperature``) are injected into every call.

    For providers like watsonx whose SDK methods (``ModelInference.achat``)
    expect these parameters inside a ``params`` dict rather than as top-level
    keyword arguments, we inject them directly into the ``params`` kwarg.

    **Important:** This must be applied *after* ``_make_prompt_validated_wrapper``
    because ``ValidatingLLMClient.generate_async`` calls ``super()._generate_async()``
    which bypasses instance-level patches on ``_generate_async``.  By wrapping
    ``generate`` / ``generate_async`` (the outermost entry points) we ensure the
    ``params`` dict is present before *any* downstream code runs.
    """
    params_dict: Dict[str, Any] = {}
    if "max_tokens" in eval_model_params:
        params_dict["max_tokens"] = eval_model_params["max_tokens"]
    if "temperature" in eval_model_params:
        params_dict["temperature"] = eval_model_params["temperature"]
    if not params_dict:
        return client

    _orig_generate = client.generate
    _orig_generate_async = client.generate_async

    def _patched_generate(prompt, **kwargs):
        # Merge into the `params` dict that watsonx SDK methods expect
        existing_params = kwargs.get("params") or {}
        if isinstance(existing_params, dict):
            merged = {**params_dict, **existing_params}  # caller overrides defaults
        else:
            merged = existing_params  # TextChatParameters object — don't touch
        kwargs["params"] = merged
        return _orig_generate(prompt, **kwargs)

    async def _patched_generate_async(prompt, **kwargs):
        existing_params = kwargs.get("params") or {}
        if isinstance(existing_params, dict):
            merged = {**params_dict, **existing_params}
        else:
            merged = existing_params
        kwargs["params"] = merged
        return await _orig_generate_async(prompt, **kwargs)

    client.generate = _patched_generate
    client.generate_async = _patched_generate_async
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
                                                                   config.get("eval_model_name"),
                                                                   config.get("eval_model_params"))

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

    def clear_llm_client_to_altk_llm_client(self, llm_client, provider: str, model_name: str,
                                               eval_model_params: Optional[Dict] = None) -> BaseLLMClient:
        """Convert CLEAR's LLM object to ALTK's LLM Object."""

        # LiteLLMClient - use ALTK's native litellm support
        if isinstance(llm_client, LiteLLMClient):
            MetricsClientCls = get_llm("litellm.output_val")
            # LiteLLM model format: provider/model_name (consistent with LiteLLMClient)
            litellm_model = f"{provider}/{model_name}"
            # Forward inference parameters (e.g. max_tokens) so that the
            # ALTK client passes them through to every litellm.completion
            # call.  Without an explicit max_tokens some providers (watsonx)
            # default to a very low value (1024) which is easily exhausted
            # by reasoning-model "thinking" tokens, leaving no room for the
            # actual response content.
            lite_kwargs: Dict[str, Any] = {}
            if eval_model_params:
                if "max_tokens" in eval_model_params:
                    lite_kwargs["max_tokens"] = eval_model_params["max_tokens"]
                if "temperature" in eval_model_params:
                    lite_kwargs["temperature"] = eval_model_params["temperature"]
            altk_client = MetricsClientCls(model_name=litellm_model, **lite_kwargs)
            # Relax validation so free-form "object" fields also accept the
            # string representation produced by the patched Pydantic model.
            _patch_validate(altk_client)
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
                altk_client = MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url,
                    space_id=llm.space_id,
                )
            elif llm.project_id:
                altk_client = MetricsClientCls(
                    model_id=llm.model_id,
                    api_key=llm.api_key._secret_value,
                    url=llm.url._secret_value,
                    project_id=llm.project_id,
                )
            else:
                raise KeyError("Either space_id or project_id must be specified for watsonx inference.")
            # Relax validation so free-form "object" fields also accept the
            # string representation produced by the patched Pydantic model.
            _patch_validate(altk_client)
            # watsonx does not support OpenAI-style response_format, so use
            # prompt-based schema validation instead.
            altk_client = _make_prompt_validated_wrapper(altk_client)
            # Inject inference parameters (e.g. max_tokens, temperature) into
            # every LLM call via the watsonx `params` dict.  Without an
            # explicit max_tokens, watsonx defaults to 1024 which is easily
            # exhausted by reasoning-model "thinking" tokens.
            # NOTE: must be applied *after* _make_prompt_validated_wrapper so
            # that this is the outermost wrapper and `params` flows through.
            if eval_model_params:
                _inject_default_generation_args(altk_client, eval_model_params)
            return altk_client

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
            print(f"=======provider: {provider}, inference_backend: {inference_backend}======")
            # model_name = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
            model_name = "gpt-4.1" if provider == "openai" else "openai/gpt-oss-120b"
            config = load_config(DEFAULT_CONFIG_PATH, user_config_path=None, provider=provider , eval_model_name=model_name, inference_backend=inference_backend)

            llm = get_eval_llm_from_config(config)

            tool_call_use_case = ToolCallEvalUseCase()
            evaluated_df = tool_call_use_case.eval_records(df.copy(), llm, config)
            evaluated_df.to_csv(sample_data_file.replace(".csv", f"_eval_{provider}_{inference_backend}.csv"), index=False)