"""Tests for inference backend selection and (optionally) live inference.

Three tiers:
1. TestBackendSelection  - config -> get_llm_client argument mapping (mocked, no credentials).
2. TestBackendValidation - backend validation errors (real get_llm_client, raises before network).
3. TestLiveInferenceWatsonx - real API calls, opt-in via RUN_LIVE_LLM_TESTS=1 AND WatsonX creds.

Live tests are skipped by default so normal `pytest` runs never make network calls.
To run them:
    RUN_LIVE_LLM_TESTS=1 pytest tests/test_inference_backends.py
Requires WATSONX_APIKEY and WATSONX_PROJECT_ID. The model can be overridden with
WATSONX_TEST_MODEL (default: meta-llama/llama-3-3-70b-instruct).
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from clear_eval.pipeline.full_pipeline import get_eval_llm_from_config, get_llm_from_config
from clear_eval.pipeline.inference_utils.llm_client import get_llm_client

RUN_LIVE = os.getenv("RUN_LIVE_LLM_TESTS") == "1"
HAS_WATSONX = all(os.getenv(v) for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID"))
WATSONX_MODEL = os.getenv("WATSONX_TEST_MODEL", "meta-llama/llama-3-3-70b-instruct")
WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1"
LIVE_WATSONX = RUN_LIVE and HAS_WATSONX


class TestBackendSelection:
    """config -> get_llm_client argument mapping. get_llm_client is mocked, so no credentials
    or network are needed."""

    def _call(self, config, eval_mode=True):
        with patch("clear_eval.pipeline.full_pipeline.get_llm_client") as mock_client:
            mock_client.return_value = MagicMock()
            if eval_mode:
                get_eval_llm_from_config(config)
            else:
                get_llm_from_config(config, eval_mode=False)
            mock_client.assert_called_once()
            return mock_client.call_args.kwargs

    def test_inference_backend_forwarded(self):
        kwargs = self._call({
            "provider": "watsonx",
            "eval_model_name": "some-model",
            "inference_backend": "langchain",
            "eval_model_params": {"max_tokens": 5},
        })
        assert kwargs["provider"] == "watsonx"
        assert kwargs["model"] == "some-model"
        assert kwargs["inference_backend"] == "langchain"
        assert kwargs["parameters"] == {"max_tokens": 5}
        assert kwargs["eval_mode"] is True

    def test_use_litellm_flag_maps_to_litellm_backend(self):
        # Backward compatibility: use_litellm=True overrides inference_backend.
        kwargs = self._call({
            "provider": "openai",
            "eval_model_name": "gpt-4o",
            "use_litellm": True,
            "eval_model_params": {},
        })
        assert kwargs["inference_backend"] == "litellm"

    def test_endpoint_url_forwarded(self):
        kwargs = self._call({
            "provider": "watsonx",
            "eval_model_name": "some-model",
            "inference_backend": "endpoint",
            "endpoint_url": WATSONX_URL,
            "eval_model_params": {},
        })
        assert kwargs["inference_backend"] == "endpoint"
        assert kwargs["endpoint_url"] == WATSONX_URL

    def test_gen_fields_used_when_not_eval_mode(self):
        kwargs = self._call({
            "provider": "openai",
            "gen_model_name": "gen-model",
            "gen_model_params": {"temperature": 0.7},
            "inference_backend": "langchain",
        }, eval_mode=False)
        assert kwargs["model"] == "gen-model"
        assert kwargs["parameters"] == {"temperature": 0.7}
        assert kwargs["eval_mode"] is False


class TestBackendValidation:
    """Validation paths in get_llm_client that raise before any network/auth, so no
    credentials are required."""

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid inference_backend"):
            get_llm_client(provider="watsonx", model="m", inference_backend="not-a-backend")

    def test_endpoint_without_url_raises(self):
        # The ValueError is caught and re-wrapped as a generic Exception by get_llm_client,
        # but the message is preserved.
        with pytest.raises(Exception, match="endpoint_url"):
            get_llm_client(provider="watsonx", model="m", inference_backend="endpoint")


@pytest.mark.skipif(
    not LIVE_WATSONX,
    reason="Set RUN_LIVE_LLM_TESTS=1 and WATSONX_APIKEY/WATSONX_PROJECT_ID to run live tests",
)
class TestLiveInferenceWatsonx:
    """Real inference against WatsonX. Opt-in only."""

    @pytest.mark.parametrize(
        "backend,endpoint_url",
        [
            ("langchain", None),
            ("litellm", None),
            ("endpoint", WATSONX_URL),
        ],
    )
    def test_invoke_returns_text(self, backend, endpoint_url):
        config = {
            "provider": "watsonx",
            "eval_model_name": WATSONX_MODEL,
            "inference_backend": backend,
            "eval_mode": True,
            "eval_model_params": {"max_tokens": 5, "temperature": 0.0, "top_p": 0.95},
        }
        if endpoint_url:
            config["endpoint_url"] = endpoint_url

        client = get_eval_llm_from_config(config)
        response = client.invoke([{"role": "user", "content": "Say hello in one word."}])

        assert isinstance(response, str)
        assert response.strip(), "Response is empty"
