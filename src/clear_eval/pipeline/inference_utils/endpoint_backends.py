# llm_backends.py
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Iterator, List, Optional, Union
from urllib.parse import urljoin

import requests

# =========
# Base types
# =========

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLMBackend(ABC):
    """
    Unified interface for LLM providers.
    Implementations should:
      - support chat(messages, stream=False, **gen_params)
      - optionally support embed(texts)
    """

    @abstractmethod
    def chat(
            self,
            messages: List[Message],
            stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        If stream=False: return a single string.
        If stream=True: return an iterator of text chunks (tokens).
        """
        ...

# =======================
# Prompt rendering helpers
# =======================

ROLE_TAGS = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}


def render_chat_as_prompt(messages: List[Message], assistant_prefix: bool = True) -> str:
    """
    Serializes a chat into a single text prompt.
    Useful for providers that don't accept a messages[] array (e.g., watsonx).
    
    Uses a simple concatenation format that lets the model's chat template handle formatting.
    """
    parts = []
    
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        
        # Simple format: just concatenate with minimal markers
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(content)

    return "\n\n".join(parts)


# ==============================
# SSE parsing (Server-Sent Events)
# ==============================

def iter_sse_data_lines(resp: requests.Response) -> Iterator[dict]:
    """
    Yields parsed JSON payloads from an SSE response where each event line is `data: {...}`.
    Stops on '[DONE]'.
    """
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if not raw.startswith("data: "):
            # ignore comments like ': keep-alive' or other lines
            continue
        payload = raw[6:].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


# ============================
# OpenAI-compatible HTTP backend
# ============================

class OpenAIStyleHTTPBackend(LLMBackend):
    """
    Works with any OpenAI-compatible endpoint:
      - OpenAI / Azure OpenAI (with appropriate base_url)
      - vLLM OpenAI API server
      - Ollama (OpenAI compat mode)
      - LiteLLM proxy exposing /v1 endpoints
      - TGI/OpenRouter/etc if they emulate OpenAI schema

    base_url should typically include '/v1', e.g. 'https://your-proxy.example.com/v1'
    """

    def __init__(
            self,
            base_url: str,
            model_name: str,
            eval_mode: bool = True,
            timeout: int = 60,
            extra_headers: Optional[Dict[str, str]] = None,
            **gen_params
    ):
        self.base_url = base_url.rstrip("/")
        # optionally auto-append /v1 if missing
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model_name = model_name
        self.eval_mode = eval_mode
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.gen_params = gen_params

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, body: dict, stream: bool = False) -> requests.Response:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        resp = requests.post(
            url,
            json=body,
            headers=self._headers(),
            timeout=self.timeout,
            stream=stream,
        )
        # Raise HTTP errors with body context
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = f" {resp.text[:500]}"
            except Exception:
                pass
            raise requests.HTTPError(f"{e}{detail}") from None
        return resp

    def chat(
            self,
            messages: List[Message],
            stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        body = {
            "model": self.model_name,
            "messages": messages,
            **({"stream": True} if stream else {}),
        }
        if self.eval_mode:
            body["temperature"] = 0
        if self.gen_params:
            body.update(**self.gen_params)

        resp = self._post("chat/completions", body=body, stream=stream)

        if not stream:
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        def generator() -> Iterator[str]:
            for evt in iter_sse_data_lines(resp):
                delta = evt["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta

        return generator()

# =================
# IBM watsonx backend
# =================

class WatsonXBackend(LLMBackend):
    """
    Watsonx text generation backend.

    Notes:
    - Requires an IAM Bearer token. You can pass `bearer_token` directly, or provide
      an IBM Cloud `api_key` and let this backend fetch the token from IBM IAM.
    - Many accounts require `X-IBM-Project-Id` or `X-IBM-Space-Id` header. Provide one of them.
    - 'version' query param is mandatory.
    - Watsonx does not (generally) provide SSE streaming for generation; stream=False only.
    """

    IAM_URL = "https://iam.cloud.ibm.com/identity/token"  # token exchange endpoint
    bearer_token = None
    def __init__(
        self,
        base_url: str,                    # e.g., "https://us-south.ml.cloud.ibm.com/ml/v1"
        model_name: str,                  # e.g., "ibm/granite-13b-instruct-v2"
        eval_mode: bool = True,
        *,
        version: str = "2024-05-01",
        timeout: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
        auto_refresh_token: bool = True,
        **gen_params,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model_name = model_name
        self.eval_mode = eval_mode
        # Get credentials from environment if not provided
        self.api_key = os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.space_id = os.getenv("WATSONX_SPACE_ID")
        
        self.version = version
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.auto_refresh_token = auto_refresh_token
        self._token_expiry_ts: Optional[float] = None  # naive cache if we exchange token

        self.gen_params = gen_params

    # ---- Auth helpers ----
    def _maybe_refresh_token(self):
        if self.api_key and self.auto_refresh_token:
            self._exchange_api_key_for_token()

    def _exchange_api_key_for_token(self):
        """
        Exchanges IBM Cloud API key for an IAM bearer token.
        """
        headers = {"Content-Type": "application/x-www-form-urlencoded", **self.extra_headers}
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key,
        }
        resp = requests.post(self.IAM_URL, data=data, headers=headers, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(f"IBM IAM token exchange failed: {e} {resp.text[:500]}") from None
        j = resp.json()
        token = j.get("access_token")
        expires_in = j.get("expires_in", 3600)
        if not token:
            raise RuntimeError(f"IBM IAM token exchange returned no access_token: {j}")
        self.bearer_token = token
        self._token_expiry_ts = time.time() + expires_in if expires_in else None

    # ---- HTTP helpers ----

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if not self.bearer_token:
            if self.auto_refresh_token:
                self._maybe_refresh_token()
            if not self.bearer_token:
                raise RuntimeError("WatsonXBackend missing bearer token. Provide bearer_token or api_key.")
        h["Authorization"] = f"Bearer {self.bearer_token}"
        if self.project_id:
            h["X-IBM-Project-Id"] = self.project_id
        if self.space_id:
            h["X-IBM-Space-Id"] = self.space_id
        return h

    def _post(self, path: str, body: dict) -> requests.Response:
        # watsonx generation endpoint: /text/generation?version=YYYY-MM-DD
        url = urljoin(self.base_url + "/", f"{path.lstrip('/')}?version={self.version}")

        resp = requests.post(url, json=body, headers=self._headers(), timeout=self.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = f" {resp.text[:500]}"
            except Exception:
                pass
            raise requests.HTTPError(f"{e}{detail}") from None
        return resp

    # ---- Interface ----

    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
    ) -> str:
        if stream:
            raise NotImplementedError("Watsonx streaming is not supported via SSE in this backend.")

        # Use defaults that match the working llm_chat_utils implementation
        parameters = {
            "decoding_method": "greedy" if self.eval_mode else "sample",
            "min_new_tokens": 1,
            "stop_sequences": ["<|eom_id|>"],
        }
        if self.gen_params:
            # Map max_tokens to max_new_tokens for consistency with LangChain/LiteLLM
            mapped_params = dict(self.gen_params)
            if "max_tokens" in mapped_params and "max_new_tokens" not in mapped_params:
                mapped_params["max_new_tokens"] = mapped_params.pop("max_tokens")
            parameters.update(mapped_params)
        # Use the /text/chat endpoint which accepts messages directly (not deprecated)
        body = {
            "model_id": self.model_name,
            "messages": messages,  # Pass messages array directly
            "parameters": parameters,
        }
        
        # WatsonX requires project_id or space_id in the request body
        if self.project_id:
            body["project_id"] = self.project_id
        elif self.space_id:
            body["space_id"] = self.space_id

        resp = self._post("text/chat", body)  # Use text/chat instead of text/generation
        j = resp.json()

        # Response shape for chat endpoint: {"choices": [{"message": {"content": "..."}}]}
        try:
            return j["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected watsonx response shape: {j}")


# =============================================================================
# Backend Factory
# =============================================================================

def create_backend(
    provider: str,
    base_url: str,
    model_name: str,
    eval_mode: bool = True,
    **gen_params
) -> LLMBackend:
    """
    Factory function to create appropriate backend based on provider.
    
    Args:
        provider: Backend provider name (openai, watsonx)
        base_url: Base URL for the API endpoint
        model_name: Model identifier
        eval_mode: bool, if true - set temperature to 0
        
    Returns:
        LLMBackend instance
        
    Raises:
        ValueError: If provider is not supported
        
    Examples:
        # OpenAI
        backend = create_backend(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4"
        )
        
        # WatsonX
        backend = create_backend(
            provider="watsonx",
            base_url="https://us-south.ml.cloud.ibm.com/ml/v1",
            model_name="ibm/granite-13b-instruct-v2"
        )
    """
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return OpenAIStyleHTTPBackend(
            base_url=base_url,
            model_name=model_name,
            eval_mode=eval_mode,
            **gen_params
        )
    elif provider_lower == "watsonx":
        return WatsonXBackend(
            base_url=base_url,
            model_name=model_name,
            eval_mode=eval_mode,
            **gen_params
        )
    else:
        raise ValueError(
            f"Unsupported endpoint provider: {provider}. "
            f"Supported providers: openai, watsonx"
        )
