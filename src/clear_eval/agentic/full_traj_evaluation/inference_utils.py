"""
Multi-Provider Async Inference Utilities
=========================================

Provides async inference support for multiple LLM providers:
- RITS (IBM Research Infrastructure)
- watsonx (IBM watsonx.ai)
- OpenAI (OpenAI API)

Maintains async/await pattern from original RITS implementation while
supporting multiple providers.
"""

import os
import asyncio
import aiohttp
from typing import Optional
from dotenv import load_dotenv
os.environ["OPENAI_API_KEY"]="sk-LSs6XhqDC9pHH4D1lCOHcw"
load_dotenv()


# =============================================================================
# Provider Configuration
# =============================================================================

class ProviderConfig:
    """Configuration for LLM providers."""
    
    # RITS Configuration
    RITS_API_KEY = os.getenv("RITS_API_KEY")
    RITS_BASE_URL_TEMPLATE = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_base}/v1"
    
    # watsonx Configuration
    WATSONX_URL = os.getenv("WATSONX_URL")
    WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_SPACE_ID = os.getenv("WATSONX_SPACE_ID")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_API_URL")
    
    # Model name mappings for RITS
    RITS_MODEL_MAPPINGS = {
        "openai/gpt-oss-20b": "gpt-oss-20b",
        "openai/gpt-oss-120b": "gpt-oss-120b",
        "deepseek-ai/DeepSeek-V3.2": "deepseek-v3-2",
        "microsoft/phi-4": "microsoft-phi-4",
        "microsoft/Phi-4-reasoning": "phi-4-reasoning",
        "mistralai/mixtral-8x7B-instruct-v0.1": "mixtral-8x7b-instruct-v01",
        "mistralai/mixtral-8x22B-instruct-v0.1": "mixtral-8x22b-instruct-v01",
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": "llama-4-mvk-17b-128e-fp8",
        "deepseek-ai/DeepSeek-V3": "deepseek-v3-h200",
        "meta-llama/Llama-3.1-8B-Instruct": "llama-3-1-8b-instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama-4-scout-17b-16e-instruct",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "mistral-small-3-1-24b-2503",
        "ibm-granite/granite-guardian-3.2-5b": "granite-guardian-3-2-5b-ris",
    }


# =============================================================================
# RITS Provider (Original async implementation)
# =============================================================================

async def call_rits_async(
    prompt: str,
    *,
    system_message: str,
    model_id: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 3,
    session: aiohttp.ClientSession,
    **extra_params
) -> Optional[str]:
    """
    Send async chat-completion request to RITS-hosted model.
    
    Args:
        prompt: User prompt
        system_message: System message
        model_id: Full model ID (e.g., "openai/gpt-oss-120b")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Number of retry attempts
        session: aiohttp session
        **extra_params: Additional provider-specific parameters
            (e.g., top_p, frequency_penalty, reasoning_effort, include_reasoning, etc.)
        
    Returns:
        Model response text or None on failure
    """
    # Get model base name for URL
    model_base = ProviderConfig.RITS_MODEL_MAPPINGS.get(model_id)
    if not model_base:
        # Fallback: derive from model ID
        model_base = model_id.split("/")[-1].replace(".", "-").lower()
    
    base_url = ProviderConfig.RITS_BASE_URL_TEMPLATE.format(model_base=model_base)
    api_key = ProviderConfig.RITS_API_KEY
    
    if not api_key:
        raise ValueError("RITS_API_KEY not set in environment")
    
    headers = {
        "RITS_API_KEY": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Build payload with defaults, then override with extra_params
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Merge extra_params - they override defaults if there are conflicts
    payload.update(extra_params)
    
    url = f"{base_url}/chat/completions"
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                await asyncio.sleep(2 ** attempt)
            
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                if resp.status == 403:
                    body = await resp.text()
                    raise ValueError(f"403 Forbidden: {body[:300]}")
                body = await resp.text()
                if attempt == max_retries - 1:
                    raise ValueError(f"HTTP {resp.status}: {body[:300]}")
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
    
    return None


# =============================================================================
# watsonx Provider (Async wrapper)
# =============================================================================

async def call_watsonx_async(
    prompt: str,
    *,
    system_message: str,
    model_id: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 3,
    session: aiohttp.ClientSession,
    **extra_params
) -> Optional[str]:
    """
    Send async chat-completion request to watsonx.
    
    Note: watsonx SDK doesn't have native async support, so we run in executor.
    For true async, consider using aiohttp directly with watsonx REST API.
    
    Args:
        **extra_params: Additional parameters (currently not used by watsonx wrapper,
            but kept for API consistency)
    """
    # Import here to avoid dependency if not using watsonx
    from langchain_ibm import ChatWatsonx
    
    url = ProviderConfig.WATSONX_URL
    apikey = ProviderConfig.WATSONX_APIKEY
    
    if not url or not apikey:
        raise ValueError("WATSONX_URL and WATSONX_APIKEY must be set")
    
    # Determine project_id or space_id
    project_id = ProviderConfig.WATSONX_PROJECT_ID
    space_id = ProviderConfig.WATSONX_SPACE_ID
    
    if not project_id and not space_id:
        raise ValueError("Either WATSONX_PROJECT_ID or WATSONX_SPACE_ID must be set")
    
    # Build LLM kwargs with defaults, then override with extra_params
    llm_kwargs = {
        "model_id": model_id,
        "url": url,
        "apikey": apikey,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "include_reasoning":True,
        "reasoning_effort":"high" # TODO!!!!!
    }
    # Merge extra_params - they override defaults
    llm_kwargs.update(extra_params)
    
    # Add space_id or project_id
    if space_id:
        llm_kwargs["space_id"] = space_id
    else:
        llm_kwargs["project_id"] = project_id
    
    # Create LLM instance
    llm = ChatWatsonx(**llm_kwargs)
    
    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    
    def _sync_call():
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        response = llm.invoke(messages)
        return response.content
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                await asyncio.sleep(2 ** attempt)
            result = await loop.run_in_executor(None, _sync_call)
            return result
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
    
    return None


# =============================================================================
# OpenAI Provider (Async)
# =============================================================================

async def call_openai_async(
    prompt: str,
    *,
    system_message: str,
    model_id: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 3,
    session: aiohttp.ClientSession,
    **extra_params
) -> Optional[str]:
    """
    Send async chat-completion request to OpenAI API.
    
    Args:
        prompt: User prompt
        system_message: System message
        model_id: Model ID (e.g., "gpt-4o", "gpt-4o-mini")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Number of retry attempts
        session: aiohttp session
        **extra_params: Additional provider-specific parameters
            (e.g., top_p, frequency_penalty, reasoning_effort, include_reasoning, etc.)
        
    Returns:
        Model response text or None on failure
    """
    api_key = ProviderConfig.OPENAI_API_KEY
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Build payload with defaults, then override with extra_params
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Merge extra_params - they override defaults if there are conflicts
    payload.update(extra_params)
    
    url = f"{os.getenv('OPENAI_BASE_URL')}/chat/completions"
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                await asyncio.sleep(2 ** attempt)
            
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                if resp.status in (401, 403):
                    body = await resp.text()
                    raise ValueError(f"HTTP {resp.status}: {body[:300]}")
                body = await resp.text()
                if attempt == max_retries - 1:
                    print("########## FAILED INFERENCE #########")
                    return None
                    #raise ValueError(f"HTTP {resp.status}: {body[:300]}")
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
    
    return None


# =============================================================================
# Unified Interface
# =============================================================================

async def call_llm_async(
    prompt: str,
    *,
    system_message: str,
    provider: str,
    model_id: str,
    temperature: float = 0.3,
    max_tokens: int = 16378,
    max_retries: int = 3,
    session: aiohttp.ClientSession,
    **extra_params
) -> Optional[str]:
    """
    Unified async interface for calling any supported LLM provider.
    
    Args:
        prompt: User prompt
        system_message: System message
        provider: Provider name ("rits", "watsonx", "openai")
        model_id: Model ID
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Number of retry attempts
        session: aiohttp session
        **extra_params: Additional provider-specific parameters
            (e.g., top_p, frequency_penalty, reasoning_effort, include_reasoning, etc.)
        
    Returns:
        Model response text or None on failure
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider == "rits":
        return await call_rits_async(
            prompt,
            system_message=system_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            session=session,
            **extra_params
        )
    elif provider == "watsonx":
        try:
            return await call_watsonx_async(
                prompt,
                system_message=system_message,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                session=session,
                **extra_params
            )
        except Exception as exc:
            print(exc)
    elif provider == "openai":
        return await call_openai_async(
            prompt,
            system_message=system_message,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            session=session,
            **extra_params
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: rits, watsonx, openai"
        )


def get_supported_providers() -> list:
    """Get list of supported provider names."""
    return ["rits", "watsonx", "openai"]

# async def probe_model_context_length(
#     model_id: str,
#     short_name: str,
#     provider: str = "rits",
#     session: Optional[aiohttp.ClientSession] = None,
# ) -> Optional[int]:
#     """
#     Probe the model's context length by sending a test request.
#
#     Args:
#         model_id: Model identifier
#         short_name: Short name for the model (RITS-specific)
#         provider: Provider name (default: "rits")
#         session: Optional aiohttp session
#
#     Returns:
#         Context length if successful, None otherwise
#     """
#     import logging
#     logger = logging.getLogger(__name__)
#
#     try:
#         test_prompt = "Hello"
#         response = await call_llm_async(
#             test_prompt,
#             system_message="You are a helpful assistant.",
#             provider=provider,
#             model_id=model_id,
#             max_tokens=10,
#             session=session,
#         )
#
#         if response:
#             logger.info(f"Successfully probed model {model_id}")
#             # For RITS, we could extract context length from response headers
#             # For now, return a default value
#             return 32768  # Common default for modern models
#         return None
#     except Exception as e:
#         logger.warning(f"Failed to probe model {model_id}: {e}")
#         return None


async def main():
    # Print sanity info
    print("Using BASE:", os.environ.get("OPENAI_BASE_URL"))
    print("Using KEY:", os.environ.get("OPENAI_API_KEY")[:5] + "... (hidden)")

    async with aiohttp.ClientSession() as session:
        print("Sending request…")
        result = await call_openai_async(
            prompt="Hello! Say 'pong'.",
            system_message="You are a test assistant.",
            model_id="Azure/gpt-5-2025-08-07",      # must match your proxy!
            session=session,
        )
        print("\n=== RESULT ===")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
