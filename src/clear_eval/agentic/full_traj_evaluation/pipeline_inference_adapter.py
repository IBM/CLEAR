"""
Pipeline Inference Adapter for Full Trajectory Evaluation
==========================================================

Provides inference interface compatible with full_traj_evaluation scripts
while using the pipeline's unified LLMClient and run_parallel infrastructure.

This adapter eliminates the need for custom inference_utils.py and async/semaphore patterns.
"""

import logging
from typing import Optional, Dict, Any

from clear_eval.pipeline.llm_client import get_llm_client, run_parallel, LLMClient

logger = logging.getLogger(__name__)


class LLMClientAdapter:
    """
    Adapter for pipeline's LLMClient.
    
    Provides simple interface for full_traj_evaluation scripts.
    """
    
    def __init__(
        self,
        provider: str,
        model_id: str,
        eval_mode: bool = True,
        use_litellm: bool = True,
        **parameters
    ):
        """
        Initialize LLM client adapter.
        
        Args:
            provider: Provider name (openai, azure, watsonx, rits, anthropic, etc.)
            model_id: Model identifier
            eval_mode: If True, use temperature=0 for deterministic output
            use_litellm: If True, use LiteLLM. If False, use LangChain.
            **parameters: Additional model parameters
        """
        self.provider = provider
        self.model_id = model_id
        self.eval_mode = eval_mode
        self.use_litellm = use_litellm
        self.parameters = parameters
        
        # Create the underlying LLM client
        self.client: LLMClient = get_llm_client(
            provider=provider,
            model=model_id,
            use_litellm=use_litellm,
            eval_mode=eval_mode,
            parameters=parameters
        )
        
        logger.debug(f"Initialized LLMClientAdapter: {provider}/{model_id}")
    
    def call(
        self,
        prompt: str,
        system_message: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **extra_params
    ) -> Optional[str]:
        """
        Synchronous call to LLM.
        
        Args:
            prompt: User prompt
            system_message: System message
            temperature: Sampling temperature (overrides eval_mode if provided)
            max_tokens: Maximum tokens to generate
            **extra_params: Additional provider-specific parameters
            
        Returns:
            Model response text or None on failure
        """
        try:
            # Build messages list
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Build kwargs
            kwargs = {**extra_params}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            
            # Call using client's invoke method
            response = self.client.invoke(messages, **kwargs)
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            return None
    
    async def call_async(
        self,
        prompt: str,
        system_message: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **extra_params
    ) -> Optional[str]:
        """
        Async call to LLM.
        
        Args:
            prompt: User prompt
            system_message: System message
            temperature: Sampling temperature (overrides eval_mode if provided)
            max_tokens: Maximum tokens to generate
            **extra_params: Additional provider-specific parameters
            
        Returns:
            Model response text or None on failure
        """
        try:
            # Build messages list
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Build kwargs
            kwargs = {**extra_params}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            
            # Call using client's async method
            response = await self.client.ainvoke(messages, **kwargs)
            return response
            
        except Exception as e:
            logger.error(f"Error in async LLM call: {e}")
            return None


# Global client cache to avoid recreating clients
_client_cache = {}


def get_llm_client_adapter(
    provider: str,
    model_id: str,
    eval_mode: bool = True,
    use_litellm: bool = True,
    **parameters
) -> LLMClientAdapter:
    """
    Get or create an LLM client adapter.

    Caches clients by (provider, model_id) to avoid recreation.

    Args:
        provider: Provider name
        model_id: Model identifier
        eval_mode: If True, use temperature=0
        use_litellm: If True, use LiteLLM. If False, use LangChain.
        **parameters: Additional model parameters

    Returns:
        LLMClientAdapter instance
    """
    cache_key = (provider, model_id, eval_mode, use_litellm)

    if cache_key not in _client_cache:
        _client_cache[cache_key] = LLMClientAdapter(
            provider=provider,
            model_id=model_id,
            eval_mode=eval_mode,
            use_litellm=use_litellm,
            **parameters
        )

    return _client_cache[cache_key]


def call_llm(
    prompt: str,
    *,
    system_message: str = "",
    provider: str,
    model_id: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **extra_params
) -> Optional[str]:
    """
    Synchronous interface for calling LLM.

    Uses the pipeline's LLMClient infrastructure.

    Args:
        prompt: User prompt
        system_message: System message
        provider: Provider name (openai, azure, watsonx, rits, anthropic, etc.)
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **extra_params: Additional provider-specific parameters

    Returns:
        Model response text or None on failure
    """
    # Get or create client
    client = get_llm_client_adapter(
        provider=provider,
        model_id=model_id,
        eval_mode=(temperature is None or temperature == 0),
        use_litellm=True,  # Use LiteLLM for better provider support
    )

    # Call synchronously
    return client.call(
        prompt=prompt,
        system_message=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra_params
    )


async def call_llm_async(
    prompt: str,
    *,
    system_message: str = "",
    provider: str,
    model_id: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    session=None,  # Kept for API compatibility but not used
    **extra_params
) -> Optional[str]:
    """
    Async interface for calling LLM (compatible with old inference_utils.py API).

    Uses the pipeline's LLMClient infrastructure.

    Args:
        prompt: User prompt
        system_message: System message
        provider: Provider name (openai, azure, watsonx, rits, anthropic, etc.)
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        session: aiohttp session (kept for compatibility, not used)
        **extra_params: Additional provider-specific parameters

    Returns:
        Model response text or None on failure
    """
    # Get or create client
    client = get_llm_client_adapter(
        provider=provider,
        model_id=model_id,
        eval_mode=(temperature is None or temperature == 0),
        use_litellm=True,  # Use LiteLLM for better provider support
    )

    # Call async
    return await client.call_async(
        prompt=prompt,
        system_message=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra_params
    )


def evaluate_batch_parallel(
    evaluate_func,
    entries: list,
    max_workers: int = 10,
    use_async: bool = False,
    progress_desc: str = "Evaluating"
) -> list:
    """
    Evaluate a batch of entries in parallel using pipeline's run_parallel.
    
    Replaces the custom async/semaphore pattern with pipeline infrastructure.
    
    Args:
        evaluate_func: Function to call for each entry.
                      For async mode, must be async function.
                      Should accept a single entry and return result dict or None.
        entries: List of entries to evaluate
        max_workers: Maximum concurrent executions
        use_async: If True, use async execution. If False, use threads.
        progress_desc: Progress bar description
        
    Returns:
        List of successful results (None results are filtered out)
    """
    if not entries:
        return []
    
    # Run in parallel
    parallel_results = run_parallel(
        func=evaluate_func,
        inputs=entries,
        use_async=use_async,
        max_workers=max_workers,
        task_timeout=300,
        error_prefix="Evaluation error",
        progress_desc=progress_desc
    )
    
    # Filter out failures and None results
    results = []
    for pr in parallel_results:
        if pr.is_success and pr.result is not None:
            results.append(pr.result)
    
    return results