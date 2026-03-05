"""
Unified LLM client interface for CLEAR eval.

Abstracts both the LLM backend (LangChain vs LiteLLM) and execution model
(threaded vs async), allowing eval_utils to use a single interface.

Usage:
    from clear_eval.pipeline.llm_client import get_llm_client, run_parallel

    # Get client based on config
    client = get_llm_client(config)

    # Single call
    content = client.invoke("What is 2+2?")
    content = client.invoke([{"role": "user", "content": "Hello"}])

    # Parallel execution (auto-selects threaded vs async based on client type)
    results = run_parallel(
        func=my_func,
        inputs=input_list,
        max_workers=10
    )
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from clear_eval.pipeline.llm_chat_utils import get_rits_base

logger = logging.getLogger(__name__)

# Suppress LiteLLM logging and prints
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm._logging._disable_debugging()  # Disables print statements
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Module-level event loop to avoid LiteLLM queue binding issues
# when asyncio.run() creates new loops
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a reusable event loop for async operations."""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
    return _event_loop


def run_async(coro):
    """
    Run an async coroutine using a shared event loop.

    This avoids issues with LiteLLM's internal logging queue binding
    to different event loops when asyncio.run() is called multiple times.
    """
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(coro)


@dataclass
class ParallelResult:
    """Result from parallel execution."""
    is_success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def invoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """
        Invoke the LLM and return content string.

        Args:
            messages: String prompt or list of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            Response content as string
        """
        pass

    async def ainvoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """Async invoke. Default implementation wraps sync invoke."""
        return await asyncio.to_thread(self.invoke, messages, **kwargs)


class LangChainClient(LLMClient):
    """LLM client wrapping LangChain chat models."""

    def __init__(self, llm):
        """
        Args:
            llm: LangChain chat model (ChatOpenAI, AzureChatOpenAI, ChatWatsonx, etc.)
        """
        self.llm = llm

    def invoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        # LangChain accepts string or list of BaseMessage objects
        # For simplicity, pass through - LangChain handles string prompts
        normalized = normalize_messages(messages)
        response = self.llm.invoke(normalized, **kwargs)
        if response is None:
            return ""
        return response.content.strip()

    async def ainvoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """Native async invoke using LangChain's ainvoke."""

        normalized = normalize_messages(messages)
        response = await self.llm.ainvoke(normalized, **kwargs)
        if response is None:
            return ""
        return response.content.strip()


def normalize_messages(messages: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Convert string prompt to message list."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


class LiteLLMClient(LLMClient):
    """LLM client using LiteLLM for 100+ providers."""

    def __init__(
        self,
        provider: str,
        model: str,
        eval_mode: bool = True,
        max_retries: int = 3,
        **params
    ):
        """
        Args:
            provider: Provider name (openai, azure, anthropic, bedrock, etc.)
            model: Model identifier
            eval_mode: If True, sets temperature=0 for deterministic output
            max_retries: Number of retries on failure
            **params: Additional model parameters
        """
        self.provider = provider
        self.model = model
        self.eval_mode = eval_mode
        self.max_retries = max_retries
        self.params = params

        self._configure_provider()
        self._litellm_model = self._get_litellm_model()

    def _get_litellm_model(self) -> str:
        """Map provider/model to litellm format: provider/model_name."""
        if self.provider == "rits":
            # RITS uses OpenAI-compatible API
            return f"openai/{self.model}"
        else:
            # Standard format: provider/model
            return f"{self.provider}/{self.model}"

    def _configure_provider(self):
        """Configure litellm for the provider."""
        import litellm

        # Reset custom config
        litellm.api_base = None
        litellm.headers = None

        if self.provider == "rits":
            rits_api_key = os.getenv("RITS_API_KEY")
            if not rits_api_key:
                raise KeyError("RITS_API_KEY env var required for RITS.")
            model_base = get_rits_base(self.model)
            litellm.api_base = f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_base}/v1"
            litellm.headers = {"RITS_API_KEY": rits_api_key}

        elif self.provider == "watsonx":
            if not os.getenv("WATSONX_URL"):
                raise KeyError("WATSONX_URL env var required.")
            watsonx_key = os.getenv("WATSONX_APIKEY") or os.getenv("WATSONX_API_KEY")
            if not watsonx_key:
                raise KeyError("WATSONX_APIKEY or WATSONX_API_KEY env var required.")
            if os.getenv("WATSONX_APIKEY") and not os.getenv("WATSONX_API_KEY"):
                os.environ["WATSONX_API_KEY"] = os.getenv("WATSONX_APIKEY")
            if not os.getenv("WATSONX_PROJECT_ID") and not os.getenv("WATSONX_SPACE_ID"):
                raise KeyError("WATSONX_PROJECT_ID or WATSONX_SPACE_ID env var required.")

        elif self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise KeyError("OPENAI_API_KEY env var required.")

        # Other providers: trust user has set credentials per litellm docs
        logger.debug(f"Configured {self.provider} provider")

    def _get_params(self, **kwargs) -> dict:
        """Get merged parameters for the call."""
        params = {**self.params, **kwargs}
        if self.eval_mode and "temperature" not in params:
            params["temperature"] = 0
        params["num_retries"] = self.max_retries

        # Watsonx-specific
        if self.provider == "watsonx":
            project_id = os.getenv("WATSONX_PROJECT_ID")
            space_id = os.getenv("WATSONX_SPACE_ID")
            if project_id:
                params["project_id"] = project_id
            elif space_id:
                params["space_id"] = space_id

        return params

    def invoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        from litellm import completion

        normalized = normalize_messages(messages)
        params = self._get_params(**kwargs)

        response = completion(
            model=self._litellm_model,
            messages=normalized,
            **params
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    async def ainvoke(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        from litellm import acompletion

        normalized = normalize_messages(messages)
        params = self._get_params(**kwargs)

        response = await acompletion(
            model=self._litellm_model,
            messages=normalized,
            **params
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


# =============================================================================
# Parallel Execution
# =============================================================================

def _run_threaded(
    func: Callable,
    inputs: List[Any],
    max_workers: int = 10,
    task_timeout: float = 300,
    error_prefix: str = "Error: ",
    progress_desc: str = "Processing"
) -> List[ParallelResult]:
    """Run function over inputs using ThreadPoolExecutor."""
    if not inputs:
        return []

    if len(inputs) == 1:
        item = inputs[0]
        try:
            if isinstance(item, tuple):
                result = func(*item)
            else:
                result = func(item)
            return [ParallelResult(is_success=True, result=result)]
        except Exception as e:
            return [ParallelResult(is_success=False, error=f"{error_prefix}: {e}")]

    results = [None] * len(inputs)
    with ThreadPoolExecutor(max_workers) as executor:
        future_to_idx = {}
        for i, item in enumerate(inputs):
            if isinstance(item, tuple):
                future = executor.submit(func, *item)
            else:
                future = executor.submit(func, item)
            future_to_idx[future] = i

        for future in tqdm(as_completed(future_to_idx), total=len(inputs), desc=progress_desc):
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=task_timeout)
                results[idx] = ParallelResult(is_success=True, result=result)
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                results[idx] = ParallelResult(is_success=False, error=f"{error_prefix}item {idx}: {e}")

    return results


async def _run_async(
    func: Callable,
    inputs: List[Any],
    max_workers: int = 10,
    task_timeout: float = 300,
    error_prefix: str = "Error: ",
    progress_desc: str = "Processing"
) -> List[ParallelResult]:
    """Run async function over inputs using asyncio."""
    if not inputs:
        return []

    semaphore = asyncio.Semaphore(max_workers)

    async def limited_call(idx: int, item) -> ParallelResult:
        async with semaphore:
            try:
                if isinstance(item, tuple):
                    result = await asyncio.wait_for(func(*item), timeout=task_timeout)
                else:
                    result = await asyncio.wait_for(func(item), timeout=task_timeout)
                return ParallelResult(is_success=True, result=result)
            except asyncio.TimeoutError:
                return ParallelResult(is_success=False, error=f"{error_prefix}item {idx}: Timeout")
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                return ParallelResult(is_success=False, error=f"{error_prefix}item {idx}: {e}")

    tasks = [limited_call(i, item) for i, item in enumerate(inputs)]
    return await tqdm_asyncio.gather(*tasks, desc=progress_desc)


def run_parallel(
    func: Callable,
    inputs: List[Any],
    use_async: bool = False,
    max_workers: int = 10,
    task_timeout: float = 300,
    error_prefix: str = "Error: ",
    progress_desc: str = "Processing"
) -> List[ParallelResult]:
    """
    Run function over inputs in parallel.

    Automatically selects threaded or async execution based on use_async flag.

    Args:
        func: Function to call. For async mode, must be async function.
        inputs: List of inputs. Each is either a single value or tuple of args.
        use_async: If True, use async execution. If False, use threads.
        max_workers: Maximum concurrent executions
        task_timeout: Timeout per task in seconds
        error_prefix: Prefix for error messages
        progress_desc: Progress bar description

    Returns:
        List of ParallelResult in same order as inputs
    """
    if use_async:
        return run_async(_run_async(
            func, inputs, max_workers, task_timeout, error_prefix, progress_desc
        ))
    else:
        return _run_threaded(
            func, inputs, max_workers, task_timeout, error_prefix, progress_desc
        )


# =============================================================================
# Factory Function
# =============================================================================

def get_llm_client(
    provider: str,
    model: str,
    use_litellm: bool = False,
    eval_mode: bool = True,
    parameters: Optional[Dict] = None
) -> LLMClient:
    """
    Get an LLM client based on configuration.

    Args:
        provider: Provider name (openai, azure, watsonx, rits, anthropic, etc.)
        model: Model identifier
        use_litellm: If True, use LiteLLM. If False, use LangChain.
        eval_mode: If True, use temperature=0 for deterministic output
        parameters: Additional model parameters

    Returns:
        LLMClient instance (LangChainClient or LiteLLMClient)
    """
    parameters = parameters or {}

    if use_litellm:
        return LiteLLMClient(
            provider=provider,
            model=model,
            eval_mode=eval_mode,
            **parameters
        )
    else:
        # Use existing LangChain factory
        from clear_eval.pipeline.llm_chat_utils import get_chat_llm
        llm = get_chat_llm(provider, model, parameters=parameters, eval_mode=eval_mode)
        return LangChainClient(llm)