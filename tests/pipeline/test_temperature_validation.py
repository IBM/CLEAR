"""
Test temperature validation across all 3 client backends.

Tests that:
1. get_llm_client creates a working client
2. The temperature probe fires and disables temperature=0 if the model rejects it
3. The client works for subsequent calls after the probe

Usage:
    python tests/test_temperature_validation.py

Requires env vars:
    OPENAI_API_KEY          — for openai provider (Azure/gpt-5-2025-08-07, Azure/gpt-4.1)
    WATSONX_APIKEY          — for watsonx provider (openai/gpt-oss-120b)
    WATSONX_URL             — watsonx endpoint URL
    WATSONX_PROJECT_ID or WATSONX_SPACE_ID
"""

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from clear_eval.pipeline.inference_utils.llm_client import get_llm_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_PROMPT = [{"role": "user", "content": "Say hello in one word."}]

TEST_CASES = [
    # (description, provider, model, backend, endpoint)
     ("LangChain  | openai | Azure/gpt-5-2025-08-07",   "openai",  "Azure/gpt-5-2025-08-07",  "langchain", None),
     ("LiteLLM    | openai | Azure/gpt-5-2025-08-07",   "openai",  "Azure/gpt-5-2025-08-07",  "litellm", None),
     ("LangChain  | openai | Azure/gpt-4.1",            "openai",  "Azure/gpt-4.1",           "langchain", None),
     ("LiteLLM    | openai | Azure/gpt-4.1",            "openai",  "Azure/gpt-4.1",           "litellm", None),
   #  ("LangChain  | watsonx | openai/gpt-oss-120b",     "watsonx", "openai/gpt-oss-120b",     "langchain", None),
   #  ("LiteLLM    | watsonx | openai/gpt-oss-120b",     "watsonx", "openai/gpt-oss-120b",     "litellm",),
#Endpoint backend only works with explicit endpoint_url; add if you have one:
   # ("Endpoint   | openai | llama3:8b", "openai", "llama3:8b", "endpoint", "http://localhost:11434/"),
]

def run_test(description, provider, model, backend, endpoint):
    logger.info("=" * 60)
    logger.info(f"TEST: {description}")
    logger.info("=" * 60)

    # Step 1: Create client with eval_mode=True (triggers temperature probe)
    try:
        client = get_llm_client(
            provider=provider,
            model=model,
            inference_backend=backend,
            eval_mode=True,
            endpoint_url=endpoint
        )
    except Exception as e:
        logger.error(f"FAIL - Client creation failed: {e}")
        return False

    # Check if temperature was disabled by the probe
    eval_mode_after = getattr(client, "eval_mode", "N/A")
    logger.info(f"  eval_mode after probe: {eval_mode_after}")

    # Step 2: Make a real call to confirm the client works
    try:
        response = client.invoke(TEST_PROMPT)
        logger.info(f"  Response: {response[:100]}")
        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error(f"  FAIL - invoke after probe failed: {e}")
        return False


def main():
    results = []
    for desc, provider, model, backend, endpoint in TEST_CASES:
        try:
            passed = run_test(desc, provider, model, backend, endpoint)
        except Exception as e:
            logger.error(f"  FAIL - Unexpected error: {e}")
            passed = False
        results.append((desc, passed))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for desc, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
