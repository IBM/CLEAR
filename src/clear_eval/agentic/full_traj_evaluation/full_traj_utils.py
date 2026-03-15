import json
import os
from pathlib import Path

CHARS_PER_TOKEN = 4
RESPONSE_RESERVED_TOKENS = 4_096
PROMPT_OVERHEAD_TOKENS = 2_500
CONTEXT_SAFETY_MARGIN = 0.90

# Default context window (can be overridden per model)
DEFAULT_CONTEXT_TOKENS = 128_000


def get_max_trajectory_chars(context_tokens: int = DEFAULT_CONTEXT_TOKENS) -> int:
    """Compute the maximum trajectory text length (in characters) for a model."""
    available_tokens = context_tokens - RESPONSE_RESERVED_TOKENS - PROMPT_OVERHEAD_TOKENS
    max_chars = int(available_tokens * CHARS_PER_TOKEN * CONTEXT_SAFETY_MARGIN)
    return max_chars




