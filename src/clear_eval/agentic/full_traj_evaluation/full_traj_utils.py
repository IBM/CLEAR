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

def discover_trajectories(
        base_dir: Path,
        filter_dataset: str | None = None,
        filter_model: str | None = None,
    ) -> list[dict]:
        """Discover all trajectory JSON files.

        Returns list of dicts: {dataset, model_name, file_path, traj_name}
        """
        results = []

        for dataset_dir in sorted(base_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            dataset_name = dataset_dir.name

            if filter_dataset and dataset_name != filter_dataset:
                continue

            for model_dir in sorted(dataset_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                model_name = model_dir.name

                if filter_model and model_name != filter_model:
                    continue

                trace_dir = model_dir / "traces_compact"

                for json_file in sorted(trace_dir.glob("*.json")):

                    traj_name = json_file.stem
                    results.append({
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "file_path": json_file,
                        "traj_name": traj_name,
                    })

        return results

def middle_out(text, limit):
        if len(text) <= limit: return text
        half = limit // 2
        return f"{text[:half]}\n\n... [TRUNCATED] ...\n\n{text[-half:]}"


def _cap_trajectory(trajectory_text: str, max_len: int) -> str:
    """Truncate trajectory text if it exceeds max_len characters.

    The max_len should be computed via ``get_max_trajectory_chars()`` so that
    each judge model gets a limit tailored to its context window.
    """
    if len(trajectory_text) > max_len:
       print(f"Trajectory too long: {len(trajectory_text)} > {max_len}")
       return middle_out(trajectory_text, max_len)
    return trajectory_text



