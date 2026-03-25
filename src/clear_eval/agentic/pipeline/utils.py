"""
Shared utilities for agentic pipelines.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from clear_eval.pipeline.config_loader import load_config


@dataclass
class InferenceConfig:
    """Configuration for LLM inference in trajectory evaluation."""
    model_id: str
    provider: str
    inference_backend: str = "litellm"
    endpoint_url: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "InferenceConfig":
        """Create InferenceConfig from pipeline config dict."""
        return cls(
            model_id=config.get('eval_model_name'),
            provider=config.get('provider'),
            inference_backend=config.get('inference_backend', 'litellm'),
            endpoint_url=config.get('endpoint_url'),
            model_params=config.get('eval_model_params', {}),
        )


# Path to shared default config
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "setup", "default_agentic_config.yaml")


def build_cli_overrides(args) -> dict:
    """
    Build CLI overrides dictionary from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of CLI overrides ready for load_config()
    """
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None and key != 'agentic_config_path'
    }


def load_pipeline_config(
    user_config_path: Optional[str] = None,
    **cli_overrides
) -> dict:
    """
    Load pipeline configuration with standard precedence.

    Precedence (lowest to highest):
        1. Default config (setup/default_agentic_config.yaml)
        2. User config file (if provided)
        3. CLI overrides

    Args:
        user_config_path: Optional path to user config file (JSON or YAML)
        **cli_overrides: CLI argument overrides

    Returns:
        Merged configuration dictionary
    """
    return load_config(
        DEFAULT_CONFIG_PATH,
        user_config_path,
        **cli_overrides
    )


def get_run_output_dir(
    base_output_dir: str,
    run_name: Optional[str] = None
) -> Tuple[Path, str]:
    """
    Get the output directory for a pipeline run.

    Creates output path under base_output_dir/<run_name>/
    If run_name is not provided, generates a timestamp-based name.

    Args:
        base_output_dir: Base output directory from config
        run_name: Optional run identifier (uses timestamp if None)

    Returns:
        Tuple of (output_dir Path, resolved run_name string)
    """
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(base_output_dir) / run_name

    return output_dir, run_name


def validate_required_config(
    config: dict,
    required_fields: list,
    parser=None
) -> None:
    """
    Validate that required configuration fields are present.

    Args:
        config: Configuration dictionary
        required_fields: List of required field names
        parser: Optional argparse parser for error reporting

    Raises:
        ValueError: If a required field is missing and no parser provided
    """
    for field in required_fields:
        if not config.get(field):
            msg = f"{field} is required (set in config or use --{field.replace('_', '-')})"
            if parser:
                parser.error(msg)
            else:
                raise ValueError(msg)
