import os
import yaml
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS_DIR = os.path.join(Path(CURRENT_DIR).parents[1], "experiments")


def load_yaml(filepath):
    """Load YAML file if it exists."""
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as file:
            return yaml.safe_load(file) or {}  # Ensure it's a dict
    return {}


def merge_configs(defaults, overrides):
    """Recursively merge two dictionaries (user config overrides defaults)."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in defaults:
            defaults[key] = merge_configs(defaults.get(key, {}), value)
        else:
            defaults[key] = value
    return defaults


def load_config(default_configs_path, user_config_path=None, **overrides):
    """Load and merge configuration."""

    default_config = load_yaml(default_configs_path)
    if user_config_path:
        user_config = load_yaml(user_config_path)
        merged_config = merge_configs(default_config, user_config)
    else:
        merged_config = default_config

    if overrides:
        merged_config = merge_configs(merged_config, overrides)

    return merged_config
