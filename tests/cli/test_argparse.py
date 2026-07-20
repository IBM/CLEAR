"""
Tests for CLEAR argument parsing and config merging.

Tests both regular and agentic flows, verifying:
- CLI argument parsing
- Config file loading (YAML/JSON)
- Config merging precedence: default YAML < user YAML < CLI overrides
- Provider-specific defaults resolution
- Boolean, JSON, and list argument types

Does NOT run any actual pipeline - only parses and verifies args.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# ============================================================================
# Imports
# ============================================================================

from clear_eval.args import (
    add_clear_args_to_parser,
    create_clear_parser,
    extract_clear_overrides,
    get_clear_arg_names,
    parse_args,
    parse_dict,
    str2bool,
)
from clear_eval.pipeline.config_loader import (
    load_config,
    load_config_file,
    merge_configs,
    resolve_provider_config,
)
from clear_eval.agentic.pipeline.argument_definitions import (
    add_agentic_pipeline_args,
    add_full_trajectory_args,
    add_preprocessing_args,
    add_unified_pipeline_args,
)
from clear_eval.agentic.pipeline.utils import (
    build_cli_overrides,
    load_pipeline_config,
    DEFAULT_CONFIG_PATH,
)

# Paths to default configs
REGULAR_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "src", "clear_eval", "pipeline", "setup", "default_config.yaml"
)
AGENTIC_DEFAULT_CONFIG = DEFAULT_CONFIG_PATH


# ============================================================================
# Helper utilities
# ============================================================================

def _create_temp_yaml(config_dict: dict) -> str:
    """Create a temporary YAML config file, return its path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config_dict, f)
    f.close()
    return f.name


def _create_temp_json(config_dict: dict) -> str:
    """Create a temporary JSON config file, return its path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_dict, f)
    f.close()
    return f.name


def _parse_regular_cli(cli_args: list) -> dict:
    """Simulate regular flow CLI parsing with given args."""
    with patch.object(sys, 'argv', ['run-clear-eval-analysis'] + cli_args):
        return parse_args()


def _parse_agentic_cli(cli_args: list) -> dict:
    """Simulate agentic flow CLI parsing, return (parsed_args, config_dict)."""
    import argparse
    parser = argparse.ArgumentParser()
    add_agentic_pipeline_args(parser)
    add_preprocessing_args(parser)
    add_unified_pipeline_args(parser)
    add_full_trajectory_args(parser)
    add_clear_args_to_parser(parser, group_name="CLEAR Configuration")

    with patch.object(sys, 'argv', ['run-clear-agentic-eval'] + cli_args):
        args = parser.parse_args()
    return args


# ============================================================================
# Test: str2bool utility
# ============================================================================

class TestStr2Bool:
    def test_true_values(self):
        for v in ("yes", "true", "t", "1", "True", "YES"):
            assert str2bool(v) is True

    def test_false_values(self):
        for v in ("no", "false", "f", "0", "False", "NO"):
            assert str2bool(v) is False

    def test_none_returns_none(self):
        assert str2bool(None) is None

    def test_bool_passthrough(self):
        assert str2bool(True) is True
        assert str2bool(False) is False

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            str2bool("maybe")


class TestParseDict:
    def test_valid_json(self):
        result = parse_dict('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            parse_dict("not json")


# ============================================================================
# Test: Regular Flow - CLI Parsing
# ============================================================================

class TestRegularFlowCLIParsing:
    """Test parse_args() returns only non-None values from CLI."""

    def test_no_args_returns_empty(self):
        result = _parse_regular_cli([])
        assert result == {}

    def test_provider_arg(self):
        result = _parse_regular_cli(["--provider", "openai"])
        assert result == {"provider": "openai"}

    def test_eval_model_name(self):
        result = _parse_regular_cli(["--eval-model-name", "gpt-4o"])
        assert result == {"eval_model_name": "gpt-4o"}

    def test_data_path(self):
        result = _parse_regular_cli(["--data-path", "/tmp/data.csv"])
        assert result == {"data_path": "/tmp/data.csv"}

    def test_multiple_args(self):
        result = _parse_regular_cli([
            "--provider", "watsonx",
            "--eval-model-name", "llama-3-70b",
            "--max-workers", "5",
            "--run-name", "test_run",
        ])
        assert result == {
            "provider": "watsonx",
            "eval_model_name": "llama-3-70b",
            "max_workers": 5,
            "run_name": "test_run",
        }

    def test_boolean_true(self):
        result = _parse_regular_cli(["--perform-generation", "true"])
        assert result == {"perform_generation": True}

    def test_boolean_false(self):
        result = _parse_regular_cli(["--is-reference-based", "false"])
        assert result == {"is_reference_based": False}

    def test_json_evaluation_criteria(self):
        criteria = '{"accuracy": "Response is correct"}'
        result = _parse_regular_cli(["--evaluation-criteria", criteria])
        assert result == {"evaluation_criteria": {"accuracy": "Response is correct"}}

    def test_json_eval_model_params(self):
        params = '{"temperature": 0.7, "max_tokens": 2000}'
        result = _parse_regular_cli(["--eval-model-params", params])
        assert result == {"eval_model_params": {"temperature": 0.7, "max_tokens": 2000}}

    def test_list_input_columns(self):
        result = _parse_regular_cli(["--input-columns", "col1", "col2", "col3"])
        assert result == {"input_columns": ["col1", "col2", "col3"]}

    def test_list_predefined_issues(self):
        result = _parse_regular_cli(["--predefined-issues", "issue1", "issue2"])
        assert result == {"predefined_issues": ["issue1", "issue2"]}

    def test_int_args(self):
        result = _parse_regular_cli(["--max-examples-to-analyze", "100"])
        assert result == {"max_examples_to_analyze": 100}

    def test_float_args(self):
        result = _parse_regular_cli(["--high-score-threshold", "0.85"])
        assert result == {"high_score_threshold": 0.85}

    def test_inference_backend(self):
        result = _parse_regular_cli(["--inference-backend", "litellm"])
        assert result == {"inference_backend": "litellm"}

    def test_endpoint_url(self):
        result = _parse_regular_cli(["--endpoint-url", "https://api.example.com/v1"])
        assert result == {"endpoint_url": "https://api.example.com/v1"}

    def test_task_choices(self):
        for task in ["general", "agent", "tool_call", "math", "rag", "external"]:
            result = _parse_regular_cli(["--task", task])
            assert result == {"task": task}

    def test_issues_format_choices(self):
        for fmt in ["shortcomings", "recommendations"]:
            result = _parse_regular_cli(["--issues-format", fmt])
            assert result == {"issues_format": fmt}

    def test_config_path_is_included(self):
        result = _parse_regular_cli(["--config-path", "/tmp/my_config.yaml"])
        assert result == {"config_path": "/tmp/my_config.yaml"}


# ============================================================================
# Test: Regular Flow - Config Merging
# ============================================================================

class TestRegularFlowConfigMerging:
    """Test load_config() merging precedence for regular flow."""

    def test_defaults_loaded(self):
        config = load_config(REGULAR_DEFAULT_CONFIG)
        # Verify known defaults from default_config.yaml
        assert config["perform_generation"] is True
        assert config["is_reference_based"] is False
        assert config["resume_enabled"] is True
        assert config["task"] == "general"
        assert config["high_score_threshold"] == 0.91

    def test_user_config_overrides_defaults(self):
        user_config = {"provider": "anthropic", "max_workers": 3}
        user_path = _create_temp_yaml(user_config)
        try:
            config = load_config(REGULAR_DEFAULT_CONFIG, user_path)
            assert config["provider"] == "anthropic"
            assert config["max_workers"] == 3
            # Other defaults remain
            assert config["task"] == "general"
        finally:
            os.unlink(user_path)

    def test_cli_overrides_user_config(self):
        user_config = {"provider": "anthropic", "max_workers": 3}
        user_path = _create_temp_yaml(user_config)
        try:
            config = load_config(REGULAR_DEFAULT_CONFIG, user_path, provider="openai", max_workers=50)
            assert config["provider"] == "openai"
            assert config["max_workers"] == 50
        finally:
            os.unlink(user_path)

    def test_cli_overrides_defaults_without_user_config(self):
        config = load_config(REGULAR_DEFAULT_CONFIG, None, provider="rits", eval_model_name="custom-model")
        assert config["provider"] == "rits"
        assert config["eval_model_name"] == "custom-model"

    def test_resolve_provider_config_openai(self):
        config = load_config(REGULAR_DEFAULT_CONFIG, None, provider="openai")
        # openai provider_defaults should fill in gen_model_name and eval_model_name
        assert config.get("gen_model_name") == "gpt-4o"
        assert config.get("eval_model_name") == "gpt-3.5-turbo"
        assert config.get("max_workers") == 20

    def test_resolve_provider_config_watsonx(self):
        config = load_config(REGULAR_DEFAULT_CONFIG, None, provider="watsonx")
        assert config.get("eval_model_name") == "meta-llama/llama-3-3-70b-instruct"
        assert config.get("gen_model_name") == "ibm/granite-3-3-8b-instruct"
        # Note: max_workers=20 is already in default_config.yaml,
        # so provider_defaults won't override it (only fills missing keys)
        assert config.get("max_workers") == 20

    def test_explicit_value_not_overridden_by_provider_defaults(self):
        # Provider defaults should NOT override explicitly set values
        config = load_config(
            REGULAR_DEFAULT_CONFIG, None,
            provider="openai", eval_model_name="my-custom-model"
        )
        assert config["eval_model_name"] == "my-custom-model"

    def test_json_config_file(self):
        user_config = {"provider": "anthropic", "task": "rag"}
        user_path = _create_temp_json(user_config)
        try:
            config = load_config(REGULAR_DEFAULT_CONFIG, user_path)
            assert config["provider"] == "anthropic"
            assert config["task"] == "rag"
        finally:
            os.unlink(user_path)

    def test_nested_dict_merging(self):
        """Test that nested dicts (like eval_model_params) merge correctly."""
        user_config = {"eval_model_params": {"temperature": 0.5}}
        user_path = _create_temp_yaml(user_config)
        try:
            config = load_config(REGULAR_DEFAULT_CONFIG, user_path)
            # User override replaces the nested dict
            assert config["eval_model_params"]["temperature"] == 0.5
            # Note: merge_configs does recursive merge, so max_tokens from default should remain
            assert config["eval_model_params"].get("max_tokens") == 8096
        finally:
            os.unlink(user_path)


# ============================================================================
# Test: Agentic Flow - CLI Parsing
# ============================================================================

class TestAgenticFlowCLIParsing:
    """Test agentic pipeline argument parsing."""

    def test_agentic_config_path(self):
        args = _parse_agentic_cli(["--agentic-config-path", "/tmp/config.yaml"])
        assert args.agentic_config_path == "/tmp/config.yaml"

    def test_data_dir(self):
        args = _parse_agentic_cli(["--data-dir", "/tmp/traces"])
        assert args.data_dir == "/tmp/traces"

    def test_results_dir(self):
        args = _parse_agentic_cli(["--results-dir", "/tmp/output"])
        assert args.results_dir == "/tmp/output"

    def test_from_raw_traces_boolean(self):
        args = _parse_agentic_cli(["--from-raw-traces", "true"])
        assert args.from_raw_traces is True

    def test_overwrite_boolean(self):
        args = _parse_agentic_cli(["--overwrite", "false"])
        assert args.overwrite is False

    def test_memory_only_boolean(self):
        args = _parse_agentic_cli(["--memory-only", "true"])
        assert args.memory_only is True


    def test_agent_framework_choices(self):
        for fw in ["langgraph", "crewai", "atif"]:
            args = _parse_agentic_cli(["--agent-framework", fw])
            assert args.agent_framework == fw

    def test_observability_framework_choices(self):
        for fw in ["mlflow", "langfuse"]:
            args = _parse_agentic_cli(["--observability-framework", fw])
            assert args.observability_framework == fw

    def test_unified_pipeline_run_step_by_step(self):
        args = _parse_agentic_cli(["--run-step-by-step", "true"])
        assert args.run_step_by_step is True

    def test_unified_pipeline_run_full_trajectory(self):
        args = _parse_agentic_cli(["--run-full-trajectory", "false"])
        assert args.run_full_trajectory is False

    def test_eval_types(self):
        args = _parse_agentic_cli(["--eval-types", "task_success", "rubric"])
        assert args.eval_types == ["task_success", "rubric"]

    def test_eval_types_all(self):
        args = _parse_agentic_cli(["--eval-types", "all"])
        assert args.eval_types == ["all"]

    def test_clear_analysis_types(self):
        args = _parse_agentic_cli(["--clear-analysis-types", "root_cause", "issues"])
        assert args.clear_analysis_types == ["root_cause", "issues"]

    def test_generate_rubrics(self):
        args = _parse_agentic_cli(["--generate-rubrics", "true"])
        assert args.generate_rubrics is True

    def test_rubric_dir(self):
        args = _parse_agentic_cli(["--rubric-dir", "/tmp/rubrics"])
        assert args.rubric_dir == "/tmp/rubrics"

    def test_context_tokens(self):
        args = _parse_agentic_cli(["--context-tokens", "128000"])
        assert args.context_tokens == 128000

    def test_max_files(self):
        args = _parse_agentic_cli(["--max-files", "5"])
        assert args.max_files == 5

    def test_clear_args_in_agentic_context(self):
        """CLEAR args like --provider and --eval-model-name work in agentic parser."""
        args = _parse_agentic_cli([
            "--provider", "openai",
            "--eval-model-name", "gpt-4o",
            "--max-workers", "15",
        ])
        assert args.provider == "openai"
        assert args.eval_model_name == "gpt-4o"
        assert args.max_workers == 15

    def test_combined_agentic_and_clear_args(self):
        """Full combination of agentic + CLEAR args."""
        args = _parse_agentic_cli([
            "--data-dir", "/data/traces",
            "--results-dir", "/output",
            "--from-raw-traces", "true",
            "--agent-framework", "langgraph",
            "--observability-framework", "mlflow",
            "--run-step-by-step", "true",
            "--run-full-trajectory", "false",
            "--provider", "watsonx",
            "--eval-model-name", "llama-3-70b",
            "--max-workers", "8",
            "--inference-backend", "litellm",
        ])
        assert args.data_dir == "/data/traces"
        assert args.results_dir == "/output"
        assert args.from_raw_traces is True
        assert args.agent_framework == "langgraph"
        assert args.observability_framework == "mlflow"
        assert args.run_step_by_step is True
        assert args.run_full_trajectory is False
        assert args.provider == "watsonx"
        assert args.eval_model_name == "llama-3-70b"
        assert args.max_workers == 8
        assert args.inference_backend == "litellm"


# ============================================================================
# Test: Agentic Flow - build_cli_overrides
# ============================================================================

class TestBuildCliOverrides:
    """Test build_cli_overrides() filters correctly."""

    def test_filters_none_values(self):
        args = _parse_agentic_cli(["--provider", "openai"])
        overrides = build_cli_overrides(args)
        assert "provider" in overrides
        # None-valued args should not be included
        assert "data_dir" not in overrides
        assert "results_dir" not in overrides

    def test_filters_agentic_config_path(self):
        args = _parse_agentic_cli(["--agentic-config-path", "/tmp/config.yaml"])
        overrides = build_cli_overrides(args)
        assert "agentic_config_path" not in overrides

    def test_includes_all_non_none(self):
        args = _parse_agentic_cli([
            "--data-dir", "/data",
            "--results-dir", "/out",
            "--provider", "openai",
            "--max-workers", "5",
        ])
        overrides = build_cli_overrides(args)
        assert overrides["data_dir"] == "/data"
        assert overrides["results_dir"] == "/out"
        assert overrides["provider"] == "openai"
        assert overrides["max_workers"] == 5


# ============================================================================
# Test: Agentic Flow - Config Merging
# ============================================================================

class TestAgenticFlowConfigMerging:
    """Test load_pipeline_config() merging for agentic flow."""

    def test_defaults_loaded(self):
        config = load_pipeline_config()
        # Verify known defaults from default_agentic_config.yaml
        assert config["from_raw_traces"] is False
        assert config["agent_framework"] == "langgraph"
        assert config["observability_framework"] == "langfuse"
        assert config["overwrite"] is False
        assert config["max_workers"] == 10
        assert config["eval_model_name"] == "gpt-4o"
        assert config["provider"] == "openai"
        assert config["run_step_by_step"] is True
        assert config["run_full_trajectory"] is True
        assert config["memory_only"] is False
        assert config["separate_tools"] is False

    def test_user_config_overrides_defaults(self):
        user_config = {
            "provider": "anthropic",
            "eval_model_name": "claude-3-opus",
            "max_workers": 20,
            "from_raw_traces": True,
        }
        user_path = _create_temp_yaml(user_config)
        try:
            config = load_pipeline_config(user_path)
            assert config["provider"] == "anthropic"
            assert config["eval_model_name"] == "claude-3-opus"
            assert config["max_workers"] == 20
            assert config["from_raw_traces"] is True
            # Other defaults remain
            assert config["overwrite"] is False
            assert config["run_step_by_step"] is True
        finally:
            os.unlink(user_path)

    def test_cli_overrides_user_config(self):
        user_config = {
            "provider": "anthropic",
            "eval_model_name": "claude-3-opus",
        }
        user_path = _create_temp_yaml(user_config)
        try:
            config = load_pipeline_config(
                user_path,
                provider="openai",
                eval_model_name="gpt-4-turbo"
            )
            assert config["provider"] == "openai"
            assert config["eval_model_name"] == "gpt-4-turbo"
        finally:
            os.unlink(user_path)

    def test_cli_overrides_without_user_config(self):
        config = load_pipeline_config(
            None,
            provider="watsonx",
            data_dir="/my/data",
            results_dir="/my/results",
        )
        assert config["provider"] == "watsonx"
        assert config["data_dir"] == "/my/data"
        assert config["results_dir"] == "/my/results"
        # defaults still present
        assert config["from_raw_traces"] is False

    def test_json_user_config(self):
        user_config = {"provider": "openai", "max_files": 10}
        user_path = _create_temp_json(user_config)
        try:
            config = load_pipeline_config(user_path)
            assert config["provider"] == "openai"
            assert config["max_files"] == 10
        finally:
            os.unlink(user_path)

    def test_eval_model_params_nested_merge(self):
        user_config = {"eval_model_params": {"temperature": 0.2}}
        user_path = _create_temp_yaml(user_config)
        try:
            config = load_pipeline_config(user_path)
            assert config["eval_model_params"]["temperature"] == 0.2
            # max_tokens from default should remain via recursive merge
            assert config["eval_model_params"]["max_tokens"] == 8096
        finally:
            os.unlink(user_path)

    def test_full_precedence_chain(self):
        """Default < user config < CLI overrides."""
        user_config = {
            "provider": "anthropic",  # overrides default "openai"
            "max_workers": 20,        # overrides default 10
            "data_dir": "/user/data",
        }
        user_path = _create_temp_yaml(user_config)
        try:
            config = load_pipeline_config(
                user_path,
                max_workers=99,           # CLI overrides user config's 20
                eval_model_name="custom", # CLI overrides default "gpt-4o"
            )
            # From user config (not overridden by CLI)
            assert config["provider"] == "anthropic"
            assert config["data_dir"] == "/user/data"
            # CLI override
            assert config["max_workers"] == 99
            assert config["eval_model_name"] == "custom"
            # From defaults (not overridden)
            assert config["from_raw_traces"] is False
        finally:
            os.unlink(user_path)


# ============================================================================
# Test: End-to-end CLI simulation for agentic flow
# ============================================================================

class TestAgenticEndToEnd:
    """Simulate full CLI -> config resolution for agentic pipeline."""

    def test_full_cli_to_config_no_user_config(self):
        """Simulate: run-clear-agentic-eval --data-dir ... --results-dir ... --provider ..."""
        import argparse
        parser = argparse.ArgumentParser()
        add_agentic_pipeline_args(parser)
        add_preprocessing_args(parser)
        add_unified_pipeline_args(parser)
        add_full_trajectory_args(parser)
        add_clear_args_to_parser(parser, group_name="CLEAR Configuration")

        cli_args = [
            "--data-dir", "/tmp/traces",
            "--results-dir", "/tmp/output",
            "--provider", "openai",
            "--eval-model-name", "gpt-4o",
            "--from-raw-traces", "true",
            "--run-step-by-step", "true",
            "--run-full-trajectory", "false",
            "--max-workers", "4",
        ]

        with patch.object(sys, 'argv', ['cmd'] + cli_args):
            args = parser.parse_args()

        cli_overrides = build_cli_overrides(args)
        config = load_pipeline_config(args.agentic_config_path, **cli_overrides)

        assert config["data_dir"] == "/tmp/traces"
        assert config["results_dir"] == "/tmp/output"
        assert config["provider"] == "openai"
        assert config["eval_model_name"] == "gpt-4o"
        assert config["from_raw_traces"] is True
        assert config["run_step_by_step"] is True
        assert config["run_full_trajectory"] is False
        assert config["max_workers"] == 4
        # Defaults preserved
        assert config["agent_framework"] == "langgraph"
        assert config["overwrite"] is False

    def test_full_cli_to_config_with_user_config(self):
        """Simulate: run-clear-agentic-eval --agentic-config-path config.yaml --eval-model-name override"""
        user_config = {
            "data_dir": "/config/data",
            "results_dir": "/config/output",
            "provider": "watsonx",
            "eval_model_name": "llama-70b",
            "from_raw_traces": True,
            "agent_framework": "crewai",
        }
        user_path = _create_temp_yaml(user_config)

        try:
            import argparse
            parser = argparse.ArgumentParser()
            add_agentic_pipeline_args(parser)
            add_preprocessing_args(parser)
            add_unified_pipeline_args(parser)
            add_full_trajectory_args(parser)
            add_clear_args_to_parser(parser, group_name="CLEAR Configuration")

            cli_args = [
                "--agentic-config-path", user_path,
                "--eval-model-name", "gpt-4o",  # CLI override
            ]

            with patch.object(sys, 'argv', ['cmd'] + cli_args):
                args = parser.parse_args()

            cli_overrides = build_cli_overrides(args)
            config = load_pipeline_config(args.agentic_config_path, **cli_overrides)

            # From user config
            assert config["data_dir"] == "/config/data"
            assert config["results_dir"] == "/config/output"
            assert config["provider"] == "watsonx"
            assert config["from_raw_traces"] is True
            assert config["agent_framework"] == "crewai"
            # CLI override wins over user config
            assert config["eval_model_name"] == "gpt-4o"
            # Default preserved (not in user config or CLI)
            assert config["overwrite"] is False
            assert config["run_step_by_step"] is True
        finally:
            os.unlink(user_path)


# ============================================================================
# Test: End-to-end CLI simulation for regular flow
# ============================================================================

class TestRegularEndToEnd:
    """Simulate full CLI -> config resolution for regular pipeline."""

    def test_full_cli_to_config_no_user_config(self):
        """Simulate: run-clear-eval-analysis --provider openai --eval-model-name gpt-4o ..."""
        overrides = _parse_regular_cli([
            "--provider", "openai",
            "--eval-model-name", "gpt-4o",
            "--data-path", "/tmp/data.csv",
            "--output-dir", "/tmp/output",
            "--perform-generation", "false",
            "--max-workers", "15",
        ])
        config = load_config(REGULAR_DEFAULT_CONFIG, overrides.pop("config_path", None), **overrides)

        assert config["provider"] == "openai"
        assert config["eval_model_name"] == "gpt-4o"
        assert config["data_path"] == "/tmp/data.csv"
        assert config["output_dir"] == "/tmp/output"
        assert config["perform_generation"] is False
        assert config["max_workers"] == 15
        # Defaults preserved
        assert config["is_reference_based"] is False
        assert config["task"] == "general"

    def test_full_cli_to_config_with_user_config(self):
        """Simulate: run-clear-eval-analysis --config-path user.yaml --provider override"""
        user_config = {
            "provider": "watsonx",
            "data_path": "/user/data.csv",
            "output_dir": "/user/output",
            "task": "rag",
        }
        user_path = _create_temp_yaml(user_config)

        try:
            overrides = _parse_regular_cli([
                "--config-path", user_path,
                "--provider", "openai",  # CLI override
            ])
            config_path = overrides.pop("config_path", None)
            config = load_config(REGULAR_DEFAULT_CONFIG, config_path, **overrides)

            # CLI override wins
            assert config["provider"] == "openai"
            # From user config
            assert config["data_path"] == "/user/data.csv"
            assert config["output_dir"] == "/user/output"
            assert config["task"] == "rag"
            # Defaults preserved
            assert config["is_reference_based"] is False
        finally:
            os.unlink(user_path)


# ============================================================================
# Test: Cross-flow Compatibility
# ============================================================================

class TestCrossFlowCompatibility:
    """Test shared args work in both flows and agentic-only args don't leak."""

    def test_shared_args_in_regular_flow(self):
        """Shared args like --provider, --eval-model-name, --max-workers work in regular flow."""
        result = _parse_regular_cli([
            "--provider", "openai",
            "--eval-model-name", "gpt-4o",
            "--max-workers", "10",
            "--inference-backend", "litellm",
        ])
        assert "provider" in result
        assert "eval_model_name" in result
        assert "max_workers" in result
        assert "inference_backend" in result

    def test_shared_args_in_agentic_flow(self):
        """Same shared args work in agentic flow."""
        args = _parse_agentic_cli([
            "--provider", "openai",
            "--eval-model-name", "gpt-4o",
            "--max-workers", "10",
            "--inference-backend", "litellm",
        ])
        assert args.provider == "openai"
        assert args.eval_model_name == "gpt-4o"
        assert args.max_workers == 10
        assert args.inference_backend == "litellm"

    def test_agentic_only_args_not_in_regular_parser(self):
        """Agentic-only args (--data-dir, --from-raw-traces, etc.) fail in regular parser."""
        with pytest.raises(SystemExit):
            _parse_regular_cli(["--data-dir", "/tmp"])

        with pytest.raises(SystemExit):
            _parse_regular_cli(["--from-raw-traces", "true"])

        with pytest.raises(SystemExit):
            _parse_regular_cli(["--run-step-by-step", "true"])

    def test_regular_only_args_not_in_agentic_parser(self):
        """Regular-only args like --data-path should still parse in agentic (since CLEAR args are added)."""
        # --data-path is a CLEAR arg added to agentic parser too
        args = _parse_agentic_cli(["--data-path", "/tmp/data.csv"])
        assert args.data_path == "/tmp/data.csv"


# ============================================================================
# Test: add_clear_args_to_parser with group_name
# ============================================================================

class TestAddClearArgsGrouping:
    """Test that CLEAR args can be added as a named group."""

    def test_without_group_name(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_clear_args_to_parser(parser)
        # All CLEAR args should be accessible
        names = get_clear_arg_names(parser)
        assert "provider" in names
        assert "eval_model_name" in names
        assert "max_workers" in names

    def test_with_group_name(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_clear_args_to_parser(parser, group_name="CLEAR Configuration")
        names = get_clear_arg_names(parser, group_name="CLEAR Configuration")
        assert "provider" in names
        assert "eval_model_name" in names
        assert "max_workers" in names

    def test_extract_clear_overrides_with_group(self):
        """extract_clear_overrides only returns non-None args from the CLEAR group."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--my-custom-arg", default="custom_value")
        add_clear_args_to_parser(parser, group_name="CLEAR Configuration")

        with patch.object(sys, 'argv', ['cmd', '--provider', 'openai', '--my-custom-arg', 'val']):
            overrides = extract_clear_overrides(parser, group_name="CLEAR Configuration")

        # Only CLEAR args should be included
        assert overrides == {"provider": "openai"}
        assert "my_custom_arg" not in overrides


# ============================================================================
# Test: Config file loading edge cases
# ============================================================================

class TestConfigFileLoading:
    """Test config file loading edge cases."""

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config_file("/nonexistent/path.yaml")

    def test_unsupported_format_raises(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("hello")
        f.close()
        try:
            with pytest.raises(ValueError):
                load_config_file(f.name)
        finally:
            os.unlink(f.name)

    def test_empty_yaml_returns_empty_dict(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        f.write("")
        f.close()
        try:
            result = load_config_file(f.name)
            assert result == {}
        finally:
            os.unlink(f.name)

    def test_none_path_returns_empty_dict(self):
        result = load_config_file(None)
        assert result == {}


# ============================================================================
# Test: merge_configs utility
# ============================================================================

class TestMergeConfigs:
    """Test recursive config merging."""

    def test_flat_override(self):
        defaults = {"a": 1, "b": 2}
        overrides = {"b": 3, "c": 4}
        result = merge_configs(defaults, overrides)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        defaults = {"nested": {"x": 1, "y": 2}, "top": "val"}
        overrides = {"nested": {"y": 3, "z": 4}}
        result = merge_configs(defaults, overrides)
        assert result == {"nested": {"x": 1, "y": 3, "z": 4}, "top": "val"}

    def test_override_replaces_non_dict_with_non_dict(self):
        defaults = {"key": "old"}
        overrides = {"key": "new"}
        result = merge_configs(defaults, overrides)
        assert result == {"key": "new"}

    def test_override_replaces_non_dict_with_dict(self):
        defaults = {"key": "string_val"}
        overrides = {"key": {"nested": True}}
        result = merge_configs(defaults, overrides)
        assert result == {"key": {"nested": True}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
