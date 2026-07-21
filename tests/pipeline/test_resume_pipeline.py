"""
Test script for verifying checkpoint-based resume logic in run_eval_pipeline / run_aggregation_from_df.

Tests all combinations of resume_enabled and output dir state:
1. Fresh run (no checkpoint, no zip)
2. Resume with checkpoint at generation stage (has model_output_column)
3. Resume with checkpoint at evaluation stage (has score)
4. Resume with checkpoint at summaries stage (has evaluation_summary)
5. Resume with checkpoint at mapping stage (has identified_shortcomings)
6. Resume with zip already present (final output exists)
7. resume_enabled=False with existing checkpoint (should ignore it)

Uses the gsm8k sample data as input (already has predictions, so perform_generation=False).
Mocks the LLM to avoid real API calls.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from clear_eval.pipeline.full_pipeline import (
    run_eval_pipeline, run_aggregation_from_df, get_run_info, CHECKPOINT_FILE_PREFIX
)
from clear_eval.pipeline.constants import (
    SCORE_COL, EVALUATION_SUMMARY_COL, IDENTIFIED_SHORTCOMING_COL, SHORTCOMING_PREFIX
)

SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "clear_eval", "sample_data", "gsm8k", "gsm8k_default_predictions.csv"
)

BASE_CONFIG = {
    "provider": "rits",
    "inference_backend": "litellm",
    "eval_model_name": "test-model",
    "gen_model_name": "test-gen-model",
    "eval_model_params": {"max_tokens": 100},
    "data_path": SAMPLE_DATA_PATH,
    "perform_generation": False,
    "is_reference_based": False,
    "resume_enabled": True,
    "task": "math",
    "question_column": "question",
    "model_output_column": "response",
    "reference_column": "ground_truth",
    "model_input_column": "model_input",
    "qid_column": "id",
    "use_full_text_for_analysis": False,
    "max_workers": 2,
    "max_shortcomings": 5,
    "min_shortcomings": 2,
    "max_eval_text_for_synthesis": 100,
    "perform_clustering": False,
    "high_score_threshold": 0.91,
    "generate_issues": True,
    "run_name": "test_resume",
    "issues_format": "shortcomings",
    "max_examples_to_analyze": None,
    "use_general_prompt": True,
    "evaluation_criteria": None,
    "predefined_issues": None,
    "documents_column": "documents",
    "synthesis_template": None,
}


def make_config(output_dir, resume_enabled=True):
    config = BASE_CONFIG.copy()
    config["output_dir"] = output_dir
    config["resume_enabled"] = resume_enabled
    return config


def load_sample_df():
    return pd.read_csv(SAMPLE_DATA_PATH).head(5)


def make_checkpoint_with_generation(df):
    """Simulate checkpoint after generation (already has response column)."""
    return df.copy()


def make_checkpoint_with_eval(df):
    """Simulate checkpoint after evaluation stage."""
    df = df.copy()
    df[SCORE_COL] = 0.5
    df["evaluation_text"] = "Mock evaluation text"
    df["error"] = "[]"
    return df


def make_checkpoint_with_summaries(df):
    """Simulate checkpoint after summaries stage."""
    df = make_checkpoint_with_eval(df)
    df[EVALUATION_SUMMARY_COL] = "Mock summary"
    return df


def make_checkpoint_with_mapping(df):
    """Simulate checkpoint after mapping stage."""
    df = make_checkpoint_with_summaries(df)
    df[IDENTIFIED_SHORTCOMING_COL] = "[]"
    df[f"{SHORTCOMING_PREFIX}1"] = 0
    df[f"{SHORTCOMING_PREFIX}2"] = 0
    return df


class MockLLM:
    async def ainvoke(self, prompt):
        return "Mock LLM response\nEvaluation score: 0.5"


def mock_produce_summaries(df, llm, config):
    df = df.copy()
    df[EVALUATION_SUMMARY_COL] = "Mock summary"
    return df


def mock_synthesize_shortcomings(df, llm, config, synthesis_template=None, format_mode=None):
    return ["Issue 1: Mock shortcoming", "Issue 2: Another mock shortcoming"]


def mock_map_shortcomings(df, llm, shortcomings_list, use_full_text, qid_col, max_workers,
                          high_score_threshold, score_col=SCORE_COL, format_mode=None,
                          checkpoint_every=0, cache_path=None):
    df = df.copy()
    df[IDENTIFIED_SHORTCOMING_COL] = "[]"
    for i in range(len(shortcomings_list)):
        df[f"{SHORTCOMING_PREFIX}{i+1}"] = 0
    return df


def get_patches():
    """Return context manager that patches all LLM-dependent functions."""
    return [
        patch("clear_eval.pipeline.full_pipeline.get_eval_llm_from_config", return_value=MockLLM()),
        patch("clear_eval.pipeline.full_pipeline.get_llm_from_config", return_value=MockLLM()),
        patch("clear_eval.pipeline.full_pipeline.produce_summaries_per_record", side_effect=mock_produce_summaries),
        patch("clear_eval.pipeline.full_pipeline.synthesize_shortcomings_from_df", side_effect=mock_synthesize_shortcomings),
        patch("clear_eval.pipeline.full_pipeline.map_shortcomings_to_records", side_effect=mock_map_shortcomings),
    ]


class TestResumeFromCheckpoint(unittest.TestCase):
    """Test that the pipeline resumes correctly from each checkpoint stage."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="clear_test_resume_")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _get_checkpoint_path(self, config):
        run_info = get_run_info(config)
        return os.path.join(config["output_dir"], f"{CHECKPOINT_FILE_PREFIX}_{run_info}.csv")

    def _get_zip_path(self, config):
        run_info = get_run_info(config)
        return os.path.join(config["output_dir"], f"analysis_results_{run_info}.zip")

    def _apply_patches(self):
        """Start all patches and return list of mocks for cleanup in tearDown."""
        patches = get_patches()
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        return mocks

    def test_fresh_run_no_checkpoint(self):
        """Full run from scratch, no existing checkpoint."""
        self._apply_patches()
        config = make_config(self.output_dir)

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task_obj.eval_records = lambda df, llm, cfg: make_checkpoint_with_eval(df)
            mock_task.return_value = mock_task_obj

            run_eval_pipeline(config)

        self.assertTrue(os.path.exists(self._get_checkpoint_path(config)))
        self.assertTrue(os.path.exists(self._get_zip_path(config)))

    def test_resume_from_generation(self):
        """Checkpoint has generation output, should skip to evaluation."""
        self._apply_patches()
        config = make_config(self.output_dir)
        df = load_sample_df()
        checkpoint_df = make_checkpoint_with_generation(df)
        checkpoint_path = self._get_checkpoint_path(config)
        checkpoint_df.to_csv(checkpoint_path, index=False)

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task_obj.eval_records = lambda df, llm, cfg: make_checkpoint_with_eval(df)
            mock_task.return_value = mock_task_obj
            with patch("clear_eval.pipeline.full_pipeline.generate_model_predictions") as mock_gen:
                run_eval_pipeline(config)
                mock_gen.assert_not_called()

        self.assertTrue(os.path.exists(self._get_zip_path(config)))

    def test_resume_from_eval(self):
        """Checkpoint has eval output, should skip to summaries."""
        self._apply_patches()
        config = make_config(self.output_dir)
        df = load_sample_df()
        checkpoint_df = make_checkpoint_with_eval(df)
        checkpoint_path = self._get_checkpoint_path(config)
        checkpoint_df.to_csv(checkpoint_path, index=False)

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task_obj.eval_records = MagicMock(
                side_effect=AssertionError("eval_records should not be called"))
            mock_task.return_value = mock_task_obj

            run_eval_pipeline(config)

        self.assertTrue(os.path.exists(self._get_zip_path(config)))

    def test_resume_from_summaries(self):
        """Checkpoint has summaries, should skip to aggregation."""
        self._apply_patches()
        config = make_config(self.output_dir)
        df = load_sample_df()
        checkpoint_df = make_checkpoint_with_summaries(df)
        checkpoint_path = self._get_checkpoint_path(config)
        checkpoint_df.to_csv(checkpoint_path, index=False)

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task_obj.eval_records = MagicMock(
                side_effect=AssertionError("eval_records should not be called"))
            mock_task.return_value = mock_task_obj
            with patch("clear_eval.pipeline.full_pipeline.produce_summaries_per_record",
                       side_effect=AssertionError("produce_summaries should not be called")):
                run_eval_pipeline(config)

        self.assertTrue(os.path.exists(self._get_zip_path(config)))

    def test_resume_from_mapping(self):
        """Checkpoint has mapping columns, should just produce UI output. No LLM created."""
        config = make_config(self.output_dir)
        df = load_sample_df()
        checkpoint_df = make_checkpoint_with_mapping(df)
        checkpoint_path = self._get_checkpoint_path(config)
        checkpoint_df.to_csv(checkpoint_path, index=False)

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task.return_value = mock_task_obj
            with patch("clear_eval.pipeline.full_pipeline.get_eval_llm_from_config",
                       side_effect=AssertionError("LLM should not be created when mapping exists")):
                run_eval_pipeline(config)

        self.assertTrue(os.path.exists(self._get_zip_path(config)))

    def test_resume_with_zip_exists(self):
        """Zip already exists, should return immediately. No LLM, no checkpoint load."""
        config = make_config(self.output_dir)
        zip_path = self._get_zip_path(config)
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        with open(zip_path, "w") as f:
            f.write("dummy")

        with patch("clear_eval.pipeline.full_pipeline.get_eval_llm_from_config",
                   side_effect=AssertionError("LLM should not be created when zip exists")):
            with patch("clear_eval.pipeline.full_pipeline.load_dataframe_from_cache",
                       side_effect=AssertionError("Checkpoint should not be loaded when zip exists")):
                run_eval_pipeline(config)

    def test_resume_disabled_ignores_checkpoint(self):
        """resume_enabled=False should ignore existing checkpoint and run from scratch."""
        self._apply_patches()
        config = make_config(self.output_dir, resume_enabled=False)
        df = load_sample_df()
        checkpoint_df = make_checkpoint_with_mapping(df)
        checkpoint_path = self._get_checkpoint_path(config)
        checkpoint_df.to_csv(checkpoint_path, index=False)

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task_obj.eval_records = lambda df, llm, cfg: make_checkpoint_with_eval(df)
            mock_task.return_value = mock_task_obj

            run_eval_pipeline(config)

        self.assertTrue(os.path.exists(self._get_zip_path(config)))


class TestAggregationResume(unittest.TestCase):
    """Test run_aggregation_from_df resume logic."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="clear_test_agg_")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _get_zip_path(self, file_name_info):
        return os.path.join(self.output_dir, f"analysis_results_{file_name_info}.zip")

    def _apply_patches(self):
        patches = get_patches()
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_aggregation_with_mapping_already_done(self):
        """If df already has mapping columns, just produce UI output. No LLM created."""
        config = make_config(self.output_dir)
        df = make_checkpoint_with_mapping(load_sample_df())

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task.return_value = mock_task_obj
            with patch("clear_eval.pipeline.full_pipeline.get_eval_llm_from_config",
                       side_effect=AssertionError("LLM should not be created")):
                run_aggregation_from_df(config, df, "test_info")

        self.assertTrue(os.path.exists(self._get_zip_path("test_info")))

    def test_aggregation_with_zip_exists(self):
        """If zip exists and resume_enabled, return immediately."""
        config = make_config(self.output_dir)
        zip_path = self._get_zip_path("test_info")
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        with open(zip_path, "w") as f:
            f.write("dummy")

        df = make_checkpoint_with_summaries(load_sample_df())

        with patch("clear_eval.pipeline.full_pipeline.get_eval_llm_from_config",
                   side_effect=AssertionError("LLM should not be created")):
            run_aggregation_from_df(config, df, "test_info")

    def test_aggregation_with_cached_issues(self):
        """If issues JSON is cached, should skip synthesis and go to mapping."""
        self._apply_patches()
        config = make_config(self.output_dir)
        df = make_checkpoint_with_summaries(load_sample_df())
        run_info = "test_info"

        issues_path = os.path.join(self.output_dir, f"shortcoming_list_{run_info}.json")
        with open(issues_path, "w") as f:
            json.dump(["Cached issue 1", "Cached issue 2"], f)

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task.return_value = mock_task_obj
            with patch("clear_eval.pipeline.full_pipeline.synthesize_shortcomings_from_df",
                       side_effect=AssertionError("Synthesis should not be called when issues are cached")):
                run_aggregation_from_df(config, df, run_info)

        self.assertTrue(os.path.exists(self._get_zip_path(run_info)))

    def test_aggregation_fresh(self):
        """No cache, should run full aggregation."""
        self._apply_patches()
        config = make_config(self.output_dir)
        df = make_checkpoint_with_summaries(load_sample_df())

        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task_obj = MagicMock()
            mock_task.return_value = mock_task_obj

            run_aggregation_from_df(config, df, "test_fresh")

        self.assertTrue(os.path.exists(self._get_zip_path("test_fresh")))

    def test_checkpoint_every_forwarded_to_mapping(self):
        """When checkpoint_every is set, full_pipeline must forward checkpoint_every
        and a derived cache_path to map_shortcomings_to_records."""
        config = make_config(self.output_dir)
        config["checkpoint_every"] = 3
        df = make_checkpoint_with_summaries(load_sample_df())

        map_mock = MagicMock(side_effect=mock_map_shortcomings)
        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task.return_value = MagicMock()
            with patch("clear_eval.pipeline.full_pipeline.get_eval_llm_from_config", return_value=MockLLM()), \
                 patch("clear_eval.pipeline.full_pipeline.synthesize_shortcomings_from_df",
                       side_effect=mock_synthesize_shortcomings), \
                 patch("clear_eval.pipeline.full_pipeline.map_shortcomings_to_records", map_mock):
                run_aggregation_from_df(config, df, "test_ckpt")

        map_mock.assert_called_once()
        _, kwargs = map_mock.call_args
        self.assertEqual(kwargs["checkpoint_every"], 3)
        self.assertIsNotNone(kwargs["cache_path"])
        self.assertTrue(str(kwargs["cache_path"]).endswith("cache_map.jsonl"))

    def test_checkpoint_every_default_disables_caching(self):
        """Without checkpoint_every, mapping is called with checkpoint_every=0 and cache_path=None."""
        config = make_config(self.output_dir)
        df = make_checkpoint_with_summaries(load_sample_df())

        map_mock = MagicMock(side_effect=mock_map_shortcomings)
        with patch("clear_eval.pipeline.use_cases.use_case_utils.get_task_data_obj") as mock_task:
            mock_task.return_value = MagicMock()
            with patch("clear_eval.pipeline.full_pipeline.get_eval_llm_from_config", return_value=MockLLM()), \
                 patch("clear_eval.pipeline.full_pipeline.synthesize_shortcomings_from_df",
                       side_effect=mock_synthesize_shortcomings), \
                 patch("clear_eval.pipeline.full_pipeline.map_shortcomings_to_records", map_mock):
                run_aggregation_from_df(config, df, "test_no_ckpt")

        map_mock.assert_called_once()
        _, kwargs = map_mock.call_args
        self.assertEqual(kwargs["checkpoint_every"], 0)
        self.assertIsNone(kwargs["cache_path"])


if __name__ == "__main__":
    unittest.main()
