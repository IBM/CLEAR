"""Tests for the issues_format feature (shortcomings vs recommendations mode)."""

import pytest
from clear_eval.pipeline.constants import DEFAULT_ISSUES_FORMAT_MODE
from clear_eval.pipeline.propmts import (
    get_synthesis_prompt,
    get_synthesis_prompt_cont,
    get_shortcomings_synthesis_prompt,
    get_recommendations_synthesis_prompt,
    get_shortcomings_clustering_prompt,
    get_issues_mapping_human_prompt,
    get_issues_mapping_system_prompt,
)
from clear_eval.pipeline.full_pipeline import get_issues_format


class TestDefaultIssuesFormat:
    """Test that default format mode is 'shortcomings'."""

    def test_default_constant_is_shortcomings(self):
        assert DEFAULT_ISSUES_FORMAT_MODE == "shortcomings"

    def test_get_issues_format_returns_default_when_not_in_config(self):
        config = {}
        assert get_issues_format(config) == "shortcomings"

    def test_get_issues_format_returns_config_value(self):
        config = {"issues_format": "recommendations"}
        assert get_issues_format(config) == "recommendations"

        config = {"issues_format": "shortcomings"}
        assert get_issues_format(config) == "shortcomings"


class TestSynthesisPrompt:
    """Test synthesis prompt generation for both modes."""

    def test_synthesis_prompt_shortcomings_mode(self):
        prompt = get_synthesis_prompt("eval text", 10, format_mode="shortcomings")
        assert "shortcomings" in prompt.lower()
        assert "recommendations" not in prompt.lower()
        assert "eval text" in prompt

    def test_synthesis_prompt_recommendations_mode(self):
        prompt = get_synthesis_prompt("eval text", 10, format_mode="recommendations")
        assert "recommendations" in prompt.lower()
        assert "Ensure" in prompt or "actionable" in prompt.lower()
        assert "eval text" in prompt

    def test_synthesis_prompt_default_is_shortcomings(self):
        prompt_default = get_synthesis_prompt("eval text", 10)
        prompt_explicit = get_synthesis_prompt("eval text", 10, format_mode="shortcomings")
        assert prompt_default == prompt_explicit

    def test_shortcomings_synthesis_prompt_matches_original_behavior(self):
        """Ensure the shortcomings prompt contains expected elements."""
        prompt = get_shortcomings_synthesis_prompt("test evaluations", 15)
        assert "shortcomings" in prompt.lower()
        assert "15" in prompt  # max_shortcomings
        assert "test evaluations" in prompt
        assert "Python list" in prompt

    def test_recommendations_synthesis_prompt_has_actionable_framing(self):
        """Ensure recommendations prompt uses actionable language."""
        prompt = get_recommendations_synthesis_prompt("test evaluations", 10)
        assert "actionable" in prompt.lower()
        assert "recommendations" in prompt.lower()
        assert "Ensure" in prompt or "Add" in prompt or "Improve" in prompt


class TestSynthesisPromptCont:
    """Test continuation synthesis prompt generation."""

    def test_synthesis_prompt_cont_shortcomings_mode(self):
        prompt = get_synthesis_prompt_cont("new evals", "existing issues", 5, format_mode="shortcomings")
        assert "shortcomings" in prompt.lower()
        assert "new evals" in prompt
        assert "existing issues" in prompt

    def test_synthesis_prompt_cont_recommendations_mode(self):
        prompt = get_synthesis_prompt_cont("new evals", "existing recs", 5, format_mode="recommendations")
        assert "recommendations" in prompt.lower()
        assert "new evals" in prompt
        assert "existing recs" in prompt

    def test_synthesis_prompt_cont_default_is_shortcomings(self):
        prompt_default = get_synthesis_prompt_cont("evals", "existing", 5)
        prompt_explicit = get_synthesis_prompt_cont("evals", "existing", 5, format_mode="shortcomings")
        assert prompt_default == prompt_explicit


class TestClusteringPrompt:
    """Test clustering/deduplication prompt generation."""

    def test_clustering_prompt_shortcomings_mode(self):
        issues = ["issue 1", "issue 2", "issue 3"]
        prompt = get_shortcomings_clustering_prompt(issues, 10, format_mode="shortcomings")
        assert "issues" in prompt.lower()
        assert "3" in prompt  # len(issues)
        assert "10" in prompt  # max_issues

    def test_clustering_prompt_recommendations_mode(self):
        issues = ["rec 1", "rec 2"]
        prompt = get_shortcomings_clustering_prompt(issues, 5, format_mode="recommendations")
        assert "recommendations" in prompt.lower()
        assert "2" in prompt  # len(issues)

    def test_clustering_prompt_default_is_shortcomings(self):
        issues = ["a", "b"]
        prompt_default = get_shortcomings_clustering_prompt(issues, 5)
        prompt_explicit = get_shortcomings_clustering_prompt(issues, 5, format_mode="shortcomings")
        assert prompt_default == prompt_explicit


class TestMappingHumanPrompt:
    """Test human prompt for mapping issues to records."""

    def test_human_prompt_shortcomings_mode(self):
        prompt = get_issues_mapping_human_prompt("eval text", 5, format_mode="shortcomings")
        assert "shortcomings" in prompt
        assert "eval text" in prompt
        assert "5" in prompt

    def test_human_prompt_recommendations_mode(self):
        prompt = get_issues_mapping_human_prompt("eval text", 5, format_mode="recommendations")
        assert "recommendations" in prompt
        assert "eval text" in prompt
        assert "5" in prompt

    def test_human_prompt_default_is_shortcomings(self):
        prompt_default = get_issues_mapping_human_prompt("text", 3)
        prompt_explicit = get_issues_mapping_human_prompt("text", 3, format_mode="shortcomings")
        assert prompt_default == prompt_explicit


class TestMappingSystemPrompt:
    """Test system prompt for mapping issues to records."""

    def test_system_prompt_shortcomings_mode(self):
        shortcomings = ["Issue A", "Issue B", "Issue C"]
        prompt = get_issues_mapping_system_prompt(shortcomings, format_mode="shortcomings")

        assert "shortcomings" in prompt.lower()
        assert "Shortcoming List:" in prompt
        assert "1. Issue A" in prompt
        assert "2. Issue B" in prompt
        assert "3. Issue C" in prompt
        assert "negative sentiment" in prompt

    def test_system_prompt_recommendations_mode(self):
        recs = ["Ensure X", "Add Y"]
        prompt = get_issues_mapping_system_prompt(recs, format_mode="recommendations")

        assert "recommendations" in prompt.lower()
        assert "Recommendation List:" in prompt
        assert "1. Ensure X" in prompt
        assert "2. Add Y" in prompt
        assert "failed to follow" in prompt.lower()

    def test_system_prompt_default_is_shortcomings(self):
        items = ["Item 1"]
        prompt_default = get_issues_mapping_system_prompt(items)
        prompt_explicit = get_issues_mapping_system_prompt(items, format_mode="shortcomings")
        assert prompt_default == prompt_explicit

    def test_system_prompt_contains_binary_instructions(self):
        """Both modes should instruct for binary output."""
        items = ["test"]

        for mode in ["shortcomings", "recommendations"]:
            prompt = get_issues_mapping_system_prompt(items, format_mode=mode)
            assert "binary" in prompt.lower()
            assert "0" in prompt and "1" in prompt
            assert "[0, 1, 0, ...]" in prompt or "[1, 0, 1, 0]" in prompt


class TestShortcomingsModeMatchesOriginalBehavior:
    """Verify that shortcomings mode produces prompts matching original implementation."""

    def test_synthesis_prompt_structure(self):
        """Check key structural elements of the synthesis prompt."""
        prompt = get_shortcomings_synthesis_prompt("evaluations here", 10)

        # Key phrases from original
        assert "expert analyst" in prompt.lower()
        assert "common themes" in prompt.lower()
        assert "Python list" in prompt
        assert "Begin Evaluation Texts" in prompt
        assert "End Evaluation Texts" in prompt

    def test_system_prompt_structure(self):
        """Check key structural elements of the system prompt."""
        prompt = get_issues_mapping_system_prompt(["issue1", "issue2"], format_mode="shortcomings")

        # Key phrases from original
        assert "expert analyst" in prompt.lower()
        assert "negative sentiment" in prompt
        assert "Valid Example" in prompt
        assert "Invalid Examples" in prompt


class TestRecommendationsModeDistinct:
    """Verify recommendations mode produces distinct, actionable prompts."""

    def test_recommendations_use_imperative_language(self):
        prompt = get_recommendations_synthesis_prompt("evals", 5)

        # Should contain action-oriented words
        action_words = ["ensure", "add", "improve", "verify", "include"]
        prompt_lower = prompt.lower()
        assert any(word in prompt_lower for word in action_words)

    def test_recommendations_mapping_logic_inverted(self):
        """Recommendations should mark 1 when model FAILED to follow."""
        prompt = get_issues_mapping_system_prompt(["rec"], format_mode="recommendations")

        assert "failed to follow" in prompt.lower()
        # Should NOT have the negative sentiment language from shortcomings
        assert "negative sentiment" not in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
