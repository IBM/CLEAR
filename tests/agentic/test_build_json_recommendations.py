"""Tests for SPARC recommendation aggregation inside
``build_json_results._process_results_dir``."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from clear_eval.agentic.pipeline import build_json_results as bjr


def _make_rec(target, *, tool=None, param=None, importance=0.5, diff=None, rationale="r"):
    return {
        "target": target,
        "tool_name": tool,
        "parameter_name": param,
        "importance": importance,
        "rationale": rationale,
        "diff": diff or f"--- a/x\n+++ b/x\n@@\n+{target} for {tool or 'system'}",
    }


def _write_results_dir(tmp_path: Path, rows):
    """Write a minimal analysis_results CSV that ``_process_results_dir`` accepts."""
    d = tmp_path / "tool_calls"
    d.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(d / "analysis_results_run.csv", index=False)
    return d


def _run(d):
    metrics = defaultdict(lambda: {"scores": [], "has_issues": []})
    return bjr._process_results_dir(
        d,
        config_dict={},
        trace_metrics=metrics,
        all_traces=set(),
        all_results_dfs=None,
    )


def _base_row(task_id="t1", step=1, recs=None):
    return {
        "task_id": task_id,
        "id": f"{task_id}_{step}",
        "step_in_trace_general": step,
        "score": 0.5,
        "evaluation_text": "",
        "evaluation_summary": "",
        "model_input": "[]",
        "response": "{}",
        "intent": "",
        "tool_or_agent": "agent",
        "meta_data": "{}",
        "recurring_issues_str": "[]",
        "sparc_recommendations": json.dumps(recs or []),
    }


class TestAggregator:
    def test_no_column_returns_none(self, tmp_path):
        d = _write_results_dir(tmp_path, [
            {k: v for k, v in _base_row().items() if k != "sparc_recommendations"}
        ])
        result = _run(d)
        assert result is not None
        assert "recommendations" not in result

    def test_empty_recs_returns_no_payload(self, tmp_path):
        d = _write_results_dir(tmp_path, [_base_row(recs=[])])
        result = _run(d)
        assert "recommendations" not in result

    def test_groups_by_tool_and_system(self, tmp_path):
        diff_a = "--- a/tool/search\n+++ b/tool/search\n@@\n+Doc fix"
        diff_b = "--- a/system_prompt\n+++ b/system_prompt\n@@\n+System fix"
        rows = [
            _base_row(step=1, recs=[
                _make_rec("tool_description", tool="search", diff=diff_a, importance=0.6),
                _make_rec("system_prompt", diff=diff_b, importance=0.7),
            ]),
            _base_row(step=2, recs=[
                _make_rec("tool_description", tool="search", diff=diff_a, importance=0.8),
            ]),
        ]
        d = _write_results_dir(tmp_path, rows)
        result = _run(d)
        recs = result["recommendations"]
        assert recs["total"] == 3
        assert "search" in recs["by_tool"]
        tool_entries = recs["by_tool"]["search"]
        assert len(tool_entries) == 1, "same diff should dedupe to one entry"
        e = tool_entries[0]
        assert e["count"] == 2
        assert e["importance_mean"] == pytest.approx(0.7, abs=1e-6)
        assert e["importance_max"] == pytest.approx(0.8, abs=1e-6)
        sys_entries = recs["by_system_prompt"]
        assert len(sys_entries) == 1
        assert sys_entries[0]["count"] == 1

    def test_param_target_grouped_under_tool(self, tmp_path):
        rows = [
            _base_row(recs=[_make_rec("parameter_description", tool="t1", param="q", importance=0.4)]),
            _base_row(step=2, recs=[_make_rec("parameter_examples", tool="t1", param="q", importance=0.9)]),
        ]
        d = _write_results_dir(tmp_path, rows)
        result = _run(d)
        recs = result["recommendations"]
        assert "t1" in recs["by_tool"]
        assert len(recs["by_tool"]["t1"]) == 2
        assert {e["target"] for e in recs["by_tool"]["t1"]} == {
            "parameter_description", "parameter_examples"
        }

    def test_importance_clamped(self, tmp_path):
        rows = [_base_row(recs=[
            _make_rec("system_prompt", importance=2.0),
            _make_rec("system_prompt", importance=-0.5, diff="--- a/x\n+++ b/x\n@@\n+other"),
        ])]
        d = _write_results_dir(tmp_path, rows)
        result = _run(d)
        recs = result["recommendations"]
        for e in recs["by_system_prompt"]:
            assert 0.0 <= e["importance_mean"] <= 1.0
            assert 0.0 <= e["importance_max"] <= 1.0

    def test_sort_order_by_importance_then_count(self, tmp_path):
        # Two distinct system-prompt diffs: one with high importance, one with high count
        diff_high = "--- a/system_prompt\n+++ b/system_prompt\n@@\n+A"
        diff_freq = "--- a/system_prompt\n+++ b/system_prompt\n@@\n+B"
        rows = [
            _base_row(recs=[_make_rec("system_prompt", diff=diff_high, importance=0.95)]),
            _base_row(step=2, recs=[_make_rec("system_prompt", diff=diff_freq, importance=0.4)]),
            _base_row(step=3, recs=[_make_rec("system_prompt", diff=diff_freq, importance=0.4)]),
            _base_row(step=4, recs=[_make_rec("system_prompt", diff=diff_freq, importance=0.4)]),
        ]
        d = _write_results_dir(tmp_path, rows)
        result = _run(d)
        sys_entries = result["recommendations"]["by_system_prompt"]
        assert sys_entries[0]["importance_mean"] >= sys_entries[1]["importance_mean"]

    def test_unknown_target_skipped(self, tmp_path):
        rows = [_base_row(recs=[
            {"target": "bogus", "diff": "--- a\n+++ b\n@@\n+x", "importance": 0.5}
        ])]
        d = _write_results_dir(tmp_path, rows)
        result = _run(d)
        assert "recommendations" not in result

    def test_invalid_json_cell_handled(self, tmp_path):
        row = _base_row()
        row["sparc_recommendations"] = "not-json"
        d = _write_results_dir(tmp_path, [row])
        result = _run(d)
        assert "recommendations" not in result
