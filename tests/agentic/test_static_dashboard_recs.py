"""The dashboard now reads SPARC recommendations from inside the
per-agent ``tools_eval`` payload (no sidecar). These tests verify the
loader and renderer follow that contract."""

from __future__ import annotations

import json
from pathlib import Path

from clear_eval.agentic.dashboard.generate_static_dashboard import (
    generate_html,
    load_json_data,
)


def _write_fake_run(tmp_path: Path, with_recs: bool) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    tools_eval = {
        "agent_summary": {"total_interactions": 3, "avg_score": 0.66},
        "issues": [
            {
                "issue_id": 1,
                "issue_text": "A problem",
                "occurrence_count": 1,
                "frequency": 0.33,
                "severity": 0.5,
                "occurrences": [],
            }
        ],
        "no_issues": [{"trace_id": "t1"}],
    }

    if with_recs:
        tools_eval["recommendations"] = {
            "by_tool": {
                "search_items": [
                    {
                        "target": "parameter_description",
                        "tool_name": "search_items",
                        "parameter_name": "query",
                        "diff": (
                            "--- a/tool/search_items#query\n"
                            "+++ b/tool/search_items#query\n"
                            "@@\n+Required. Must be grounded in user input."
                        ),
                        "rationale": "Prevent fabricated queries.",
                        "importance_mean": 0.9,
                        "importance_max": 0.9,
                        "count": 2,
                    }
                ]
            },
            "by_system_prompt": [
                {
                    "target": "system_prompt",
                    "diff": (
                        "--- a/system_prompt\n+++ b/system_prompt\n@@\n"
                        "+Call search before book."
                    ),
                    "rationale": "Reinforce search-first.",
                    "importance_mean": 0.8,
                    "importance_max": 0.8,
                    "count": 3,
                }
            ],
            "total": 5,
        }

    clear_results = {
        "metadata": {
            "statistics": {"total_traces": 1, "total_interactions_analyzed": 3},
        },
        "agents": {"tool_calling": {"tools_eval": tools_eval}},
    }
    (run_dir / "clear_results.json").write_text(json.dumps(clear_results))
    return run_dir


class TestLoaderReadsInPayloadRecs:
    def test_loader_attaches_recs_when_present(self, tmp_path):
        run_dir = _write_fake_run(tmp_path, with_recs=True)
        data = load_json_data(run_dir / "clear_results.json")
        entry = data["agents"]["tool_calling:tools"]
        assert entry["recommendations"] is not None
        assert entry["recommendations"]["total"] == 5
        assert "search_items" in entry["recommendations"]["by_tool"]

    def test_loader_leaves_recs_none_when_absent(self, tmp_path):
        run_dir = _write_fake_run(tmp_path, with_recs=False)
        data = load_json_data(run_dir / "clear_results.json")
        entry = data["agents"]["tool_calling:tools"]
        assert entry["recommendations"] is None


class TestHtmlRenders:
    def test_html_contains_recs_section_when_present(self, tmp_path):
        run_dir = _write_fake_run(tmp_path, with_recs=True)
        out = run_dir / "clear_results.html"
        generate_html(run_dir / "clear_results.json", out)
        html = out.read_text()
        assert "SPARC Recommendations" in html
        assert "renderRecommendations" in html
        assert "search_items" in html
        assert "Call search before book" in html
        assert "tool/search_items#query" in html

    def test_html_omits_recs_payload_when_absent(self, tmp_path):
        run_dir = _write_fake_run(tmp_path, with_recs=False)
        out = run_dir / "clear_results.html"
        generate_html(run_dir / "clear_results.json", out)
        html = out.read_text()
        assert '"recommendations": null' in html or '"recommendations":null' in html
