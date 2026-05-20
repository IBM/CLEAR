#!/usr/bin/env python3
"""Convert exgentic OpenTelemetry trace JSON into CLEAR tool-calling SPARC CSV.

Emits the **unified row format**: one row per LLM call.  When the LLM
produced tool calls, ``response`` is a JSON object
``{"content": "...", "tool_calls": [...]}``; otherwise it is the raw
assistant text.  ``tool_or_agent`` is always ``"agent"``.

This matches ``build_csv_rows`` in
  src/clear_eval/agentic/pipeline/preprocess_traces/trace_utils.py
so downstream SPARC reformatting (reasoning injection / per-tool-call
splitting) happens inside ``run_clear_step_analysis.convert_to_clear_format``
when the pipeline's ``separate_tools`` flag is enabled.

Usage:
    python traces_to_clear_csv.py <trace.json> [...] --output-dir <dir> \\
        [--specs-dir <dir>]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CSV_FIELDNAMES = [
    "id", "Name", "intent", "task_id", "step_in_trace_general",
    "llm_call_index", "model_input", "response", "tool_or_agent",
    "api_spec", "meta_data", "traj_score",
]

INTENT_LIMIT = 500  # matches _INTENT_LIMIT in trace_utils.py
MESSAGE_TOOL = "message"  # pseudo-tool: agent text reply
INITIAL_OBS = "initial_observation"  # pseudo-tool: first user message


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def find_session_span(spans: list[dict]) -> dict:
    for span in spans:
        if span.get("parent_id") in (None, "null"):
            return span
    # fallback: largest span by children
    return max(spans, key=lambda s: len(s.get("attributes", {})))


def child_spans_sorted(spans: list[dict], session_span_id: str) -> list[dict]:
    children = [s for s in spans if s.get("parent_id") == session_span_id]
    children.sort(key=lambda s: s.get("start_time", ""))
    return children


def get_attr(span: dict, key: str, default: Any = None) -> Any:
    return span.get("attributes", {}).get(key, default)


def collect_session_tool_names(session_attrs: dict) -> dict[str, dict]:
    """Return {tool_name: {description, is_message, is_finish}} from attrs."""
    pattern = re.compile(r"^exgentic\.session\.action\.([^.]+)\.name$")
    tools: dict[str, dict] = defaultdict(dict)
    for key, val in session_attrs.items():
        m = pattern.match(key)
        if not m:
            continue
        slot = m.group(1)
        name = val
        prefix = f"exgentic.session.action.{slot}."
        tools[name] = {
            "description": session_attrs.get(prefix + "description", ""),
            "is_message": session_attrs.get(prefix + "is_message", False),
            "is_finish": session_attrs.get(prefix + "is_finish", False),
        }
    return dict(tools)


def load_tool_specs(
    specs_dir: Path, slug: str, subset: str, session_tools: dict[str, dict]
) -> list[dict]:
    """Baseline from specs_dir/<slug>_<subset>.json or <slug>.json; merge in
    any session-declared tools missing from baseline (minimal entry)."""
    baseline: list[dict] = []
    for candidate in (f"{slug}_{subset}.json", f"{slug}.json"):
        path = specs_dir / candidate
        if path.is_file():
            with path.open() as f:
                baseline = json.load(f)
            break

    have = {t["function"]["name"] for t in baseline}
    for name, info in session_tools.items():
        if name in have:
            continue
        if name == MESSAGE_TOOL or info.get("is_message") or info.get("is_finish"):
            continue  # skip pseudo-tools
        baseline.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info.get("description", ""),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        })
    # filter out any baseline entry named "message" in case a spec file
    # accidentally includes it
    baseline = [t for t in baseline if t["function"]["name"] != MESSAGE_TOOL]
    return baseline


def parse_tool_result(raw: Any) -> Any:
    """gen_ai.tool.result is usually a JSON string; decode once if so."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def extract_user_reply_from_result(parsed: Any) -> str | None:
    """Exgentic's result shape: [{"invoking_actions": [...], "result": {...}}].
    Return a user reply string if the result holds one, else None."""
    if isinstance(parsed, list) and parsed:
        item = parsed[0]
        if isinstance(item, dict):
            res = item.get("result")
            if isinstance(res, dict) and res.get("sender") == "user":
                return res.get("message")
    return None


def extract_tool_result_payload(parsed: Any) -> str:
    """Return the 'result' field (JSON-stringified) for feeding back as a
    tool message. Falls back to the full raw form."""
    if isinstance(parsed, list) and parsed:
        item = parsed[0]
        if isinstance(item, dict) and "result" in item:
            res = item["result"]
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
    return json.dumps(parsed, ensure_ascii=False) if not isinstance(parsed, str) else parsed


def parse_arguments(raw: Any) -> tuple[str, dict]:
    """Return (raw_args_string, parsed_dict). Arguments may already be a dict."""
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False), raw
    if isinstance(raw, str):
        try:
            return raw, json.loads(raw)
        except json.JSONDecodeError:
            return raw, {}
    return "{}", {}


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "trace"


def _build_response(content: str, tool_calls: list[dict]) -> str:
    """Render an LLM call's response in the unified format.

    When tool_calls are present, return a JSON object with ``content``
    and ``tool_calls`` keys; otherwise return the plain content string.
    """
    if tool_calls:
        return json.dumps(
            {"content": content or "", "tool_calls": tool_calls},
            ensure_ascii=False,
        )
    return content


class _Ctx:
    """Shared per-trace context (api_spec string, meta base, etc.)."""

    def __init__(self, task_id, agent_name, api_spec_str, base_meta, traj_score):
        self.task_id = task_id
        self.agent_name = agent_name
        self.api_spec_str = api_spec_str
        self.base_meta = base_meta
        self.traj_score = traj_score
        self.step = 0
        self.llm_call = 0
        self.intent = ""

    def make_row(self, model_input_list, content, tool_calls, row_meta):
        """Emit one CSV row per LLM call in the unified row format."""
        self.step += 1
        self.llm_call += 1
        return {
            "id": f"{self.task_id}_{self.step}",
            "Name": self.agent_name,
            "intent": self.intent,
            "task_id": self.task_id,
            "step_in_trace_general": self.step,
            "llm_call_index": self.llm_call,
            "model_input": json.dumps(model_input_list, ensure_ascii=False),
            "response": _build_response(content, tool_calls),
            "tool_or_agent": "agent",
            "api_spec": self.api_spec_str,
            "meta_data": json.dumps(row_meta, ensure_ascii=False),
            "traj_score": self.traj_score,
        }


def _convert_tau2_style(children: list[dict], ctx: _Ctx, session_attrs: dict) -> list[dict]:
    """Tau2-style harness: each child is an `execute_tool` span for one
    LLM-produced action (``message``, ``initial_observation``, or a real tool)."""
    rows: list[dict] = []
    messages: list[dict] = []
    policy = str(session_attrs.get("exgentic.context.policy", ""))
    if policy:
        messages.append({"role": "system", "content": policy})

    for span in children:
        tool_name = get_attr(span, "gen_ai.tool.name")
        tool_id = get_attr(span, "gen_ai.tool.id", "")
        raw_params = get_attr(span, "gen_ai.tool.parameters", "{}")
        raw_result = get_attr(span, "gen_ai.tool.result", "")
        parsed_result = parse_tool_result(raw_result)

        if tool_name == INITIAL_OBS:
            user_msg = extract_user_reply_from_result(parsed_result)
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
                if not ctx.intent:
                    ctx.intent = user_msg[:INTENT_LIMIT]
            continue

        model_input_snapshot = copy.deepcopy(messages)
        args_str, args_dict = parse_arguments(raw_params)
        row_meta = {
            **ctx.base_meta,
            "span_id": span.get("span_id"),
            "tool_call_id": tool_id,
            "start_time": span.get("start_time"),
            "end_time": span.get("end_time"),
        }

        if tool_name == MESSAGE_TOOL:
            content = args_dict.get("content", "")
            rows.append(ctx.make_row(model_input_snapshot, content, [], row_meta))
            messages.append({"role": "assistant", "content": content})
            user_reply = extract_user_reply_from_result(parsed_result)
            if user_reply:
                messages.append({"role": "user", "content": user_reply})
        else:
            tool_call_dict = {
                "id": tool_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": args_str},
            }
            rows.append(ctx.make_row(
                model_input_snapshot, "", [tool_call_dict], row_meta,
            ))
            messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call_dict]})
            tool_payload = extract_tool_result_payload(parsed_result)
            messages.append({"role": "tool", "tool_call_id": tool_id, "content": tool_payload})

    return rows


def _normalize_tool_call(tc: dict) -> dict:
    """Coerce a tool call dict to canonical OpenAI shape (arguments as str)."""
    fn = tc.get("function", {}) or {}
    args = fn.get("arguments", "")
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    elif not isinstance(args, str):
        args = "" if args is None else str(args)
    return {
        "id": tc.get("id", tc.get("call_id", "")),
        "type": "function",
        "function": {"name": fn.get("name", ""), "arguments": args},
    }


def _convert_chat_style(children: list[dict], ctx: _Ctx, session_attrs: dict) -> list[dict]:
    """Appworld-style harness: children include ``chat <model>`` spans that
    carry ``gen_ai.input.messages`` (pre-reconstructed conversation). The
    assistant response of chat[i] appears as newly-appended messages in
    chat[i+1]'s input; the tail is recovered from sibling ``execute_tool``
    spans that ran after the last chat. Successive chats with identical
    inputs (retries) are deduplicated.

    Emits the unified row format: one row per chat turn carrying both the
    assistant text (``content``) and any tool calls bundled inside ``response``.
    Per-tool-call splitting + reasoning injection happen later inside the
    pipeline at ``run_clear_step_analysis.convert_to_clear_format`` when
    ``separate_tools`` is enabled."""
    rows: list[dict] = []

    task_text = str(session_attrs.get("exgentic.session.task", "") or "")
    if task_text:
        ctx.intent = task_text[:INTENT_LIMIT]

    chats = [
        c for c in children
        if c.get("name", "").startswith("chat ")
        and "gen_ai.input.messages" in c.get("attributes", {})
    ]
    if not chats:
        return rows

    def parse_msgs(span: dict) -> list[dict]:
        raw = span["attributes"].get("gen_ai.input.messages", "[]")
        if isinstance(raw, list):
            return raw
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return []

    deduped: list[dict] = []
    last_key = None
    for c in chats:
        key = c["attributes"].get("gen_ai.input.messages")
        if key == last_key:
            continue
        deduped.append(c)
        last_key = key

    if not ctx.intent and deduped:
        first_msgs = parse_msgs(deduped[0])
        first_user = next((m for m in first_msgs if m.get("role") == "user"), None)
        if first_user and isinstance(first_user.get("content"), str):
            ctx.intent = first_user["content"][:INTENT_LIMIT]

    et_spans = sorted(
        (c for c in children if c.get("name", "").startswith("execute_tool")),
        key=lambda s: s.get("start_time", ""),
    )

    def response_turn(i: int) -> dict | None:
        cur = parse_msgs(deduped[i])
        if i + 1 < len(deduped):
            nxt = parse_msgs(deduped[i + 1])
            added = nxt[len(cur):] if len(nxt) >= len(cur) else []
            assistant = next((m for m in added if m.get("role") == "assistant"), None)
            return assistant
        start = deduped[i].get("start_time", "")
        tail = [e for e in et_spans if e.get("start_time", "") >= start]
        if not tail:
            return None
        tool_calls = []
        for e in tail:
            name = get_attr(e, "gen_ai.tool.name")
            if not name:
                continue
            args_str, _ = parse_arguments(get_attr(e, "gen_ai.tool.parameters", "{}"))
            tool_calls.append({
                "id": get_attr(e, "gen_ai.tool.id", ""),
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
        return {"role": "assistant", "tool_calls": tool_calls} if tool_calls else None

    for i, chat in enumerate(deduped):
        model_input = parse_msgs(chat)
        turn = response_turn(i)
        if turn is None:
            continue
        content = turn.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        tool_calls_raw = turn.get("tool_calls") or []
        tool_calls = [_normalize_tool_call(tc) for tc in tool_calls_raw if tc]

        if not content.strip() and not tool_calls:
            continue

        row_meta = {
            **ctx.base_meta,
            "span_id": chat.get("span_id"),
            "start_time": chat.get("start_time"),
            "end_time": chat.get("end_time"),
        }
        rows.append(ctx.make_row(model_input, content, tool_calls, row_meta))

    return rows


def _is_chat_style(children: list[dict]) -> bool:
    for c in children:
        if c.get("name", "").startswith("chat ") \
                and "gen_ai.input.messages" in c.get("attributes", {}):
            return True
    return False


def convert_trace(trace: dict, specs_dir: Path) -> tuple[list[dict], str]:
    """Return (rows, suggested_output_basename)."""
    spans = trace.get("spans", [])
    if not spans:
        raise ValueError("trace has no spans")

    session_span = find_session_span(spans)
    session_attrs = session_span.get("attributes", {})
    session_id = session_span.get("span_id", "")
    children = child_spans_sorted(spans, session_id)

    slug = str(session_attrs.get("exgentic.benchmark.slug_name", "unknown"))
    subset = str(session_attrs.get("exgentic.benchmark.subset", ""))
    task_id = str(
        session_attrs.get("exgentic.session.task_id")
        or session_attrs.get("exgentic.session.id")
        or trace.get("trace_id")
        or "unknown"
    )
    agent_name = str(session_attrs.get("exgentic.benchmark.agent.name", "agent"))
    model = str(session_attrs.get("gen_ai.request.model", ""))
    try:
        traj_score = float(session_attrs.get("exgentic.score", ""))
    except (TypeError, ValueError):
        traj_score = ""

    session_tools = collect_session_tool_names(session_attrs)
    api_spec_list = load_tool_specs(specs_dir, slug, subset, session_tools)
    api_spec_str = json.dumps(api_spec_list, ensure_ascii=False) if api_spec_list else ""

    base_meta = {
        "model": model,
        "benchmark_slug": slug,
        "benchmark_subset": subset,
        "session_steps": session_attrs.get("exgentic.session.steps"),
        "agent_cost": session_attrs.get("exgentic.agent.agent_cost"),
    }

    ctx = _Ctx(task_id, agent_name, api_spec_str, base_meta, traj_score)
    if _is_chat_style(children):
        rows = _convert_chat_style(children, ctx, session_attrs)
    else:
        rows = _convert_tau2_style(children, ctx, session_attrs)

    basename = f"{sanitize_filename(task_id)}__{sanitize_filename(slug)}_{sanitize_filename(subset)}"
    return rows, basename


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", type=Path, help="input trace JSON files")
    ap.add_argument("--output-dir", type=Path, required=True, help="directory for CSV outputs")
    ap.add_argument("--specs-dir", type=Path, default=here / "tool_specs", help="dir containing <slug>[_<subset>].json tool specs")
    args = ap.parse_args(argv)

    rc = 0
    for trace_path in args.traces:
        try:
            trace = load_json(trace_path)
            rows, basename = convert_trace(trace, args.specs_dir)
            out_path = args.output_dir / f"{basename}__{sanitize_filename(trace_path.stem)}.csv"
            write_csv(rows, out_path)
            print(f"[ok] {trace_path} -> {out_path} ({len(rows)} rows)")
        except Exception as e:
            print(f"[err] {trace_path}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
