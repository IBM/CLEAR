"""Regression tests for CodeQL py/redos alerts in parse_shortcoming_list_response.

The quoted-list patterns used to let `".*?"` span item separators, so an
unterminated input such as '[""' + '","' * n could be split into items in 2**n
ways, all tried before the match failed. The expected values below were checked
against the pre-fix patterns: the rewrite must change speed, not results.
"""

import time

import pytest

from clear_eval.pipeline.eval_utils import (
    _DOUBLE_QUOTED_LIST_RE,
    _SINGLE_QUOTED_LIST_RE,
    parse_shortcoming_list_response,
)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param('[""' + '","' * 5000, id="double-quoted"),
        pytest.param("[''," + "','" * 5000, id="single-quoted"),
        pytest.param('[ ""' + ' , ""' * 5000, id="spaced-separators"),
        pytest.param('["' + "x" * 100_000, id="long-unterminated-item"),
        pytest.param('["a", ' * 5000, id="repeated-list-openings"),
    ],
)
def test_unterminated_lists_fail_fast(payload):
    start = time.monotonic()
    assert parse_shortcoming_list_response(payload) is None
    assert time.monotonic() - start < 1.0


def test_long_valid_list_parses_fast():
    payload = "[" + ", ".join(f'"item {i}"' for i in range(5000)) + "]"
    start = time.monotonic()
    result = parse_shortcoming_list_response(payload)
    assert time.monotonic() - start < 1.0
    assert result == [f"item {i}" for i in range(5000)]


@pytest.mark.parametrize(
    "response, expected",
    [
        pytest.param('["a","b"]', ["a", "b"], id="adjacent-items"),
        pytest.param('[ "a" ,\n   "b" ]', ["a", "b"], id="whitespace-and-newlines"),
        pytest.param('["slow, verbose", "wrong"]', ["slow, verbose", "wrong"], id="comma-inside-item"),
        pytest.param('["cites [1] wrongly", "b"]', ["cites [1] wrongly", "b"], id="brackets-inside-item"),
        pytest.param('["uses ] oddly", "b"]', ["uses ] oddly", "b"], id="close-bracket-inside-item"),
        pytest.param('Here is the list:\n["a", "b"]\nHope this helps.', ["a", "b"], id="prose-around-list"),
        pytest.param('["a"] and later ["b"]', ["a"], id="first-list-wins"),
        # Lenient, as before: a quote not followed by `,` or `]` is item text.
        pytest.param('["a" "b"]', ["a", "b"], id="missing-separator"),
        pytest.param('See [1]. ["a", "b"]', ["a", "b"], id="skips-non-string-list"),
        pytest.param('["", "  ", "a"]', ["a"], id="drops-blank-items"),
        pytest.param('["  padded  "]', ["padded"], id="strips-items"),
        pytest.param('["naïve — résumé", "日本"]', ["naïve — résumé", "日本"], id="unicode"),
        pytest.param("['it's slow', 'doesn't cite']", ["it's slow", "doesn't cite"], id="apostrophes"),
        pytest.param("['a',\n 'b']", ["a", "b"], id="single-quoted-multiline"),
        pytest.param("[]", [], id="empty-list"),
        pytest.param("[  \n ]", [], id="empty-list-with-whitespace"),
    ],
)
def test_parsing_results_unchanged(response, expected):
    assert parse_shortcoming_list_response(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("no list here", id="no-list"),
        pytest.param('["a", "b"', id="missing-close-bracket"),
        pytest.param('["a", \'b\']', id="mixed-quote-styles"),
        pytest.param('["a", "b",]', id="trailing-comma"),
    ],
)
def test_unparseable_responses_return_none(response):
    assert parse_shortcoming_list_response(response) is None


@pytest.mark.parametrize(
    "response, captured",
    [
        # A quote not followed by `,` or `]` stays inside the item...
        pytest.param('["he said "hi" ok", "b"]', '"he said "hi" ok", "b"', id="inner-quotes"),
        # ...and one followed by `,` always closes it, so `","` is a boundary.
        pytest.param('["a","b"]', '"a","b"', id="separator-is-boundary"),
        pytest.param('["first\nsecond", "b"]', '"first\nsecond", "b"', id="newline-inside-item"),
    ],
)
def test_double_quoted_capture(response, captured):
    assert _DOUBLE_QUOTED_LIST_RE.search(response).group(1) == captured


def test_single_quoted_capture_keeps_apostrophes():
    match = _SINGLE_QUOTED_LIST_RE.search("['it's slow', 'b']")
    assert match.group(1) == "'it's slow', 'b'"


def test_double_quoted_pattern_rejects_unterminated():
    assert _DOUBLE_QUOTED_LIST_RE.search('[""' + '","' * 50) is None
