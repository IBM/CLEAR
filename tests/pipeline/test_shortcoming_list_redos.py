"""Regression tests for CodeQL py/redos alerts in parse_shortcoming_list_response."""

import time

from clear_eval.pipeline.eval_utils import parse_shortcoming_list_response


def test_quoted_list_regexes_do_not_backtrack_exponentially():
    # The old `".*?"(?:\s*,\s*".*?")*` form took ~10x longer per 4 extra
    # repetitions on these unterminated inputs.
    for payload in ('[""' + '","' * 5000, "[''," + "','" * 5000):
        start = time.monotonic()
        parse_shortcoming_list_response(payload)
        assert time.monotonic() - start < 1.0


def test_quoted_lists_still_parse():
    assert parse_shortcoming_list_response('Here:\n["a",\n "b"]\nDone') == ["a", "b"]
    assert parse_shortcoming_list_response("['it's slow', 'b']") == ["it's slow", "b"]
    assert parse_shortcoming_list_response("[]") == []
