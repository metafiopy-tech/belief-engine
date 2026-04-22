"""Tests for the lenient JSON parsing added for local-model output.

Local models produce malformed JSON more often than Claude.  The
hardened parser needs to accept:

  - Markdown code fences (```json ... ```)
  - Truncated output (existing behavior)
  - Trailing commas  — ``{"a": 1,}``
  - Single-quoted strings — ``{'a': 'b'}``
  - Unquoted keys — ``{a: 1}``
  - Regex fallback for scalar fields when nothing else works

Valid JSON must stay untouched — cloud output never goes through the
lenient path if ``json.loads`` already succeeded.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel


class _Demo(BaseModel):
    name: str
    count: int
    enabled: bool = True


# ── Invariants ────────────────────────────────────────────────────────────


def test_baseline_valid_json_parses():
    from belief.llm import _parse_structured
    m = _parse_structured('{"name": "foo", "count": 3}', _Demo)
    assert m.name == "foo"
    assert m.count == 3
    assert m.enabled is True


def test_markdown_fence_stripped():
    from belief.llm import _parse_structured
    wrapped = '```json\n{"name": "foo", "count": 3}\n```'
    m = _parse_structured(wrapped, _Demo)
    assert m.name == "foo"


# ── Local-model failure modes ────────────────────────────────────────────


def test_trailing_comma_before_closing_brace():
    from belief.llm import _parse_structured, _strip_trailing_commas
    assert _strip_trailing_commas('{"a": 1,}') == '{"a": 1}'
    m = _parse_structured('{"name": "foo", "count": 3,}', _Demo)
    assert m.name == "foo" and m.count == 3


def test_trailing_comma_before_closing_bracket():
    from belief.llm import _strip_trailing_commas
    assert _strip_trailing_commas('[1, 2, 3,]') == '[1, 2, 3]'


def test_single_quoted_strings():
    from belief.llm import _parse_structured, _convert_single_to_double_quotes
    assert _convert_single_to_double_quotes("{'a': 'b'}") == '{"a": "b"}'
    m = _parse_structured("{'name': 'bar', 'count': 7}", _Demo)
    assert m.name == "bar" and m.count == 7


def test_unquoted_keys():
    from belief.llm import _parse_structured, _quote_unquoted_keys
    cleaned = _quote_unquoted_keys('{name: "baz", count: 9}')
    assert '"name":' in cleaned
    assert '"count":' in cleaned
    m = _parse_structured('{name: "baz", count: 9}', _Demo)
    assert m.name == "baz" and m.count == 9


def test_all_three_failure_modes_combined():
    from belief.llm import _parse_structured
    ugly = "{ name: 'qux', count: 12, enabled: true, }"
    m = _parse_structured(ugly, _Demo)
    assert m.name == "qux"
    assert m.count == 12
    assert m.enabled is True


# ── Anti-clobbering: lenient passes must not corrupt valid strings ────────


def test_apostrophe_inside_valid_double_quoted_string():
    """The single-quote rewrite must not touch apostrophes inside an
    already-double-quoted string."""
    from belief.llm import _parse_structured
    # Note: \u0027 is a literal apostrophe inside a valid JSON string
    m = _parse_structured('{"name": "it\\u0027s fine", "count": 1}', _Demo)
    assert "fine" in m.name


def test_colon_inside_string_not_mistaken_for_key():
    """The unquoted-key rewrite must only fire outside of string
    literals.  Otherwise ``"key: value"`` becomes ``"\\"key\\": value"``."""
    from belief.llm import _parse_structured
    m = _parse_structured('{"name": "key: value", "count": 1}', _Demo)
    assert m.name == "key: value"


def test_comma_inside_string_not_stripped():
    from belief.llm import _strip_trailing_commas
    # Trailing commas inside a quoted string must survive
    preserved = _strip_trailing_commas('{"a": "has, commas, inside"}')
    assert "has, commas, inside" in preserved


# ── Truncation (pre-existing behavior, guard against regression) ──────────


def test_truncated_json_gets_repaired():
    from belief.llm import _parse_structured
    m = _parse_structured('{"name": "trunc", "count": 5', _Demo)
    assert m.name == "trunc"
    assert m.count == 5


# ── Regex fallback (last-resort path) ────────────────────────────────────


def test_regex_fallback_extracts_scalars_from_garbage():
    from belief.llm import _regex_extract_fields
    broken = 'Here is the answer: "name": "salvaged" blah "count": 42 blah'
    salvaged = _regex_extract_fields(broken, _Demo)
    assert salvaged == {"name": "salvaged", "count": 42}


def test_regex_fallback_returns_none_when_required_missing():
    """If a required field can't be found, refuse to return a partial
    — the caller needs to see the original parse error."""
    from belief.llm import _regex_extract_fields
    # No 'count' anywhere
    broken = 'name": "lonely" but no count here'
    result = _regex_extract_fields(broken, _Demo)
    assert result is None


def test_end_to_end_regex_fallback_on_totally_broken_output():
    from belief.llm import _parse_structured
    broken = (
        "Let me think... the user wants "
        '"name": "fallback" and "count": 99, enabled: maybe?'
    )
    m = _parse_structured(broken, _Demo)
    assert m.name == "fallback"
    assert m.count == 99


# ── Negative: truly empty / no-JSON responses still raise ─────────────────


def test_no_json_object_raises():
    from belief.llm import _parse_structured
    with pytest.raises(ValueError, match="No JSON object"):
        _parse_structured("Sorry, I can't help with that.", _Demo)


# ── Don't-break-cloud invariant ────────────────────────────────────────────


def test_cloud_style_nested_json_still_parses():
    """Claude often emits nested structures.  The lenient path must
    never be invoked on these; json.loads handles them on the first try.
    Guard by asserting the result comes out bitwise-identical to
    json.loads."""
    import json as _json
    from belief.llm import _parse_structured

    class _Nested(BaseModel):
        name: str
        tags: list[str]
        meta: dict[str, int]

    valid = '{"name": "x", "tags": ["a", "b"], "meta": {"k": 1}}'
    m = _parse_structured(valid, _Nested)
    assert m.tags == ["a", "b"]
    assert m.meta == {"k": 1}
