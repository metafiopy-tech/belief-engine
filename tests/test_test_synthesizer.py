"""belief.memory.test_synthesizer — injectable-client Haiku prompt shape."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from belief.memory.test_synthesizer import (
    DEFAULT_MAX_TESTS,
    SynthesisResult,
    synthesize_tests,
)


GOOD_OUTPUT = json.dumps(
    [
        {"description": "basic case", "kwargs": {"a": 1, "b": 2}, "expected": 3},
        {"description": "zeros", "kwargs": {"a": 0, "b": 0}, "expected": 0},
        {"description": "negative", "kwargs": {"a": -1, "b": 1}, "expected": 0},
    ]
)

OUTPUT_IN_PROSE = (
    "Sure, here are your tests:\n\n"
    + GOOD_OUTPUT
    + "\n\nLet me know if you want more edge cases!"
)

OUTPUT_MALFORMED_ROWS = json.dumps(
    [
        {"description": "ok", "kwargs": {"x": 1}, "expected": 1},     # valid
        {"description": "no kwargs", "expected": 1},                  # missing kwargs
        {"kwargs": {"x": 2}, "expected": 2},                          # missing desc (ok — desc is optional)
        "not a dict",                                                  # not a dict
        {"description": "no expected", "kwargs": {"x": 3}},           # missing expected
    ]
)


@dataclass
class FakeClient:
    responses: list[str] = field(default_factory=list)
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def generate_text(self, *, system: str, prompt: str, max_tokens: int) -> str:
        self.calls.append((system, prompt, max_tokens))
        return self.responses.pop(0) if self.responses else ""


# ---------------------------------------------------------------------------
# Client behavior
# ---------------------------------------------------------------------------


def test_no_client_returns_empty_result() -> None:
    result = synthesize_tests({"main.py": "def f(): pass"}, "Do the thing")
    assert isinstance(result, SynthesisResult)
    assert result.tests == []


def test_valid_response_yields_three_entries() -> None:
    client = FakeClient(responses=[GOOD_OUTPUT])
    result = synthesize_tests(
        {"main.py": "def add(a, b): return a + b"},
        "Add two numbers",
        client=client,
        n_tests=5,
    )
    assert len(result) == 3
    assert result.dropped == 0
    # Kwargs and expected preserved
    assert result.tests[0]["kwargs"] == {"a": 1, "b": 2}
    assert result.tests[0]["expected"] == 3


def test_json_extracted_from_prose_response() -> None:
    client = FakeClient(responses=[OUTPUT_IN_PROSE])
    result = synthesize_tests(
        {"main.py": "..."}, "Add", client=client,
    )
    assert len(result) == 3


def test_malformed_rows_dropped() -> None:
    client = FakeClient(responses=[OUTPUT_MALFORMED_ROWS])
    result = synthesize_tests(
        {"main.py": "..."}, "Add", client=client,
    )
    # "ok", "no desc" -> 2 valid; "no kwargs", "not a dict", "no expected" -> 3 dropped
    assert len(result) == 2
    assert result.dropped == 3


def test_respects_n_tests_cap() -> None:
    client = FakeClient(responses=[GOOD_OUTPUT])
    result = synthesize_tests(
        {"main.py": "..."}, "Add", client=client, n_tests=2,
    )
    assert len(result) == 2


def test_client_exception_returns_empty() -> None:
    class Boom:
        def generate_text(self, *, system: str, prompt: str, max_tokens: int) -> str:
            raise RuntimeError("rate limit")

    result = synthesize_tests(
        {"main.py": "..."}, "Add", client=Boom(),
    )
    assert result.tests == []


def test_prompt_contains_goal_and_code() -> None:
    client = FakeClient(responses=["[]"])
    synthesize_tests(
        {"main.py": "def f(): return 1"},
        goal="Return a constant",
        client=client,
    )
    assert client.calls, "client must be called"
    _system, prompt, _max_tokens = client.calls[0]
    assert "Return a constant" in prompt
    assert "def f(): return 1" in prompt


def test_code_truncated_at_4k() -> None:
    long_code = "# " + "x" * 8000
    client = FakeClient(responses=["[]"])
    synthesize_tests(
        {"main.py": long_code}, "long", client=client,
    )
    _system, prompt, _max_tokens = client.calls[0]
    # Prompt carries the truncation marker and is under a reasonable length
    assert "<truncated>" in prompt
    assert len(prompt) < 5000


def test_default_max_tests_is_spec() -> None:
    assert DEFAULT_MAX_TESTS == 5
