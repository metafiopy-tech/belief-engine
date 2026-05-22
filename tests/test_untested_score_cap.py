"""Session B (handoff Q3): no-tests score cap.

Hermetic — no network, no subprocess. Exercises the testable-logic heuristic,
the test-file discovery mirror, the in-module-test detector, and the cap that
stops a testable-but-untested build from scoring 1.00.
"""

from __future__ import annotations

import logging

import pytest

from belief.agents.validator import (
    DEFAULT_UNTESTED_SCORE_CAP,
    _apply_untested_cap,
    _discover_test_files,
    _has_testable_logic,
    _modules_with_inline_tests,
    _untested_score_cap,
)
from belief.models.artifacts import ValidationResult, ValidationVerdict


# ── testable-logic heuristic ─────────────────────────────────────────────────


def test_has_testable_logic_true_for_function():
    assert _has_testable_logic({"app.py": "def add(a, b):\n    return a + b\n"}) is True


def test_has_testable_logic_true_for_class():
    assert _has_testable_logic({"models.py": "class User:\n    pass\n"}) is True


def test_has_testable_logic_false_for_constants_only():
    assert _has_testable_logic({"config.py": "DEBUG = True\nNAME = 'x'\n"}) is False


def test_has_testable_logic_false_for_imports_only():
    assert _has_testable_logic({"deps.py": "import os\nimport sys\n"}) is False


def test_has_testable_logic_ignores_test_files():
    # A def that lives in a test file is not "untested logic".
    assert _has_testable_logic({"test_app.py": "def test_x():\n    assert True\n"}) is False


# ── test-file discovery mirror ───────────────────────────────────────────────


def test_discover_test_files_picks_up_explicit_and_pathnamed():
    code = {"app.py": "def f(): pass", "tests/test_app.py": "def test_f(): pass"}
    tf = {"test_extra.py": "def test_e(): pass"}
    discovered = _discover_test_files(code, tf)
    assert "test_extra.py" in discovered
    assert "tests/test_app.py" in discovered
    assert "app.py" not in discovered


def test_discover_test_files_empty_when_none():
    assert _discover_test_files({"app.py": "def f(): pass"}, {}) == {}


# ── cap config ───────────────────────────────────────────────────────────────


def test_cap_default(monkeypatch):
    monkeypatch.delenv("BELIEF_UNTESTED_SCORE_CAP", raising=False)
    assert _untested_score_cap() == pytest.approx(DEFAULT_UNTESTED_SCORE_CAP)


def test_cap_env_override(monkeypatch):
    monkeypatch.setenv("BELIEF_UNTESTED_SCORE_CAP", "0.3")
    assert _untested_score_cap() == pytest.approx(0.3)


def test_cap_env_clamped_and_invalid(monkeypatch):
    monkeypatch.setenv("BELIEF_UNTESTED_SCORE_CAP", "5")
    assert _untested_score_cap() == pytest.approx(1.0)
    monkeypatch.setenv("BELIEF_UNTESTED_SCORE_CAP", "junk")
    assert _untested_score_cap() == pytest.approx(DEFAULT_UNTESTED_SCORE_CAP)


# ── the cap itself ───────────────────────────────────────────────────────────


def _perfect_result() -> ValidationResult:
    return ValidationResult(verdict=ValidationVerdict.PASS, weighted_score=1.0, issues=[])


def test_testable_but_untested_is_capped(monkeypatch):
    monkeypatch.delenv("BELIEF_UNTESTED_SCORE_CAP", raising=False)
    result = _perfect_result()
    _apply_untested_cap(result, {"app.py": "def add(a, b):\n    return a + b\n"}, {})
    assert result.weighted_score <= DEFAULT_UNTESTED_SCORE_CAP
    assert result.verdict == ValidationVerdict.FAIL_FIXABLE
    assert "No test files generated for testable logic" in result.issues


def test_config_only_build_not_penalized():
    result = _perfect_result()
    _apply_untested_cap(result, {"config.py": "DEBUG = True\n"}, {})
    assert result.weighted_score == pytest.approx(1.0)
    assert result.verdict == ValidationVerdict.PASS
    assert "No test files generated for testable logic" not in result.issues


def test_build_with_discovered_tests_untouched():
    result = _perfect_result()
    code = {"app.py": "def add(a, b):\n    return a + b\n"}
    tf = {"test_app.py": "def test_add():\n    assert True\n"}
    _apply_untested_cap(result, code, tf)
    assert result.weighted_score == pytest.approx(1.0)
    assert result.verdict == ValidationVerdict.PASS


def test_inline_test_module_logs_discovery_gap(caplog):
    code = {
        "device.py": (
            "import unittest\n"
            "def control():\n    return 1\n"
            "class TestDevice(unittest.TestCase):\n    pass\n"
        )
    }
    assert _modules_with_inline_tests(code) == ["device.py"]
    result = _perfect_result()
    with caplog.at_level(logging.WARNING, logger="belief.agents.validator"):
        _apply_untested_cap(result, code, {})
    # in-module tests are NOT discovered tests → build is still capped
    assert result.weighted_score <= DEFAULT_UNTESTED_SCORE_CAP
    assert any("discovery gap" in r.message for r in caplog.records)
