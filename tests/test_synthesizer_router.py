"""Hermetic tests for Session 4 (v3.2) — synthesizer router.

The router decides whether to polish based on cheap local signals.
Every test constructs a synthetic state dict and verifies the
``(bool, reason)`` return shape.  No LLM, no subprocess (except the
genuine ruff/radon invocations which we stub via missing-CLI paths
when needed).

Run with::

    python3 -m pytest tests/test_synthesizer_router.py -v
"""

from __future__ import annotations

import os

import pytest

from belief.synthesizer_router import (
    CYCLOMATIC_COMPLEXITY_THRESHOLD,
    LINES_ADDED_THRESHOLD,
    RUFF_ERROR_THRESHOLD,
    WALLCLOCK_BUDGET_S,
    route_enabled,
    should_polish,
)


# ---------------------------------------------------------------------------
# State builder helpers
# ---------------------------------------------------------------------------


def _state(
    *,
    tests_failed: int = 0,
    code_files: dict[str, str] | None = None,
    wallclock_s: float = 30.0,
) -> dict:
    """Synthetic pipeline state with just the fields the router reads."""
    return {
        "execution_result": {
            "success": tests_failed == 0,
            "tests_failed": tests_failed,
        },
        "code_files": code_files or {"main.py": "print('ok')\n"},
        "agent_timings": {"builder": wallclock_s},
    }


# ---------------------------------------------------------------------------
# Route enabled / disabled
# ---------------------------------------------------------------------------


class TestRouteEnabled:
    def test_default_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SYNTHESIZER_ROUTE_ENABLED", raising=False)
        assert route_enabled() is True

    def test_zero_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNTHESIZER_ROUTE_ENABLED", "0")
        assert route_enabled() is False

    def test_false_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYNTHESIZER_ROUTE_ENABLED", "false")
        assert route_enabled() is False

    def test_disabled_forces_polish_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the router is disabled, should_polish always returns True
        regardless of signals — that's the pre-session-4 behaviour."""
        monkeypatch.setenv("SYNTHESIZER_ROUTE_ENABLED", "0")
        polish, _ = should_polish(_state())
        assert polish is True


# ---------------------------------------------------------------------------
# Trigger: tests failed
# ---------------------------------------------------------------------------


class TestTestsFailed:
    def test_zero_tests_failed_does_not_trigger(self) -> None:
        polish, reason = should_polish(_state(tests_failed=0))
        # May still skip on other signals; main thing is tests_failed
        # alone doesn't force polish.
        assert polish is False or "tests_failed" not in reason.split(":")[0]

    def test_any_tests_failed_triggers(self) -> None:
        polish, reason = should_polish(_state(tests_failed=1))
        assert polish is True
        assert "tests_failed" in reason


# ---------------------------------------------------------------------------
# Trigger: ruff errors > threshold
# ---------------------------------------------------------------------------


class TestRuffTrigger:
    def test_ruff_clean_does_not_trigger(self) -> None:
        clean = "def main() -> None:\n    pass\n"
        polish, _ = should_polish(_state(code_files={"main.py": clean}))
        assert polish is False

    def test_many_ruff_violations_trigger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ruff to report >3 errors via monkey-patching the count helper."""
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 10)
        polish, reason = should_polish(_state())
        assert polish is True
        assert "ruff_errors" in reason


# ---------------------------------------------------------------------------
# Trigger: cyclomatic complexity
# ---------------------------------------------------------------------------


class TestComplexityTrigger:
    def test_low_complexity_does_not_trigger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 5)
        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 0)
        polish, _ = should_polish(_state())
        assert polish is False

    def test_high_complexity_triggers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 0)
        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 18)
        polish, reason = should_polish(_state())
        assert polish is True
        assert "max_cc" in reason


# ---------------------------------------------------------------------------
# Trigger: lines added
# ---------------------------------------------------------------------------


class TestLinesTrigger:
    def test_short_code_does_not_trigger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 0)
        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 0)
        polish, _ = should_polish(_state(code_files={"x.py": "pass\n"}))
        assert polish is False

    def test_large_code_triggers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 0)
        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 0)
        big = "\n".join(f"x{i} = {i}" for i in range(LINES_ADDED_THRESHOLD + 10))
        polish, reason = should_polish(_state(code_files={"big.py": big}))
        assert polish is True
        assert "lines_added" in reason


# ---------------------------------------------------------------------------
# Suppressor: wallclock over budget
# ---------------------------------------------------------------------------


class TestWallclockSuppressor:
    def test_under_budget_allows_polish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 10)
        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 0)
        polish, _ = should_polish(_state(wallclock_s=30.0))
        assert polish is True

    def test_over_budget_suppresses_polish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Over the wallclock budget, every other trigger is overridden —
        the build is already expensive, no point adding more cost."""
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 100)
        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 50)
        polish, reason = should_polish(
            _state(tests_failed=5, wallclock_s=WALLCLOCK_BUDGET_S + 10)
        )
        assert polish is False
        assert "budget" in reason


# ---------------------------------------------------------------------------
# Integration — combined signal sanity
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_all_clean_skips_polish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 0)
        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 0)
        polish, reason = should_polish(
            _state(tests_failed=0, wallclock_s=30.0,
                   code_files={"main.py": "pass\n"})
        )
        assert polish is False
        assert "skip-polish" in reason

    def test_thresholds_match_session_doc(self) -> None:
        """Session-4 spec pins these numbers.  If someone changes them
        accidentally, this test flags it."""
        assert RUFF_ERROR_THRESHOLD == 3
        assert CYCLOMATIC_COMPLEXITY_THRESHOLD == 12
        assert LINES_ADDED_THRESHOLD == 150
        assert WALLCLOCK_BUDGET_S == 180.0


# ---------------------------------------------------------------------------
# Live ruff/radon probe (sandbox-optional)
# ---------------------------------------------------------------------------


class TestLiveProbe:
    """These call ruff / radon as subprocesses.  They pass on CI where
    both are installed and are skipped if either binary is missing."""

    def test_ruff_count_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        if shutil.which("ruff") is None:
            pytest.skip("ruff not on PATH")
        from belief.synthesizer_router import _count_ruff_errors

        # Known-bad code: several F rules, several E rules.
        bad = """\
from os import path, getcwd, sep
x = 1
y=2
z =3
def f():
    return unknown_name
"""
        n = _count_ruff_errors({"bad.py": bad})
        assert n >= 3, f"expected ruff to find ≥3 issues, found {n}"

    def test_radon_cc_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        if shutil.which("radon") is None:
            pytest.skip("radon not on PATH")
        from belief.synthesizer_router import _max_cyclomatic_complexity

        # A function with lots of branches.
        code = """
def f(x):
    if x > 1:
        if x > 2:
            if x > 3:
                if x > 4:
                    if x > 5:
                        return 'a'
                    return 'b'
                return 'c'
            return 'd'
        return 'e'
    return 'f'
"""
        cc = _max_cyclomatic_complexity({"complex.py": code})
        assert cc is not None and cc >= 6
