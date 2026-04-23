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


import pytest

from belief.synthesizer_router import (
    CYCLOMATIC_COMPLEXITY_THRESHOLD,
    LINES_ADDED_THRESHOLD,
    RUFF_ERROR_THRESHOLD,
    RuffInvocationError,
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

    def test_over_budget_suppresses_polish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Over the wallclock budget, every other trigger is overridden —
        the build is already expensive, no point adding more cost."""
        import belief.synthesizer_router as sr

        monkeypatch.setattr(sr, "_count_ruff_errors", lambda _files: 100)
        monkeypatch.setattr(sr, "_max_cyclomatic_complexity", lambda _files: 50)
        polish, reason = should_polish(_state(tests_failed=5, wallclock_s=WALLCLOCK_BUDGET_S + 10))
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
            _state(tests_failed=0, wallclock_s=30.0, code_files={"main.py": "pass\n"})
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


# ---------------------------------------------------------------------------
# Session 0 (v3.2) — ruff invocation hardening
# ---------------------------------------------------------------------------


class _FakeCompleted:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestRuffHardening:
    """Session 0 audit finding: _count_ruff_errors must

    1. pass ``--no-cache`` so an unwritable .ruff_cache cannot silently
       fail-open the router, and
    2. raise ``RuffInvocationError`` on exit code ≥ 2 (internal error)
       rather than returning 0 indistinguishably from a clean run.
    """

    def _captured_argv(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        """Patch subprocess.run in the router module and capture the argv
        list of every invocation. Returns the (mutating) list."""
        import shutil

        import belief.synthesizer_router as sr

        # Guarantee the "ruff missing" short-circuit is NOT taken —
        # shutil.which is imported locally inside _count_ruff_errors
        # via `import shutil as _shutil`, so patching shutil.which on
        # the real module works because Python returns the cached
        # module from sys.modules on the inner import.
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: "/usr/bin/" + name if name == "ruff" else None,
        )

        captured: list[list[str]] = []

        def fake_run(argv, **_kwargs):  # noqa: ANN001 — mirrors subprocess.run
            captured.append(list(argv))
            return _FakeCompleted(returncode=0, stdout="[]")

        monkeypatch.setattr(sr.subprocess, "run", fake_run)
        return captured

    def test_no_cache_flag_is_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every ruff invocation must include ``--no-cache``. Directly
        asserts the argv — the invariant that made the old fail-open
        bug possible is that the cache was being used."""
        from belief.synthesizer_router import _count_ruff_errors

        captured = self._captured_argv(monkeypatch)
        n = _count_ruff_errors({"main.py": "x = 1\n"})

        assert n == 0  # stdout "[]" → zero findings
        assert captured, "subprocess.run was never called"
        argv = captured[0]
        assert "--no-cache" in argv, f"--no-cache missing from argv: {argv}"
        # Sanity: we're still invoking the right subcommand and rules.
        assert "check" in argv
        assert "--output-format" in argv
        assert "json" in argv

    def test_exit_code_2_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ruff exit code 2 = internal error (e.g. unwritable cache dir,
        malformed argv, panic). The old code returned 0 in this case,
        silently fail-opening the polish router. Must now raise."""
        import shutil

        import belief.synthesizer_router as sr

        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: "/usr/bin/" + name if name == "ruff" else None,
        )

        def fake_run(_argv, **_kwargs):  # noqa: ANN001
            return _FakeCompleted(
                returncode=2,
                stdout="",
                stderr="error: unable to initialize cache at ./.ruff_cache\n",
            )

        monkeypatch.setattr(sr.subprocess, "run", fake_run)

        from belief.synthesizer_router import _count_ruff_errors

        with pytest.raises(RuffInvocationError) as excinfo:
            _count_ruff_errors({"main.py": "x = 1\n"})
        # The error message must name the exit code and include a
        # stderr tail — operators reading logs need enough to diagnose.
        msg = str(excinfo.value)
        assert "2" in msg
        assert "internal error" in msg.lower() or "cache" in msg.lower()

    def test_should_polish_routes_toward_polish_on_ruff_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The module docstring contract: should_polish never raises.
        If ruff itself errors, route *toward* polish (fail-safe, not
        fail-open)."""
        import shutil

        import belief.synthesizer_router as sr

        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: "/usr/bin/" + name if name == "ruff" else None,
        )

        def fake_run(_argv, **_kwargs):  # noqa: ANN001
            return _FakeCompleted(returncode=2, stdout="", stderr="ruff panic\n")

        monkeypatch.setattr(sr.subprocess, "run", fake_run)

        polish, reason = should_polish(_state())
        assert polish is True
        assert "ruff-internal-error" in reason
        assert "RuffInvocationError" in reason

    def test_exit_code_1_with_findings_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit 1 = ruff ran fine, found lint issues. This is the normal
        path and must not raise — only the count must be returned."""
        import shutil

        import belief.synthesizer_router as sr

        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: "/usr/bin/" + name if name == "ruff" else None,
        )

        def fake_run(_argv, **_kwargs):  # noqa: ANN001
            # Ruff returns exit 1 when findings exist. Stdout has JSON
            # list of findings.
            return _FakeCompleted(
                returncode=1,
                stdout='[{"code":"F401"},{"code":"E501"},{"code":"E302"},{"code":"B007"}]',
                stderr="",
            )

        monkeypatch.setattr(sr.subprocess, "run", fake_run)

        from belief.synthesizer_router import _count_ruff_errors

        n = _count_ruff_errors({"main.py": "x=1\n"})
        assert n == 4
