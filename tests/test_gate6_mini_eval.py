"""Hermetic tests for ``audit/gate6/run_mini_eval.py``.

Bug 1 regression coverage. Verifies that the engine-build score
capture works in three shapes:

1. Archive lookup returns an outcome → use those numbers.
2. Archive empty / no run_id → fall back to regex-parsing
   stdout *and* stderr (the validator log goes to stderr).
3. Both empty → record zeros without raising.

These tests do not touch the network, run no subprocesses, and use no
on-disk archive. They monkeypatch ``subprocess.run`` and
``AgentArchive`` so the runner exercises only its own parsing logic.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

# audit/ is not a regular package (no __init__.py) but is importable
# as a namespace package when the repo root is on sys.path. Pytest
# normally adds the repo root, but be defensive in case the test is
# invoked from a different cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audit.gate6 import run_mini_eval as rme  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_metrics_from_streams — regex fallback
# ---------------------------------------------------------------------------


class TestParseMetricsFromStreams:
    def test_validator_line_in_stderr_is_picked_up(self) -> None:
        """The original Bug 1: validator emits its line via logger.info
        which writes to stderr. Stdout-only parsing missed it. The
        fallback must scan stderr too."""
        stderr = (
            "INFO    belief.thermal              thermal_gate: ok\n"
            "INFO    belief.agents.validator     "
            "Validator: pass, 9/9 tests, weighted=1.00\n"
            "INFO    belief.agents.validator     Validator completed in 1.3s\n"
        )
        stdout = (
            "============================================================\n"
            "  BUILD COMPLETE — 187.3s\n"
            "  Verdict: pass\n"
        )
        m = rme._parse_metrics_from_streams(stdout, stderr)
        assert m["verdict"] == "pass"
        assert m["tests_passed"] == 9
        assert m["tests_total"] == 9
        assert abs(m["weighted_score"] - 1.0) < 1e-6

    def test_partial_pass_is_parsed(self) -> None:
        """Real audit log: Validator: pass, 9/10 tests, weighted=0.88."""
        stderr = "INFO belief.agents.validator Validator: pass, 9/10 tests, weighted=0.88"
        m = rme._parse_metrics_from_streams("", stderr)
        assert m["tests_passed"] == 9
        assert m["tests_total"] == 10
        assert abs(m["weighted_score"] - 0.88) < 1e-3

    def test_validator_line_only_no_verdict_print(self) -> None:
        """If stdout never printed Verdict:, take the validator line's
        own verdict so we don't end up with verdict=unknown."""
        stderr = "Validator: fail_fixable, 2/5 tests, weighted=0.40"
        m = rme._parse_metrics_from_streams("", stderr)
        assert m["verdict"] == "fail_fixable"
        assert m["tests_passed"] == 2
        assert m["tests_total"] == 5

    def test_empty_streams_return_zeros(self) -> None:
        m = rme._parse_metrics_from_streams("", "")
        assert m == {
            "verdict": "unknown",
            "weighted_score": 0.0,
            "tests_passed": 0,
            "tests_total": 0,
        }


# ---------------------------------------------------------------------------
# _lookup_outcome_in_archive — archive primary path
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, payload: dict | None) -> None:
        self._payload = payload

    def get(self, *, ids: list[str], include: list[str]) -> dict:
        if self._payload is None:
            return {"metadatas": []}
        return {"metadatas": [self._payload]}


class _FakeArchive:
    """Stand-in for AgentArchive in lookup tests."""

    def __init__(self, payload: dict | None) -> None:
        self._collection = _FakeCollection(payload)

    def _ensure(self) -> None:
        pass


def _install_fake_archive(monkeypatch: pytest.MonkeyPatch, payload: dict | None) -> None:
    def _factory() -> _FakeArchive:
        return _FakeArchive(payload)

    # The runner imports AgentArchive locally inside the helper, so we
    # patch the symbol on belief.archive.store (the source module).
    monkeypatch.setattr("belief.archive.store.AgentArchive", _factory)


class TestLookupOutcomeInArchive:
    def test_outcome_json_round_trip_extracts_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from belief.archive.outcome import BuildOutcome

        outcome = BuildOutcome(
            run_id="belief-deadbeef",
            goal="Build a fizzbuzz",
            verdict="pass",
            tests_passed=9,
            tests_total=9,
            weighted_score=1.0,
        )
        _install_fake_archive(
            monkeypatch,
            {
                "outcome_json": outcome.to_json(),
                "verdict": "pass",
                "weighted_score": 1.0,
            },
        )
        m = rme._lookup_outcome_in_archive("belief-deadbeef")
        assert m is not None
        assert m["verdict"] == "pass"
        assert m["tests_passed"] == 9
        assert m["tests_total"] == 9
        assert abs(m["weighted_score"] - 1.0) < 1e-6

    def test_metadata_only_fallback_when_outcome_json_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_archive(
            monkeypatch,
            {"verdict": "fail_fixable", "weighted_score": 0.4},
        )
        m = rme._lookup_outcome_in_archive("belief-cafebabe")
        # Without outcome_json we still get verdict + score, but
        # tests_passed/total default to 0 — caller will then choose to
        # fall through to regex parsing.
        assert m == {
            "verdict": "fail_fixable",
            "weighted_score": 0.4,
            "tests_passed": 0,
            "tests_total": 0,
        }

    def test_empty_archive_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_archive(monkeypatch, None)
        assert rme._lookup_outcome_in_archive("belief-missing") is None

    def test_blank_run_id_returns_none(self) -> None:
        # Cheap guard — no archive call should be made if there's nothing
        # to look up.
        assert rme._lookup_outcome_in_archive("") is None


# ---------------------------------------------------------------------------
# run_engine — end-to-end metric capture (subprocess + archive monkeypatched)
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_subprocess(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> None:
    import subprocess as _sp

    def _fake_run(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(_sp, "run", _fake_run)


class TestRunEngineMetricCapture:
    def test_archive_path_supplies_score_and_tests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the archive has the outcome, the runner must use it
        even if stdout/stderr have nothing parseable."""
        from belief.archive.outcome import BuildOutcome

        outcome = BuildOutcome(
            run_id="belief-archived",
            goal="Build something",
            verdict="pass",
            tests_passed=9,
            tests_total=9,
            weighted_score=1.0,
        )
        _install_fake_archive(
            monkeypatch,
            {"outcome_json": outcome.to_json()},
        )
        # Stdout has the run_id line but no validator line; stderr is
        # empty. Archive must still produce the score.
        _install_fake_subprocess(
            monkeypatch,
            _FakeProc(stdout="Run ID: belief-archived\n", stderr=""),
        )
        m = asyncio.run(rme.run_engine("Build something", seed=42))
        assert m["verdict"] == "pass"
        assert m["weighted_score"] == 1.0
        assert m["tests_passed"] == 9
        assert m["tests_total"] == 9

    def test_regex_fallback_when_archive_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No archive row — runner must still recover from stderr."""
        _install_fake_archive(monkeypatch, None)
        _install_fake_subprocess(
            monkeypatch,
            _FakeProc(
                stdout="Run ID: belief-noarchive\n  Verdict: pass\n",
                stderr="INFO Validator: pass, 7/8 tests, weighted=0.88\n",
            ),
        )
        m = asyncio.run(rme.run_engine("Build something", seed=42))
        assert m["verdict"] == "pass"
        assert m["tests_passed"] == 7
        assert m["tests_total"] == 8
        assert abs(m["weighted_score"] - 0.88) < 1e-3

    def test_no_run_id_still_parses_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If subprocess didn't print a run_id (e.g. logger config
        suppressed it), regex should still capture the validator line
        from stderr and return non-zero counts — not the original
        bug's silent zeroing."""
        _install_fake_archive(monkeypatch, None)
        _install_fake_subprocess(
            monkeypatch,
            _FakeProc(
                stdout="  Verdict: pass\n",
                stderr="Validator: pass, 9/9 tests, weighted=1.00\n",
            ),
        )
        m = asyncio.run(rme.run_engine("Build", seed=42))
        assert m["tests_total"] == 9
        assert m["weighted_score"] == 1.0

    def test_failed_subprocess_records_error_but_zeros(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subprocess returns non-zero with nothing parseable → zeros
        + stderr tail recorded as error."""
        _install_fake_archive(monkeypatch, None)
        _install_fake_subprocess(
            monkeypatch,
            _FakeProc(stdout="", stderr="boom: ollama unreachable", returncode=1),
        )
        m = asyncio.run(rme.run_engine("Build", seed=42))
        assert m["verdict"] == "unknown"
        assert m["tests_passed"] == 0
        assert m["tests_total"] == 0
        assert m["weighted_score"] == 0.0
        assert "ollama unreachable" in (m.get("error") or "")
