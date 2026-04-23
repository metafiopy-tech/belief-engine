"""Session 4 (v3.2) — conditional router for the synthesizer agent.

The synthesizer agent historically ran unconditionally after the
executor, burning 180–300s of wall clock on "polish" that usually
didn't change the build outcome.  The research report cites Aider,
Claude Code, Cursor, OpenHands, SWE-agent — none of them have a
separate polish pass, because post-success polish rarely changes
tests.

This module adds :func:`should_polish` — a cheap (<100ms) signal
test that decides whether the synthesizer has anything useful to do.
Every trigger is a failure signal; if none fire, skip directly to
the validator and save ~3 minutes.

Triggers (ANY → polish)
-----------------------

* ``tests_failed > 0`` — execution surfaced failing tests.
* ``ruff check`` finds > 3 errors (E, F, B, UP rules).
* ``radon cc -s`` finds any function with cyclomatic complexity > 12.
* ``lines_added > 150`` — polish helps most on larger surfaces.

Suppressor (always → skip)
--------------------------

* ``wallclock_so_far >= 180s`` — we're already over a reasonable
  per-build budget; further polish compounds the cost with
  diminishing returns.

Toggle
------

* Env var ``SYNTHESIZER_ROUTE_ENABLED`` (default ``"1"``) — set to
  ``"0"`` to restore the pre-router behaviour (always polish).  Used
  by the ablation harness for the ``builder_plus_synth`` condition.

The router never raises from :func:`should_polish` — every probe is
wrapped in a defensive ``try/except`` that resolves to a safe
default.  "Safe" is asymmetric by probe:

* **Missing tools** (ruff / radon absent) → treat signal as absent
  (``0``).  Best we can do when the tool simply isn't installed.
* **Soft failures** (subprocess timeout, malformed JSON, unreadable
  state dict) → same; treat as 0.
* **Ruff internal error** (exit code ≥ 2 — unwritable cache, panic,
  argv malformed) → fail *safe*, not *open*.  :func:`_count_ruff_errors`
  raises :class:`RuffInvocationError`; :func:`should_polish` catches
  it and routes **toward** polish with reason ``ruff-internal-error``.
  We can't claim the code is clean, so we shouldn't skip polish on
  that basis.

Worst case under the old (pre-Session-0) fail-open behaviour: ruff
cache unwritable → exit 2 → silent 0 errors → router skipped polish
on code that might have been dirty.  That's now a loud, traced route
decision instead of a silent routing bug.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.synthesizer_router")

# Thresholds — matched to the session-4 spec.  Centralised here so the
# ablation harness can monkey-patch them for sensitivity analyses.
RUFF_ERROR_THRESHOLD = 3
CYCLOMATIC_COMPLEXITY_THRESHOLD = 12
LINES_ADDED_THRESHOLD = 150
WALLCLOCK_BUDGET_S = 180.0

# Ruff rules to check.  E/F/B/UP are roughly "errors that matter";
# D1xx (docstrings) and ANN1xx (full annotations) are skipped per the
# session doc — those are style-only and the polish pass is not about
# style pedantry.
RUFF_SELECT = "E,F,B,UP"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_enabled() -> bool:
    """Check the ``SYNTHESIZER_ROUTE_ENABLED`` env var (default True)."""
    v = os.environ.get("SYNTHESIZER_ROUTE_ENABLED", "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def should_polish(state: dict[str, Any]) -> tuple[bool, str]:
    """Return ``(polish?, reason)`` given the current pipeline state.

    Reason string is for logging — human-readable single line.
    """
    # Master switch — if disabled, always polish (matches pre-session-4 behaviour).
    if not route_enabled():
        return True, "route-disabled (SYNTHESIZER_ROUTE_ENABLED=0) — polish always"

    # Wallclock budget — hard suppressor.
    wallclock_s = _wallclock_so_far(state)
    if wallclock_s >= WALLCLOCK_BUDGET_S:
        return False, (
            f"skip-polish: wallclock={wallclock_s:.0f}s ≥ budget={WALLCLOCK_BUDGET_S:.0f}s "
            "(over budget, polish cost outweighs expected benefit)"
        )

    # Trigger 1 — tests failed.
    tests_failed = _tests_failed(state)
    if tests_failed > 0:
        return True, f"polish: tests_failed={tests_failed}"

    # Trigger 2 — ruff errors.
    code_files = state.get("code_files") or {}
    try:
        ruff_err = _count_ruff_errors(code_files)
    except RuffInvocationError as e:
        # Fail *safe*, not *open*: if ruff itself broke, we can't claim
        # the code is clean.  Route toward polish and surface the reason
        # in the log so operators can fix the cache/install problem.
        logger.warning("ruff invocation failed; routing toward polish: %s", e)
        return True, f"polish: ruff-internal-error ({type(e).__name__})"
    if ruff_err > RUFF_ERROR_THRESHOLD:
        return True, f"polish: ruff_errors={ruff_err} > {RUFF_ERROR_THRESHOLD}"

    # Trigger 3 — cyclomatic complexity.
    max_cc = _max_cyclomatic_complexity(code_files)
    if max_cc is not None and max_cc > CYCLOMATIC_COMPLEXITY_THRESHOLD:
        return True, f"polish: max_cc={max_cc} > {CYCLOMATIC_COMPLEXITY_THRESHOLD}"

    # Trigger 4 — lines added.
    lines = _lines_added(state)
    if lines > LINES_ADDED_THRESHOLD:
        return True, f"polish: lines_added={lines} > {LINES_ADDED_THRESHOLD}"

    return False, (
        f"skip-polish: tests_passed, ruff_err={ruff_err}, max_cc={max_cc}, "
        f"lines={lines}, wallclock={wallclock_s:.0f}s — nothing to polish"
    )


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------


def _tests_failed(state: dict[str, Any]) -> int:
    """Extract tests_failed from execution_result in a shape-tolerant way."""
    exec_r = state.get("execution_result")
    if exec_r is None:
        return 0
    if isinstance(exec_r, dict):
        failed = exec_r.get("tests_failed", 0)
    else:
        failed = getattr(exec_r, "tests_failed", 0)
    try:
        return int(failed or 0)
    except (TypeError, ValueError):
        return 0


def _lines_added(state: dict[str, Any]) -> int:
    """Total lines across ``state['code_files']``.  Proxy for "how much
    code did we generate this build" — there's no real diff against a
    base because the builder generates from scratch.
    """
    code_files = state.get("code_files") or {}
    total = 0
    for content in code_files.values():
        if isinstance(content, str):
            total += content.count("\n") + 1
    return total


def _wallclock_so_far(state: dict[str, Any]) -> float:
    """Sum of per-agent durations from ``state['agent_timings']``."""
    timings = state.get("agent_timings") or {}
    if not isinstance(timings, dict):
        return 0.0
    total = 0.0
    for v in timings.values():
        try:
            total += float(v or 0.0)
        except (TypeError, ValueError):
            continue
    return total


class RuffInvocationError(RuntimeError):
    """Raised when ruff itself fails to run (exit 2), not when it finds errors.

    Session 0 (v3.2) audit finding: the router was silently fail-opening
    when ruff's cache dir was unwritable — exit 2 was indistinguishable
    from exit 0 (clean code). That routed builds *away* from the
    synthesizer when they should have been routed *toward* it. Raise
    loudly so operators fix the cache problem instead of shipping
    unpolished code they think was polished.
    """


def _count_ruff_errors(code_files: dict[str, str]) -> int:
    """Count ruff errors across every .py file.

    Uses ruff's JSON output for robust parsing. Returns 0 on zero
    findings; raises :class:`RuffInvocationError` when ruff itself
    fails to run (exit code 2). Missing ruff binary or timeout still
    fail-opens (returns 0) — those are environment issues, not
    configuration bugs.

    Session 0 (v3.2) hardening:
    * ``--no-cache`` — don't rely on .ruff_cache being writable. The
      router runs once per build so the cache adds no value and has
      caused silent fail-open in directories with permission quirks.
    * exit code 2 → raise. ruff's "internal error" return code was
      being treated as "clean file" because stdout was empty.
    """
    import shutil as _shutil
    ruff = _shutil.which("ruff")
    if ruff is None:
        logger.debug("ruff missing; treating as 0 errors (cannot gate on this signal)")
        return 0
    py_files = {n: c for n, c in code_files.items() if n.endswith(".py") and isinstance(c, str)}
    if not py_files:
        return 0

    # Write files to a temp dir so we can run ruff against them as a
    # group (faster than one subprocess per file).
    try:
        with tempfile.TemporaryDirectory(prefix="belief_router_ruff_") as td:
            td_path = Path(td)
            for fname, content in py_files.items():
                p = td_path / Path(fname).name
                p.write_text(content)
            proc = subprocess.run(
                [
                    ruff, "check",
                    "--no-cache",
                    "--select", RUFF_SELECT,
                    "--output-format", "json",
                    str(td_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ruff check timed out/failed: %s", e)
        return 0

    # ruff exit codes:
    #   0 = no errors (stdout may be empty or "[]")
    #   1 = errors found (stdout has JSON findings)
    #   2 = internal error (unwritable cache, malformed argv, panic, …)
    # Anything ≥ 2 is NOT a signal about the code — it's a signal
    # about our invocation. Fail loud so the caller can fix it instead
    # of polishing-or-not based on noise.
    if proc.returncode >= 2:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or ["<no stderr>"]
        raise RuffInvocationError(
            f"ruff exited {proc.returncode} (internal error, not a lint finding). "
            f"argv={[ruff, 'check', '--no-cache', '--select', RUFF_SELECT, '--output-format', 'json', str(td_path)]!r}; "
            f"stderr tail: {stderr_tail[-1]}"
        )

    if not proc.stdout:
        return 0
    try:
        import json as _json
        findings = _json.loads(proc.stdout)
        return len(findings) if isinstance(findings, list) else 0
    except Exception as e:
        logger.debug("ruff output parse failed: %s", e)
        return 0


def _max_cyclomatic_complexity(code_files: dict[str, str]) -> int | None:
    """Maximum cyclomatic complexity across every function in every
    .py file, computed via ``radon cc -s -j``.  Returns None if radon
    is missing or the sweep fails.
    """
    import shutil as _shutil
    radon = _shutil.which("radon")
    if radon is None:
        logger.debug("radon missing; skipping cyclomatic complexity signal")
        return None
    py_files = {n: c for n, c in code_files.items() if n.endswith(".py") and isinstance(c, str)}
    if not py_files:
        return None

    max_cc = 0
    try:
        with tempfile.TemporaryDirectory(prefix="belief_router_radon_") as td:
            td_path = Path(td)
            for fname, content in py_files.items():
                p = td_path / Path(fname).name
                p.write_text(content)
            proc = subprocess.run(
                [radon, "cc", "-s", "-j", str(td_path)],
                capture_output=True, text=True, timeout=10,
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("radon cc timed out/failed: %s", e)
        return None

    if not proc.stdout:
        return 0
    try:
        import json as _json
        data = _json.loads(proc.stdout)
    except Exception as e:
        logger.debug("radon output parse failed: %s", e)
        return None

    if not isinstance(data, dict):
        return 0
    for _fname, blocks in data.items():
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict):
                cc = block.get("complexity")
                try:
                    max_cc = max(max_cc, int(cc or 0))
                except (TypeError, ValueError):
                    continue
    return max_cc


__all__ = [
    "CYCLOMATIC_COMPLEXITY_THRESHOLD",
    "LINES_ADDED_THRESHOLD",
    "RUFF_ERROR_THRESHOLD",
    "RuffInvocationError",
    "WALLCLOCK_BUDGET_S",
    "route_enabled",
    "should_polish",
]
