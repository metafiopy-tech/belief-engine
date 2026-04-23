"""
Danger-theory self-modification gate (Session 16).

Borrowed from immunology: the classical model said the immune system
distinguishes *self* from *non-self*; Matzinger's danger theory says
it actually responds to *tissue damage* — it only mounts an attack
when something is clearly going wrong nearby.

Applied to the Belief Engine: self-modifications (SICA proposals,
NEW_TOOL suggestions, jitterbug integrations) should only fire when
three conditions are simultaneously true — "tissue damage" is real,
localized, and trending worse:

    1. Test failures are LOCALIZED to the target module.
       A broad wave of failures across the whole codebase isn't a
       signal about *this* module — it's a signal about the
       environment (bad commit elsewhere, flaky CI, a dep breakage).
    2. Uncertainty (confidence-probe output) is RISING in this area.
       Flat or falling uncertainty means the engine's predictions
       are improving on their own; no intervention needed.
    3. The module is NOT in :data:`CRITICAL_FILES`.
       Even with perfect signals we refuse to let the system auto-
       mutate its benchmark, its hardening primitives, or anything
       else the CLAUDE.md marks as off-limits.

The gate is *advisory*, not blocking (Session-16 constraint).
Callers log the verdict and defer (never reject outright) so a
proposal that's borderline today can resurface when the signals
catch up to it.

No ChromaDB, no network, pure stdlib.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

logger = logging.getLogger("belief.safety.danger_theory")


# ── CRITICAL_FILES ─────────────────────────────────────────────────────────


# Files the engine is **never** allowed to auto-modify.  The top-of-file
# project instructions list ``benchmark.py`` and ``hardening.py``
# explicitly; :data:`CRITICAL_FILES` is a superset that also covers the
# safety primitives themselves so the immune system can't disable
# itself in a self-modification cycle.  Extend this list — don't
# shrink it — when in doubt.
CRITICAL_FILES: frozenset[str] = frozenset(
    {
        # Explicit constraints from CLAUDE.md
        "belief/benchmark.py",
        "belief/hardening.py",
        # Safety-critical primitives — the immune system mustn't edit
        # its own detectors.
        "belief/safety/overseer.py",
        "belief/safety/probes.py",
        "belief/safety/goodhart_canary.py",
        "belief/safety/danger_theory.py",
        "belief/safety/pheromones.py",
        # Evaluation/scoring: editing these would let SICA Goodhart its
        # own benchmark.
        "belief/evolution/archive.py",
        "belief/evolution/cascade.py",
    }
)


def _normalize_path(module_path: str) -> str:
    """Canonicalise a path for CRITICAL_FILES / failure-trace lookup.

    Accepts absolute paths, repo-relative paths, or module notation
    (``belief.benchmark`` → ``belief/benchmark.py``).  Trailing
    whitespace and leading ``./`` are stripped; backslashes are
    converted to forward slashes so the comparison is platform
    independent.
    """
    if not module_path:
        return ""
    p = str(module_path).strip().replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    # If it's in dotted notation with no slashes, convert the dots
    # that precede the final path element.  We don't touch paths
    # that clearly already have slashes.
    if "/" not in p and p.endswith(".py") is False and "." in p:
        # belief.benchmark → belief/benchmark.py
        p = p.replace(".", "/") + ".py"
    return p


def is_critical(module_path: str) -> bool:
    """Whether ``module_path`` is in the do-not-auto-modify list.

    Matches by canonical suffix — callers may pass an absolute path,
    a repo-relative path, or a dotted module name; all three resolve
    to the same critical-file slug.
    """
    canonical = _normalize_path(module_path)
    if not canonical:
        return False
    if canonical in CRITICAL_FILES:
        return True
    # Suffix match so /Users/foo/Desktop/belief-engine/belief/benchmark.py
    # still resolves.
    return any(canonical.endswith(cf) for cf in CRITICAL_FILES)


# ── Signal 1: localized test failures ─────────────────────────────────────


def _failure_mentions_module(failure: Any, module_path: str) -> bool:
    """Heuristic: does this failure record blame the target module?

    Accepts duck-typed records with any subset of:

        module / module_path / path  — the file the failure pins to
        failing_file / target_file   — aliases used by different runners
        error / error_message / traceback / tb
                                     — free-text; we substring-match

    Returns True when any of those fields match the normalized
    ``module_path``.  The comparison is suffix-based so a long
    traceback path matches a shorter repo-relative one.
    """
    canon = _normalize_path(module_path)
    if not canon:
        return False

    # Direct path-like fields
    for attr in ("module", "module_path", "path", "failing_file", "target_file", "file"):
        value = None
        if isinstance(failure, dict):
            value = failure.get(attr)
        else:
            value = getattr(failure, attr, None)
        if not value:
            continue
        v = _normalize_path(str(value))
        if v == canon or v.endswith(canon) or canon.endswith(v):
            return True

    # Free-text fields (traceback, error_message)
    for attr in ("error", "error_message", "traceback", "tb", "message"):
        value = None
        if isinstance(failure, dict):
            value = failure.get(attr)
        else:
            value = getattr(failure, attr, None)
        if value and canon in str(value):
            return True
    return False


def has_localized_failures(
    module_path: str,
    recent_failures: Iterable[Any],
    min_fraction: float = 0.3,
    min_count: int = 2,
) -> bool:
    """Whether recent failures are concentrated on ``module_path``.

    Localization is defined as two thresholds holding simultaneously:

    * At least ``min_count`` recent failures mention this module (a
      single flake isn't damage).
    * At least ``min_fraction`` (default 30%) of all recent failures
      mention it — rules out the "everything is failing" case where
      no single module is the cause.

    An empty ``recent_failures`` iterable returns False: nothing has
    broken, so there's nothing localized to fix.
    """
    failures = list(recent_failures)
    if not failures:
        return False

    blamed = sum(1 for f in failures if _failure_mentions_module(f, module_path))
    if blamed < min_count:
        return False
    fraction = blamed / len(failures)
    return fraction >= min_fraction


# ── Signal 2: uncertainty rising ──────────────────────────────────────────


def uncertainty_rising(
    trend: Sequence[float],
    min_samples: int = 3,
    min_slope: float = 0.0,
) -> bool:
    """Simple linear regression: slope of ``trend`` vs index must be > 0.

    Uses ordinary least squares on index-vs-value; avoids pulling in
    numpy for a three-line formula.  Returns False when there aren't
    enough samples to fit — the spec is "rising", and you can't call
    a single point a trend.

    Args:
        trend:       Sequence of uncertainty readings (earlier first).
        min_samples: Refuse to judge a trend shorter than this.
        min_slope:   Minimum slope to count as "rising".  Strict ``>``.
    """
    n = len(trend)
    if n < max(2, int(min_samples)):
        return False

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(trend) / n
    num = sum((xs[i] - mean_x) * (trend[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return False
    slope = num / den
    return slope > min_slope


# ── Canonical spec function: is_danger_zone ───────────────────────────────


def is_danger_zone(
    module_path: str,
    recent_failures: list[dict],
    uncertainty_trend: list[float],
    *,
    min_fraction: float = 0.3,
    min_samples: int = 3,
) -> bool:
    """Danger-theory gate — see the module docstring.

    Returns True **only when all three conditions hold**:

        1. ``has_localized_failures(module_path, recent_failures)``
        2. ``uncertainty_rising(uncertainty_trend)``
        3. ``not is_critical(module_path)``

    Kept as a pure function so SICA tests can supply canned signals
    without reaching into live subsystems.
    """
    if is_critical(module_path):
        logger.info(f"danger_theory: {module_path} is CRITICAL — never a danger zone")
        return False
    if not has_localized_failures(
        module_path,
        recent_failures,
        min_fraction=min_fraction,
    ):
        return False
    if not uncertainty_rising(uncertainty_trend, min_samples=min_samples):
        return False
    return True


# ── Combined decision surface (used by SICA / NEW_TOOL / jitterbug) ────────


@dataclass
class DangerSignals:
    """Per-decision snapshot of the three danger-theory inputs.

    Callers assemble this once per SICA iteration (or per self-mod
    attempt) and pass it to :func:`evaluate` which returns a
    ``(allow, reason)`` tuple suitable for logging.
    """

    recent_failures: list[Any] = field(default_factory=list)
    uncertainty_trend: list[float] = field(default_factory=list)
    min_fraction: float = 0.3
    min_samples: int = 3


def evaluate(
    module_path: str,
    signals: DangerSignals,
) -> tuple[bool, str]:
    """Decide whether a self-modification targeting ``module_path`` is justified.

    Returns ``(should_permit, reason)``.  ``should_permit`` is True
    only when the module is in a danger zone; ``reason`` is a
    human-readable log line either way so the caller can surface it
    to the audit log.

    The reason distinguishes the four failure modes:
        * "critical" — module is on the CRITICAL_FILES list
        * "no-localized-failures" — test failures aren't concentrated
        * "no-uncertainty-rise" — probe confidence isn't trending up
        * "danger-zone" — all three signals fire
    """
    if is_critical(module_path):
        return False, f"critical: {module_path} is on CRITICAL_FILES"
    if not has_localized_failures(
        module_path,
        signals.recent_failures,
        min_fraction=signals.min_fraction,
    ):
        return False, (
            f"no-localized-failures: recent test failures are not concentrated on {module_path}"
        )
    if not uncertainty_rising(
        signals.uncertainty_trend,
        min_samples=signals.min_samples,
    ):
        return False, (
            f"no-uncertainty-rise: confidence probe is stable or improving around {module_path}"
        )
    return True, f"danger-zone: self-modification of {module_path} justified"
