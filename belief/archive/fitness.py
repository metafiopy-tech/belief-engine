"""SICA-style scalar utility function — Session 6 (v3.2).

Per Robeyns et al. (arXiv:2504.15228), a self-improving agent needs
a single scalar to rank candidates.  Without a scalar, there's no
selection pressure and the archive is noise.

The default weights are one opinionated balance of quality / cost /
latency / rule-fidelity.  Override via env vars (see below) if your
priorities differ from mine — they're MY values for THIS project,
not laws of physics.
"""

from __future__ import annotations

import os
from typing import Any

from belief.archive.outcome import BuildOutcome


# Default weights — must sum to 1.0.  Override per-weight via env var.
_DEFAULT_WEIGHTS = {
    "U_W_SCORE": 0.5,      # weighted_score (tests passed)
    "U_W_COST": 0.2,       # 1 - normalised cost (lower is better)
    "U_W_TIME": 0.15,      # 1 - normalised wallclock
    "U_W_COVENANT": 0.15,  # covenant-fidelity rate
}

# Normalisation ceilings.  Cost normalises against a $10/build budget
# (set by the CLI's default), time against a 10-minute ceiling.
_COST_CEILING_USD = 10.0
_TIME_CEILING_S = 600.0


def _weight(key: str) -> float:
    """Resolve a weight from env var or fall back to default."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return _DEFAULT_WEIGHTS[key]
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_WEIGHTS[key]


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def covenant_rate(outcome: BuildOutcome, expected_covenants: int = 7) -> float:
    """Fraction of expected covenants that fired without violation.

    We can't see "expected" covenants directly from the outcome —
    the minimum we can assume is 7 (the static set in session 1/2).
    Callers that know the dynamic covenant count can pass it in.
    """
    if expected_covenants <= 0:
        return 1.0
    violations = len(outcome.covenant_violations)
    hits = max(0, expected_covenants - violations)
    return _clip01(hits / expected_covenants)


def utility(
    outcome: BuildOutcome,
    *,
    expected_covenants: int = 7,
) -> float:
    """SICA-style scalar U(outcome) in [0, 1].

    Higher is better.  Breakdown:

    * ``U_W_SCORE × weighted_score`` — raw quality signal.
    * ``U_W_COST × (1 - cost/10)`` — cheaper is better, clipped at $10.
    * ``U_W_TIME × (1 - wallclock/600)`` — faster is better, clipped
      at 10 minutes.
    * ``U_W_COVENANT × covenant_rate`` — fraction of covenants that
      fired cleanly.

    Callers can override any weight via env vars U_W_SCORE,
    U_W_COST, U_W_TIME, U_W_COVENANT.
    """
    w_score = _weight("U_W_SCORE")
    w_cost = _weight("U_W_COST")
    w_time = _weight("U_W_TIME")
    w_cov = _weight("U_W_COVENANT")

    quality = _clip01(outcome.weighted_score)
    cost_term = _clip01(1.0 - outcome.estimated_cost_usd / _COST_CEILING_USD)
    time_term = _clip01(1.0 - outcome.wallclock_s / _TIME_CEILING_S)
    cov_term = covenant_rate(outcome, expected_covenants=expected_covenants)

    u = (
        w_score * quality
        + w_cost * cost_term
        + w_time * time_term
        + w_cov * cov_term
    )
    return _clip01(u)


__all__ = ["covenant_rate", "utility"]
