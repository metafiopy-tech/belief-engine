"""Shared utility scoring for ecology organs (v3.3).

Used by Predator (Session 2) to decide what to soft-tombstone and by
Garbage Collector (Session 2.x) for related decisions. Lives in a shared
module so the formula is one source of truth.

Per spec §3.1, the v1 utility function is::

    utility = w_usage    * usage_count_normalized
            + w_retr     * fsrs_retrievability
            + w_recency  * recency_factor
            - w_failure  * known_failure_rate

Weights are persisted in ``~/.belief-engine/ecology_weights.json`` so
the future Economist self-tuning loop (Session 5) can adjust them.
A malformed/missing file falls back to the spec defaults below.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("belief.ecology.utility")

# ── Defaults (per spec §3.1) ───────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "usage": 0.5,
    "retrievability": 0.3,
    "recency": 0.2,
    "failure": 0.4,  # subtracted, not added
}

# Cap reinforcement_count for normalization. A nutrient reused 10+ times
# is "fully utilized" — further reuse doesn't grow the usage signal.
USAGE_NORM_CAP: int = 10

# Recency falls linearly to zero over this many days since last_reinforced.
RECENCY_HALFLIFE_DAYS: float = 30.0

_DEFAULT_WEIGHTS_PATH = Path.home() / ".belief-engine" / "ecology_weights.json"

_REQUIRED_WEIGHT_KEYS = ("usage", "retrievability", "recency", "failure")


# ── Data ───────────────────────────────────────────────────────────────────


@dataclass
class UtilityBreakdown:
    """Per-component contribution to the final utility score.

    Useful for audit logs and CLI explanations — when Predator
    tombstones something, the breakdown explains *why*.
    """

    nutrient_id: str
    usage: float
    retrievability: float
    recency: float
    failure: float
    total: float


# ── Weight loading ─────────────────────────────────────────────────────────


def load_weights(weights_path: Path | None = None) -> dict[str, float]:
    """Load weights from JSON, falling back to defaults on any error.

    The file shape is::

        {"usage": 0.5, "retrievability": 0.3, "recency": 0.2, "failure": 0.4}

    Missing keys are filled in from defaults. Malformed JSON or wrong
    types log a warning and return defaults entirely. Callers should
    treat the return value as immutable (don't write back through it).
    """
    path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("weights file must contain a JSON object")
        out = dict(DEFAULT_WEIGHTS)
        for key in _REQUIRED_WEIGHT_KEYS:
            if key in raw:
                v = raw[key]
                if not isinstance(v, (int, float)):
                    raise ValueError(f"weight {key!r} must be a number, got {type(v).__name__}")
                out[key] = float(v)
        return out
    except FileNotFoundError:
        return dict(DEFAULT_WEIGHTS)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning(
            "Ecology weights file %s unreadable (%s); using defaults.",
            path,
            e,
        )
        return dict(DEFAULT_WEIGHTS)


# ── Component computations ────────────────────────────────────────────────


def _usage_norm(reinforcement_count: int, cap: int = USAGE_NORM_CAP) -> float:
    """Normalize reinforcement_count to [0, 1] using a hard cap."""
    if reinforcement_count <= 0:
        return 0.0
    return min(1.0, reinforcement_count / float(cap))


def _recency_factor(
    last_reinforced_ts: float,
    halflife_days: float = RECENCY_HALFLIFE_DAYS,
    now_ts: float | None = None,
) -> float:
    """Linear decay from 1.0 at last_reinforced=now to 0.0 at +halflife days.

    Past the halflife, returns 0.0 (not negative). Mirrors the FSRS
    retrievability shape but with a sharp floor — Predator wants a
    distinct "hasn't been touched in a month" signal separate from FSRS.
    """
    now = now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp()
    if last_reinforced_ts <= 0 or halflife_days <= 0:
        return 0.0
    elapsed_days = max(0.0, (now - last_reinforced_ts) / 86400.0)
    if elapsed_days >= halflife_days:
        return 0.0
    return 1.0 - elapsed_days / halflife_days


def _failure_rate(reinforcement_count: int, lapse_count: int) -> float:
    """Empirical failure rate: lapses / (reinforcements + lapses).

    A nutrient with 0 lapses scores 0 here regardless of usage. A
    nutrient with only lapses (rare; should never reinforce in practice)
    scores 1.0.
    """
    total = max(0, reinforcement_count) + max(0, lapse_count)
    if total == 0:
        return 0.0
    return max(0.0, lapse_count) / total


# ── Public scorer ──────────────────────────────────────────────────────────


def compute_utility(
    nutrient,  # belief.memory.nutrients.Nutrient — duck-typed below
    weights: dict[str, float] | None = None,
    now_ts: float | None = None,
) -> UtilityBreakdown:
    """Score a single nutrient. Higher = keep, lower = candidate for prune.

    The function is duck-typed against ``Nutrient``: it reads
    ``nutrient_id``, ``reinforcement_count``, ``lapse_count``,
    ``last_reinforced``, and calls ``nutrient.retrievability()``.
    Anything implementing those works (e.g., test stubs).

    Returns an ``UtilityBreakdown`` so audit logs can show per-component
    contributions, not just a final number.
    """
    w = weights if weights is not None else load_weights()

    usage_n = _usage_norm(int(getattr(nutrient, "reinforcement_count", 0)))
    retr = float(nutrient.retrievability())
    recency = _recency_factor(
        float(getattr(nutrient, "last_reinforced", 0.0)),
        now_ts=now_ts,
    )
    failure = _failure_rate(
        int(getattr(nutrient, "reinforcement_count", 0)),
        int(getattr(nutrient, "lapse_count", 0)),
    )

    total = (
        w["usage"] * usage_n
        + w["retrievability"] * retr
        + w["recency"] * recency
        - w["failure"] * failure
    )
    return UtilityBreakdown(
        nutrient_id=str(getattr(nutrient, "nutrient_id", "?")),
        usage=usage_n,
        retrievability=retr,
        recency=recency,
        failure=failure,
        total=total,
    )
