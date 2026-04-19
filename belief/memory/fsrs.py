"""
FSRS-4.5 — Free Spaced Repetition Scheduler for nutrient memory decay.

Standalone implementation of the FSRS-4.5 algorithm adapted for the
Belief Engine's metabolization architecture.  Nutrients (patterns,
covenants, etc.) are "reviewed" each time they are successfully reused
(reinforce) or cause a failure (lapse).  FSRS tracks per-nutrient
stability and difficulty so the recomposer can prioritise fresh,
reliable knowledge and let stale or unreliable knowledge decay.

Core concepts:
  stability  — days until retrievability drops to 90 %
  difficulty — 1.0 (trivial) to 10.0 (architectural)
  retrievability — probability the nutrient is still valid right now
  grade — 1=again, 2=hard, 3=good, 4=easy

No external dependencies beyond stdlib + dataclasses.

Reference: https://github.com/open-spaced-repetition/fsrs4anki
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


# ── Pure functions ──────────────────────────────────────────────────────────


def retrievability(stability: float, elapsed_days: float) -> float:
    """Probability that a nutrient is still retrievable after *elapsed_days*.

    Uses the FSRS power-law forgetting curve:

        R(t, S) = (1 + 19/81 * t / S) ^ (-0.5)

    Properties:
      - R(0, S) = 1.0   (just reviewed)
      - R(S, S) ~ 0.9   (at stability boundary)
      - Monotonically decreasing in *elapsed_days*

    Args:
        stability:    Days until retrievability drops to ~90 %.
        elapsed_days: Days since last review.

    Returns:
        Float in [0.0, 1.0].
    """
    if stability <= 0:
        return 0.0
    if elapsed_days <= 0:
        return 1.0
    inner = 1.0 + (19.0 / 81.0) * elapsed_days / stability
    return inner ** (-0.5)


def update_stability_on_success(s: float, d: float, r: float) -> float:
    """Grow stability after a successful review.

    Formula:
        S' = S * (1 + e^0.1 * (11 - D) * S^(-0.2) * (e^((1-R)*0.9) - 1))

    Higher existing stability gives smaller proportional gain (S^-0.2).
    Lower retrievability (longer gap) gives a bigger spacing-effect boost.
    Lower difficulty accelerates growth.

    Args:
        s: Current stability (days).
        d: Current difficulty (1.0-10.0).
        r: Current retrievability (0.0-1.0).

    Returns:
        New stability (always >= *s*).
    """
    growth = (
        math.exp(0.1)
        * (11.0 - d)
        * (s ** (-0.2))
        * (math.exp((1.0 - r) * 0.9) - 1.0)
    )
    return s * (1.0 + growth)


def update_stability_on_failure(s: float, d: float, r: float) -> float:
    """Collapse stability after a lapse (failed review).

    Formula:
        S' = max(0.1, S * D^(-0.15) * ((S+1)^0.15 - 1) * e^((1-R)*0.35))

    Stability drops significantly but never below 0.1 days — a pattern
    with 20 successes and 1 failure should retain some credibility.

    Args:
        s: Current stability (days).
        d: Current difficulty (1.0-10.0).
        r: Current retrievability (0.0-1.0).

    Returns:
        New stability (>= 0.1).
    """
    new_s = (
        s
        * (d ** (-0.15))
        * ((s + 1.0) ** 0.15 - 1.0)
        * math.exp((1.0 - r) * 0.35)
    )
    return max(0.1, new_s)


def update_difficulty(d: float, grade: int) -> float:
    """Adjust difficulty towards the mean based on review grade.

    Formula:
        D' = clamp(D + 0.1 * (grade - 3), 1.0, 10.0)

    Grade 3 ("good") leaves difficulty unchanged.  Lower grades push
    difficulty up; higher grades push it down.

    Args:
        d:     Current difficulty (1.0-10.0).
        grade: Review grade (1=again, 2=hard, 3=good, 4=easy).

    Returns:
        New difficulty clamped to [1.0, 10.0].
    """
    return max(1.0, min(10.0, d + 0.1 * (grade - 3)))


def schedule_next_review(stability: float, desired_retention: float = 0.9) -> float:
    """Days until the next review should occur to maintain *desired_retention*.

    Inverts the retrievability formula to find the interval at which
    R drops to *desired_retention*:

        interval = S * (R_target^(-1/0.5) - 1) * 81/19

    Args:
        stability:         Current stability (days).
        desired_retention: Target retrievability (default 0.9).

    Returns:
        Days until next review (always positive).
    """
    if stability <= 0:
        return 0.0
    if desired_retention <= 0 or desired_retention >= 1.0:
        return stability  # Fallback
    return stability * (desired_retention ** (-1.0 / 0.5) - 1.0) * (81.0 / 19.0)


# ── Stateful container ──────────────────────────────────────────────────────


@dataclass
class FSRSState:
    """Mutable state for one item tracked by FSRS.

    Fields:
        stability:   Days until retrievability drops to ~90 %.
        difficulty:  1.0 (trivial idiom) to 10.0 (architectural pattern).
        reps:        Total successful reviews.
        lapses:      Total failed reviews (grade 1).
        last_review: Timestamp of last review (None if never reviewed).
        next_review: Scheduled timestamp for next review (None if unscheduled).
        decay_state: Lifecycle phase — one of:
                     "new"      — never reviewed
                     "learning" — fewer than 3 successful reviews
                     "stable"   — 3+ successful reviews, no recent lapse
                     "lapsed"   — failed a review since becoming stable
    """

    stability: float = 1.0
    difficulty: float = 5.0
    reps: int = 0
    lapses: int = 0
    last_review: Optional[datetime] = None
    next_review: Optional[datetime] = None
    decay_state: str = "new"


def review(state: FSRSState, grade: int, now: Optional[datetime] = None) -> FSRSState:
    """Perform a review and return the updated state.

    Applies the FSRS-4.5 update rules:
      1. Compute elapsed days since last review.
      2. Compute current retrievability.
      3. Update stability (success or failure path).
      4. Update difficulty.
      5. Increment reps or lapses.
      6. Schedule next review.
      7. Transition decay_state.

    Args:
        state: Current FSRSState (not mutated).
        grade: 1=again, 2=hard, 3=good, 4=easy.
        now:   Override for current time (defaults to UTC now).

    Returns:
        A *new* FSRSState with all fields updated.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    grade = max(1, min(4, grade))

    # Elapsed time
    if state.last_review is not None:
        elapsed = max(0.0, (now - state.last_review).total_seconds() / 86400.0)
    else:
        elapsed = 0.0

    # Current retrievability
    r = retrievability(state.stability, elapsed)

    # Stability update
    if grade == 1:
        # Failure
        new_stability = update_stability_on_failure(state.stability, state.difficulty, r)
        new_lapses = state.lapses + 1
        new_reps = state.reps
    else:
        # Success (grade 2, 3, or 4)
        new_stability = update_stability_on_success(state.stability, state.difficulty, r)
        new_lapses = state.lapses
        new_reps = state.reps + 1

    # Difficulty update
    new_difficulty = update_difficulty(state.difficulty, grade)

    # Schedule
    interval_days = schedule_next_review(new_stability)
    new_next_review = now + timedelta(days=interval_days)

    # Decay state transitions
    if grade == 1:
        if state.decay_state in ("stable", "learning"):
            new_decay_state = "lapsed"
        else:
            new_decay_state = state.decay_state  # stay in current
    else:
        if state.decay_state == "new":
            new_decay_state = "learning"
        elif state.decay_state == "learning":
            new_decay_state = "stable" if new_reps >= 3 else "learning"
        elif state.decay_state == "lapsed":
            new_decay_state = "learning"  # recovery
        else:
            new_decay_state = state.decay_state  # stay stable

    return FSRSState(
        stability=new_stability,
        difficulty=new_difficulty,
        reps=new_reps,
        lapses=new_lapses,
        last_review=now,
        next_review=new_next_review,
        decay_state=new_decay_state,
    )
