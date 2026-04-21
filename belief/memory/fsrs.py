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

Session 13 extension: clade-productivity weighting.  A nutrient's
retention is proportional to how often its DESCENDANTS succeed in
future builds.  ``clade_productivity()`` walks the lineage graph and
returns a generative-value score; ``review()`` accepts an optional
``productivity`` argument that scales stability growth, so nutrients
whose descendants succeed retain longer even without direct access.

No external dependencies beyond stdlib + dataclasses.  ``soil`` is
duck-typed (any object exposing ``iter_all_nutrients()``) so fsrs.py
stays free of ChromaDB/Pydantic imports and remains importable in
the sandbox.

Reference: https://github.com/open-spaced-repetition/fsrs4anki
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional


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


def update_stability_on_success(
    s: float,
    d: float,
    r: float,
    productivity_weight: float = 1.0,
) -> float:
    """Grow stability after a successful review.

    Formula:
        S' = S * (1 + e^0.1 * (11 - D) * S^(-0.2) * (e^((1-R)*0.9) - 1) * W)

    Higher existing stability gives smaller proportional gain (S^-0.2).
    Lower retrievability (longer gap) gives a bigger spacing-effect boost.
    Lower difficulty accelerates growth.  The ``productivity_weight``
    (``W``) multiplies the growth term and is supplied by the
    clade-productivity extension (Session 13) — default ``1.0`` keeps
    the classical FSRS-4.5 behaviour.

    Args:
        s: Current stability (days).
        d: Current difficulty (1.0-10.0).
        r: Current retrievability (0.0-1.0).
        productivity_weight: Multiplier on the stability growth term
            (clamped to ``>= 0.0``).  ``1.0`` = classical FSRS, higher
            values amplify growth for generatively-valuable nutrients.

    Returns:
        New stability (always >= *s*).
    """
    w = max(0.0, float(productivity_weight))
    growth = (
        math.exp(0.1)
        * (11.0 - d)
        * (s ** (-0.2))
        * (math.exp((1.0 - r) * 0.9) - 1.0)
    )
    return s * (1.0 + growth * w)


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


def _productivity_to_weight(productivity: float) -> float:
    """Convert a raw clade-productivity score into a stability-growth multiplier.

    Raw productivity is an unbounded score ``sum(success_rate * use_count) / n``
    which grows with descendant activity.  We map it to a bounded
    multiplier in ``[1.0, 3.0]``:

        weight = 1.0 + min(max(productivity, 0.0), 10.0) * 0.2

    ``productivity == 0`` gives ``1.0`` (no effect, classical FSRS).
    ``productivity == 10`` (heavy descendant activity) gives the cap
    at ``3.0`` — triple stability growth.

    The cap keeps a runaway nutrient from dominating soil retention.
    """
    p = max(0.0, float(productivity))
    return 1.0 + min(p, 10.0) * 0.2


def review(
    state: FSRSState,
    grade: int,
    now: Optional[datetime] = None,
    productivity: float = 0.0,
) -> FSRSState:
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
        state:        Current FSRSState (not mutated).
        grade:        1=again, 2=hard, 3=good, 4=easy.
        now:          Override for current time (defaults to UTC now).
        productivity: Clade-productivity score for this nutrient.  Scales
            stability growth on success — high-productivity nutrients
            retain longer (Session 13).  Default ``0.0`` preserves the
            classical FSRS-4.5 behaviour.  Lapses are intentionally
            *not* softened by productivity: a failing nutrient whose
            descendants work is still a failure.

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
        # Failure — productivity intentionally ignored (no reward for bad nutrients).
        new_stability = update_stability_on_failure(state.stability, state.difficulty, r)
        new_lapses = state.lapses + 1
        new_reps = state.reps
    else:
        # Success (grade 2, 3, or 4) — scale growth by clade productivity.
        weight = _productivity_to_weight(productivity)
        new_stability = update_stability_on_success(
            state.stability, state.difficulty, r, productivity_weight=weight
        )
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


# ── Clade-productivity weighting (Session 13) ──────────────────────────────
#
# Nutrients form a lineage DAG: each nutrient lists the IDs of the nutrients
# it inherited from (``lineage_parent_ids``).  The "clade" rooted at a given
# nutrient is the set of its descendants — nutrients that list this
# nutrient's ID anywhere in their provenance.  A nutrient is generatively
# valuable if its descendants keep succeeding in future builds.
#
# ``clade_productivity`` quantifies this: for each descendant we compute
# ``success_rate * use_count`` (``success_rate = reps / (reps+lapses)`` with
# a zero-division guard; ``use_count = reps + lapses``), sum those, and
# divide by the number of descendants.  The result is a non-negative score
# that plugs into ``review()``'s ``productivity`` argument so stability
# growth scales with generative value.


def _successor_score(descendant: Any) -> float:
    """Compute ``success_rate * use_count`` for a descendant nutrient.

    Duck-typed on ``reinforcement_count`` / ``lapse_count`` so tests can
    pass simple fakes and production passes :class:`Nutrient` instances.
    Missing fields fall back to ``0``.
    """
    reps = float(getattr(descendant, "reinforcement_count", 0) or 0)
    lapses = float(getattr(descendant, "lapse_count", 0) or 0)
    use_count = reps + lapses
    if use_count <= 0:
        return 0.0
    success_rate = reps / use_count
    return success_rate * use_count  # = reps (algebraically), but keep shape


def _descendant_parent_ids(descendant: Any) -> Iterable[str]:
    """Best-effort extraction of a descendant's parent-ID list."""
    parents = getattr(descendant, "lineage_parent_ids", None) or ()
    return parents


def _collect_descendants(
    root_id: str, soil: Any, _seen: Optional[set] = None
) -> list[Any]:
    """Return all descendants of ``root_id`` (transitive, deduped).

    Walks the lineage DAG breadth-first.  ``soil`` must expose
    ``iter_all_nutrients()`` returning an iterable of nutrient-like
    objects with ``nutrient_id`` and ``lineage_parent_ids`` attributes.
    """
    nutrients = list(soil.iter_all_nutrients())
    # Build parent -> children index once so lookups are O(|nutrients|) total
    # rather than O(|nutrients|^2) for the BFS below.
    children_by_parent: dict[str, list[Any]] = {}
    for n in nutrients:
        nid = getattr(n, "nutrient_id", None)
        if nid is None:
            continue
        for parent_id in _descendant_parent_ids(n):
            children_by_parent.setdefault(parent_id, []).append(n)

    descendants: list[Any] = []
    seen: set = _seen if _seen is not None else set()
    frontier = list(children_by_parent.get(root_id, ()))
    while frontier:
        node = frontier.pop(0)
        nid = getattr(node, "nutrient_id", None)
        if nid is None or nid in seen or nid == root_id:
            continue
        seen.add(nid)
        descendants.append(node)
        frontier.extend(children_by_parent.get(nid, ()))
    return descendants


def clade_productivity(
    nutrient_id: str,
    soil: Any,
    cache: Optional[dict] = None,
) -> float:
    """Compute how generatively valuable this nutrient is.

    Walks the lineage graph: for each descendant (nutrients that list
    this nutrient's ID anywhere in their ``lineage_parent_ids``), compute
    ``success_rate * use_count``.  The productivity score is::

        productivity = sum(descendant.success_rate * descendant.use_count)
                       / max(1, total_descendants)

    where ``success_rate = reinforcement_count / (reinforcement_count +
    lapse_count)`` (zero when neither has occurred) and
    ``use_count = reinforcement_count + lapse_count``.

    Args:
        nutrient_id: The root nutrient whose clade we score.
        soil:        Any object with ``iter_all_nutrients()``.  In
                     production this is a :class:`~belief.memory.soil.Soil`.
        cache:       Optional ``dict`` for memoisation across repeated
                     calls with the same ``soil`` snapshot.  If ``None``
                     no caching is applied.

    Returns:
        Non-negative float.  ``0.0`` when the nutrient has no
        descendants or when all descendants have zero usage.
    """
    if cache is not None and nutrient_id in cache:
        return cache[nutrient_id]

    descendants = _collect_descendants(nutrient_id, soil)
    if not descendants:
        result = 0.0
    else:
        weighted = sum(_successor_score(d) for d in descendants)
        result = weighted / max(1, len(descendants))

    if cache is not None:
        cache[nutrient_id] = result
    return result


def compute_clade_productivity_map(soil: Any) -> dict[str, float]:
    """Compute clade productivity for every nutrient in a single pass.

    More efficient than calling :func:`clade_productivity` per id when
    many nutrients need scoring (e.g. during a soil maintenance cycle):
    the descendants index is built once and reused.

    Args:
        soil: Any object with ``iter_all_nutrients()``.

    Returns:
        Dict ``{nutrient_id: productivity}``.  Roots with no descendants
        map to ``0.0``.
    """
    nutrients = list(soil.iter_all_nutrients())
    # Parent -> direct-children index
    children_by_parent: dict[str, list[Any]] = {}
    for n in nutrients:
        nid = getattr(n, "nutrient_id", None)
        if nid is None:
            continue
        for parent_id in _descendant_parent_ids(n):
            children_by_parent.setdefault(parent_id, []).append(n)

    scores: dict[str, float] = {}
    for n in nutrients:
        root_id = getattr(n, "nutrient_id", None)
        if root_id is None:
            continue
        # BFS from this root using the shared index.
        seen: set = set()
        descendants: list[Any] = []
        frontier = list(children_by_parent.get(root_id, ()))
        while frontier:
            node = frontier.pop(0)
            nid = getattr(node, "nutrient_id", None)
            if nid is None or nid in seen or nid == root_id:
                continue
            seen.add(nid)
            descendants.append(node)
            frontier.extend(children_by_parent.get(nid, ()))
        if not descendants:
            scores[root_id] = 0.0
        else:
            weighted = sum(_successor_score(d) for d in descendants)
            scores[root_id] = weighted / max(1, len(descendants))
    return scores
