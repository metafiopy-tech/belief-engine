"""Cross-domain novelty gate (SE Session 6).

Thin wrapper around :meth:`BiologicalPrimitiveStore.novelty_score`
that turns a continuous similarity score into a discrete cycle-level
gate decision. Called from ``cycle.py`` immediately before
``write_session`` -- if the candidate mechanism is too close to one
already in the bio store, the cycle aborts with reason
``cross_domain_redundant`` and the bundle's source rows get marked
``rejected`` rather than promoted.

The default threshold is 0.30 -- novelty < 0.30 means the candidate
is at least 70% similar to its nearest neighbor, which is high enough
that the synthesizer almost certainly re-derived a known mechanism
rather than discovering something new. Tunable per-call via the
``threshold`` kwarg.

Out of scope for Session 6:
  - Adaptive thresholds based on bio_store size / age.
  - Per-domain thresholds (e.g. "biology+computing" should be more
    permissive than "biology+biology").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from belief.photosynthesis.synthesis.structural_mechanism import StructuralMechanism


logger = logging.getLogger("belief.photosynthesis.synthesis.cross_domain_novelty")


DEFAULT_NOVELTY_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class NoveltyVerdict:
    """Outcome of the novelty gate."""

    accepted: bool
    novelty_score: float
    threshold: float
    reason: str = ""

    @property
    def rejected(self) -> bool:
        return not self.accepted

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "novelty_score": self.novelty_score,
            "threshold": self.threshold,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gate(
    mechanism: StructuralMechanism,
    *,
    bio_store: Any,
    threshold: float = DEFAULT_NOVELTY_THRESHOLD,
) -> NoveltyVerdict:
    """Run the novelty gate.

    Returns a :class:`NoveltyVerdict`. ``accepted=True`` when
    ``bio_store.novelty_score(mechanism) >= threshold``. When the
    bio_store is None or the score query raises, accept by default --
    the gate must not block synthesis on its own infrastructure
    failures.
    """
    if bio_store is None:
        return NoveltyVerdict(
            accepted=True,
            novelty_score=1.0,
            threshold=threshold,
            reason="bio_store_unavailable",
        )
    try:
        score = float(bio_store.novelty_score(mechanism))
    except Exception as exc:
        logger.warning("novelty_score query failed (accepting by default): %s", exc)
        return NoveltyVerdict(
            accepted=True,
            novelty_score=1.0,
            threshold=threshold,
            reason=f"bio_store_error:{type(exc).__name__}",
        )

    score = max(0.0, min(1.0, score))
    if score >= threshold:
        return NoveltyVerdict(
            accepted=True,
            novelty_score=score,
            threshold=threshold,
            reason="novel",
        )
    return NoveltyVerdict(
        accepted=False,
        novelty_score=score,
        threshold=threshold,
        reason="cross_domain_redundant",
    )


__all__ = [
    "DEFAULT_NOVELTY_THRESHOLD",
    "NoveltyVerdict",
    "gate",
]
