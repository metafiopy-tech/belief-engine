"""Combined-value ranker.

Weighted sum (weights from spec):

    value = 0.40 * novelty
          + 0.35 * zpd_fit
          + 0.15 * coverage_gain
          + 0.10 * source_quality

    coverage_gain = 1 - |seed_tags & top20_archive_tags| / |seed_tags|
    source_quality = log-normalized (stars|citations|score|downloads)

Bittensor bias: if the seed is closer to the 'bittensor_swebench'
centroid (cosine) than a configurable threshold, multiply the final
value by 1.5. The centroid embedding is passed in by the caller (or
None to disable).

Reject if value < 0.45.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


ACCEPT_THRESHOLD = 0.45
BITTENSOR_BIAS_MULTIPLIER = 1.5
BITTENSOR_BIAS_COSINE_CUTOFF = 0.70  # seeds >= this cosine to centroid get the boost


@dataclass
class RankerResult:
    value: float
    accepted: bool
    components: dict[str, float] = field(default_factory=dict)
    bittensor_boosted: bool = False


def source_quality(seed: dict[str, Any]) -> float:
    """Log-normalize the most salient source metric into [0, 1].

    Falls through to 0.5 if no numeric signal is present — mid-value
    is the safest fallback (doesn't spuriously boost or suppress).
    """
    metrics = (
        seed.get("stars"),
        seed.get("citations"),
        seed.get("score"),
        seed.get("downloads"),
    )
    for m in metrics:
        if isinstance(m, (int, float)) and m >= 0:
            # log10(x+1) / log10(10001) gives ~0 at 0, ~0.5 at 100, ~1.0 at 10k+
            return max(0.0, min(1.0, math.log10(float(m) + 1.0) / 4.0))
    return 0.5


def coverage_gain(
    seed_tags: list[str],
    top_archive_tags: list[str],
) -> float:
    """Fraction of seed tags NOT already saturated in the archive.

    Returns 1.0 if seed_tags is empty (no evidence of redundancy, treat
    as full gain). This is deliberately optimistic — a seed without
    tags hasn't proven it's a duplicate either.
    """
    if not seed_tags:
        return 1.0
    seen = set(top_archive_tags or [])
    overlap = sum(1 for t in seed_tags if t in seen)
    return 1.0 - (overlap / len(seed_tags))


def combined_value(
    *,
    novelty: float,
    zpd_fit: float,
    coverage_gain: float,  # noqa: A002
    source_quality: float,  # noqa: A002
    bittensor_cosine: Optional[float] = None,
) -> RankerResult:
    """Run the weighted sum + Bittensor bias + accept gate."""
    # Clamp each component to [0, 1] defensively.
    n = max(0.0, min(1.0, float(novelty)))
    z = max(0.0, min(1.0, float(zpd_fit)))
    cg = max(0.0, min(1.0, float(coverage_gain)))
    sq = max(0.0, min(1.0, float(source_quality)))

    raw = 0.40 * n + 0.35 * z + 0.15 * cg + 0.10 * sq

    boosted = False
    if (
        bittensor_cosine is not None
        and bittensor_cosine >= BITTENSOR_BIAS_COSINE_CUTOFF
    ):
        raw *= BITTENSOR_BIAS_MULTIPLIER
        boosted = True

    # Cap at 1.0 for the accept gate — boost can push above 1, but the
    # heap + generator treat value as a relative ordering, not probability.
    value = min(1.0, raw)
    accepted = value >= ACCEPT_THRESHOLD

    return RankerResult(
        value=value,
        accepted=accepted,
        components={
            "novelty": n,
            "zpd_fit": z,
            "coverage_gain": cg,
            "source_quality": sq,
            "raw": raw,
            "bittensor_cosine": float(bittensor_cosine or 0.0),
        },
        bittensor_boosted=boosted,
    )


__all__ = [
    "ACCEPT_THRESHOLD",
    "BITTENSOR_BIAS_COSINE_CUTOFF",
    "BITTENSOR_BIAS_MULTIPLIER",
    "RankerResult",
    "combined_value",
    "coverage_gain",
    "source_quality",
]
