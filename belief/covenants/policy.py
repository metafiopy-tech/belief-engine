"""Session 8 (v3.2) — covenant proposer policy thresholds.

Centralised so test code and the CLI share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatePolicy:
    """Precision-gate thresholds for a covenant proposal.

    All four must be satisfied for a proposal to be marked
    ``auto_pass``; anything failing gets ``auto_fail`` plus the
    measured metrics so a human can review why.
    """

    min_would_have_prevented: int = 5   # past failures the rule would have caught
    max_would_have_broken: int = 0      # past passing builds the rule would break
    min_precision: float = 1.0          # prevented / (prevented + broken)
    min_cluster_size: int = 5           # clusterer's minimum samples


DEFAULT_POLICY = GatePolicy()


__all__ = ["GatePolicy", "DEFAULT_POLICY"]
