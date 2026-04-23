"""Boltzmann-weighted parent sampler — Session 6 (v3.2).

The DGM paper (Zhang et al., arXiv:2505.22954) shows that
parent-selection biased toward high utility but not purely greedy
strictly beats "pick the best every time" — the archive preserves
stepping stones that a greedy selector would discard.

We use a standard Boltzmann (aka softmax-with-temperature) scheme::

    P(i) ∝ exp(U_i / τ)    with τ = 0.2 by default

τ = 0.2 is aggressive enough that the top entries dominate but leaves
noise-floor probability for lower-utility entries to surface
occasionally — enough diversity to matter, not enough to devolve to
random sampling.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any

from belief.archive.fitness import utility
from belief.archive.store import AgentArchive


_DEFAULT_TEMPERATURE = 0.2


def parent_sample(
    goal: str,
    *,
    archive: AgentArchive | None = None,
    k: int = 5,
    temperature: float | None = None,
    rng: random.Random | None = None,
    verdicts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample up to ``k`` past BuildOutcomes as parents for a new build.

    Semantic similarity filters the candidate set; Boltzmann sampling
    orders what remains.  Returns a list of query-result dicts (same
    shape as :meth:`AgentArchive.query_by_goal`) ordered by sample
    probability.

    With τ → 0 the sampling collapses to pure argmax (greedy).
    With τ → ∞ the sampling collapses to uniform random.  τ = 0.2 is
    the empirical sweet spot from the DGM paper's hyperparam sweep.
    """
    if archive is None:
        archive = AgentArchive()
    if temperature is None:
        tau_env = os.environ.get("BELIEF_PARENT_TAU", "").strip()
        try:
            temperature = float(tau_env) if tau_env else _DEFAULT_TEMPERATURE
        except ValueError:
            temperature = _DEFAULT_TEMPERATURE
    if rng is None:
        rng = random.Random()
    if temperature <= 0:
        temperature = 1e-6  # avoid div-by-zero; effectively greedy

    # Pull a larger candidate set (up to 3k) so the Boltzmann pass
    # actually has something to sample from.
    candidates = archive.query_by_goal(goal, k=max(k * 3, 15), verdicts=verdicts)
    if not candidates:
        return []

    # Utility lookup — prefer the one on the BuildOutcome, fall back
    # to the metadata cache (both are stored at persist time).
    def _u(c: dict[str, Any]) -> float:
        outcome = c.get("outcome")
        if outcome is not None:
            return utility(outcome)
        meta = c.get("metadata") or {}
        try:
            return float(meta.get("utility_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    utilities = [_u(c) for c in candidates]

    # Numerically-stable softmax: subtract max before exp.
    m = max(utilities) if utilities else 0.0
    weights = [math.exp((u - m) / temperature) for u in utilities]
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]

    # Weighted sample WITHOUT replacement — the distribution is small
    # enough that a straightforward loop is fine; a proper Gumbel-Top-K
    # is overkill here.
    chosen: list[int] = []
    pool_indices = list(range(len(candidates)))
    pool_probs = list(probs)
    for _ in range(min(k, len(candidates))):
        s = sum(pool_probs)
        if s <= 0:
            break
        pick = rng.random() * s
        acc = 0.0
        for j, p in enumerate(pool_probs):
            acc += p
            if acc >= pick:
                chosen.append(pool_indices[j])
                pool_indices.pop(j)
                pool_probs.pop(j)
                break

    return [candidates[i] for i in chosen]


__all__ = ["parent_sample"]
