"""Q-Value Store — MemRL-Style Value-Aware Memory.

Extends ChromaDB retrieval with Q-value scoring. Each stored experience
gets a Q-value that's updated based on whether retrieval led to successful
builds. Two-phase retrieval: semantic filter → value re-rank.

Research basis:
- MemRL (arXiv 2601.03192): 56% relative improvement over RAG
- Update rule: Q_new ← Q_old + α · (reward − Q_old)
- Two-phase: cosine_sim > δ filter, then score = (1−λ)·ẑ_sim + λ·Q̂
- Optimal λ = 0.5 (equal weight to similarity and value)

Usage:
    from belief.memory.q_value_store import QValueStore
    store = QValueStore()
    results = store.retrieve("build a REST API", n=5)
    # After build succeeds/fails:
    store.update_q_values(retrieved_ids, reward=1.0)  # success
    store.update_q_values(retrieved_ids, reward=0.0)  # failure
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.memory.q_value")


@dataclass
class QValueExperience:
    """An experience with Q-value for value-aware retrieval."""
    id: str
    content: str
    q_value: float = 0.5  # Initial Q-value (neutral)
    similarity: float = 0.0  # Cosine similarity to query
    combined_score: float = 0.0  # (1−λ)·sim + λ·Q
    retrieval_count: int = 0
    success_count: int = 0
    tags: list[str] = field(default_factory=list)


class QValueStore:
    """Value-aware memory store using MemRL update rule.

    Wraps ChromaDB soil with Q-value tracking. Experiences that lead
    to successful builds get Q-values pushed toward 1.0, failures
    toward 0.0. Retrieval combines semantic similarity with learned
    Q-values for better-than-RAG performance.
    """

    def __init__(
        self,
        q_values_path: str | Path | None = None,
        alpha: float = 0.3,  # Learning rate for Q-value updates
        lambda_weight: float = 0.5,  # Weight between similarity and Q-value
    ):
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self._q_values: dict[str, float] = {}
        self._retrieval_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}

        # Persist Q-values to disk
        if q_values_path is None:
            q_values_path = Path("~/.belief-engine/q_values.json").expanduser()
        self._path = Path(q_values_path)
        self._load()

    def retrieve(
        self,
        query: str,
        n: int = 5,
        min_similarity: float = 0.3,
    ) -> list[QValueExperience]:
        """Two-phase retrieval: semantic filter → Q-value re-rank.

        Phase A: Retrieve top-K by cosine similarity from ChromaDB (K = 3×n)
        Phase B: Re-rank by combined score = (1−λ)·ẑ_sim + λ·Q̂
        Return top-n by combined score.
        """
        try:
            from belief.memory.soil import Soil
            soil_path = Path("~/.belief-engine/soil").expanduser()
            soil = Soil(soil_path)

            # Phase A: Broad semantic retrieval
            raw_results = soil.retrieve(query, n=n * 3)

            if not raw_results:
                return []

            # Build experiences with Q-values
            experiences = []
            for r in raw_results:
                exp_id = getattr(r, "id", str(hash(r.content[:100])))
                q = self._q_values.get(exp_id, 0.5)
                sim = getattr(r, "similarity", 0.5)

                experiences.append(QValueExperience(
                    id=exp_id,
                    content=r.content,
                    q_value=q,
                    similarity=sim,
                    retrieval_count=self._retrieval_counts.get(exp_id, 0),
                    success_count=self._success_counts.get(exp_id, 0),
                    tags=getattr(r, "tags", []),
                ))

            # Filter by minimum similarity
            experiences = [e for e in experiences if e.similarity >= min_similarity]

            if not experiences:
                return []

            # Phase B: Z-normalize similarity and Q-values, combine
            sim_values = [e.similarity for e in experiences]
            q_values = [e.q_value for e in experiences]

            sim_z = _z_normalize_list(sim_values)
            q_z = _z_normalize_list(q_values)

            for i, exp in enumerate(experiences):
                exp.combined_score = (
                    (1 - self.lambda_weight) * sim_z[i] +
                    self.lambda_weight * q_z[i]
                )

            # Sort by combined score
            experiences.sort(key=lambda e: e.combined_score, reverse=True)

            # Track retrievals
            for exp in experiences[:n]:
                self._retrieval_counts[exp.id] = self._retrieval_counts.get(exp.id, 0) + 1

            self._save()

            logger.info(
                f"Q-value retrieval: {len(experiences)} candidates → top {n}, "
                f"avg Q={sum(e.q_value for e in experiences[:n])/max(n,1):.2f}"
            )

            return experiences[:n]

        except Exception as e:
            logger.debug(f"Q-value retrieval failed: {e}")
            return []

    def update_q_values(
        self,
        experience_ids: list[str],
        reward: float,
    ) -> None:
        """Update Q-values for retrieved experiences based on build outcome.

        MemRL update rule: Q_new ← Q_old + α · (reward − Q_old)

        Args:
            experience_ids: IDs of experiences that were retrieved for this build
            reward: 1.0 for successful build, 0.0 for failure
        """
        for exp_id in experience_ids:
            old_q = self._q_values.get(exp_id, 0.5)
            new_q = old_q + self.alpha * (reward - old_q)
            self._q_values[exp_id] = round(new_q, 4)

            if reward > 0.5:
                self._success_counts[exp_id] = self._success_counts.get(exp_id, 0) + 1

        self._save()

        avg_q = sum(self._q_values[eid] for eid in experience_ids if eid in self._q_values) / max(len(experience_ids), 1)
        logger.info(
            f"Q-value update: {len(experience_ids)} experiences, "
            f"reward={reward:.1f}, avg Q after={avg_q:.3f}"
        )

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the Q-value store."""
        q_vals = list(self._q_values.values())
        return {
            "total_experiences": len(self._q_values),
            "avg_q_value": sum(q_vals) / max(len(q_vals), 1),
            "high_value": sum(1 for q in q_vals if q > 0.7),
            "low_value": sum(1 for q in q_vals if q < 0.3),
            "total_retrievals": sum(self._retrieval_counts.values()),
            "total_successes": sum(self._success_counts.values()),
        }

    def _load(self) -> None:
        """Load Q-values from disk."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._q_values = data.get("q_values", {})
                self._retrieval_counts = data.get("retrieval_counts", {})
                self._success_counts = data.get("success_counts", {})
            except Exception:
                pass

    def _save(self) -> None:
        """Persist Q-values to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_values": self._q_values,
            "retrieval_counts": self._retrieval_counts,
            "success_counts": self._success_counts,
        }
        self._path.write_text(json.dumps(data, indent=2))


def _z_normalize_list(values: list[float]) -> list[float]:
    """Z-score normalize a list of values."""
    if not values:
        return []
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if std < 1e-10:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]
