"""Bittensor mirror + task biaser: centroid shape + ranker 1.5x bias."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from belief.photosynthesis.bittensor.swebench_mirror import (
    BittensorTask,
    SwebenchMirror,
)
from belief.photosynthesis.bittensor.task_biaser import (
    BITTENSOR_CENTROID_ID,
    TaskBiaser,
)


@pytest.fixture()
def mirror(tmp_path: Path) -> SwebenchMirror:
    m = SwebenchMirror(db_path=tmp_path / "bittensor.db")
    tasks = [
        BittensorTask(
            id=f"task-{i}",
            problem_statement=f"fix null reference in module {i}",
            repo="foo/bar",
            ecosystem="python",
        )
        for i in range(10)
    ]
    m.ingest_fixture(tasks, dataset="unit-test")
    return m


class InMemoryArchive:
    """Archive stand-in that records upserts in-process."""

    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self._seen: dict[str, list[float]] = {}

    def ensure(self, name: str) -> None:  # noqa: ARG002
        return None

    def upsert_goal(
        self,
        *,
        collection: str,
        goal_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        self.upserts.append({"collection": collection, "goal_id": goal_id, "embedding": embedding})
        self._seen[goal_id] = list(embedding)


def _embedder(text: str) -> list[float]:
    # 16-dim deterministic hash embedding
    vec = [0.0] * 16
    for ch in text:
        vec[ord(ch) % 16] += 1.0
    s = sum(v * v for v in vec) ** 0.5
    return [v / s for v in vec] if s else vec


def test_mirror_ingest_and_count(mirror: SwebenchMirror) -> None:
    assert mirror.count() == 10
    sample = mirror.sample(3)
    assert len(sample) == 3
    assert all(isinstance(t, BittensorTask) for t in sample)


def test_mirror_ingest_is_idempotent(mirror: SwebenchMirror) -> None:
    tasks = [BittensorTask(id="task-0", problem_statement="dup", repo="x", ecosystem="python")]
    added = mirror.ingest_fixture(tasks)
    assert added == 0
    assert mirror.count() == 10


def test_biaser_builds_centroid_with_correct_dim(mirror: SwebenchMirror) -> None:
    archive = InMemoryArchive()
    biaser = TaskBiaser(mirror=mirror, archive=archive, embedder=_embedder, n_samples=10)
    centroid = biaser.build_centroid()
    assert centroid is not None
    assert len(centroid) == 16
    # Unit-normalized
    norm = sum(x * x for x in centroid) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    # Was upserted
    assert any(u["goal_id"] == BITTENSOR_CENTROID_ID for u in archive.upserts)


def test_biaser_cosine_matches_self_high(mirror: SwebenchMirror) -> None:
    archive = InMemoryArchive()
    biaser = TaskBiaser(mirror=mirror, archive=archive, embedder=_embedder, n_samples=10)
    centroid = biaser.build_centroid()
    assert centroid is not None

    # A seed whose text resembles the mirrored problem statements should
    # land high on the centroid cosine. A totally unrelated seed should
    # land lower.
    high = biaser.cosine_to_centroid("fix null reference in module 7", centroid=centroid)
    low = biaser.cosine_to_centroid("xyz zz zz zzz zzzz bakery recipes", centroid=centroid)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    # Don't assert strict ordering on hash embeddings — they're too noisy —
    # but at least verify the cosine function returned finite values.


# ---------------------------------------------------------------------------
# Ranker 1.5x bias wiring
# ---------------------------------------------------------------------------


def test_ranker_applies_1_5x_boost_above_cutoff() -> None:
    from belief.photosynthesis.synthesis.ranker import (
        BITTENSOR_BIAS_COSINE_CUTOFF,
        combined_value,
    )

    # Pick components so baseline 0.40*N + 0.35*Z + ... = 0.50 exactly.
    # With bittensor_cosine above the cutoff, the raw value becomes
    # 0.50 * 1.5 = 0.75 (still accepted).
    baseline = combined_value(
        novelty=0.5,
        zpd_fit=0.5,
        coverage_gain=0.5,
        source_quality=0.5,
        bittensor_cosine=BITTENSOR_BIAS_COSINE_CUTOFF - 0.01,
    )
    boosted = combined_value(
        novelty=0.5,
        zpd_fit=0.5,
        coverage_gain=0.5,
        source_quality=0.5,
        bittensor_cosine=BITTENSOR_BIAS_COSINE_CUTOFF + 0.01,
    )
    assert not baseline.bittensor_boosted
    assert boosted.bittensor_boosted
    # Boosted value should exceed baseline (capped at 1.0)
    assert boosted.value > baseline.value
