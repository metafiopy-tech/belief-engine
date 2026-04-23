"""Bittensor centroid construction + ranker bias lookup.

Builds a single centroid embedding from N random SWE-Bench problem
statements and stores it as `domain_profile/bittensor_swebench` in
ChromaDB. The Session-4 ranker's `bittensor_cosine` argument is exactly
this: the cosine similarity between a seed's embedding and this
centroid.

Operationally:

    biaser = TaskBiaser(mirror, archive, embedder=embed_fn)
    centroid = biaser.build_centroid(n_samples=50)
    sim = biaser.cosine_to_centroid(seed_text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


logger = logging.getLogger("belief.photosynthesis.bittensor.task_biaser")


BITTENSOR_CENTROID_ID = "bittensor_swebench"
CENTROID_COLLECTION = "domain_profile"


@dataclass
class TaskBiaser:
    mirror: Any  # SwebenchMirror
    archive: Any  # ArchiveManager
    embedder: Callable[[str], Any]
    n_samples: int = 50

    def build_centroid(self, *, n_samples: Optional[int] = None) -> Optional[list[float]]:
        """Sample tasks, embed each, average into a unit centroid."""
        count = int(n_samples or self.n_samples)
        tasks = self.mirror.sample(count)
        if not tasks:
            logger.info("swebench mirror empty; cannot build centroid")
            return None

        vectors: list[list[float]] = []
        for t in tasks:
            try:
                v = self.embedder(t.problem_statement)
            except Exception as exc:
                logger.warning("embed failed for task %s: %s", t.id, exc)
                continue
            vectors.append(_to_list(v))
        if not vectors:
            return None

        centroid = _mean_vector(vectors)
        centroid = _normalize(centroid)

        # Upsert the centroid into the domain_profile collection for
        # downstream lookups. Archive manager is a no-op if chromadb
        # isn't installed — we still return the vector so callers that
        # pass it through directly (tests) work.
        try:
            self.archive.ensure(CENTROID_COLLECTION)
            self.archive.upsert_goal(
                collection="goal_archive"  # archive manager validates collection names
                if CENTROID_COLLECTION not in _known_archive_names(self.archive)
                else CENTROID_COLLECTION,
                goal_id=BITTENSOR_CENTROID_ID,
                embedding=centroid,
                document="bittensor_swebench centroid",
                metadata={
                    "title": "bittensor_swebench centroid",
                    "kind": "centroid",
                    "n_samples": len(vectors),
                },
            )
        except Exception as exc:
            logger.warning("archive upsert failed: %s", exc)

        return centroid

    def cosine_to_centroid(
        self, seed_text: str, *, centroid: Optional[list[float]] = None
    ) -> float:
        """Cosine between a seed embedding and the centroid.

        If `centroid` is None, computes a fresh one on the fly.
        """
        c = centroid if centroid is not None else self.build_centroid()
        if not c:
            return 0.0
        try:
            v = _normalize(_to_list(self.embedder(seed_text)))
        except Exception as exc:
            logger.warning("embed failed on seed: %s", exc)
            return 0.0
        return _cosine(v, c)


def _to_list(v: Any) -> list[float]:
    """Coerce numpy arrays / torch tensors / lists to a plain list[float]."""
    if isinstance(v, list):
        return [float(x) for x in v]
    try:
        return [float(x) for x in v]  # iterable
    except TypeError:
        return []


def _mean_vector(vs: Iterable[list[float]]) -> list[float]:
    it = iter(vs)
    first = next(it)
    dim = len(first)
    total = list(first)
    count = 1
    for v in it:
        if len(v) != dim:
            continue
        for i in range(dim):
            total[i] += v[i]
        count += 1
    return [x / count for x in total]


def _normalize(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5
    if norm <= 0:
        return list(v)
    return [x / norm for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def _known_archive_names(archive: Any) -> set[str]:
    try:
        from belief.photosynthesis.synthesis.archives import ARCHIVE_NAMES

        return set(ARCHIVE_NAMES)
    except Exception:
        return set()


__all__ = [
    "BITTENSOR_CENTROID_ID",
    "CENTROID_COLLECTION",
    "TaskBiaser",
]
