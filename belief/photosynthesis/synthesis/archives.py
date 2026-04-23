"""ChromaDB archive collection management for synthesis.

Four collections keep the retrieval corpus clean:

  goal_archive      successfully-built goals — the primary retrieval target
  failed_gen        goals that failed Pydantic schema validation on generator
  failed_interest   goals the interestingness judge rejected
  failed_build      goals that shipped to Grinder but failed to build

Strict separation matters: mixing failed-build goals into goal_archive
poisons novelty retrieval (near-duplicates of known failures look like
"distinct" goals at vector-distance levels). Every writer path asserts
its collection.

ChromaDB is lazy-loaded. When chromadb isn't installed (lighter test
extra), ArchiveManager.ensure() logs a warning and downstream callers
see an empty archive — novelty defaults to "distinct", which is the
safest fallback (it doesn't spuriously block new goals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("belief.photosynthesis.synthesis.archives")


ARCHIVE_NAMES = (
    "goal_archive",
    "failed_gen",
    "failed_interest",
    "failed_build",
)


@dataclass
class Neighbor:
    """A single hit from a goal-archive similarity query."""

    goal_id: str
    title: str
    cosine: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {"goal_id": self.goal_id, "title": self.title, "cosine": self.cosine}


class ArchiveManager:
    """Thin wrapper around ChromaDB's persistent client.

    One instance per daemon. Collections are created on first use with
    cosine metric metadata. If chromadb isn't installed, every method
    degrades gracefully: queries return empty, upserts are no-ops, but
    no exception propagates to the caller.
    """

    def __init__(
        self,
        persist_dir: Path | str,
        *,
        embedding_fn: Any = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = embedding_fn
        self._client: Any = None
        self._collections: dict[str, Any] = {}
        self._available: Optional[bool] = None

    # -------------------------------------------------------------- availability
    def _ensure_client(self) -> None:
        if self._available is False:
            return
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("chromadb not available; ArchiveManager is running in no-op mode.")
            self._available = False
            return
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._available = True

    def ensure(self, name: str) -> Any:
        """Get-or-create a collection with cosine metric."""
        self._ensure_client()
        if self._available is False:
            return None
        if name in self._collections:
            return self._collections[name]
        col = self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        self._collections[name] = col
        return col

    def ensure_all(self) -> None:
        for n in ARCHIVE_NAMES:
            self.ensure(n)

    # -------------------------------------------------------------- query
    def query_neighbors(
        self,
        collection: str,
        embedding: Any,
        *,
        top_k: int = 10,
    ) -> list[Neighbor]:
        """Return top-k neighbors sorted by ascending distance.

        Falls back to [] if ChromaDB isn't installed or the collection
        is empty. Cosine = 1 - distance (ChromaDB returns cosine distance
        when the collection metadata says 'hnsw:space': 'cosine').
        """
        col = self.ensure(collection)
        if col is None:
            return []
        try:
            if col.count() == 0:
                return []
            res = col.query(query_embeddings=[embedding], n_results=top_k)
        except Exception as exc:
            logger.warning("archive query on %s failed: %s", collection, exc)
            return []

        out: list[Neighbor] = []
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, goal_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            out.append(
                Neighbor(
                    goal_id=str(goal_id),
                    title=str((meta or {}).get("title", "")),
                    cosine=float(1.0 - dist),
                    metadata=meta or {},
                )
            )
        return out

    # -------------------------------------------------------------- upserts
    def upsert_goal(
        self,
        collection: str,
        *,
        goal_id: str,
        embedding: Any,
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a goal into the named collection."""
        if collection not in ARCHIVE_NAMES:
            raise ValueError(f"unknown archive collection: {collection!r}. Valid: {ARCHIVE_NAMES}")
        col = self.ensure(collection)
        if col is None:
            return  # no-op when chromadb unavailable
        try:
            col.upsert(
                ids=[goal_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as exc:
            logger.warning("upsert into %s failed: %s", collection, exc)

    def count(self, collection: str) -> int:
        col = self.ensure(collection)
        if col is None:
            return 0
        try:
            return int(col.count())
        except Exception:
            return 0

    def distinct_cosine(
        self,
        collection: str,
        embedding: Any,
        threshold: float = 0.90,
    ) -> bool:
        """True if the embedding's top-1 cosine against the collection is < threshold.

        Used by the generator's post-expansion duplicate check — if the
        freshly-written spec is already near-identical to an existing
        goal, drop it into failed_interest instead of goal_archive.
        """
        hits = self.query_neighbors(collection, embedding, top_k=1)
        if not hits:
            return True
        return hits[0].cosine < threshold

    def top_tags(self, collection: str, top_n: int = 20) -> list[str]:
        """Return the top-N most common domain tags across the collection.

        Used by the ranker's coverage_gain term. ChromaDB doesn't expose
        a native aggregate, so we page through .get() and count in-process.
        Fine at the archive's expected size (<1e4 rows for v3.0).
        """
        col = self.ensure(collection)
        if col is None:
            return []
        try:
            res = col.get(include=["metadatas"], limit=10_000)
        except Exception:
            return []

        tag_counts: dict[str, int] = {}
        for meta in res.get("metadatas") or []:
            for tag in (meta or {}).get("domain_tags", []) or []:
                tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
        sorted_tags = sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [t for t, _ in sorted_tags[:top_n]]


__all__ = ["ARCHIVE_NAMES", "ArchiveManager", "Neighbor"]
