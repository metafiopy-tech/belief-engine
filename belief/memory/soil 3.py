"""
Soil — ChromaDB-backed nutrient store for the Metabolization Architecture.

The soil stores verified nutrients (patterns, antipatterns, skeletons,
covenants) extracted from builds. It handles:
  - Deposit with deduplication (cosine sim >0.92 → reinforce existing)
  - Retrieval with composite re-ranking (similarity + retrievability + recency)
  - Profile assembly for architect context injection
  - Decay maintenance with archiving (never delete, archive to separate collection)
  - Lineage subsumption (similarity >0.85 AND new tier > old tier → set parent IDs)

Uses a single ChromaDB collection with metadata filtering — no cross-collection
queries needed. Embedding: ChromaDB default (all-MiniLM-L6-v2) for zero-config
startup, upgradable to voyage-code-3 later.

Source: METABOLIZATION_BUILD_PLAN.md Phase 2
"""

from __future__ import annotations

import hashlib
import logging
import struct
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from belief.memory.nutrients import (
    Nutrient,
    NutrientProfile,
    NutrientTier,
    NutrientType,
    _now_ts,
)

logger = logging.getLogger("belief.memory.soil")


class _HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic n-gram hash embedding that works fully offline.

    Produces 384-dimensional vectors from character trigram hashes.
    Not as semantically rich as a neural embedding model, but:
    - Zero network dependency (no model download)
    - Deterministic (same text → same embedding)
    - Decent for dedup detection (similar text → similar vectors)
    - Swappable: pass a real EmbeddingFunction to Soil() for production

    For production use, swap to voyage-code-3 or all-MiniLM-L6-v2.
    """

    DIM = 384

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(text) for text in input]

    def _embed_one(self, text: str) -> list[float]:
        text = text.lower().strip()
        vec = [0.0] * self.DIM
        if not text:
            return vec

        # Character trigram hashing into dimensions
        for i in range(len(text) - 2):
            trigram = text[i : i + 3]
            h = hashlib.md5(trigram.encode()).digest()
            # Map hash to a dimension index and a value
            idx = struct.unpack("<H", h[:2])[0] % self.DIM
            val = struct.unpack("<h", h[2:4])[0] / 32768.0
            vec[idx] += val

        # Also hash word unigrams for coarser semantic signal
        for word in text.split():
            if len(word) < 2:
                continue
            h = hashlib.md5(word.encode()).digest()
            idx = struct.unpack("<H", h[:2])[0] % self.DIM
            val = struct.unpack("<h", h[2:4])[0] / 32768.0
            vec[idx] += val * 2.0  # Words weighted more than trigrams

        # L2 normalize
        magnitude = sum(v * v for v in vec) ** 0.5
        if magnitude > 0:
            vec = [v / magnitude for v in vec]

        return vec

# Deduplication threshold — nutrients with cosine similarity above this
# are considered duplicates (reinforce existing instead of creating new)
_DEDUP_THRESHOLD = 0.92

# Lineage subsumption threshold — if a new nutrient at a higher tier has
# >0.85 similarity to an existing lower-tier nutrient, the new nutrient
# inherits lineage from the old one (review correction #6)
_LINEAGE_THRESHOLD = 0.85

# Default retrieval counts per nutrient type for profile assembly
_PROFILE_LIMITS = {
    NutrientType.COVENANT: 50,     # All covenants (effectively unlimited)
    NutrientType.ANTIPATTERN: 3,   # Top 3 most relevant
    NutrientType.PATTERN: 5,       # Top 5 most relevant
    NutrientType.SKELETON: 1,      # Closest match only
}


class Soil:
    """ChromaDB-backed nutrient store.

    Usage:
        soil = Soil(Path("~/.belief-engine/soil"))
        soil.deposit(nutrient)
        profile = soil.retrieve_profile("build a FastAPI pipeline")
        context = profile.format_context_block(complexity=3)
    """

    def __init__(
        self,
        persist_dir: Path,
        embedding_fn: Optional[EmbeddingFunction] = None,
    ) -> None:
        self._persist_dir = Path(persist_dir).expanduser()
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir)
        )

        # Default to hash-based embeddings (works offline, no model download)
        # For production: pass voyage-code-3 or all-MiniLM-L6-v2
        ef = embedding_fn or _HashEmbeddingFunction()

        # Main collection — all active nutrients
        self._collection = self._client.get_or_create_collection(
            name="nutrients",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )

        # Archive collection — decayed nutrients preserved for diagnostics
        # (review correction #4: archive, never delete)
        self._archive = self._client.get_or_create_collection(
            name="archived_nutrients",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )

    # ── Deposit ──────────────────────────────────────────────────────────────

    def deposit(self, nutrient: Nutrient) -> str:
        """Store a nutrient in the soil.

        Deduplication: if cosine sim >0.92 with an existing nutrient,
        reinforce the existing one instead of creating a duplicate.

        Lineage subsumption (review correction #6): if the new nutrient's
        tier > an existing similar nutrient's tier (sim >0.85), set
        lineage_parent_ids to include the existing nutrient's ID.

        Returns the nutrient_id (either new or the reinforced existing one).
        """
        # Check for duplicates and lineage candidates
        existing = self._collection.query(
            query_texts=[nutrient.embedding_text],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        if existing["ids"] and existing["ids"][0]:
            for i, doc_id in enumerate(existing["ids"][0]):
                distance = existing["distances"][0][i]
                similarity = 1.0 - distance  # ChromaDB cosine distance = 1 - similarity

                meta = existing["metadatas"][0][i]
                existing_tier = meta.get("tier", 1)

                # Dedup: high similarity → reinforce existing
                if similarity >= _DEDUP_THRESHOLD:
                    self.reinforce(doc_id)
                    logger.info(
                        f"Soil: reinforced existing {doc_id} "
                        f"(sim={similarity:.3f}, type={meta.get('nutrient_type')})"
                    )
                    return doc_id

                # Lineage subsumption: moderate similarity + higher tier
                if (
                    similarity >= _LINEAGE_THRESHOLD
                    and nutrient.tier.value > existing_tier
                    and doc_id not in nutrient.lineage_parent_ids
                ):
                    nutrient.lineage_parent_ids.append(doc_id)
                    logger.info(
                        f"Soil: lineage link {nutrient.nutrient_id} → {doc_id} "
                        f"(sim={similarity:.3f}, tier {existing_tier}→{nutrient.tier.value})"
                    )

        # No duplicate found — store as new nutrient
        self._collection.upsert(
            ids=[nutrient.nutrient_id],
            documents=[nutrient.embedding_text],
            metadatas=[nutrient.to_chromadb_metadata()],
        )

        logger.info(
            f"Soil: deposited {nutrient.nutrient_id} "
            f"(type={nutrient.nutrient_type.value}, tier={nutrient.tier.value})"
        )
        return nutrient.nutrient_id

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n: int = 10,
        nutrient_type: Optional[NutrientType] = None,
        min_retrievability: float = 0.3,
    ) -> list[Nutrient]:
        """Semantic search with metadata filtering and composite re-ranking.

        Over-fetches 2x, reconstructs Nutrient objects, filters by
        retrievability, then re-ranks by composite score:
          0.5 × similarity + 0.3 × retrievability + 0.2 × recency_bonus
        """
        if self._collection.count() == 0:
            return []

        # Build where filter
        where = {}
        if nutrient_type is not None:
            where["nutrient_type"] = nutrient_type.value

        # Over-fetch 2x for re-ranking headroom
        fetch_n = min(n * 2, self._collection.count())
        if fetch_n == 0:
            return []

        try:
            if not query:
                # Empty query: return all nutrients (no similarity ranking)
                results = self._collection.get(
                    where=where if where else None,
                    include=["documents", "metadatas"],
                    limit=fetch_n,
                )
                # Reshape to match query() output format
                if results["ids"]:
                    results = {
                        "ids": [results["ids"]],
                        "documents": [results["documents"]],
                        "metadatas": [results["metadatas"]],
                        "distances": [[0.0] * len(results["ids"])],  # No distance for get()
                    }
                else:
                    return []
            else:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=fetch_n,
                    where=where if where else None,
                    include=["documents", "metadatas", "distances"],
                )
        except Exception as e:
            logger.warning(f"Soil retrieve error: {e}")
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        # Reconstruct and score
        scored: list[tuple[float, Nutrient]] = []
        now = _now_ts()

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            nutrient = Nutrient.from_chromadb(doc_id, doc, meta)

            # Filter by retrievability
            r = nutrient.retrievability()
            if r < min_retrievability:
                continue

            # Recency bonus: 1.0 for today, decays to 0 over ~30 days
            days_old = (now - nutrient.last_reinforced) / 86400.0
            recency = max(0.0, 1.0 - days_old / 30.0)

            # Composite score
            score = 0.5 * similarity + 0.3 * r + 0.2 * recency
            scored.append((score, nutrient))

        # Sort by composite score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:n]]

    def retrieve_profile(
        self,
        goal: str,
        complexity: int = 3,
        max_tokens: Optional[int] = None,
    ) -> NutrientProfile:
        """Retrieve a complete nutrient profile for a build goal.

        Priority: covenants (all) > antipatterns (top 3) > patterns (top 5) > skeletons (top 1)
        Returns empty NutrientProfile if soil is empty.
        """
        if self._collection.count() == 0:
            return NutrientProfile()

        profile = NutrientProfile()

        for ntype, limit in _PROFILE_LIMITS.items():
            nutrients = self.retrieve(
                query=goal,
                n=limit,
                nutrient_type=ntype,
                min_retrievability=0.1 if ntype == NutrientType.COVENANT else 0.3,
            )
            if ntype == NutrientType.COVENANT:
                profile.covenants = nutrients
            elif ntype == NutrientType.ANTIPATTERN:
                profile.antipatterns = nutrients
            elif ntype == NutrientType.PATTERN:
                profile.patterns = nutrients
            elif ntype == NutrientType.SKELETON:
                profile.skeletons = nutrients

        return profile

    # ── Reinforcement / Lapse ────────────────────────────────────────────────

    def reinforce(self, nutrient_id: str) -> None:
        """Mark a nutrient as successfully reused — grows its stability."""
        result = self._collection.get(
            ids=[nutrient_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            logger.warning(f"Soil: reinforce failed — {nutrient_id} not found")
            return

        nutrient = Nutrient.from_chromadb(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
        )
        nutrient.reinforce()

        # Update in-place
        self._collection.update(
            ids=[nutrient_id],
            metadatas=[nutrient.to_chromadb_metadata()],
        )

    def lapse(self, nutrient_id: str) -> None:
        """Mark a nutrient as having led to a failure — drops its stability."""
        result = self._collection.get(
            ids=[nutrient_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            logger.warning(f"Soil: lapse failed — {nutrient_id} not found")
            return

        nutrient = Nutrient.from_chromadb(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
        )
        nutrient.lapse()

        self._collection.update(
            ids=[nutrient_id],
            metadatas=[nutrient.to_chromadb_metadata()],
        )

    # ── Decay Maintenance ────────────────────────────────────────────────────

    def decay_all(self, archive_threshold: float = 0.1) -> dict[str, int]:
        """Recalculate retrievability for all nutrients.

        Nutrients with retrievability < archive_threshold are moved to
        the archived_nutrients collection (review correction #4: never delete).

        Returns dict with counts: {"total": N, "archived": M, "active": P}
        """
        total = self._collection.count()
        if total == 0:
            return {"total": 0, "archived": 0, "active": 0}

        # Fetch all nutrients in batches
        archived_count = 0
        batch_size = 100
        to_archive_ids = []

        all_results = self._collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )

        for i, doc_id in enumerate(all_results["ids"]):
            meta = all_results["metadatas"][i]
            doc = all_results["documents"][i]

            nutrient = Nutrient.from_chromadb(doc_id, doc, meta)
            r = nutrient.retrievability()

            if r < archive_threshold:
                to_archive_ids.append(i)

        # Archive low-retrievability nutrients
        for idx in to_archive_ids:
            doc_id = all_results["ids"][idx]
            doc = all_results["documents"][idx]
            meta = all_results["metadatas"][idx]

            # Move to archive
            self._archive.upsert(
                ids=[doc_id],
                documents=[doc],
                metadatas=[meta],
            )
            # Remove from active
            self._collection.delete(ids=[doc_id])
            archived_count += 1

        active = total - archived_count
        if archived_count > 0:
            logger.info(
                f"Soil maintenance: archived {archived_count} nutrients "
                f"(below {archive_threshold} retrievability), {active} active"
            )

        return {
            "total": total,
            "archived": archived_count,
            "active": active,
        }

    # ── Stats ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total active nutrients."""
        return self._collection.count()

    def count_archived(self) -> int:
        """Total archived nutrients."""
        return self._archive.count()

    def count_by_type(self) -> dict[str, int]:
        """Count active nutrients by type."""
        counts = {}
        for ntype in NutrientType:
            try:
                result = self._collection.get(
                    where={"nutrient_type": ntype.value},
                    include=[],
                )
                counts[ntype.value] = len(result["ids"])
            except Exception:
                counts[ntype.value] = 0
        return counts

    def get(self, nutrient_id: str) -> Optional[Nutrient]:
        """Get a single nutrient by ID."""
        result = self._collection.get(
            ids=[nutrient_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return Nutrient.from_chromadb(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
        )
