"""
Soil — ChromaDB-backed nutrient store for the Metabolization Architecture.

The soil stores verified nutrients (patterns, antipatterns, skeletons,
covenants) extracted from builds. It handles:
  - Deposit with deduplication (cosine sim >0.92 -> reinforce existing)
  - Retrieval with composite re-ranking (similarity + retrievability + recency)
  - Profile assembly for architect context injection
  - Decay maintenance with archiving (never delete, archive to separate collection)
  - Lineage subsumption (similarity >0.85 AND new tier > old tier -> set parent IDs)

Uses 5 purpose-specific ChromaDB collections (belief_tools, belief_episodes,
belief_principles, belief_failures, belief_covenants) routed by nutrient type.
Backward compatible: auto-migrates from the legacy single collection on first access.

Embedding: _HashEmbeddingFunction for deterministic offline operation,
upgradable to voyage-code-3 later.

Source: METABOLIZATION_BUILD_PLAN.md Phase 2
"""

from __future__ import annotations

import hashlib
import logging
import struct
import time
from datetime import datetime, timezone
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
from belief.memory.fsrs import (
    FSRSState,
    review as fsrs_review,
    retrievability as fsrs_retrievability,
    schedule_next_review,
)
from belief.memory.collections import (
    COLLECTION_CONFIGS,
    collection_for_nutrient_type,
    get_or_create_collections,
    migrate_from_legacy,
    _add_fsrs_defaults,
)

logger = logging.getLogger("belief.memory.soil")


class _HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic n-gram hash embedding that works fully offline.

    Produces 384-dimensional vectors from character trigram hashes.
    Not as semantically rich as a neural embedding model, but:
    - Zero network dependency (no model download)
    - Deterministic (same text -> same embedding)
    - Decent for dedup detection (similar text -> similar vectors)
    - Swappable: pass a real EmbeddingFunction to Soil() for production

    For production use, swap to voyage-code-3 or all-MiniLM-L6-v2.
    """

    DIM = 384

    def __init__(self) -> None:
        pass

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

# Deduplication threshold -- nutrients with cosine similarity above this
# are considered duplicates (reinforce existing instead of creating new)
_DEDUP_THRESHOLD = 0.92

# Lineage subsumption threshold -- if a new nutrient at a higher tier has
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

# Nutrient type -> collection name routing
_TYPE_TO_COLLECTION = {
    NutrientType.PATTERN: "belief_principles",
    NutrientType.ANTIPATTERN: "belief_failures",
    NutrientType.SKELETON: "belief_tools",
    NutrientType.COVENANT: "belief_covenants",
}

_DEFAULT_COLLECTION = "belief_episodes"


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
        persist_dir: Optional[Path] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        collections: Optional[dict[str, chromadb.Collection]] = None,
    ) -> None:
        if persist_dir is None:
            persist_dir = Path("~/.belief-engine/soil")
        self._persist_dir = Path(persist_dir).expanduser()
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir)
        )

        # Default to hash-based embeddings (works offline, no model download)
        # For production: pass voyage-code-3 or all-MiniLM-L6-v2
        self._ef = embedding_fn or _HashEmbeddingFunction()

        if collections is not None:
            # Caller provided pre-created collections
            self._collections = collections
        else:
            # Create the 5 new collections
            self._collections = get_or_create_collections(self._client, self._ef)

        # Archive collection -- decayed nutrients preserved for diagnostics
        # (review correction #4: archive, never delete)
        self._archive = self._client.get_or_create_collection(
            name="archived_nutrients",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )

        # Backward compatibility: also keep the legacy single collection
        # accessible for reading (used by lineage, recombination, etc.)
        self._collection = self._client.get_or_create_collection(
            name="nutrients",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )

        # Auto-migrate from legacy if new collections are empty and legacy has data
        self._maybe_migrate()

    def _maybe_migrate(self) -> None:
        """Auto-migrate from legacy single collection if needed."""
        new_total = sum(c.count() for c in self._collections.values())
        legacy_count = self._collection.count()

        if new_total == 0 and legacy_count > 0:
            logger.info(
                f"Auto-migrating {legacy_count} nutrients from legacy collection"
            )
            migrate_from_legacy(self._client, "nutrients", self._ef)
            # Re-fetch collections to pick up migrated data
            self._collections = get_or_create_collections(self._client, self._ef)

    def _route_collection(self, nutrient_type: NutrientType) -> chromadb.Collection:
        """Return the correct collection for a given nutrient type."""
        name = _TYPE_TO_COLLECTION.get(nutrient_type, _DEFAULT_COLLECTION)
        return self._collections[name]

    def _route_collection_by_str(self, nutrient_type: str) -> chromadb.Collection:
        """Return the correct collection for a nutrient type string."""
        name = collection_for_nutrient_type(nutrient_type)
        return self._collections[name]

    # -- Deposit ------------------------------------------------------------------

    def deposit(self, nutrient: Nutrient) -> str:
        """Store a nutrient in the soil.

        Deduplication: if cosine sim >0.92 with an existing nutrient,
        reinforce the existing one instead of creating a duplicate.

        Lineage subsumption (review correction #6): if the new nutrient's
        tier > an existing similar nutrient's tier (sim >0.85), the new
        nutrient inherits lineage from the old one.

        Returns the nutrient_id (either new or the reinforced existing one).
        """
        target_col = self._route_collection(nutrient.nutrient_type)

        # Check for duplicates and lineage candidates in the target collection
        if target_col.count() > 0:
            existing = target_col.query(
                query_texts=[nutrient.embedding_text],
                n_results=min(5, target_col.count()),
                include=["documents", "metadatas", "distances"],
            )

            if existing["ids"] and existing["ids"][0]:
                for i, doc_id in enumerate(existing["ids"][0]):
                    distance = existing["distances"][0][i]
                    similarity = 1.0 - distance  # ChromaDB cosine distance = 1 - similarity

                    meta = existing["metadatas"][0][i]
                    existing_tier = meta.get("tier", 1)

                    # Dedup: high similarity -> reinforce existing
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
                            f"Soil: lineage link {nutrient.nutrient_id} -> {doc_id} "
                            f"(sim={similarity:.3f}, tier {existing_tier}->{nutrient.tier.value})"
                        )

        # No duplicate found -- store as new nutrient
        metadata = nutrient.to_chromadb_metadata()
        metadata = _add_fsrs_defaults(metadata)

        target_col.upsert(
            ids=[nutrient.nutrient_id],
            documents=[nutrient.embedding_text],
            metadatas=[metadata],
        )

        # Also keep in legacy collection for backward compat
        self._collection.upsert(
            ids=[nutrient.nutrient_id],
            documents=[nutrient.embedding_text],
            metadatas=[metadata],
        )

        logger.info(
            f"Soil: deposited {nutrient.nutrient_id} "
            f"(type={nutrient.nutrient_type.value}, tier={nutrient.tier.value})"
        )
        return nutrient.nutrient_id

    # -- Retrieval ----------------------------------------------------------------

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
          0.5 x similarity + 0.3 x retrievability + 0.2 x recency_bonus

        If *nutrient_type* is specified, queries only the relevant collection.
        Otherwise, queries all collections and merges results.
        """
        if nutrient_type is not None:
            collections_to_query = [self._route_collection(nutrient_type)]
        else:
            collections_to_query = list(self._collections.values())

        # Aggregate results across collections
        all_results: list[tuple[float, Nutrient]] = []
        now = _now_ts()

        for col in collections_to_query:
            if col.count() == 0:
                continue

            fetch_n = min(n * 2, col.count())
            if fetch_n == 0:
                continue

            try:
                if not query:
                    results = col.get(
                        include=["documents", "metadatas"],
                        limit=fetch_n,
                    )
                    if results["ids"]:
                        results = {
                            "ids": [results["ids"]],
                            "documents": [results["documents"]],
                            "metadatas": [results["metadatas"]],
                            "distances": [[0.0] * len(results["ids"])],
                        }
                    else:
                        continue
                else:
                    results = col.query(
                        query_texts=[query],
                        n_results=fetch_n,
                        include=["documents", "metadatas", "distances"],
                    )
            except Exception as e:
                logger.warning(f"Soil retrieve error: {e}")
                continue

            if not results["ids"] or not results["ids"][0]:
                continue

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
                all_results.append((score, nutrient))

        # Sort by composite score descending
        all_results.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in all_results[:n]]

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
        total = sum(c.count() for c in self._collections.values())
        if total == 0:
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

    # -- Reinforcement / Lapse ----------------------------------------------------

    def reinforce(self, nutrient_id: str) -> None:
        """Mark a nutrient as successfully reused -- grows its stability."""
        nutrient, col = self._find_nutrient(nutrient_id)
        if nutrient is None or col is None:
            logger.warning(f"Soil: reinforce failed -- {nutrient_id} not found")
            return

        nutrient.reinforce()
        metadata = nutrient.to_chromadb_metadata()
        metadata = _add_fsrs_defaults(metadata)

        col.update(ids=[nutrient_id], metadatas=[metadata])
        # Also update legacy
        try:
            self._collection.update(ids=[nutrient_id], metadatas=[metadata])
        except Exception as e:
            logger.debug(f"Legacy collection update skipped: {e}")

    def lapse(self, nutrient_id: str) -> None:
        """Mark a nutrient as having led to a failure -- drops its stability."""
        nutrient, col = self._find_nutrient(nutrient_id)
        if nutrient is None or col is None:
            logger.warning(f"Soil: lapse failed -- {nutrient_id} not found")
            return

        nutrient.lapse()
        metadata = nutrient.to_chromadb_metadata()
        metadata = _add_fsrs_defaults(metadata)
        metadata["fsrs_decay_state"] = "lapsed"
        metadata["fsrs_lapses"] = nutrient.lapse_count

        col.update(ids=[nutrient_id], metadatas=[metadata])
        try:
            self._collection.update(ids=[nutrient_id], metadatas=[metadata])
        except Exception as e:
            logger.debug(f"Legacy collection lapse update skipped: {e}")

    def _find_nutrient(
        self, nutrient_id: str
    ) -> tuple[Optional[Nutrient], Optional[chromadb.Collection]]:
        """Find a nutrient across all collections.  Returns (nutrient, collection)."""
        for col in self._collections.values():
            try:
                result = col.get(
                    ids=[nutrient_id],
                    include=["documents", "metadatas"],
                )
                if result["ids"]:
                    n = Nutrient.from_chromadb(
                        result["ids"][0],
                        result["documents"][0],
                        result["metadatas"][0],
                    )
                    return n, col
            except Exception as e:
                logger.debug(f"Collection search failed for {nutrient_id}: {e}")
                continue
        return None, None

    # -- FSRS Review --------------------------------------------------------------

    def review_nutrient(
        self,
        nutrient_id: str,
        collection_name: str,
        grade: int,
    ) -> None:
        """Update FSRS state for a nutrient after review.

        Args:
            nutrient_id:     ID of the nutrient to review.
            collection_name: Name of the collection containing it.
            grade:           1=again, 2=hard, 3=good, 4=easy.
        """
        col = self._collections.get(collection_name)
        if col is None:
            logger.warning(f"Soil: review_nutrient -- unknown collection {collection_name}")
            return

        result = col.get(ids=[nutrient_id], include=["metadatas"])
        if not result["ids"]:
            logger.warning(f"Soil: review_nutrient -- {nutrient_id} not found")
            return

        meta = result["metadatas"][0]

        # Build current FSRS state from metadata
        last_review_ts = meta.get("fsrs_last_review", 0.0)
        state = FSRSState(
            stability=meta.get("fsrs_stability", meta.get("stability", 1.0)),
            difficulty=meta.get("fsrs_difficulty", meta.get("difficulty", 5.0)),
            reps=meta.get("fsrs_reps", meta.get("reinforcement_count", 0)),
            lapses=meta.get("fsrs_lapses", meta.get("lapse_count", 0)),
            last_review=(
                datetime.fromtimestamp(last_review_ts, tz=timezone.utc)
                if last_review_ts > 0 else None
            ),
            decay_state=meta.get("fsrs_decay_state", "new"),
        )

        now = datetime.now(timezone.utc)
        new_state = fsrs_review(state, grade, now)

        # Write back
        meta["fsrs_stability"] = new_state.stability
        meta["fsrs_difficulty"] = new_state.difficulty
        meta["fsrs_reps"] = new_state.reps
        meta["fsrs_lapses"] = new_state.lapses
        meta["fsrs_last_review"] = now.timestamp()
        meta["fsrs_next_review"] = (
            new_state.next_review.timestamp() if new_state.next_review else 0.0
        )
        meta["fsrs_decay_state"] = new_state.decay_state

        # Also update the native nutrient fields for backward compat
        meta["stability"] = new_state.stability
        meta["difficulty"] = new_state.difficulty

        col.update(ids=[nutrient_id], metadatas=[meta])

    def get_due_reviews(
        self,
        collection_name: str,
        now: Optional[datetime] = None,
    ) -> list[dict]:
        """Return records due for review in *collection_name*.

        A record is due when ``fsrs_next_review <= now`` (as a UTC timestamp).

        Returns list of dicts with keys: id, nutrient_type, content,
        fsrs_stability, fsrs_difficulty, fsrs_decay_state, fsrs_next_review.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        col = self._collections.get(collection_name)
        if col is None or col.count() == 0:
            return []

        now_ts = now.timestamp()

        # ChromaDB where filter: next_review <= now
        try:
            results = col.get(
                where={"fsrs_next_review": {"$lte": now_ts}},
                include=["metadatas"],
                limit=col.count(),
            )
        except Exception:
            # Fallback: fetch all and filter in Python
            results = col.get(include=["metadatas"], limit=col.count())
            if results["ids"]:
                filtered_ids = []
                filtered_metas = []
                for i, meta in enumerate(results["metadatas"]):
                    if meta.get("fsrs_next_review", 0.0) <= now_ts:
                        filtered_ids.append(results["ids"][i])
                        filtered_metas.append(meta)
                results = {"ids": filtered_ids, "metadatas": filtered_metas}

        due: list[dict] = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            due.append({
                "id": doc_id,
                "nutrient_type": meta.get("nutrient_type", ""),
                "content": meta.get("content", ""),
                "fsrs_stability": meta.get("fsrs_stability", 1.0),
                "fsrs_difficulty": meta.get("fsrs_difficulty", 5.0),
                "fsrs_decay_state": meta.get("fsrs_decay_state", "new"),
                "fsrs_next_review": meta.get("fsrs_next_review", 0.0),
            })

        return due

    # -- Soil Health Metrics ------------------------------------------------------

    def get_soil_health(self) -> dict:
        """Compute soil health metrics across all collections.

        Returns dict with:
          duplicate_rate: fraction of records with nearest-neighbor cosine < 0.05
          staleness: fraction with last_used > 90 days ago
          lapse_rate: fraction in "lapsed" decay_state
          total_count: dict mapping collection name -> count
        """
        now = _now_ts()
        ninety_days_ago = now - (90 * 86400.0)

        total_records = 0
        near_dupes = 0
        stale_count = 0
        lapsed_count = 0
        per_collection: dict[str, int] = {}

        for name, col in self._collections.items():
            count = col.count()
            per_collection[name] = count
            total_records += count

            if count == 0:
                continue

            all_data = col.get(
                include=["documents", "metadatas"],
                limit=count,
            )

            for i, doc_id in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]
                doc = all_data["documents"][i]

                # Staleness: last_reinforced (or last_review) > 90 days ago
                last_used = meta.get(
                    "last_reinforced",
                    meta.get("fsrs_last_review", meta.get("created_at", now)),
                )
                if last_used < ninety_days_ago:
                    stale_count += 1

                # Lapse rate
                if meta.get("fsrs_decay_state", "") == "lapsed":
                    lapsed_count += 1

                # Duplicate check: nearest neighbor distance
                if count > 1 and doc:
                    try:
                        nn = col.query(
                            query_texts=[doc],
                            n_results=2,
                            include=["distances"],
                        )
                        if nn["distances"] and len(nn["distances"][0]) >= 2:
                            # First result is self (distance ~0), second is nearest neighbor
                            nn_distance = nn["distances"][0][1]
                            if nn_distance < 0.05:
                                near_dupes += 1
                    except Exception as e:
                        logger.debug(f"Duplicate check query failed: {e}")

        return {
            "duplicate_rate": near_dupes / total_records if total_records > 0 else 0.0,
            "staleness": stale_count / total_records if total_records > 0 else 0.0,
            "lapse_rate": lapsed_count / total_records if total_records > 0 else 0.0,
            "total_count": per_collection,
        }

    # -- Decay Maintenance --------------------------------------------------------

    def decay_all(self, archive_threshold: float = 0.1) -> dict[str, int]:
        """Recalculate retrievability for all nutrients.

        Nutrients with retrievability < archive_threshold are moved to
        the archived_nutrients collection (review correction #4: never delete).

        Returns dict with counts: {"total": N, "archived": M, "active": P}
        """
        total = 0
        archived_count = 0

        for col in self._collections.values():
            col_count = col.count()
            if col_count == 0:
                continue

            all_results = col.get(
                include=["documents", "metadatas"],
                limit=col_count,
            )

            to_archive: list[int] = []
            for i, doc_id in enumerate(all_results["ids"]):
                meta = all_results["metadatas"][i]
                doc = all_results["documents"][i]

                nutrient = Nutrient.from_chromadb(doc_id, doc, meta)
                r = nutrient.retrievability()

                if r < archive_threshold:
                    to_archive.append(i)

            for idx in to_archive:
                doc_id = all_results["ids"][idx]
                doc = all_results["documents"][idx]
                meta = all_results["metadatas"][idx]

                self._archive.upsert(
                    ids=[doc_id],
                    documents=[doc],
                    metadatas=[meta],
                )
                col.delete(ids=[doc_id])
                # Also remove from legacy
                try:
                    self._collection.delete(ids=[doc_id])
                except Exception as e:
                    logger.debug(f"Legacy collection delete skipped for {doc_id}: {e}")
                archived_count += 1

            total += col_count

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

    # -- Stats --------------------------------------------------------------------

    def count(self) -> int:
        """Total active nutrients across all collections."""
        return sum(c.count() for c in self._collections.values())

    def count_archived(self) -> int:
        """Total archived nutrients."""
        return self._archive.count()

    def count_by_type(self) -> dict[str, int]:
        """Count active nutrients by type."""
        counts = {}
        for ntype in NutrientType:
            col = self._route_collection(ntype)
            try:
                result = col.get(
                    where={"nutrient_type": ntype.value},
                    include=[],
                )
                counts[ntype.value] = len(result["ids"])
            except Exception as e:
                logger.debug(f"Count by type failed for {ntype.value}: {e}")
                counts[ntype.value] = 0
        return counts

    def get(self, nutrient_id: str) -> Optional[Nutrient]:
        """Get a single nutrient by ID (searches all collections)."""
        nutrient, _ = self._find_nutrient(nutrient_id)
        return nutrient

    def get_profile(self, goal: str, complexity: int = 3) -> NutrientProfile:
        """Alias for retrieve_profile for backward compatibility."""
        return self.retrieve_profile(goal, complexity)
