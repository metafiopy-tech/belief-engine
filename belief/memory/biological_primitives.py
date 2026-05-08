"""BiologicalPrimitiveStore -- 6th memory collection (SE Session 4).

Holds ``StructuralMechanism`` instances (from cross-domain synthesis)
with FSRS-4.5 decay metadata and AskNature taxonomy tags. The
cross-domain generator queries it for in-context priming on each new
synthesis call so the engine metabolizes its own outputs and gets
sharper over time.

Sits alongside the existing 5 belief_* collections from
:mod:`belief.memory.collections` (belief_tools, belief_episodes,
belief_principles, belief_failures, belief_covenants). The collection
name is ``belief_biological_primitives``; nothing in the existing
soil migration touches it, so existing collections continue to work
as-is.

Sandbox-safe: this module imports only ``chromadb`` and the pure-stdlib
:mod:`belief.memory.fsrs` -- no pydantic-loading dependencies. The
StructuralMechanism import is lazy to keep the module loadable even
when the photosynthesis layer isn't on the path.

Out of scope for Session 4:
  - Migrating mechanisms from the legacy ``nutrients`` collection
    (no historical data to migrate; cross-domain synthesis is new).
  - Per-collection embedding routing (uses the chroma default; the
    voyage-3-large routing in soil.py applies to text collections
    and isn't load-bearing here).
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import chromadb

from belief.memory.asknature_taxonomy import validate_tags
from belief.memory.fsrs import retrievability

if TYPE_CHECKING:  # pragma: no cover
    from belief.photosynthesis.synthesis.structural_mechanism import (
        StructuralMechanism,
    )


logger = logging.getLogger("belief.memory.biological_primitives")


COLLECTION_NAME = "belief_biological_primitives"
DEFAULT_TOP_K = 10
DEFAULT_PERSIST_DIR = Path("~/.belief-engine/biological_primitives")

# FSRS defaults mirroring belief.memory.collections._DEFAULT_FSRS_FIELDS.
_FSRS_INIT = {
    "fsrs_stability": 1.0,
    "fsrs_difficulty": 5.0,
    "fsrs_reps": 0,
    "fsrs_lapses": 0,
    "fsrs_decay_state": "new",
    "fsrs_last_review": 0.0,
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


def _make_hash_embedder(dim: int = 384) -> Any:
    """Build the offline-safe deterministic hash embedder.

    Inherits from ``chromadb.EmbeddingFunction`` so the chromadb 1.x
    query path (``embed_query``) and ingest path (``__call__``) both
    resolve correctly. Mirrors :class:`belief.memory.soil._HashEmbeddingFunction`
    but inlined so this module doesn't depend on soil (which pulls
    pydantic + voyage routing at import time and breaks the sandbox).

    Production callers can override at construction time with any
    chromadb EmbeddingFunction (voyage / openai / etc.). The default
    here just guarantees offline-safe behavior in tests + fresh
    installs without an ONNX model download.
    """
    from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

    class HashEmbed(EmbeddingFunction[Documents]):
        DIM = 384

        def __init__(self, d: int = dim) -> None:
            self.dim = int(d)

        def name(self) -> str:
            return "belief_hash_embed_v1"

        @staticmethod
        def build_from_config(config: dict) -> "HashEmbed":
            return HashEmbed(d=int(config.get("dim", HashEmbed.DIM)))

        def get_config(self) -> dict:
            return {"dim": self.dim}

        def __call__(self, input: Documents) -> Embeddings:
            return [self._embed_one(text) for text in input]

        def _embed_one(self, text: str) -> list[float]:
            text = (text or "").lower().strip()
            d = self.dim
            vec = [0.0] * d
            if not text:
                return vec
            for i in range(len(text) - 2):
                tri = text[i : i + 3]
                h = hashlib.md5(tri.encode()).digest()
                idx = struct.unpack("<H", h[:2])[0] % d
                val = struct.unpack("<h", h[2:4])[0] / 32768.0
                vec[idx] += val
            for word in text.split():
                if len(word) < 2:
                    continue
                h = hashlib.md5(word.encode()).digest()
                idx = struct.unpack("<H", h[:2])[0] % d
                val = struct.unpack("<h", h[2:4])[0] / 32768.0
                vec[idx] += val * 2.0
            return vec

    return HashEmbed()


@dataclass
class NeighborMechanism:
    """One result from query_nearest -- mechanism + distance + FSRS-weighted score."""

    mechanism: "StructuralMechanism"
    cosine_distance: float
    fsrs_retrievability: float
    weighted_score: float
    taxonomy_tags: list[str]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class BiologicalPrimitiveStore:
    """ChromaDB-backed store of cross-domain mechanisms.

    Usage::

        store = BiologicalPrimitiveStore()  # ephemeral
        store.add(mechanism, taxonomy_tags=["process_information"])
        neighbors = store.query_nearest("mantis_shrimp pre_classify_signal", top_k=5)
        novelty = store.novelty_score(candidate_mechanism)

    ``persist_dir=None`` uses an ephemeral in-memory client (good for
    tests). A real path uses :class:`chromadb.PersistentClient`.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_fn: Any = None,
        client: Optional[Any] = None,
    ) -> None:
        if client is not None:
            self._client = client
        elif persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._persist_dir = Path(persist_dir).expanduser()
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))

        kwargs: dict[str, Any] = {"metadata": {"hnsw:space": "cosine"}}
        # Default to the offline-safe hash embedder so tests + fresh
        # installs don't try to download an ONNX model. Caller-supplied
        # embedding_fn (voyage / openai / etc.) wins for production use.
        if embedding_fn is None:
            embedding_fn = _make_hash_embedder()
        kwargs["embedding_function"] = embedding_fn

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        mechanism: "StructuralMechanism",
        *,
        taxonomy_tags: Optional[list[str]] = None,
    ) -> str:
        """Insert ``mechanism`` into the store with FSRS init.

        ``taxonomy_tags`` is validated against AskNature -- unknown
        tags raise ``ValueError`` from ``validate_tags`` before any
        write happens, so the store never holds vocabulary drift.

        Returns the document id (= ``mechanism.mechanism_id`` if set,
        else a fresh UUID hex).
        """
        tags = list(taxonomy_tags or [])
        validate_tags(tags)

        doc_id = mechanism.mechanism_id or f"mech-{uuid.uuid4().hex[:12]}"
        document = _mechanism_to_document(mechanism)
        metadata = _build_metadata(mechanism, tags)

        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata],
        )
        return doc_id

    def query_nearest(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[NeighborMechanism]:
        """Return up to ``top_k`` nearest mechanisms, FSRS-reranked.

        Cosine distance from ChromaDB is the primary score; FSRS
        retrievability multiplies it so an old, low-stability hit
        sinks below a fresher one with the same cosine.

        ``query`` is matched against the document text (the predicate
        signature + domain pair) -- a free-form string is fine.
        """
        if top_k < 1:
            return []

        # Over-fetch and re-rank so FSRS reweighting can change the
        # final order. 3x is enough headroom for typical FSRS effects.
        n_request = max(top_k * 3, top_k)
        try:
            res = self._collection.query(
                query_texts=[query],
                n_results=n_request,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("query_nearest failed: %s", exc)
            return []

        ids = (res.get("ids") or [[]])[0]
        if not ids:
            return []
        distances = (res.get("distances") or [[0.0] * len(ids)])[0]
        metadatas = (res.get("metadatas") or [[{}] * len(ids)])[0]

        now_ts = time.time()
        out: list[NeighborMechanism] = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = float(distances[i]) if i < len(distances) else 0.0

            mechanism = _metadata_to_mechanism(meta)
            if mechanism is None:
                continue

            stability = float(meta.get("fsrs_stability", 1.0))
            last_review = float(meta.get("fsrs_last_review", 0.0))
            elapsed_days = (now_ts - last_review) / 86400.0 if last_review else 0.0
            r = retrievability(stability, max(0.0, elapsed_days))

            # cosine distance is ~0 (close) to ~2 (opposite). Convert to
            # similarity-style 1.0 - dist so larger is closer; multiply
            # by retrievability so stale hits sink.
            similarity = max(0.0, 1.0 - dist)
            weighted = similarity * r

            out.append(
                NeighborMechanism(
                    mechanism=mechanism,
                    cosine_distance=dist,
                    fsrs_retrievability=r,
                    weighted_score=weighted,
                    taxonomy_tags=list(_split_tags(meta.get("taxonomy_tags", ""))),
                )
            )

        out.sort(key=lambda n: n.weighted_score, reverse=True)
        return out[:top_k]

    def novelty_score(self, mechanism: "StructuralMechanism") -> float:
        """Return ``1.0 - similarity_to_nearest`` in ``[0.0, 1.0]``.

        ``1.0`` -- the mechanism is fresh; no nearby neighbor exists.
        ``0.0`` -- an exact match is already in the store.

        Returns 1.0 when the store is empty (everything is novel).
        Distance is the cosine distance from ChromaDB; 0.0 dist ->
        1.0 similarity -> 0.0 novelty. FSRS retrievability does NOT
        weight the novelty score -- novelty is about coverage, not
        freshness, so an old-but-stored exact match should still
        report novelty=0.
        """
        query_text = _mechanism_to_document(mechanism)
        try:
            res = self._collection.query(
                query_texts=[query_text],
                n_results=1,
                include=["distances"],
            )
        except Exception as exc:
            logger.warning("novelty_score query failed: %s", exc)
            return 1.0

        ids = (res.get("ids") or [[]])[0]
        if not ids:
            return 1.0
        distances = (res.get("distances") or [[0.0]])[0]
        if not distances:
            return 1.0

        dist = float(distances[0])
        similarity = max(0.0, min(1.0, 1.0 - dist))
        novelty = 1.0 - similarity
        return max(0.0, min(1.0, novelty))

    def count(self) -> int:
        """Number of mechanisms currently in the store."""
        try:
            return int(self._collection.count())
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _mechanism_to_document(mechanism: "StructuralMechanism") -> str:
    """Serialize a mechanism to a short embedding-friendly string.

    The embedding model needs enough text to discriminate between
    mechanisms but not so much that the predicate signature gets
    drowned out. Format:

        ``<source> <-> <target> :: <name>/<arity> :: <roles>``

    Plus the higher-order relation names (which encode the structural
    claim) on a second line so the embedding picks up causal context.
    """
    p = mechanism.predicate_in_source
    sig = f"{p.name}/{p.arity}"
    roles = ",".join(p.roles)
    rel_names = ",".join(r.name for r in mechanism.higher_order_relations)
    return (
        f"{mechanism.source_domain} <-> {mechanism.target_domain} :: "
        f"{sig} :: {roles}\n"
        f"relations: {rel_names}\n"
        f"marr_level: {p.marr_level}"
    )


def _build_metadata(mechanism: "StructuralMechanism", tags: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "mechanism_id": mechanism.mechanism_id,
        "source_domain": mechanism.source_domain,
        "target_domain": mechanism.target_domain,
        "predicate_name": mechanism.predicate_in_source.name,
        "predicate_arity": int(mechanism.predicate_in_source.arity),
        "marr_level": mechanism.predicate_in_source.marr_level,
        # ChromaDB metadata values must be scalar (str/int/float/bool).
        # Tags are joined with ``|`` so they round-trip cleanly via
        # _split_tags below.
        "taxonomy_tags": "|".join(tags),
        # Full mechanism JSON so query_nearest can reconstruct it
        # without re-hitting whichever upstream emitted it.
        "mechanism_json": mechanism.model_dump_json(),
        "captured_at": time.time(),
    }
    for k, v in _FSRS_INIT.items():
        meta[k] = v
    return meta


def _split_tags(joined: str) -> list[str]:
    if not joined:
        return []
    return [t for t in joined.split("|") if t]


def _metadata_to_mechanism(meta: dict[str, Any]) -> Optional["StructuralMechanism"]:
    """Reconstruct a StructuralMechanism from stored metadata."""
    raw = meta.get("mechanism_json")
    if not raw:
        return None
    try:
        # Lazy import -- keeps this module sandbox-safe when the
        # photosynthesis layer isn't available.
        from belief.photosynthesis.synthesis.structural_mechanism import (
            StructuralMechanism,
        )

        return StructuralMechanism.model_validate(json.loads(raw))
    except Exception as exc:
        logger.warning("could not reconstruct mechanism from metadata: %s", exc)
        return None


__all__ = [
    "BiologicalPrimitiveStore",
    "COLLECTION_NAME",
    "DEFAULT_PERSIST_DIR",
    "DEFAULT_TOP_K",
    "NeighborMechanism",
]
