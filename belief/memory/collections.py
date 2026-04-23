"""
Collection definitions and legacy migration for the 5-collection architecture.

Replaces the single ``nutrients`` ChromaDB collection with five
purpose-specific collections so retrieval can be scoped per concern
(tools vs. episodes vs. principles, etc.) and FSRS decay can be tuned
independently.

Collections:
  belief_tools       — self-authored tools, validators, extractors
  belief_episodes    — build traces (what happened during each build)
  belief_principles  — soft knowledge: patterns, insights, guidelines
  belief_failures    — failure traces with root cause analysis
  belief_covenants   — crystallised hard rules (AST validators, regex, schemas)

Migration:
  ``migrate_from_legacy()`` reads the old ``nutrients`` collection and
  classifies each record by its ``nutrient_type`` metadata into the
  appropriate new collection.  The old collection is NOT deleted (kept
  as a backup).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import chromadb
from chromadb.api.types import EmbeddingFunction

logger = logging.getLogger("belief.memory.collections")

# ── Collection configuration ───────────────────────────────────────────────

COLLECTION_CONFIGS = {
    "belief_tools": {
        "description": "Self-authored tools, validators, extractors",
        "hnsw_space": "cosine",
    },
    "belief_episodes": {
        "description": "Build traces — what happened during each build",
        "hnsw_space": "cosine",
    },
    "belief_principles": {
        "description": "Soft knowledge — patterns, insights, guidelines",
        "hnsw_space": "cosine",
    },
    "belief_failures": {
        "description": "Failure traces with root cause analysis",
        "hnsw_space": "cosine",
    },
    "belief_covenants": {
        "description": "Crystallized hard rules (AST validators, regex, schemas)",
        "hnsw_space": "cosine",
    },
}

# Mapping from legacy nutrient_type → new collection name
NUTRIENT_TYPE_TO_COLLECTION = {
    "pattern": "belief_principles",
    "antipattern": "belief_failures",
    "skeleton": "belief_tools",
    "covenant": "belief_covenants",
}

# Default: anything not matched goes to episodes
DEFAULT_COLLECTION = "belief_episodes"


# ── Public API ──────────────────────────────────────────────────────────────


def get_or_create_collections(
    client: chromadb.Client,
    embedding_fn: Optional[Union[EmbeddingFunction, dict[str, EmbeddingFunction]]] = None,
) -> dict[str, chromadb.Collection]:
    """Create or retrieve all 5 collections with cosine distance.

    Args:
        client:       A chromadb.Client (persistent or ephemeral).
        embedding_fn: Optional embedding function.  Three shapes are
                      accepted:

                      * ``None`` — ChromaDB uses its built-in default.
                      * A single :class:`EmbeddingFunction` — applied to
                        every collection (legacy behaviour).
                      * A ``dict[name, EmbeddingFunction]`` — each named
                        collection uses its own EF (Session 13: enables
                        per-collection routing such as ``voyage-code-3``
                        for code collections and ``voyage-3-large`` for
                        text collections).  Names missing from the dict
                        fall back to the ChromaDB default.

    Returns:
        Dict mapping collection name → chromadb.Collection.
    """
    collections: dict[str, chromadb.Collection] = {}

    is_per_collection_map = isinstance(embedding_fn, dict)

    for name, cfg in COLLECTION_CONFIGS.items():
        kwargs = {}
        if is_per_collection_map:
            ef = embedding_fn.get(name)  # type: ignore[union-attr]
            if ef is not None:
                kwargs["embedding_function"] = ef
        elif embedding_fn is not None:
            kwargs["embedding_function"] = embedding_fn

        col = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": cfg["hnsw_space"]},
            **kwargs,
        )
        collections[name] = col

    return collections


def collection_for_nutrient_type(nutrient_type: str) -> str:
    """Return the collection name for a given nutrient type string.

    Args:
        nutrient_type: One of "pattern", "antipattern", "skeleton",
                       "covenant", or any future type.

    Returns:
        Collection name (falls back to ``belief_episodes``).
    """
    return NUTRIENT_TYPE_TO_COLLECTION.get(nutrient_type, DEFAULT_COLLECTION)


# ── Default FSRS metadata ──────────────────────────────────────────────────

_DEFAULT_FSRS_FIELDS = {
    "fsrs_stability": 1.0,
    "fsrs_difficulty": 5.0,
    "fsrs_reps": 0,
    "fsrs_lapses": 0,
    "fsrs_decay_state": "new",
    "fsrs_last_review": 0.0,
    "fsrs_next_review": 0.0,
}


def _add_fsrs_defaults(metadata: dict) -> dict:
    """Merge default FSRS fields into *metadata* without overwriting."""
    enriched = dict(metadata)
    for key, default in _DEFAULT_FSRS_FIELDS.items():
        if key not in enriched:
            enriched[key] = default
    return enriched


# ── Legacy migration ───────────────────────────────────────────────────────


def migrate_from_legacy(
    client: chromadb.Client,
    legacy_collection_name: str = "belief_soil",
    embedding_fn: Optional[EmbeddingFunction] = None,
) -> dict[str, int]:
    """Migrate records from the old single collection into the 5 new ones.

    Reads all records from *legacy_collection_name*, classifies each by
    its ``nutrient_type`` metadata field, and copies it (with all
    metadata preserved plus default FSRS fields) into the appropriate
    new collection.

    The old collection is **not** deleted — it is kept as a backup.

    Also attempts migration from the ``"nutrients"`` collection name
    (used in early versions of soil.py) if *legacy_collection_name*
    doesn't exist.

    Args:
        client:                  ChromaDB client.
        legacy_collection_name:  Name of the old collection.
        embedding_fn:            Optional embedding function.

    Returns:
        Dict mapping new collection name → number of records migrated.
    """
    # Ensure target collections exist
    targets = get_or_create_collections(client, embedding_fn)

    # Try to find the legacy collection
    try:
        legacy = client.get_collection(
            name=legacy_collection_name,
            **({"embedding_function": embedding_fn} if embedding_fn else {}),
        )
    except Exception:
        # Try the alternative legacy name
        try:
            legacy = client.get_collection(
                name="nutrients",
                **({"embedding_function": embedding_fn} if embedding_fn else {}),
            )
        except Exception:
            logger.info("No legacy collection found — nothing to migrate")
            return {name: 0 for name in COLLECTION_CONFIGS}

    total = legacy.count()
    if total == 0:
        logger.info("Legacy collection is empty — nothing to migrate")
        return {name: 0 for name in COLLECTION_CONFIGS}

    logger.info(f"Migrating {total} records from legacy collection")

    # Fetch all records
    all_records = legacy.get(
        include=["documents", "metadatas"],
        limit=total,
    )

    counts: dict[str, int] = {name: 0 for name in COLLECTION_CONFIGS}

    for i, doc_id in enumerate(all_records["ids"]):
        doc = all_records["documents"][i]
        meta = all_records["metadatas"][i] or {}

        # Classify by nutrient_type
        ntype = meta.get("nutrient_type", "")
        target_name = collection_for_nutrient_type(ntype)

        # Enrich with FSRS defaults
        enriched = _add_fsrs_defaults(meta)

        # Copy to new collection
        targets[target_name].upsert(
            ids=[doc_id],
            documents=[doc],
            metadatas=[enriched],
        )
        counts[target_name] += 1

    logger.info(f"Migration complete: {counts}")
    return counts
