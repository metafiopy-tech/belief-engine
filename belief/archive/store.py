"""AgentArchive — ChromaDB-backed store for BuildOutcomes (Session 6 v3.2).

Thin wrapper that owns one ChromaDB collection named ``agent_archive``
and exposes three operations:

* :meth:`persist(outcome)` — writes a BuildOutcome, embedding its
  goal+planner-config as the document.
* :meth:`query_by_goal(goal, k)` — returns the top-k semantically
  similar past outcomes.
* :meth:`size()` — count.

Two design choices worth calling out:

1. **Embedding model.**  We use ChromaDB's default embedding function
   (``DefaultEmbeddingFunction``) — ONNX-backed all-MiniLM-L6-v2.  It
   ships with chromadb and needs no extra deps.  Switching to voyage
   or to a sentence-transformers model is a one-line swap.

2. **In-memory vs persistent.**  Default is persistent at
   ``~/.belief-engine/agent_archive/``.  Tests pass a ``path=None``
   override and use an EphemeralClient so no disk artefacts leak
   across runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.archive.store")


_DEFAULT_PERSIST_DIR = Path.home() / ".belief-engine" / "agent_archive"
_COLLECTION_NAME = "agent_archive"


class AgentArchive:
    """Thin ChromaDB wrapper for the session-6 archive.

    The internal ChromaDB client is lazily built on first use so that
    importing :mod:`belief.archive` doesn't force chromadb to
    initialise (chromadb's first import is ~2s and pulls ONNX
    runtime).  Callers who only read dataclasses from
    :mod:`belief.archive.config` don't pay that cost.
    """

    def __init__(
        self,
        *,
        path: Path | None = None,
        ephemeral: bool = False,
        embedding_function: Any = None,
    ) -> None:
        self._path = path or _DEFAULT_PERSIST_DIR
        self._ephemeral = ephemeral
        self._client: Any = None
        self._collection: Any = None
        # Optional embedding-function override — tests pass a
        # deterministic stub to avoid the ONNX download on first run.
        self._embedding_function_override = embedding_function

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure(self) -> None:
        if self._collection is not None:
            return
        import chromadb  # local import keeps module load cheap

        if self._ephemeral:
            self._client = chromadb.EphemeralClient()
        else:
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._path))

        # DefaultEmbeddingFunction ships with chromadb; no extra deps.
        if self._embedding_function_override is not None:
            ef = self._embedding_function_override
        else:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            ef = DefaultEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=ef,
            metadata={"purpose": "session-6 DGM-style agent-outcome archive"},
        )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def persist(self, outcome: "BuildOutcome") -> None:  # noqa: F821 - forward ref
        """Write a BuildOutcome as one ChromaDB row.

        The collection's primary key is ``outcome.run_id``.  If a row
        with that ID exists, ChromaDB will overwrite it — which is the
        right semantics (re-running a challenge under the same run_id
        replaces the old outcome).
        """
        from belief.archive.fitness import utility

        self._ensure()
        doc = outcome.embedding_text()
        u = utility(outcome)
        metadata = {
            "goal": outcome.goal[:500],
            "verdict": outcome.verdict,
            "weighted_score": float(outcome.weighted_score),
            "wallclock_s": float(outcome.wallclock_s),
            "cost_usd": float(outcome.estimated_cost_usd),
            "utility_score": float(u),
            "trajectory_signature": outcome.trajectory_signature,
            "outcome_json": outcome.to_json(),  # full round-trip payload
        }
        self._collection.upsert(
            ids=[outcome.run_id],
            documents=[doc],
            metadatas=[metadata],
        )
        logger.info(
            "AgentArchive: persisted %s (verdict=%s, score=%.2f, U=%.3f)",
            outcome.run_id,
            outcome.verdict,
            outcome.weighted_score,
            u,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_by_goal(
        self,
        goal: str,
        *,
        k: int = 5,
        verdicts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k most semantically similar past outcomes.

        ``verdicts`` filters on the ``verdict`` metadata — default is
        ``["pass", "fail_fixable"]`` so we don't learn from hard
        failures.  Pass ``None`` or ``[]`` to disable the filter.

        Returns a list of dicts with keys ``id``, ``distance``,
        ``metadata``, and ``outcome`` (a rehydrated BuildOutcome).
        """
        from belief.archive.outcome import BuildOutcome

        self._ensure()
        size = self._collection.count()
        if size == 0:
            return []
        effective_k = min(k, size)
        where: dict[str, Any] | None = None
        allowed = verdicts if verdicts is not None else ["pass", "fail_fixable"]
        if allowed:
            where = {"verdict": {"$in": list(allowed)}}

        try:
            result = self._collection.query(
                query_texts=[f"GOAL: {goal}"],
                n_results=effective_k,
                where=where,
            )
        except Exception as e:
            logger.debug("AgentArchive query failed: %s", e)
            return []

        out: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        distances = (
            result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)
        )
        metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else [{}] * len(ids)
        for i, id_ in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            outcome_obj: BuildOutcome | None = None
            raw = meta.get("outcome_json") if meta else None
            if raw:
                try:
                    outcome_obj = BuildOutcome.from_json(raw)
                except Exception as e:
                    logger.debug("outcome rehydration failed for %s: %s", id_, e)
            out.append(
                {
                    "id": id_,
                    "distance": distances[i] if i < len(distances) else 0.0,
                    "metadata": meta or {},
                    "outcome": outcome_obj,
                }
            )
        return out

    def size(self) -> int:
        self._ensure()
        return int(self._collection.count())


__all__ = ["AgentArchive"]
