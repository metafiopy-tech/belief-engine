"""Per-generation soil-embedding snapshots for the STARVED-arm experiment.

Produces the matrix ``X_n`` (one row per active nutrient) that the variance-decay
metrics in :mod:`belief.experiments.variance_decay` consume, and persists it to
disk so metrics are computed offline and never perturb a run.

Design (see ``docs/experiments/starved_arm_design.md`` §2.3):

- The **measurement encoder is independent of the soil's storage EF.** ChromaDB
  may store hash-EF or Voyage embeddings for dedup/retrieval, but the snapshot
  re-embeds each nutrient's ``embedding_text`` with a single **pinned** encoder
  so PR / Hill measure semantic spread on a frozen basis. The default is a
  pinned ``all-MiniLM-L6-v2`` revision.
- The encoder exposes a **fingerprint** (model + revision + dim) that is written
  into every snapshot. The experiment driver refuses to proceed if the pilot and
  full-run fingerprints differ — an encoder that drifts silently voids the
  pre-registration.
- The encoder is **injectable** so the matrix extractor and persistence are unit
  testable without downloading a model (tests pass a deterministic fake); the
  real MiniLM path is exercised on the Mac where ``sentence-transformers`` is
  installed (it ships in the ``photosynthesis`` / ``full`` extras, not the core
  hard gate).

This module is import-light: ``sentence-transformers`` is imported lazily inside
:class:`MiniLMEncoder`, so importing this module never requires the heavy dep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

# Pinned MiniLM identity. The revision is asserted into every snapshot's
# fingerprint; pin it to an exact commit before the pilot so pilot and full-run
# snapshots are guaranteed to share an encoder.
DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# NOTE: pin to an exact HF commit SHA before the pilot. "main" is a placeholder
# that must be replaced — the driver's fingerprint check is what enforces it.
DEFAULT_MINILM_REVISION = "main"


@runtime_checkable
class SnapshotEncoder(Protocol):
    """A frozen text encoder used to build snapshot matrices.

    Implementations must be deterministic for a given input and expose a stable
    ``fingerprint`` identifying the exact model/revision/dim.
    """

    @property
    def fingerprint(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float matrix."""
        ...


class MiniLMEncoder:
    """Pinned ``all-MiniLM-L6-v2`` encoder (lazy ``sentence-transformers``).

    The model is loaded on first ``encode`` so merely constructing the encoder
    (and importing this module) is cheap and dependency-free.
    """

    def __init__(
        self,
        model: str = DEFAULT_MINILM_MODEL,
        revision: str = DEFAULT_MINILM_REVISION,
        *,
        normalize: bool = True,
    ) -> None:
        self._model_name = model
        self._revision = revision
        self._normalize = normalize
        self._model = None  # lazily constructed SentenceTransformer
        self._dim: Optional[int] = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, revision=self._revision)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def fingerprint(self) -> str:
        dim = self._dim if self._dim is not None else "?"
        norm = "norm" if self._normalize else "raw"
        return f"{self._model_name}@{self._revision}:dim{dim}:{norm}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        if len(texts) == 0:
            return np.zeros((0, self._dim or 0), dtype=np.float64)
        vecs = model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Matrix extraction
# ---------------------------------------------------------------------------


def extract_soil_matrix(
    soil,
    encoder: SnapshotEncoder,
    *,
    include_invalidated: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Re-embed the active soil cloud into ``(nutrient_ids, X)``.

    Walks ``soil.iter_all_nutrients`` (active-only by default — invalidated and
    archived nutrients are excluded so the matrix reflects live retention),
    embeds each nutrient's ``embedding_text`` with the frozen ``encoder``, and
    returns the ids and the ``(n, dim)`` matrix. Rows are sorted by nutrient_id
    so the matrix is deterministic regardless of collection iteration order.
    """
    items = [
        (n.nutrient_id, n.embedding_text or n.content or "")
        for n in soil.iter_all_nutrients(include_invalidated=include_invalidated)
    ]
    items.sort(key=lambda t: t[0])
    ids = [i for i, _ in items]
    texts = [t for _, t in items]
    X = encoder.encode(texts)
    if X.shape[0] != len(ids):
        raise ValueError(f"encoder returned {X.shape[0]} rows for {len(ids)} nutrients")
    return ids, X


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationSnapshot:
    """One persisted per-(generation, arm) embedding snapshot.

    ``X`` is the embedding matrix; ``nutrient_ids`` aligns row-for-row with it.
    ``encoder_fingerprint`` and ``kmeans_k`` are carried so downstream
    adjudication can assert the encoder never drifted and the clustering
    hyperparameter stayed frozen.
    """

    gen: int
    arm: str
    nutrient_ids: list[str]
    X: np.ndarray
    encoder_fingerprint: str
    kmeans_k: int

    @property
    def n_nutrients(self) -> int:
        return len(self.nutrient_ids)


def _meta_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".json")


def save_snapshot(snapshot: GenerationSnapshot, dest_dir: Path) -> Path:
    """Persist a snapshot as ``<arm>_gen<NN>.npz`` + a JSON sidecar.

    The matrix goes to the ``.npz`` (compact, exact float round-trip); the
    sidecar holds ids, fingerprint, k, and shape for human inspection and for
    cheap metadata scans without loading the matrix.
    """
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = f"{snapshot.arm}_gen{snapshot.gen:03d}"
    npz_path = dest_dir / f"{name}.npz"
    np.savez_compressed(npz_path, X=snapshot.X)
    meta = {
        "gen": snapshot.gen,
        "arm": snapshot.arm,
        "nutrient_ids": snapshot.nutrient_ids,
        "encoder_fingerprint": snapshot.encoder_fingerprint,
        "kmeans_k": snapshot.kmeans_k,
        "n_nutrients": snapshot.n_nutrients,
        "shape": list(snapshot.X.shape),
    }
    _meta_path(npz_path).write_text(json.dumps(meta, indent=2, sort_keys=True))
    return npz_path


def load_snapshot(npz_path: Path) -> GenerationSnapshot:
    """Load a snapshot written by :func:`save_snapshot`."""
    npz_path = Path(npz_path).expanduser()
    meta = json.loads(_meta_path(npz_path).read_text())
    with np.load(npz_path) as data:
        X = np.asarray(data["X"], dtype=np.float64)
    return GenerationSnapshot(
        gen=int(meta["gen"]),
        arm=str(meta["arm"]),
        nutrient_ids=list(meta["nutrient_ids"]),
        X=X,
        encoder_fingerprint=str(meta["encoder_fingerprint"]),
        kmeans_k=int(meta["kmeans_k"]),
    )


def snapshot_soil(
    soil,
    encoder: SnapshotEncoder,
    *,
    gen: int,
    arm: str,
    kmeans_k: int,
    dest_dir: Optional[Path] = None,
    include_invalidated: bool = False,
) -> GenerationSnapshot:
    """Extract the active soil cloud and (optionally) persist it.

    Convenience wrapper: build the matrix with the frozen encoder, stamp it with
    the encoder fingerprint + frozen ``kmeans_k``, and write it to ``dest_dir``
    if given. Returns the in-memory snapshot either way.
    """
    ids, X = extract_soil_matrix(soil, encoder, include_invalidated=include_invalidated)
    snap = GenerationSnapshot(
        gen=gen,
        arm=arm,
        nutrient_ids=ids,
        X=X,
        encoder_fingerprint=encoder.fingerprint,
        kmeans_k=kmeans_k,
    )
    if dest_dir is not None:
        save_snapshot(snap, Path(dest_dir))
    return snap
