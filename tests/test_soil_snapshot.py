"""Tests for STARVED-arm soil isolation + snapshot extraction (Session 2).

Gate-safe: the pure resolver and the extractor/persistence paths run anywhere
(fake soil + fake deterministic encoder). The test that constructs a real Soil
is guarded by ``importorskip("chromadb")`` so it runs on the Mac gate and skips
cleanly in the lightweight sandbox; the real MiniLM encode path is verified on
the Mac (sentence-transformers is not a core dep).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from belief.experiments.soil_snapshot import (
    GenerationSnapshot,
    MiniLMEncoder,
    extract_soil_matrix,
    load_snapshot,
    save_snapshot,
    snapshot_soil,
)
from belief.memory.soil import default_soil_dir


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeNutrient:
    def __init__(self, nutrient_id: str, embedding_text: str = "", content: str = ""):
        self.nutrient_id = nutrient_id
        self.embedding_text = embedding_text
        self.content = content


class _FakeSoil:
    """Duck-typed stand-in exposing only iter_all_nutrients."""

    def __init__(self, active, invalidated=()):
        self._active = list(active)
        self._invalidated = list(invalidated)

    def iter_all_nutrients(self, include_invalidated: bool = True):
        yield from self._active
        if include_invalidated:
            yield from self._invalidated


class _FakeEncoder:
    """Deterministic, process-independent encoder for tests (dim=8)."""

    fingerprint = "fake@v0:dim8:norm"

    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts):
        self.seen = list(texts)
        rows = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).normal(size=8))
        return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 8))


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_default_soil_dir_unset_is_historical(monkeypatch):
    monkeypatch.delenv("BELIEF_SOIL_PATH", raising=False)
    assert default_soil_dir().name == "soil"
    assert str(default_soil_dir()).endswith("/.belief-engine/soil")


def test_default_soil_dir_empty_is_historical(monkeypatch):
    monkeypatch.setenv("BELIEF_SOIL_PATH", "   ")
    assert str(default_soil_dir()).endswith("/.belief-engine/soil")


def test_default_soil_dir_override(monkeypatch, tmp_path):
    target = tmp_path / "arm_fed" / "soil"
    monkeypatch.setenv("BELIEF_SOIL_PATH", str(target))
    assert default_soil_dir() == target


def test_default_soil_dir_expands_user(monkeypatch):
    monkeypatch.setenv("BELIEF_SOIL_PATH", "~/some/arm/soil")
    resolved = default_soil_dir()
    assert "~" not in str(resolved)
    assert str(resolved).endswith("/some/arm/soil")


def test_soil_constructor_honors_env(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    from belief.memory.soil import Soil

    target = tmp_path / "starved" / "soil"
    monkeypatch.setenv("BELIEF_SOIL_PATH", str(target))
    soil = Soil()  # persist_dir=None -> resolver
    assert soil._persist_dir == target
    assert target.exists()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def test_extract_matrix_matches_nutrients():
    soil = _FakeSoil(
        [
            _FakeNutrient("c", "gamma pattern"),
            _FakeNutrient("a", "alpha pattern"),
            _FakeNutrient("b", "beta pattern"),
        ]
    )
    enc = _FakeEncoder()
    ids, X = extract_soil_matrix(soil, enc)
    assert ids == ["a", "b", "c"]  # sorted by id for determinism
    assert X.shape == (3, 8)
    # Encoder saw texts in id-sorted order.
    assert enc.seen == ["alpha pattern", "beta pattern", "gamma pattern"]


def test_extract_excludes_invalidated_by_default():
    soil = _FakeSoil(
        [_FakeNutrient("a", "alpha")],
        invalidated=[_FakeNutrient("z", "zombie")],
    )
    ids, X = extract_soil_matrix(soil, _FakeEncoder())
    assert ids == ["a"]
    assert X.shape == (1, 8)


def test_extract_can_include_invalidated():
    soil = _FakeSoil(
        [_FakeNutrient("a", "alpha")],
        invalidated=[_FakeNutrient("z", "zombie")],
    )
    ids, _ = extract_soil_matrix(soil, _FakeEncoder(), include_invalidated=True)
    assert ids == ["a", "z"]


def test_extract_falls_back_to_content_when_no_embedding_text():
    soil = _FakeSoil([_FakeNutrient("a", "", content="fallback body")])
    enc = _FakeEncoder()
    extract_soil_matrix(soil, enc)
    assert enc.seen == ["fallback body"]


def test_extract_empty_soil():
    ids, X = extract_soil_matrix(_FakeSoil([]), _FakeEncoder())
    assert ids == []
    assert X.shape == (0, 8)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _sample_snapshot() -> GenerationSnapshot:
    rng = np.random.default_rng(0)
    return GenerationSnapshot(
        gen=7,
        arm="STARVED",
        nutrient_ids=["a", "b", "c"],
        X=rng.normal(size=(3, 8)),
        encoder_fingerprint="fake@v0:dim8:norm",
        kmeans_k=8,
    )


def test_save_load_roundtrip(tmp_path):
    snap = _sample_snapshot()
    npz = save_snapshot(snap, tmp_path)
    assert npz.exists()
    assert npz.with_suffix(".json").exists()
    loaded = load_snapshot(npz)
    assert loaded.gen == snap.gen
    assert loaded.arm == snap.arm
    assert loaded.nutrient_ids == snap.nutrient_ids
    assert loaded.encoder_fingerprint == snap.encoder_fingerprint
    assert loaded.kmeans_k == snap.kmeans_k
    assert np.array_equal(loaded.X, snap.X)


def test_snapshot_filename_encodes_arm_and_gen(tmp_path):
    npz = save_snapshot(_sample_snapshot(), tmp_path)
    assert npz.name == "STARVED_gen007.npz"


def test_snapshot_soil_wrapper_writes_and_returns(tmp_path):
    soil = _FakeSoil([_FakeNutrient("a", "alpha"), _FakeNutrient("b", "beta")])
    snap = snapshot_soil(soil, _FakeEncoder(), gen=2, arm="FED", kmeans_k=8, dest_dir=tmp_path)
    assert snap.n_nutrients == 2
    assert snap.encoder_fingerprint == "fake@v0:dim8:norm"
    assert snap.kmeans_k == 8
    assert (tmp_path / "FED_gen002.npz").exists()


def test_snapshot_soil_wrapper_no_persist_when_dest_none():
    soil = _FakeSoil([_FakeNutrient("a", "alpha")])
    snap = snapshot_soil(soil, _FakeEncoder(), gen=0, arm="FED", kmeans_k=8)
    assert snap.n_nutrients == 1


# ---------------------------------------------------------------------------
# MiniLM fingerprint (no model download)
# ---------------------------------------------------------------------------


def test_minilm_fingerprint_format_before_load():
    enc = MiniLMEncoder(revision="abc123")
    fp = enc.fingerprint
    assert "all-MiniLM-L6-v2" in fp
    assert "@abc123" in fp
    assert ":norm" in fp  # default normalize=True


def test_minilm_fingerprint_reflects_normalize_flag():
    enc = MiniLMEncoder(normalize=False)
    assert enc.fingerprint.endswith(":raw")
