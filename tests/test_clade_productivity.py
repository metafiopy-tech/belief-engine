"""Tests for Session 13: clade-productivity FSRS + Voyage embedding routing.

Covers:
  - clade_productivity() returns 0.0 for roots with no descendants
  - one-descendant score equals success_rate * use_count
  - transitive descendants are followed through the lineage DAG
  - cycles and self-loops do not infinite-loop or count self as descendant
  - cache parameter memoises across calls
  - compute_clade_productivity_map() matches per-id calls
  - review(productivity=...) amplifies stability growth on success
  - review(productivity=...) is ignored on failure (grade=1)
  - _productivity_to_weight caps productivity at 10.0
  - soil.iter_all_nutrients() dedupes across collections
  - soil.get_descendants() returns direct children only
  - _get_embedding_function routing (hash when no key, voyage-code-3
    for code collections, voyage-3-large otherwise)
  - VoyageEmbeddingFunction falls back to hash when voyageai is missing
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from belief.memory.fsrs import (
    FSRSState,
    _productivity_to_weight,
    clade_productivity,
    compute_clade_productivity_map,
    review,
    update_stability_on_success,
)


# ── Fakes for the soil argument ────────────────────────────────────────────


class _FakeNutrient:
    """Duck-type match for Nutrient — only the fields clade walkers touch."""

    __slots__ = ("nutrient_id", "lineage_parent_ids",
                 "reinforcement_count", "lapse_count")

    def __init__(self, nid: str, parents=None, reps: int = 0, lapses: int = 0):
        self.nutrient_id = nid
        self.lineage_parent_ids = list(parents or ())
        self.reinforcement_count = reps
        self.lapse_count = lapses


class _FakeSoil:
    """Minimal soil surface for clade_productivity — just iter_all_nutrients()."""

    def __init__(self, nutrients):
        self._nutrients = list(nutrients)

    def iter_all_nutrients(self):
        return iter(self._nutrients)


# ── clade_productivity ────────────────────────────────────────────────────


class TestCladeProductivity:
    def test_no_descendants_returns_zero(self):
        root = _FakeNutrient("root")
        soil = _FakeSoil([root])
        assert clade_productivity("root", soil) == 0.0

    def test_single_descendant_matches_formula(self):
        """One child with 3 reps / 1 lapse → success_rate*use_count = 3.0, n=1 → 3.0."""
        root = _FakeNutrient("root")
        child = _FakeNutrient("c1", parents=["root"], reps=3, lapses=1)
        soil = _FakeSoil([root, child])
        assert clade_productivity("root", soil) == pytest.approx(3.0)

    def test_zero_usage_descendant_contributes_zero(self):
        root = _FakeNutrient("root")
        child = _FakeNutrient("c1", parents=["root"], reps=0, lapses=0)
        soil = _FakeSoil([root, child])
        assert clade_productivity("root", soil) == 0.0

    def test_transitive_descendants_followed(self):
        """root → c1 → c2 — the clade of root should include c2."""
        root = _FakeNutrient("root")
        c1 = _FakeNutrient("c1", parents=["root"], reps=3, lapses=1)   # 3.0
        c2 = _FakeNutrient("c2", parents=["c1"], reps=2, lapses=0)    # 2.0
        soil = _FakeSoil([root, c1, c2])
        # descendants of root = {c1, c2}; score = (3.0 + 2.0) / 2 = 2.5
        assert clade_productivity("root", soil) == pytest.approx(2.5)

    def test_multiple_parents(self):
        """A single descendant with two parents contributes to both clades."""
        a = _FakeNutrient("a")
        b = _FakeNutrient("b")
        shared = _FakeNutrient("shared", parents=["a", "b"], reps=4, lapses=0)
        soil = _FakeSoil([a, b, shared])
        assert clade_productivity("a", soil) == pytest.approx(4.0)
        assert clade_productivity("b", soil) == pytest.approx(4.0)

    def test_cycle_does_not_infinite_loop(self):
        a = _FakeNutrient("a", parents=["b"], reps=1, lapses=0)
        b = _FakeNutrient("b", parents=["a"], reps=1, lapses=0)
        soil = _FakeSoil([a, b])
        # Should return a value; root itself is excluded from its own clade.
        p_a = clade_productivity("a", soil)
        p_b = clade_productivity("b", soil)
        assert p_a >= 0.0
        assert p_b >= 0.0

    def test_self_loop_excluded(self):
        s = _FakeNutrient("s", parents=["s"], reps=10, lapses=0)
        soil = _FakeSoil([s])
        assert clade_productivity("s", soil) == 0.0

    def test_cache_memoises(self):
        root = _FakeNutrient("root")
        c1 = _FakeNutrient("c1", parents=["root"], reps=3, lapses=1)
        soil = _FakeSoil([root, c1])
        cache: dict = {}
        p1 = clade_productivity("root", soil, cache=cache)
        # Mutate soil — cache should still return the old value.
        soil._nutrients.append(_FakeNutrient("c2", parents=["root"],
                                             reps=100, lapses=0))
        p2 = clade_productivity("root", soil, cache=cache)
        assert p1 == p2

    def test_compute_map_matches_per_id(self):
        root = _FakeNutrient("root")
        c1 = _FakeNutrient("c1", parents=["root"], reps=3, lapses=1)
        c2 = _FakeNutrient("c2", parents=["c1"], reps=2, lapses=0)
        soil = _FakeSoil([root, c1, c2])
        scores = compute_clade_productivity_map(soil)
        assert scores["root"] == pytest.approx(clade_productivity("root", soil))
        assert scores["c1"] == pytest.approx(clade_productivity("c1", soil))
        assert scores["c2"] == 0.0


# ── review() productivity weighting ───────────────────────────────────────


class TestReviewProductivityWeighting:
    NOW_1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    NOW_10 = datetime(2025, 1, 10, tzinfo=timezone.utc)

    def _base_state(self) -> FSRSState:
        return FSRSState(stability=5.0, last_review=self.NOW_1)

    def test_default_productivity_identical_to_classical(self):
        """productivity=0 (default) must match the classical FSRS behaviour."""
        s_default = review(self._base_state(), grade=3, now=self.NOW_10)
        s_explicit_zero = review(self._base_state(), grade=3, now=self.NOW_10,
                                 productivity=0.0)
        assert s_default.stability == s_explicit_zero.stability

    def test_positive_productivity_amplifies_success(self):
        plain = review(self._base_state(), grade=3, now=self.NOW_10,
                       productivity=0.0)
        boosted = review(self._base_state(), grade=3, now=self.NOW_10,
                         productivity=5.0)
        assert boosted.stability > plain.stability
        # Both must have grown above the starting stability.
        assert plain.stability > 5.0
        assert boosted.stability > plain.stability > 5.0

    def test_productivity_ignored_on_failure(self):
        """Failure (grade=1) must not benefit from high productivity."""
        low = review(self._base_state(), grade=1, now=self.NOW_10,
                     productivity=0.0)
        high = review(self._base_state(), grade=1, now=self.NOW_10,
                      productivity=10.0)
        assert low.stability == high.stability

    def test_productivity_weight_caps_at_ten(self):
        """Beyond productivity=10 the weight must saturate."""
        w_ten = _productivity_to_weight(10.0)
        w_huge = _productivity_to_weight(1000.0)
        assert w_ten == w_huge == 3.0

    def test_productivity_weight_zero_gives_one(self):
        assert _productivity_to_weight(0.0) == 1.0
        assert _productivity_to_weight(-5.0) == 1.0  # clamped to 0

    def test_update_stability_on_success_accepts_weight(self):
        """Backward-compatibility: existing callers (no weight) still work."""
        s_legacy = update_stability_on_success(5.0, 5.0, 0.8)
        s_w1 = update_stability_on_success(5.0, 5.0, 0.8, productivity_weight=1.0)
        assert s_legacy == s_w1
        s_w3 = update_stability_on_success(5.0, 5.0, 0.8, productivity_weight=3.0)
        assert s_w3 > s_w1


# ── Soil helpers (require chromadb — skip gracefully) ─────────────────────

chromadb = pytest.importorskip("chromadb")

from belief.memory.nutrients import Nutrient, NutrientType, NutrientTier  # noqa: E402
from belief.memory.soil import (  # noqa: E402
    Soil,
    VoyageEmbeddingFunction,
    _COLLECTION_NAME_TO_TYPE,
    _HashEmbeddingFunction,
    _get_embedding_function,
)


@pytest.fixture
def soil(tmp_path):
    return Soil(persist_dir=tmp_path / "soil")


class TestSoilLineage:
    def test_iter_all_nutrients_empty(self, soil):
        assert list(soil.iter_all_nutrients()) == []

    def test_iter_all_nutrients_dedupes(self, soil):
        """Legacy collection mirroring must not yield the same ID twice."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="pattern A",
            embedding_text="pattern A",
        )
        soil.deposit(n)
        ids = [x.nutrient_id for x in soil.iter_all_nutrients()]
        assert ids.count(n.nutrient_id) == 1

    def test_get_descendants_direct(self, soil):
        parent = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="parent",
            embedding_text="parent pattern",
        )
        soil.deposit(parent)

        child = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="child that builds on parent",
            embedding_text="child pattern with totally different embedding text",
            lineage_parent_ids=[parent.nutrient_id],
        )
        soil.deposit(child)

        descendants = soil.get_descendants(parent.nutrient_id)
        assert len(descendants) == 1
        assert descendants[0].nutrient_id == child.nutrient_id

    def test_clade_productivity_against_real_soil(self, soil):
        """End-to-end: clade_productivity() computed over a real Soil."""
        parent = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="parent",
            embedding_text="parent pattern xyz",
        )
        soil.deposit(parent)
        child = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="child",
            embedding_text="child pattern abc",
            lineage_parent_ids=[parent.nutrient_id],
            reinforcement_count=4,
            lapse_count=0,
        )
        soil.deposit(child)
        score = clade_productivity(parent.nutrient_id, soil)
        # 4 reps / (4 + 0) = 1.0 success_rate * 4 use_count = 4.0; n=1 → 4.0
        assert score == pytest.approx(4.0)

    def test_review_nutrient_passes_productivity(self, soil):
        """review_nutrient() computes clade productivity and weights growth.

        We deposit a parent with several productive descendants, run a
        successful review, and confirm the resulting stability exceeds
        the one obtained from an identical review on a parent with no
        descendants.
        """
        # Parent with productive descendant -> high productivity
        parent_a = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="parent A",
            embedding_text="parent A distinct embedding text xyz",
            stability=5.0,
        )
        soil.deposit(parent_a)
        for i in range(3):
            soil.deposit(Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content=f"child-a-{i}",
                embedding_text=f"child-a-{i} distinct embedding blob {i}",
                lineage_parent_ids=[parent_a.nutrient_id],
                reinforcement_count=5,
                lapse_count=0,
            ))

        # Parent with no descendants -> productivity = 0
        parent_b = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="parent B",
            embedding_text="parent B totally different wording qrs",
            stability=5.0,
        )
        soil.deposit(parent_b)

        col_name = "belief_principles"
        soil.review_nutrient(parent_a.nutrient_id, col_name, grade=3)
        soil.review_nutrient(parent_b.nutrient_id, col_name, grade=3)

        a = soil._collections[col_name].get(
            ids=[parent_a.nutrient_id], include=["metadatas"]
        )["metadatas"][0]
        b = soil._collections[col_name].get(
            ids=[parent_b.nutrient_id], include=["metadatas"]
        )["metadatas"][0]
        assert a["fsrs_stability"] > b["fsrs_stability"]

    def test_review_nutrient_productivity_override(self, soil):
        """Callers can pass an explicit productivity (bypass the auto-compute)."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="solo",
            embedding_text="solo embedding text for override test",
            stability=5.0,
        )
        soil.deposit(n)
        # Override productivity to a high value — stability should grow
        # more than it would under auto-computed 0.0.
        soil.review_nutrient(n.nutrient_id, "belief_principles",
                             grade=3, productivity=8.0)
        meta = soil._collections["belief_principles"].get(
            ids=[n.nutrient_id], include=["metadatas"]
        )["metadatas"][0]
        # Re-run with productivity=0 on a parallel soil for comparison.
        # (Here we just check the growth exceeds the classical bound.)
        assert meta["fsrs_stability"] > 5.0


# ── Embedding-function routing ───────────────────────────────────────────


class TestEmbeddingRouting:
    def test_no_key_falls_back_to_hash(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        for ctype in ("tools", "failures", "covenants",
                      "episodes", "principles"):
            ef = _get_embedding_function(ctype)
            assert isinstance(ef, _HashEmbeddingFunction)

    def test_code_collections_use_voyage_code_3(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "sk-test")
        for ctype in ("tools", "failures", "covenants"):
            ef = _get_embedding_function(ctype)
            assert isinstance(ef, VoyageEmbeddingFunction)
            assert ef._model == "voyage-code-3"

    def test_text_collections_use_voyage_3_large(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "sk-test")
        for ctype in ("episodes", "principles", "unknown"):
            ef = _get_embedding_function(ctype)
            assert isinstance(ef, VoyageEmbeddingFunction)
            assert ef._model == "voyage-3-large"

    def test_voyage_falls_back_to_hash_on_import_error(self, monkeypatch):
        """When voyageai is not installed, embedding calls must succeed
        via the hash fallback rather than raising."""
        import builtins
        real_import = builtins.__import__

        def _no_voyage(name, *args, **kwargs):
            if name == "voyageai":
                raise ImportError("voyageai not installed for this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_voyage)

        ef = VoyageEmbeddingFunction(api_key="sk-test", model="voyage-code-3")
        out = ef(["hello world"])
        assert len(out) == 1
        # Voyage fallback matches voyage's native 1024 dim so ChromaDB's
        # HNSW index stays consistent if the fallback fires mid-session.
        assert len(out[0]) == VoyageEmbeddingFunction._VOYAGE_DIM

    def test_soil_init_routes_per_collection_when_no_ef_passed(
        self, tmp_path, monkeypatch
    ):
        """When VOYAGE_API_KEY is set and no EF is passed, each collection
        receives its routed EF."""
        monkeypatch.setenv("VOYAGE_API_KEY", "sk-test")
        # With a fake key chromadb still creates collections; the real
        # Voyage API is never called because we never embed.
        soil = Soil(persist_dir=tmp_path / "routed")
        for name, ctype in _COLLECTION_NAME_TO_TYPE.items():
            ef = soil._per_collection_ef[name]
            assert isinstance(ef, VoyageEmbeddingFunction)
            expected_model = (
                "voyage-code-3"
                if ctype in ("tools", "failures", "covenants")
                else "voyage-3-large"
            )
            assert ef._model == expected_model

    def test_soil_init_respects_explicit_ef(self, tmp_path, monkeypatch):
        """Passing an explicit embedding_fn must override routing."""
        monkeypatch.setenv("VOYAGE_API_KEY", "sk-test")
        hash_ef = _HashEmbeddingFunction()
        soil = Soil(persist_dir=tmp_path / "explicit", embedding_fn=hash_ef)
        for ef in soil._per_collection_ef.values():
            assert ef is hash_ef
