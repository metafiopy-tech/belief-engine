"""Tests for Soil.revalidate_nutrient() (added in v3.3 Session 2).

These exercise the real ChromaDB-backed Soil and skip when chromadb is
not installed (sandbox without the full dep stack). Mirrors the fixture
style of tests/test_collections.py and tests/test_bitemporal_manifold.py.
"""

from __future__ import annotations

import pytest

# Skip the whole module if chromadb (and its deps) aren't installed.
pytest.importorskip("chromadb")

from belief.memory.nutrients import Nutrient, NutrientType  # noqa: E402
from belief.memory.soil import Soil  # noqa: E402


@pytest.fixture
def soil(tmp_path):
    return Soil(persist_dir=tmp_path / "soil")


def _deposit(soil: Soil, content: str = "x") -> str:
    n = Nutrient(
        nutrient_type=NutrientType.PATTERN,
        content=content,
        embedding_text=content,
    )
    soil.deposit(n)
    return n.nutrient_id


def test_revalidate_restores_after_invalidate(soil: Soil) -> None:
    nid = _deposit(soil, "test pattern")
    assert soil.invalidate_nutrient(nid, "predator: low utility") is True
    fetched = soil.get(nid)
    assert fetched is not None
    assert fetched.valid_until > 0  # invalidated
    assert fetched.invalidation_reason == "predator: low utility"

    assert soil.revalidate_nutrient(nid) is True
    after = soil.get(nid)
    assert after is not None
    assert after.valid_until == 0.0
    assert after.invalidation_reason == ""
    assert after.is_active()


def test_revalidate_missing_returns_false(soil: Soil) -> None:
    assert soil.revalidate_nutrient("nutrient-that-does-not-exist") is False


def test_revalidate_already_active_is_noop(soil: Soil) -> None:
    """Calling revalidate on a never-invalidated nutrient returns False, no error."""
    nid = _deposit(soil, "active")
    assert soil.revalidate_nutrient(nid) is False
    fetched = soil.get(nid)
    assert fetched is not None
    assert fetched.is_active()


def test_revalidated_nutrient_returns_to_retrieve_results(soil: Soil) -> None:
    """The whole point: revalidated nutrients re-enter live retrieval."""
    nid = _deposit(soil, "needle pattern that survives revalidation")
    soil.invalidate_nutrient(nid, "test")
    # Confirm it's filtered out while invalidated.
    hits_pre = [n for n in soil.iter_all_nutrients(include_invalidated=False)]
    assert all(n.nutrient_id != nid for n in hits_pre)

    soil.revalidate_nutrient(nid)
    hits_post = [n for n in soil.iter_all_nutrients(include_invalidated=False)]
    assert any(n.nutrient_id == nid for n in hits_post)
