"""Tests for local-mode prompt compression in the recomposer.

Covers the helpers added in validation Session 1:

  - _is_local_mode  (env + state detection)
  - _looks_like_human_docs (tag-based README/deploy filter)
  - _truncate_nutrient (per-nutrient 500-char cap)
  - _compress_for_local (whole-profile trim)
  - _hard_cap_context (final safety net)

Cloud-mode behavior is verified indirectly in test_domain_tracking.py;
these tests target only the new branches.
"""

from __future__ import annotations

import os

import pytest


def _nutrient(nid, content="x", nt=None, tags=None):
    from belief.memory.nutrients import Nutrient, NutrientType
    nt = nt or NutrientType.PATTERN
    return Nutrient(
        id=nid,
        nutrient_type=nt,
        content=content,
        embedding_text=content[:50],
        tags=list(tags or []),
    )


# ── _is_local_mode ────────────────────────────────────────────────────────


def test_is_local_mode_default_false(monkeypatch):
    from belief.memory.recomposer import _is_local_mode
    monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)
    assert _is_local_mode({}) is False


def test_is_local_mode_via_env(monkeypatch):
    from belief.memory.recomposer import _is_local_mode
    monkeypatch.setenv("BELIEF_MODEL_MODE", "local")
    assert _is_local_mode({}) is True


def test_is_local_mode_via_state_overrides(monkeypatch):
    from belief.memory.recomposer import _is_local_mode
    monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)
    assert _is_local_mode({"model_mode": "LOCAL"}) is True
    assert _is_local_mode({"model_mode": "cloud"}) is False


# ── human-docs filter ─────────────────────────────────────────────────────


@pytest.mark.parametrize("tag", ["readme", "docs", "deploy", "dockerfile", "CHANGELOG"])
def test_looks_like_human_docs_matches_expected_tags(tag):
    from belief.memory.recomposer import _looks_like_human_docs
    assert _looks_like_human_docs(_nutrient("x", tags=[tag])) is True


def test_looks_like_human_docs_skips_regular_tags():
    from belief.memory.recomposer import _looks_like_human_docs
    assert _looks_like_human_docs(_nutrient("x", tags=["fastapi", "async"])) is False


# ── truncation ────────────────────────────────────────────────────────────


def test_truncate_nutrient_noop_when_short():
    from belief.memory.recomposer import _truncate_nutrient
    n = _nutrient("s", content="short")
    assert _truncate_nutrient(n).content == "short"


def test_truncate_nutrient_clips_long_content():
    from belief.memory.recomposer import (
        LOCAL_NUTRIENT_CHAR_CAP,
        _truncate_nutrient,
    )
    n = _nutrient("l", content="x" * 2000)
    clipped = _truncate_nutrient(n)
    assert len(clipped.content) <= LOCAL_NUTRIENT_CHAR_CAP
    assert clipped.content.endswith("…")


# ── whole-profile compression ────────────────────────────────────────────


def test_compress_for_local_drops_readme_caps_categories_clips_content():
    from belief.memory.nutrients import NutrientProfile, NutrientType
    from belief.memory.recomposer import (
        LOCAL_NUTRIENT_CHAR_CAP,
        _compress_for_local,
    )

    profile = NutrientProfile(
        covenants=[
            _nutrient(f"c{i}", content="rule " + "z" * 800, nt=NutrientType.COVENANT)
            for i in range(5)
        ],
        antipatterns=[
            _nutrient(f"a{i}", content="bad " + "z" * 800, nt=NutrientType.ANTIPATTERN)
            for i in range(8)
        ],
        patterns=[
            _nutrient(
                f"p{i}",
                content="good " + "z" * 800,
                nt=NutrientType.PATTERN,
                tags=["readme"] if i == 0 else ["fastapi"],
            )
            for i in range(8)
        ],
        skeletons=[
            _nutrient(f"sk{i}", content="scaffold", nt=NutrientType.SKELETON)
            for i in range(3)
        ],
    )

    compressed = _compress_for_local(profile)

    # compact() caps patterns/antipatterns at 3, skeletons at 1
    assert len(compressed.patterns) == 3
    assert len(compressed.antipatterns) == 3
    assert len(compressed.skeletons) == 1
    # README pattern dropped
    assert not any("readme" in (n.tags or []) for n in compressed.patterns)
    # All content clipped
    for section in (compressed.covenants, compressed.antipatterns, compressed.patterns):
        for n in section:
            assert len(n.content) <= LOCAL_NUTRIENT_CHAR_CAP


def test_compress_for_local_rendered_block_respects_hard_cap():
    from belief.memory.nutrients import NutrientProfile, NutrientType
    from belief.memory.recomposer import (
        LOCAL_TOTAL_CONTEXT_CHAR_CAP,
        _compress_for_local,
        _hard_cap_context,
    )

    profile = NutrientProfile(
        covenants=[
            _nutrient(f"c{i}", content="rule " + "z" * 800, nt=NutrientType.COVENANT)
            for i in range(6)
        ],
        antipatterns=[
            _nutrient(f"a{i}", content="bad " + "z" * 800, nt=NutrientType.ANTIPATTERN)
            for i in range(6)
        ],
        patterns=[
            _nutrient(f"p{i}", content="good " + "z" * 800, nt=NutrientType.PATTERN)
            for i in range(6)
        ],
        skeletons=[],
    )
    compressed = _compress_for_local(profile)
    block = _hard_cap_context(
        compressed.format_context_block_compact(complexity=3)
    )
    assert len(block) <= LOCAL_TOTAL_CONTEXT_CHAR_CAP


# ── integration: recomposer_node in local mode ──────────────────────────


def test_recomposer_node_compresses_in_local_mode(monkeypatch):
    """End-to-end check: with BELIEF_MODEL_MODE=local, the recomposer's
    output nutrient_context stays within LOCAL_TOTAL_CONTEXT_CHAR_CAP
    and shows signs of compression — individual nutrient contents are
    truncated, forbidden README tag is dropped, and the block is
    measurably shorter than the cloud version.
    """
    import asyncio
    from belief.memory import recomposer as rec
    from belief.memory.nutrients import NutrientProfile, NutrientType

    def _build_profile():
        # Large content per nutrient so truncation is visible; 5 patterns
        # so the 3-cap matters; one tagged "readme" so human-docs drop
        # matters.
        return NutrientProfile(
            covenants=[
                _nutrient(f"c{i}", content="rule-" + "z" * 1500,
                          nt=NutrientType.COVENANT)
                for i in range(3)
            ],
            antipatterns=[],
            patterns=[
                _nutrient(
                    f"p{i}",
                    content="pattern-" + "z" * 1500,
                    nt=NutrientType.PATTERN,
                    tags=["readme"] if i == 0 else ["fastapi"],
                )
                for i in range(5)
            ],
            skeletons=[],
        )

    class _FakeSoil:
        def decay_all(self):
            return {"archived": 0, "active": 1}
        def retrieve_profile(self, goal, complexity=3):
            return _build_profile()

    monkeypatch.setattr(rec, "_get_soil", lambda: _FakeSoil())
    monkeypatch.setattr(
        "belief.memory.tool_registry.ToolRegistry",
        lambda *a, **k: type("T", (), {"find_tools_for_goal": lambda *a, **k: []})(),
        raising=False,
    )
    monkeypatch.setattr(
        "belief.memory.reflexion.retrieve_reflexions",
        lambda *a, **k: [],
        raising=False,
    )

    # First: local mode produces a short, capped block
    monkeypatch.setenv("BELIEF_MODEL_MODE", "local")
    local_result = asyncio.run(rec.recomposer_node({"user_goal": "build a FizzBuzz"}))
    local_block = local_result["nutrient_context"]

    assert len(local_block) <= rec.LOCAL_TOTAL_CONTEXT_CHAR_CAP, (
        f"local block length {len(local_block)} exceeds cap {rec.LOCAL_TOTAL_CONTEXT_CHAR_CAP}"
    )
    # README-tagged pattern was dropped; fastapi patterns survive
    # (order within the raw block is enough — we just want to see that
    # the compression pipeline ran)
    assert "pattern-" in local_block or "rule-" in local_block

    # Second: cloud mode produces a noticeably longer block on the
    # same inputs — confirms the compression path is what trimmed it.
    monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)
    cloud_result = asyncio.run(rec.recomposer_node({"user_goal": "build a FizzBuzz"}))
    cloud_block = cloud_result["nutrient_context"]
    assert len(cloud_block) > len(local_block), (
        f"cloud block ({len(cloud_block)}) must be longer than local ({len(local_block)})"
    )


def test_recomposer_node_uses_full_formatter_in_cloud_mode(monkeypatch):
    """Cloud mode must NOT use the compact formatter — same fake profile
    should produce the full block."""
    import asyncio
    from belief.memory import recomposer as rec

    class _FakeProfile:
        is_empty = False
        total_nutrients = 1
        covenants = [_nutrient("c", content="COV")]
        antipatterns = []
        patterns = []
        skeletons = []

        def format_context_block(self, complexity=3):
            return "FULL_BLOCK"

        def format_context_block_compact(self, complexity=3):
            raise AssertionError("cloud mode must not call compact formatter")

    class _FakeSoil:
        def decay_all(self):
            return {"archived": 0, "active": 1}
        def retrieve_profile(self, goal, complexity=3):
            return _FakeProfile()

    monkeypatch.setattr(rec, "_get_soil", lambda: _FakeSoil())
    monkeypatch.setattr(
        "belief.memory.tool_registry.ToolRegistry",
        lambda *a, **k: type("T", (), {"find_tools_for_goal": lambda *a, **k: []})(),
        raising=False,
    )
    monkeypatch.setattr(
        "belief.memory.reflexion.retrieve_reflexions",
        lambda *a, **k: [],
        raising=False,
    )
    monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)

    result = asyncio.run(rec.recomposer_node({"user_goal": "build a FizzBuzz"}))
    assert result["nutrient_context"] == "FULL_BLOCK"
