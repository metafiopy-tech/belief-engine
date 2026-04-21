"""Tests for OMNI-EPIC novelty bands and interestingness judge parsing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from belief.photosynthesis.synthesis.archives import ArchiveManager, Neighbor
from belief.photosynthesis.synthesis.novelty import (
    DISTINCT_THRESHOLD,
    HARD_DUP_THRESHOLD,
    _parse_verdict,
    async_score_novelty,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeArchive:
    """Just enough of ArchiveManager to satisfy the novelty pipeline."""

    def __init__(self, neighbors: list[Neighbor]) -> None:
        self._neighbors = neighbors

    def query_neighbors(self, collection: str, embedding: Any, *, top_k: int = 10):
        return list(self._neighbors[:top_k])


def _embedder(text: str) -> list[float]:
    # The tests only care that the embedder is callable; the FakeArchive
    # doesn't use the embedding at all.
    return [0.0] * 8


# ---------------------------------------------------------------------------
# Band routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_archive_accepts_with_max_novelty() -> None:
    arch = FakeArchive([])
    seed = {"title": "new MCP server", "summary": "exposes weather tool"}
    r = await async_score_novelty(seed, archive=arch, embedder=_embedder)
    assert r.accepted is True
    assert r.reason == "archive_empty"
    assert r.novelty == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_hard_duplicate_band_rejects_without_judge_call() -> None:
    arch = FakeArchive(
        [Neighbor(goal_id="g1", title="similar", cosine=0.95)]
    )
    called = []

    async def judge(prompt: str) -> str:
        called.append(prompt)
        return '{"interesting": true, "category": "a"}'

    seed = {"title": "near-duplicate", "summary": "x"}
    r = await async_score_novelty(
        seed, archive=arch, embedder=_embedder, llm_judge=judge
    )
    assert r.accepted is False
    assert r.reason == "hard_duplicate"
    assert r.novelty == 0.0
    assert called == [], "judge must not be invoked in hard-duplicate band"


@pytest.mark.asyncio
async def test_distinct_band_accepts_without_judge_call() -> None:
    arch = FakeArchive(
        [Neighbor(goal_id="g1", title="unrelated", cosine=0.40)]
    )
    called = []

    async def judge(prompt: str) -> str:
        called.append(prompt)
        return ""

    seed = {"title": "clearly novel", "summary": "x"}
    r = await async_score_novelty(
        seed, archive=arch, embedder=_embedder, llm_judge=judge
    )
    assert r.accepted is True
    assert r.reason == "distinct"
    assert r.novelty > 0.5
    assert called == [], "judge must not be invoked below the mid-band"


@pytest.mark.asyncio
async def test_mid_band_without_judge_rejects() -> None:
    arch = FakeArchive(
        [Neighbor(goal_id="g1", title="related", cosine=0.85)]
    )
    seed = {"title": "kinda similar", "summary": "x"}
    r = await async_score_novelty(seed, archive=arch, embedder=_embedder)
    assert r.accepted is False
    assert r.reason == "mid_no_judge"


@pytest.mark.asyncio
async def test_mid_band_with_interesting_judge_accepts() -> None:
    arch = FakeArchive(
        [Neighbor(goal_id="g1", title="related", cosine=0.82)]
    )

    async def judge(_prompt: str) -> str:
        return json.dumps(
            {
                "interesting": True,
                "category": "b",
                "nearest_archived_goal_id": "g1",
                "one_line_justification": "combines MCP + FSRS novelly",
            }
        )

    seed = {"title": "combo", "summary": "x"}
    r = await async_score_novelty(
        seed, archive=arch, embedder=_embedder, llm_judge=judge
    )
    assert r.accepted is True
    assert r.reason == "judge_b"
    assert r.judge_verdict is not None
    assert r.judge_verdict["interesting"] is True


@pytest.mark.asyncio
async def test_mid_band_with_judge_saying_not_interesting_rejects() -> None:
    arch = FakeArchive(
        [Neighbor(goal_id="g1", title="related", cosine=0.82)]
    )

    async def judge(_prompt: str) -> str:
        return '{"interesting": false, "category": "x"}'

    r = await async_score_novelty(
        {"title": "refactor", "summary": "x"},
        archive=arch,
        embedder=_embedder,
        llm_judge=judge,
    )
    assert r.accepted is False
    assert r.reason == "judge_x"
    assert r.novelty == 0.0


# ---------------------------------------------------------------------------
# Verdict parsing robustness
# ---------------------------------------------------------------------------


def test_parse_verdict_strict_json() -> None:
    v = _parse_verdict('{"interesting": true, "category": "a"}')
    assert v is not None and v["interesting"] is True


def test_parse_verdict_pulls_json_from_prose() -> None:
    raw = 'Sure! Here is my verdict:\n{"interesting": false, "category": "z"}\nThanks.'
    v = _parse_verdict(raw)
    assert v is not None and v["interesting"] is False


def test_parse_verdict_rejects_non_dict() -> None:
    assert _parse_verdict("[1,2,3]") is None


def test_parse_verdict_rejects_missing_interesting_key() -> None:
    assert _parse_verdict('{"category": "a"}') is None


def test_parse_verdict_rejects_non_bool_interesting() -> None:
    assert _parse_verdict('{"interesting": "yes"}') is None


def test_parse_verdict_normalizes_unknown_category() -> None:
    v = _parse_verdict('{"interesting": true, "category": "weird"}')
    assert v is not None and v["category"] == "?"
