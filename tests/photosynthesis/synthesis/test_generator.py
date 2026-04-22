"""Tests for the goal-spec generator (Sonnet sampler + post-dup check)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from belief.photosynthesis.synthesis.archives import Neighbor
from belief.photosynthesis.synthesis.generator import (
    GoalSpec,
    synthesize,
)


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


GOOD_SPEC_JSON = {
    "goal_id": "mcp-echo-fastapi",
    "title": "Mount a FastMCP echo server on FastAPI with MCP tool",
    "one_paragraph_description": (
        "Build a FastAPI app that mounts a FastMCP server exposing an echo "
        "tool and a health endpoint. Serve over Streamable HTTP, validate "
        "inputs with Pydantic v2, and persist request counts to SQLite."
    ),
    "artifact_type": "api",
    "primary_libraries": ["fastapi", "fastmcp", "pydantic"],
    "new_libraries_introduced": ["fastmcp"],
    "acceptance_criteria": [
        {"kind": "endpoint", "spec": "POST /mcp handles MCP protocol"},
        {"kind": "test", "spec": "pytest tests verify echo tool returns input"},
        {"kind": "endpoint", "spec": "GET /health returns 200 with ok body"},
    ],
    "estimated_build_time_min": 60,
    "estimated_difficulty": 3,
    "prerequisite_skills": ["fastapi", "fastmcp-basics"],
    "relevance_rationale": "MCP is the primary tool-use protocol the engine targets.",
    "novelty_rationale": "No FastMCP goals currently in archive.",
    "source_citation": "github.com/user/repo",
}


class FakeArchive:
    def __init__(
        self,
        *,
        existing_neighbors: list[Neighbor] | None = None,
        post_hits: list[Neighbor] | None = None,
        top_tags_val: list[str] | None = None,
    ) -> None:
        self._existing = existing_neighbors or []
        self._post_hits = post_hits or []
        self._top_tags = top_tags_val or []
        self.upserts: list[dict[str, Any]] = []

    def query_neighbors(
        self, collection: str, embedding: Any, *, top_k: int = 10
    ) -> list[Neighbor]:
        return list(self._post_hits[:top_k])

    def top_tags(self, collection: str, top_n: int = 20) -> list[str]:
        return list(self._top_tags)

    def upsert_goal(self, collection: str, **kwargs: Any) -> None:
        self.upserts.append({"collection": collection, **kwargs})


def _embedder(text: str) -> list[float]:
    return [0.0] * 8


def _make_generator(outputs: list[str]) -> Any:
    """Return an async callable that replays pre-canned outputs FIFO."""
    i = [0]

    async def gen(prompt: str, *, temperature: float, max_tokens: int) -> str:
        if i[0] >= len(outputs):
            return ""
        out = outputs[i[0]]
        i[0] += 1
        return out

    return gen


# ---------------------------------------------------------------------------
# Core flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_spec_on_first_sample() -> None:
    raw = json.dumps(GOOD_SPEC_JSON)
    archive = FakeArchive(post_hits=[])  # no dup match
    result = await synthesize(
        {"title": "something", "summary": "x", "domain_tags": ["fastapi", "mcp"]},
        novelty_score=0.8,
        zpd_fit=0.8,
        pred_time_min=60,
        neighbors=[],
        archive=archive,
        embedder=_embedder,
        generator_client=_make_generator([raw, raw, raw, raw]),
    )
    assert result.reason == "accepted"
    assert result.spec is not None
    assert result.spec.goal_id == "mcp-echo-fastapi"
    assert result.ranker is not None and result.ranker.accepted is True


@pytest.mark.asyncio
async def test_post_expansion_duplicate_diverts_to_failed_interest() -> None:
    raw = json.dumps(GOOD_SPEC_JSON)
    archive = FakeArchive(
        post_hits=[Neighbor(goal_id="old", title="close dup", cosine=0.95)]
    )
    result = await synthesize(
        {"title": "something", "summary": "x"},
        novelty_score=0.8,
        zpd_fit=0.8,
        pred_time_min=60,
        neighbors=[],
        archive=archive,
        embedder=_embedder,
        generator_client=_make_generator([raw] * 4),
    )
    assert result.reason == "post_dup"
    assert result.spec is None
    assert any(u["collection"] == "failed_interest" for u in archive.upserts)


@pytest.mark.asyncio
async def test_all_samples_invalid_returns_no_valid_sample() -> None:
    archive = FakeArchive()
    result = await synthesize(
        {"title": "x", "summary": "y"},
        novelty_score=0.8,
        zpd_fit=0.8,
        pred_time_min=60,
        neighbors=[],
        archive=archive,
        embedder=_embedder,
        generator_client=_make_generator(
            ["nope", "{}", "{'bad': json}", "still bad"]
        ),
    )
    assert result.reason == "no_valid_sample"
    assert result.spec is None


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


def test_goalspec_rejects_bad_artifact_type() -> None:
    bad = dict(GOOD_SPEC_JSON, artifact_type="mainframe")
    with pytest.raises(Exception):
        GoalSpec.model_validate(bad)


def test_goalspec_rejects_empty_acceptance_criteria() -> None:
    bad = dict(GOOD_SPEC_JSON, acceptance_criteria=[])
    with pytest.raises(Exception):
        GoalSpec.model_validate(bad)


def test_goalspec_rejects_out_of_range_difficulty() -> None:
    bad = dict(GOOD_SPEC_JSON, estimated_difficulty=9)
    with pytest.raises(Exception):
        GoalSpec.model_validate(bad)


def test_goalspec_valid_minimum_fields() -> None:
    spec = GoalSpec.model_validate(GOOD_SPEC_JSON)
    assert spec.goal_id == "mcp-echo-fastapi"
    assert len(spec.acceptance_criteria) == 3
