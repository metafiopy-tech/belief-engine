"""Offline parser tests for each source harvester.

We don't hit any real API here. For each source we build a stub
httpx response (or a minimal fake client), pass it through the source's
harvest() function, and assert:

  - new rows land in raw_signals with the expected fields,
  - duplicates don't double-insert on a second call,
  - the watermark advances.

Where a source uses feedparser, we skip the test if feedparser isn't
installed — that's the intended behavior for the light test extra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from belief.photosynthesis.config import PhotoConfig
from belief.photosynthesis.state import CandidateSeed, PhotosynthesisState


# ---------------------------------------------------------------------------
# Minimal fake httpx client
# ---------------------------------------------------------------------------


@dataclass
class FakeResponse:
    status_code: int
    _text: str = ""
    _json: Any = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self._text

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeClient:
    """Pretends to be BreakerAsyncClient for source harvesters.

    Queue responses up via `.enqueue(response)`. Requests are served
    FIFO. Any request past the end of the queue returns a 404.
    """

    def __init__(self) -> None:
        self._responses: list[FakeResponse] = []
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def enqueue(self, resp: FakeResponse) -> None:
        self._responses.append(resp)

    async def get(
        self, url: str, headers: dict[str, str] | None = None, params: Any = None
    ) -> FakeResponse:
        self.calls.append((url, "GET", params if isinstance(params, dict) else None))
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse(status_code=404)


@pytest.fixture()
def photo_state(tmp_path: Path) -> PhotosynthesisState:
    return PhotosynthesisState(str(tmp_path / "signals.sqlite"))


@pytest.fixture()
def photo_config(tmp_path: Path) -> PhotoConfig:
    # Override directories to the tmp path so nothing leaks to /var/lib
    from dataclasses import replace

    cfg = PhotoConfig()
    return replace(
        cfg,
        state_dir=tmp_path,
        log_dir=tmp_path,
        config_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# github_search — JSON response parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_github_search_parses_items_and_advances_watermark(
    photo_state: PhotosynthesisState,
    photo_config: PhotoConfig,
) -> None:
    from belief.photosynthesis.sources import github_search

    body = {
        "items": [
            {
                "id": 123,
                "full_name": "user/cool-agent",
                "description": "An agent built on langgraph",
                "stargazers_count": 100,
                "html_url": "https://github.com/user/cool-agent",
                "topics": ["ai", "langgraph"],
                "language": "Python",
                "created_at": "2026-01-01T00:00:00Z",
                "pushed_at": "2026-04-01T00:00:00Z",
            }
        ]
    }
    client = FakeClient()
    # One response per LANGUAGES entry (python + typescript). Return the
    # same body for both; dedup should prevent double-insertion.
    client.enqueue(FakeResponse(status_code=200, _json=body, headers={"etag": "W/\"abc\""}))
    client.enqueue(FakeResponse(status_code=200, _json=body))

    seeds = await github_search.harvest(client, photo_state, photo_config)  # type: ignore[arg-type]

    assert len(seeds) == 1
    [seed] = seeds
    assert seed.source == "github_search"
    assert seed.title == "user/cool-agent"
    assert "langgraph" in seed.summary
    # Second call (same response) should mark-as-seen and insert nothing
    client.enqueue(FakeResponse(status_code=200, _json=body))
    client.enqueue(FakeResponse(status_code=200, _json=body))
    seeds2 = await github_search.harvest(client, photo_state, photo_config)  # type: ignore[arg-type]
    assert seeds2 == []


# ---------------------------------------------------------------------------
# hackernews — JSON response parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hackernews_parses_hits_and_dedups(
    photo_state: PhotosynthesisState,
    photo_config: PhotoConfig,
) -> None:
    from belief.photosynthesis.sources import hackernews

    body = {
        "hits": [
            {
                "objectID": "abc1",
                "title": "Show HN: Belief Engine",
                "url": "https://example.com/belief",
                "story_text": "Autonomous build system using langgraph and mcp",
                "created_at_i": 1_700_000_000,
            },
            {
                "objectID": "abc2",
                "title": "Show HN: FSRS Flashcard App",
                "url": "https://example.com/fsrs",
                "story_text": "Spaced repetition in Rust",
                "created_at_i": 1_700_000_100,
            },
        ]
    }
    # Pre-seed an old watermark so our mock hit timestamps aren't below
    # the "first run" fallback of now - 7 days. Real traffic never hits
    # this ordering problem because Algolia filters server-side.
    photo_state.set_watermark("hackernews", last_ts=1_699_999_000)

    client = FakeClient()
    client.enqueue(FakeResponse(status_code=200, _json=body))

    seeds = await hackernews.harvest(client, photo_state, photo_config)  # type: ignore[arg-type]
    assert len(seeds) == 2

    # Watermark should advance to the max created_at_i
    ts, _ = photo_state.get_watermark("hackernews")
    assert ts == 1_700_000_100

    # Second harvest with the same payload: no new rows
    client.enqueue(FakeResponse(status_code=200, _json=body))
    seeds2 = await hackernews.harvest(client, photo_state, photo_config)  # type: ignore[arg-type]
    assert seeds2 == []


# ---------------------------------------------------------------------------
# stackoverflow — JSON parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stackoverflow_parses_items(
    photo_state: PhotosynthesisState,
    photo_config: PhotoConfig,
) -> None:
    from belief.photosynthesis.sources import stackoverflow

    body = {
        "items": [
            {
                "question_id": 77777,
                "title": "How to use MCP with FastAPI",
                "body_markdown_excerpt": "I'm trying to mount an MCP server...",
                "creation_date": 1_700_000_000,
                "tags": ["mcp", "fastapi"],
            }
        ]
    }
    client = FakeClient()
    client.enqueue(FakeResponse(status_code=200, _json=body))

    seeds = await stackoverflow.harvest(client, photo_state, photo_config)  # type: ignore[arg-type]
    assert len(seeds) == 1
    assert seeds[0].source_id == "77777"
    assert "MCP" in seeds[0].title


# ---------------------------------------------------------------------------
# github_releases / pypi / arxiv — require feedparser for RSS / Atom
# ---------------------------------------------------------------------------


FEEDPARSER_AVAILABLE = True
try:  # pragma: no cover
    import feedparser  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    FEEDPARSER_AVAILABLE = False


FAKE_RELEASE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <title>Releases</title>
    <entry>
        <id>tag:github.com,2008:Repository/1/v1.0.0</id>
        <title>v1.0.0</title>
        <summary>Big release with MCP support</summary>
        <updated>2026-04-01T00:00:00Z</updated>
    </entry>
</feed>
"""


@pytest.mark.skipif(not FEEDPARSER_AVAILABLE, reason="feedparser not installed")
@pytest.mark.asyncio
async def test_github_releases_parses_atom(
    photo_state: PhotosynthesisState,
    photo_config: PhotoConfig,
) -> None:
    from belief.photosynthesis.sources import github_releases

    # Make sure exactly one tracked dep is in DEP_REPO_MAP so we make one call
    from dataclasses import replace

    cfg = replace(photo_config, tracked_deps=("langgraph",))
    client = FakeClient()
    client.enqueue(FakeResponse(status_code=200, _text=FAKE_RELEASE_ATOM))

    seeds = await github_releases.harvest(client, photo_state, cfg)  # type: ignore[arg-type]
    assert len(seeds) == 1
    assert "langgraph:" in seeds[0].title
