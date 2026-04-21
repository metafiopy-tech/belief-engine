"""GitHub release atoms for tracked dependencies.

Atom feeds don't count against the core 5000/hr REST quota. Conditional
GET returns 304 cheaply when there's nothing new. Cadence: 15 min.

For each tracked dep we fetch `https://github.com/<repo>/releases.atom`.
The mapping of dep -> repo is hardcoded here (it's a short, stable list).
If a new dep is added to PhotoConfig.tracked_deps without a mapping,
we silently skip it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from belief.photosynthesis.sources._common import parse_feed_text, summarize
from belief.photosynthesis.state import CandidateSeed

if TYPE_CHECKING:  # pragma: no cover
    from belief.core.http import BreakerAsyncClient
    from belief.photosynthesis.config import PhotoConfig
    from belief.photosynthesis.state import PhotosynthesisState


SOURCE_NAME = "github_releases"

DEP_REPO_MAP: dict[str, str] = {
    "langgraph": "langchain-ai/langgraph",
    "langchain": "langchain-ai/langchain",
    "anthropic": "anthropics/anthropic-sdk-python",
    "pydantic": "pydantic/pydantic",
    "fastapi": "tiangolo/fastapi",
    "click": "pallets/click",
    "chromadb": "chroma-core/chroma",
    "httpx": "encode/httpx",
    "typer": "tiangolo/typer",
    "openai": "openai/openai-python",
    "uvicorn": "encode/uvicorn",
    "dspy": "stanfordnlp/dspy",
    "rich": "Textualize/rich",
    "ollama": "ollama/ollama",
}


async def harvest(
    client: "BreakerAsyncClient",
    state: "PhotosynthesisState",
    config: "PhotoConfig",
) -> list[CandidateSeed]:
    """Poll every tracked dep's releases.atom feed with conditional GET."""
    new_seeds: list[CandidateSeed] = []

    for dep in config.tracked_deps:
        repo = DEP_REPO_MAP.get(dep)
        if not repo:
            continue

        subkey = f"{SOURCE_NAME}:{dep}"
        _, cursor = state.get_watermark(subkey)
        # cursor stores the latest entry id we've already seen
        headers: dict[str, str] = {}

        url = f"https://github.com/{repo}/releases.atom"
        try:
            resp = await client.get(url, headers=headers)
        except Exception:
            # A single dep failing must not abort the rest.
            continue
        if resp.status_code == 304 or resp.status_code >= 400:
            continue

        feed = parse_feed_text(resp.text)
        latest_id = cursor
        for entry in getattr(feed, "entries", []) or []:
            entry_id = getattr(entry, "id", "") or getattr(entry, "link", "")
            if not entry_id:
                continue
            if entry_id == cursor:
                break  # we've hit the last one we saw; atoms are newest-first
            if not state.mark_if_new(subkey, entry_id):
                continue

            title = getattr(entry, "title", "") or ""
            summary_text = getattr(entry, "summary", "") or ""
            seed = CandidateSeed(
                source=SOURCE_NAME,
                source_id=entry_id,
                title=f"{dep}: {title}"[:400],
                summary=summarize(summary_text),
                raw_excerpt=summary_text[:4000],
            )
            rowid = state.insert_signal(seed)
            if rowid is not None:
                new_seeds.append(seed)
            if not latest_id:
                latest_id = entry_id

        if latest_id and latest_id != cursor:
            state.set_watermark(subkey, last_ts=int(time.time()), last_cursor=latest_id)

    return new_seeds


__all__ = ["DEP_REPO_MAP", "SOURCE_NAME", "harvest"]
