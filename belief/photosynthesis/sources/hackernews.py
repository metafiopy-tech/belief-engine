"""Show HN via the Algolia HN search index.

Algolia's `search_by_date` endpoint supports a `numericFilters` clause
shaped like `created_at_i>WATERMARK`, which gives us strict
incremental fetches without any auth. Cadence: 15 min.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from belief.photosynthesis.sources._common import summarize
from belief.photosynthesis.state import CandidateSeed

if TYPE_CHECKING:  # pragma: no cover
    from belief.core.http import BreakerAsyncClient
    from belief.photosynthesis.config import PhotoConfig
    from belief.photosynthesis.state import PhotosynthesisState


SOURCE_NAME = "hackernews"
API_URL = "https://hn.algolia.com/api/v1/search_by_date"


async def harvest(
    client: "BreakerAsyncClient",
    state: "PhotosynthesisState",
    config: "PhotoConfig",
) -> list[CandidateSeed]:
    """Pull Show HN posts created after the watermark."""
    last_ts, _ = state.get_watermark(SOURCE_NAME)
    since = last_ts or int(time.time() - 7 * 86400)

    params: dict[str, str] = {
        "tags": config.hn_tags,
        "numericFilters": f"created_at_i>{since}",
        "hitsPerPage": "100",
    }

    try:
        resp = await client.get(API_URL, params=params)
    except Exception:
        return []
    if resp.status_code >= 400:
        return []

    try:
        payload = resp.json()
    except ValueError:
        return []

    new_seeds: list[CandidateSeed] = []
    max_created = since

    for hit in payload.get("hits") or []:
        obj_id = hit.get("objectID")
        if not obj_id:
            continue
        obj_id = str(obj_id)
        if not state.mark_if_new(SOURCE_NAME, obj_id):
            continue

        title = hit.get("title") or hit.get("story_title") or ""
        url = hit.get("url") or ""
        body = hit.get("story_text") or hit.get("comment_text") or ""
        created = int(hit.get("created_at_i") or 0)
        max_created = max(max_created, created)

        seed = CandidateSeed(
            source=SOURCE_NAME,
            source_id=obj_id,
            title=title[:400],
            summary=summarize(f"{url} {body}" if url else body),
            raw_excerpt=body[:4000],
            captured_at=created or int(time.time()),
        )
        if state.insert_signal(seed) is not None:
            new_seeds.append(seed)

    state.set_watermark(SOURCE_NAME, last_ts=max_created)
    return new_seeds


__all__ = ["SOURCE_NAME", "harvest"]
