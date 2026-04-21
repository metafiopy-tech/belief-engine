"""Stack Overflow questions on the tracked tag set.

We use the v2.3 API with a custom filter id (see
https://api.stackexchange.com/docs/create-filter) that trims the payload
down to: question_id, title, body_markdown_excerpt, tags, creation_date,
score. That's ~10% of the default filter size. Cadence: 30 min.

An app key raises the daily quota from 300 to 10,000 — STACKEX_KEY is
honored if present, otherwise we stay unauthenticated.
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


SOURCE_NAME = "stackoverflow"
API_URL = "https://api.stackexchange.com/2.3/questions"

# Narrow filter id — pre-created on StackExchange. Honored when the env
# var STACKEX_FILTER is set; otherwise we send no filter and the API
# returns the default (larger) shape, which still works.
DEFAULT_FILTER = None


async def harvest(
    client: "BreakerAsyncClient",
    state: "PhotosynthesisState",
    config: "PhotoConfig",
) -> list[CandidateSeed]:
    """Pull questions tagged with any of the tracked tags since the watermark."""
    last_ts, _ = state.get_watermark(SOURCE_NAME)
    fromdate = last_ts or int(time.time() - 7 * 86400)

    import os

    params: dict[str, str] = {
        "site": "stackoverflow",
        "order": "desc",
        "sort": "creation",
        "fromdate": str(fromdate),
        "tagged": ";".join(config.stackoverflow_tags),
        "pagesize": "50",
    }
    key = os.environ.get(config.stackex_key_env, "")
    if key:
        params["key"] = key
    filter_id = os.environ.get("STACKEX_FILTER", "") or DEFAULT_FILTER
    if filter_id:
        params["filter"] = filter_id

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
    max_created = fromdate

    for item in payload.get("items") or []:
        qid = item.get("question_id")
        if qid is None:
            continue
        qid_str = str(qid)
        if not state.mark_if_new(SOURCE_NAME, qid_str):
            continue

        title = item.get("title") or ""
        body = item.get("body_markdown_excerpt") or item.get("body_markdown") or ""
        created = int(item.get("creation_date") or 0)
        max_created = max(max_created, created)

        seed = CandidateSeed(
            source=SOURCE_NAME,
            source_id=qid_str,
            title=title[:400],
            summary=summarize(body),
            raw_excerpt=(body or "")[:4000],
            captured_at=created or int(time.time()),
        )
        if state.insert_signal(seed) is not None:
            new_seeds.append(seed)

    state.set_watermark(SOURCE_NAME, last_ts=max_created)
    return new_seeds


__all__ = ["SOURCE_NAME", "harvest"]
