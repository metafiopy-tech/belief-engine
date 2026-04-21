"""Trending-repo proxy via the GitHub search API.

GitHub deprecated a stable JSON /trending endpoint years ago — the only
supported stable substitute is the search API with a created-date
window plus a min-stars gate. Spec:

    /search/repositories?q=created:>{date}+stars:>20+language:python&sort=stars

Cadence: 6h. Conditional GET via ETag. No unauthenticated calls when
we have a token; GitHub's search limit is 30 req/min authenticated,
10 req/min unauth.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from belief.photosynthesis.sources._common import (
    gh_auth_headers,
    safe_get,
    summarize,
)
from belief.photosynthesis.state import CandidateSeed

if TYPE_CHECKING:  # pragma: no cover
    from belief.core.http import BreakerAsyncClient
    from belief.photosynthesis.config import PhotoConfig
    from belief.photosynthesis.state import PhotosynthesisState


SOURCE_NAME = "github_search"
API_URL = "https://api.github.com/search/repositories"
LANGUAGES = ("python", "typescript")


async def harvest(
    client: "BreakerAsyncClient",
    state: "PhotosynthesisState",
    config: "PhotoConfig",
) -> list[CandidateSeed]:
    """Query created:>N AND stars:>20, per language, dedup, insert."""
    last_ts, etag = state.get_watermark(SOURCE_NAME)
    since = _since_from_watermark(last_ts)

    headers = gh_auth_headers(config.github_token_env)
    if etag:
        headers["If-None-Match"] = etag

    new_seeds: list[CandidateSeed] = []
    latest_etag = etag

    for lang in LANGUAGES:
        params = {
            "q": f"created:>{since} stars:>20 language:{lang}",
            "sort": "stars",
            "order": "desc",
            "per_page": 30,
        }
        resp = await client.get(API_URL, headers=headers, params=params)
        # 304 => nothing new; skip parsing this language and continue.
        if resp.status_code == 304:
            continue
        if resp.status_code >= 400:
            # Don't abort the whole harvest — log and move on.
            continue

        latest_etag = (resp.headers or {}).get("etag", latest_etag)
        payload = resp.json()

        for item in payload.get("items") or []:
            repo_id = str(item.get("id") or safe_get(item, "full_name"))
            if not repo_id or not state.mark_if_new(SOURCE_NAME, repo_id):
                continue

            seed = CandidateSeed(
                source=SOURCE_NAME,
                source_id=repo_id,
                title=str(item.get("full_name") or ""),
                summary=summarize(item.get("description") or ""),
                raw_excerpt=json.dumps(
                    {
                        "full_name": item.get("full_name"),
                        "description": item.get("description"),
                        "stargazers_count": item.get("stargazers_count"),
                        "html_url": item.get("html_url"),
                        "topics": item.get("topics") or [],
                        "language": item.get("language"),
                        "created_at": item.get("created_at"),
                        "pushed_at": item.get("pushed_at"),
                    }
                )[:4000],
            )
            rowid = state.insert_signal(seed)
            if rowid is not None:
                new_seeds.append(seed)

    state.set_watermark(SOURCE_NAME, last_ts=int(time.time()), last_cursor=latest_etag)
    return new_seeds


def _since_from_watermark(last_ts: int | None) -> str:
    """Build the `created:>DATE` window.

    First run: last 7 days. Subsequent runs: from the watermark minus a
    small overlap so we don't drop items right on the boundary.
    """
    if last_ts:
        dt = datetime.fromtimestamp(last_ts, tz=timezone.utc) - timedelta(hours=1)
    else:
        dt = datetime.now(timezone.utc) - timedelta(days=7)
    return dt.strftime("%Y-%m-%d")


__all__ = ["SOURCE_NAME", "harvest"]
