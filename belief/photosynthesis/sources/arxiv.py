"""arXiv cs.AI / cs.CL / cs.SE harvester over :mod:`belief.core.http`.

Cadence: 6h, one query per cycle.  All HTTP goes through the shared
``BreakerAsyncClient`` passed in by the photosynthesis daemon, which
gives us tenacity retry, pybreaker circuit-breaker, and the domain
allowlist enforcement that :mod:`belief.core.http` provides.

**Session 0.5 (2026-04-23)**: the ``arxiv`` pip package used to be the
preferred path, with HTTP as a fallback.  The package exposes only a
private ``_session: requests.Session`` attribute and offers no public
injection hook, so every call via the package bypassed the shared
retry / breaker / allowlist stack.  The package's features we actually
used — pagination and a 3s politeness delay — aren't load-bearing at a
6h cadence with one request per cycle, so the package path was removed.
If you hit arXiv outside this module (*please don't*), note that
``export.arxiv.org`` is in the default allowlist and 3s politeness is
on you.
"""

from __future__ import annotations

import time
import urllib.parse
from typing import TYPE_CHECKING

from belief.photosynthesis.sources._common import parse_feed_text, summarize
from belief.photosynthesis.state import CandidateSeed

if TYPE_CHECKING:  # pragma: no cover
    from belief.core.http import BreakerAsyncClient
    from belief.photosynthesis.config import PhotoConfig
    from belief.photosynthesis.state import PhotosynthesisState


SOURCE_NAME = "arxiv"


async def harvest(
    client: "BreakerAsyncClient",
    state: "PhotosynthesisState",
    config: "PhotoConfig",
) -> list[CandidateSeed]:
    """Pull recent papers in cs.AI / cs.CL / cs.SE published since the watermark."""
    last_ts, _ = state.get_watermark(SOURCE_NAME)
    since = last_ts or int(time.time() - 2 * 86400)

    cat_query = " OR ".join(f"cat:{c}" for c in config.arxiv_categories)
    try:
        hits = await _fetch_via_http(client, cat_query, since)
    except Exception:
        hits = []

    new_seeds: list[CandidateSeed] = []
    max_published = since

    for hit in hits:
        paper_id = hit["id"]
        published_ts = hit["published_ts"]
        if not state.mark_if_new(SOURCE_NAME, paper_id):
            continue
        max_published = max(max_published, published_ts)

        seed = CandidateSeed(
            source=SOURCE_NAME,
            source_id=paper_id,
            title=hit.get("title", "")[:400],
            summary=summarize(hit.get("summary", "")),
            raw_excerpt=(hit.get("summary") or "")[:4000],
            captured_at=published_ts or int(time.time()),
        )
        if state.insert_signal(seed) is not None:
            new_seeds.append(seed)

    state.set_watermark(SOURCE_NAME, last_ts=max_published)
    return new_seeds


async def _fetch_via_http(
    client: "BreakerAsyncClient", cat_query: str, since: int
) -> list[dict]:
    """HTTP query against ``export.arxiv.org/api/query`` (Atom feed)."""
    params = {
        "search_query": cat_query,
        "start": "0",
        "max_results": "50",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    try:
        resp = await client.get(url)
    except Exception:
        return []
    if resp.status_code >= 400:
        return []
    feed = parse_feed_text(resp.text)

    out: list[dict] = []
    for entry in getattr(feed, "entries", []) or []:
        eid = getattr(entry, "id", "") or ""
        # published is an ISO-8601 string on arXiv entries
        published = getattr(entry, "published", "") or getattr(entry, "updated", "")
        ts = 0
        try:
            import datetime as _dt

            ts = int(_dt.datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp())
        except Exception:
            ts = 0
        if ts and ts <= since:
            continue
        out.append(
            {
                "id": eid,
                "title": getattr(entry, "title", "") or "",
                "summary": getattr(entry, "summary", "") or "",
                "published_ts": ts,
            }
        )
    return out


__all__ = ["SOURCE_NAME", "harvest"]
