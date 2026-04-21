"""arXiv cs.AI / cs.CL / cs.SE harvester via the `arxiv` pip package.

Cadence: 6h. The pip package enforces a `delay_seconds` between API hits
(default 3s per request). For a single category sweep we make ~1 call per
cycle, so rate limits are easy.

If `arxiv` isn't installed (e.g. the `photosynthesis-test` extra was used
instead of the full `photosynthesis` extra), we fall back to hitting the
arXiv API directly over HTTP. That's slower and slightly more error-prone
but keeps the source usable.
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
    # Try the arxiv pip package first — it handles pagination and retries.
    try:
        hits = _fetch_via_arxiv_package(cat_query, since)
    except ImportError:
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


def _fetch_via_arxiv_package(cat_query: str, since: int) -> list[dict]:
    """Use the arxiv pip package. Raises ImportError if not installed."""
    import arxiv  # type: ignore[import-untyped]

    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=cat_query,
        max_results=50,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    out: list[dict] = []
    for result in client.results(search):
        published = getattr(result, "published", None)
        published_ts = int(published.timestamp()) if published else 0
        if published_ts and published_ts <= since:
            break
        out.append(
            {
                "id": getattr(result, "entry_id", "") or result.get_short_id(),
                "title": getattr(result, "title", "") or "",
                "summary": getattr(result, "summary", "") or "",
                "published_ts": published_ts,
            }
        )
    return out


async def _fetch_via_http(
    client: "BreakerAsyncClient", cat_query: str, since: int
) -> list[dict]:
    """HTTP fallback using the public arXiv query API (Atom)."""
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
