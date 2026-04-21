"""PyPI new releases via the public RSS feed.

Cadence: 10 min. The RSS feed (pypi.org/rss/updates.xml) lists the latest
releases across *all* packages — roughly 40 entries at a time. We filter
in-process by keyword-match against the signal domain_keywords list and
only issue the packument JSON fetch on a match.

That keeps traffic low: a single 304 on the RSS + at most a handful of
JSON fetches per 10-minute cycle.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from belief.photosynthesis.sources._common import parse_feed_text, summarize
from belief.photosynthesis.state import CandidateSeed

if TYPE_CHECKING:  # pragma: no cover
    from belief.core.http import BreakerAsyncClient
    from belief.photosynthesis.config import PhotoConfig
    from belief.photosynthesis.state import PhotosynthesisState


SOURCE_NAME = "pypi"
RSS_URL = "https://pypi.org/rss/updates.xml"


async def harvest(
    client: "BreakerAsyncClient",
    state: "PhotosynthesisState",
    config: "PhotoConfig",
) -> list[CandidateSeed]:
    """Pull PyPI RSS, filter by tracked-dep / keyword match, fetch JSON on hit."""
    # Load keyword set for the in-process keyword gate. The full cascade
    # lives in filter/cascade.py; this is a narrower pre-filter so we
    # don't spam packument fetches on irrelevant releases.
    keywords = _load_keyword_set(config)

    last_ts, _ = state.get_watermark(SOURCE_NAME)
    try:
        resp = await client.get(RSS_URL)
    except Exception:
        return []
    if resp.status_code == 304 or resp.status_code >= 400:
        return []

    feed = parse_feed_text(resp.text)
    new_seeds: list[CandidateSeed] = []

    for entry in getattr(feed, "entries", []) or []:
        entry_id = getattr(entry, "id", "") or getattr(entry, "link", "")
        title = getattr(entry, "title", "") or ""
        summary_text = getattr(entry, "summary", "") or ""

        if not entry_id:
            continue

        # Gate: does the title match any tracked dep or keyword?
        name = title.split(" ")[0].strip() if title else ""
        blob = f"{title} {summary_text}".lower()
        if not _passes_keyword_gate(name, blob, keywords, config.tracked_deps):
            continue

        if not state.mark_if_new(SOURCE_NAME, entry_id):
            continue

        # Optionally fetch the packument JSON for richer metadata. We
        # don't retry hard — a failure just means we settle for RSS data.
        packument = await _maybe_fetch_packument(client, name)

        seed = CandidateSeed(
            source=SOURCE_NAME,
            source_id=entry_id,
            title=title[:400],
            summary=summarize(summary_text),
            raw_excerpt=_excerpt(summary_text, packument),
        )
        if state.insert_signal(seed) is not None:
            new_seeds.append(seed)

    state.set_watermark(SOURCE_NAME, last_ts=int(time.time()))
    return new_seeds


def _passes_keyword_gate(
    name: str,
    blob: str,
    keywords: set[str],
    tracked_deps: tuple[str, ...],
) -> bool:
    """Cheap pre-filter: accept if name is a tracked dep or any keyword hits."""
    if not name:
        return False
    name_lower = name.lower()
    if name_lower in tracked_deps:
        return True
    # Keyword fast check on name + blob — any single hit passes.
    for kw in keywords:
        if kw in blob or kw in name_lower:
            return True
    return False


def _load_keyword_set(config: "PhotoConfig") -> set[str]:
    """Load domain keywords from the yaml file into a lowercased set.

    If PyYAML isn't installed or the file is missing, fall back to an
    empty set (the keyword gate then only passes tracked deps, which is
    still useful). Errors here must not take down the source.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return set()
    try:
        data = yaml.safe_load(config.keywords_file.read_text())
    except (FileNotFoundError, OSError):
        return set()
    return {str(k).lower() for k in (data or {}).get("keywords", [])}


async def _maybe_fetch_packument(
    client: "BreakerAsyncClient", name: str
) -> Optional[dict]:
    """Fetch https://pypi.org/pypi/{name}/json for extra metadata. Best-effort."""
    if not name:
        return None
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        resp = await client.get(url)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _excerpt(rss_summary: str, packument: Optional[dict]) -> str:
    """Prefer PyPI's description; fall back to the RSS summary."""
    if packument:
        info = packument.get("info") or {}
        desc = info.get("summary") or info.get("description") or ""
        if desc:
            return desc[:4000]
    return (rss_summary or "")[:4000]


__all__ = ["SOURCE_NAME", "harvest"]
