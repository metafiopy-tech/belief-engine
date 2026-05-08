"""Research dispatcher -- parallel fan-out over external sources (SE S5).

Takes the :class:`ResearchPrompt` objects produced by
:mod:`belief.photosynthesis.synthesis.atomizer` and dispatches each
across one or more search-capable sources (arXiv / GitHub / PyPI).
Results are deduplicated by document URL/id and returned as a flat
list of :class:`RetrievedDoc` objects.

Two design points worth noting:

1. **Source abstraction.** :class:`SearchableSource` is a tiny
   protocol: ``async search(client, query) -> list[RetrievedDoc]``.
   Concrete implementations for arxiv / github / pypi ship below.
   Tests inject stubs without httpx so the dispatcher can be
   exercised hermetically. The existing harvester contract
   (``async harvest(client, state, config) -> list[CandidateSeed]``)
   stays untouched -- harvesters are scheduled fetchers tied to the
   daemon's watermark model; ad-hoc queries don't fit that contract.

2. **Parallelism.** ``dispatch`` issues all queries via
   ``asyncio.gather`` so wall-clock latency = max(per-query latency)
   instead of sum. With 30 prompts at ~1s each this is the
   difference between 30s and 1-2s.

Out of scope for Session 5:
  - Retry / backoff on transient errors -- failures are logged and
    the prompt's results are dropped from the corpus. Future S5.5
    can add tenacity if rate limiting becomes an issue.
  - Cross-source semantic dedup -- current dedup is exact URL/id
    match. Approximate dedup belongs in S6's incompleteness pass.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from belief.photosynthesis.synthesis.atomizer import ResearchPrompt


logger = logging.getLogger("belief.photosynthesis.synthesis.research_dispatcher")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievedDoc:
    """One document returned by a SearchableSource."""

    url: str
    title: str
    summary: str
    source: str  # "arxiv" / "github" / "pypi" / ...
    raw_excerpt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Which research prompt(s) surfaced this doc -- populated by the
    # dispatcher during dedup so callers can trace back to axis/word.
    prompts: list[ResearchPrompt] = field(default_factory=list)

    def doc_key(self) -> str:
        """Stable dedup key. URL is canonical when present; falls back
        to ``source:title`` so source-id'd hits without URLs still
        collapse cleanly."""
        if self.url:
            return self.url.strip().rstrip("/")
        return f"{self.source}:{self.title}"


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchableSource(Protocol):
    """Anything the dispatcher can fan a query out to."""

    name: str

    async def search(self, client: Any, query: str) -> list[RetrievedDoc]: ...


# ---------------------------------------------------------------------------
# Concrete sources -- network-backed implementations
# ---------------------------------------------------------------------------


class ArxivSearch:
    """ad-hoc arXiv search helper.

    Hits ``export.arxiv.org/api/query?search_query=...`` -- the same
    endpoint the daemon's harvester uses, but with a free-form query
    string instead of the per-category cadence.
    """

    name = "arxiv"

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = int(max_results)

    async def search(self, client: Any, query: str) -> list[RetrievedDoc]:
        if client is None:
            return []
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(self.max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        try:
            resp = await client.get(url)
        except Exception as exc:
            logger.warning("arxiv search '%s' failed: %s", query[:50], exc)
            return []
        status = getattr(resp, "status_code", 200)
        if status >= 400:
            return []

        try:
            from belief.photosynthesis.sources._common import parse_feed_text

            feed = parse_feed_text(resp.text)
        except Exception as exc:
            logger.warning("arxiv feed parse failed: %s", exc)
            return []

        out: list[RetrievedDoc] = []
        for entry in (getattr(feed, "entries", []) or [])[: self.max_results]:
            eid = getattr(entry, "id", "") or ""
            title = (getattr(entry, "title", "") or "").strip()
            summary = (getattr(entry, "summary", "") or "").strip()
            out.append(
                RetrievedDoc(
                    url=eid,
                    title=title[:400],
                    summary=summary[:1000],
                    source=self.name,
                    raw_excerpt=summary[:4000],
                )
            )
        return out


class GithubSearch:
    """ad-hoc GitHub repository search."""

    name = "github"

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = int(max_results)

    async def search(self, client: Any, query: str) -> list[RetrievedDoc]:
        if client is None:
            return []
        try:
            from belief.photosynthesis.sources._common import gh_auth_headers
        except ImportError:
            gh_auth_headers = lambda: {}  # noqa: E731

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": str(self.max_results),
        }
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(params)
        try:
            resp = await client.get(url, headers=gh_auth_headers())
        except Exception as exc:
            logger.warning("github search '%s' failed: %s", query[:50], exc)
            return []
        status = getattr(resp, "status_code", 200)
        if status >= 400:
            return []
        try:
            data = resp.json()
        except Exception:
            return []

        out: list[RetrievedDoc] = []
        for item in (data.get("items") or [])[: self.max_results]:
            if not isinstance(item, dict):
                continue
            out.append(
                RetrievedDoc(
                    url=str(item.get("html_url") or ""),
                    title=str(item.get("full_name") or "")[:400],
                    summary=str(item.get("description") or "")[:1000],
                    source=self.name,
                    raw_excerpt=str(item.get("description") or "")[:4000],
                    metadata={
                        "stars": int(item.get("stargazers_count") or 0),
                        "language": item.get("language"),
                    },
                )
            )
        return out


class PypiSearch:
    """ad-hoc PyPI search via the JSON metadata API.

    PyPI doesn't expose a full-text search endpoint; we use the
    package metadata API on a name-match assumption (the dispatcher
    extracts the most plausible package name from the query). Returns
    an empty list for queries that don't look like package names --
    that's fine, dedup just sees fewer docs.
    """

    name = "pypi"

    def __init__(self, max_results: int = 1) -> None:
        self.max_results = int(max_results)

    async def search(self, client: Any, query: str) -> list[RetrievedDoc]:
        if client is None:
            return []
        # Simple heuristic: try the lowest-cased single-word slug from
        # the query as a package name. The atomizer's templates often
        # produce queries like "How does mantis_shrimp work..." -- the
        # word token is right there at the front.
        words = [w.strip().lower() for w in query.replace("?", " ").split() if w.strip()]
        if not words:
            return []
        candidate = words[0].rstrip(",.")
        url = f"https://pypi.org/pypi/{urllib.parse.quote(candidate)}/json"
        try:
            resp = await client.get(url)
        except Exception:
            return []
        status = getattr(resp, "status_code", 404)
        if status != 200:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        info = data.get("info") or {}
        name = info.get("name") or candidate
        return [
            RetrievedDoc(
                url=info.get("project_url") or info.get("home_page") or url,
                title=str(name)[:400],
                summary=str(info.get("summary") or "")[:1000],
                source=self.name,
                raw_excerpt=str(info.get("description") or "")[:4000],
                metadata={"version": info.get("version")},
            )
        ]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def default_sources() -> list[SearchableSource]:
    """The canonical S5 source set."""
    return [ArxivSearch(), GithubSearch(), PypiSearch()]


async def dispatch(
    prompts: list[ResearchPrompt],
    *,
    client: Any = None,
    sources: Optional[list[SearchableSource]] = None,
    respect_source_hints: bool = True,
) -> list[RetrievedDoc]:
    """Fan ``prompts`` out across ``sources`` in parallel; return deduped docs.

    ``client`` is an httpx-compatible async client, passed through to
    each source's ``search()``. Tests pass a stub or ``None``.

    ``respect_source_hints`` (default True) routes each prompt only
    to sources whose name appears in ``prompt.source_hints`` (or to
    every source if hints is empty). Set False to fan every prompt
    to every source -- useful when comparing source quality.

    Dedup is by :meth:`RetrievedDoc.doc_key` (URL when present,
    ``source:title`` otherwise). When the same doc is returned by
    multiple prompts, all surfacing prompts are recorded in
    ``doc.prompts`` so callers can trace which axis hit it.
    """
    if not prompts:
        return []
    src_list = sources if sources is not None else default_sources()
    sources_by_name = {s.name: s for s in src_list}

    # Build the full (prompt, source) call list.
    calls: list[tuple[ResearchPrompt, SearchableSource]] = []
    for prompt in prompts:
        eligible: list[SearchableSource]
        if respect_source_hints and prompt.source_hints:
            eligible = [
                sources_by_name[name] for name in prompt.source_hints if name in sources_by_name
            ]
            if not eligible:
                # Hints didn't match any configured source -- fan out
                # broadly so we don't drop the prompt silently.
                eligible = list(src_list)
        else:
            eligible = list(src_list)
        for s in eligible:
            calls.append((prompt, s))

    async def _one(
        prompt: ResearchPrompt, src: SearchableSource
    ) -> tuple[ResearchPrompt, list[RetrievedDoc]]:
        try:
            docs = await src.search(client, prompt.query)
        except Exception as exc:
            logger.warning(
                "source %s on prompt '%s...' failed: %s", src.name, prompt.query[:40], exc
            )
            docs = []
        return prompt, list(docs)

    results = await asyncio.gather(
        *[_one(p, s) for (p, s) in calls],
        return_exceptions=False,
    )

    # Deduplicate by doc_key, accumulating which prompts surfaced each.
    by_key: dict[str, RetrievedDoc] = {}
    for prompt, docs in results:
        for d in docs:
            key = d.doc_key()
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                d.prompts = [prompt]
                by_key[key] = d
            else:
                existing.prompts.append(prompt)
    return list(by_key.values())


__all__ = [
    "ArxivSearch",
    "GithubSearch",
    "PypiSearch",
    "RetrievedDoc",
    "SearchableSource",
    "default_sources",
    "dispatch",
]
