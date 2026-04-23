"""Helpers shared across source harvesters.

`parse_feed_text` wraps feedparser in a lazy-import guard so tests can
monkey-patch it without installing feedparser. `gh_auth_headers` centralizes
the GitHub token lookup. `stub_etag` is used by offline fixtures.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def gh_auth_headers(github_token_env: str = "GH_TOKEN") -> dict[str, str]:
    """Build the headers dict for a GitHub REST request.

    Returns an empty dict if the token env var isn't set — GitHub's
    unauthenticated quota is tiny (60/hr) but still nonzero, so we
    don't hard-fail.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "belief-engine-photosynthesis/1",
    }
    tok = os.environ.get(github_token_env, "")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def parse_feed_text(text: str) -> Any:
    """feedparser.parse() wrapper with a helpful lazy-import error."""
    try:
        import feedparser  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "feedparser is required for RSS/Atom sources. Install the [photosynthesis] extra."
        ) from exc
    return feedparser.parse(text)


def summarize(text: str, limit: int = 240) -> str:
    """Collapse whitespace and truncate to `limit` chars."""
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(limit - 1, 1)].rstrip() + "…"


def safe_get(mapping: Any, *keys: str, default: Optional[str] = "") -> Any:
    """Best-effort nested .get chain that tolerates missing / non-dict nodes."""
    cur = mapping
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
        if cur is None:
            return default
    return cur


__all__ = ["gh_auth_headers", "parse_feed_text", "safe_get", "summarize"]
