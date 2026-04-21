"""
AST parse cache — share ``ast.parse`` work across covenant firings (Session 17).

The covenant registry fires dozens of checkers per build, and most
of them call :func:`ast.parse` on the same handful of source files
independently.  A typical build parses the same 7 files 50 times —
350 wasted parses.  With the cache, each source is parsed at most
once per process.

Callers opt in by replacing ``ast.parse(source)`` with
:func:`parse_cached(source)`.  The cache is process-scoped and
keyed by an SHA-1 of the source text, so identical strings share
a parsed tree; distinct strings never collide.  Cached trees are
immutable ``ast.Module`` objects — the AST module guarantees a
parsed tree can be walked repeatedly, so sharing is safe.

The cache has a soft size limit (``MAX_CACHE_ENTRIES``); once it
fills, older entries are evicted in LRU order.  Callers can clear
it manually with :func:`clear_parse_cache` between benchmark runs
so memory doesn't accumulate across long-lived processes.

Everything here is stdlib-only.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger("belief.validators.ast_cache")


# Soft cap on number of distinct sources we'll keep parsed.  Each
# entry is an ``ast.Module`` — cheap to hold, but a runaway grinder
# loop could accumulate thousands over days.  Eviction is LRU once
# we exceed this count.
MAX_CACHE_ENTRIES = 512


# ── Internal state ────────────────────────────────────────────────────────


_cache: "OrderedDict[str, ast.Module]" = OrderedDict()
_errors: "OrderedDict[str, SyntaxError]" = OrderedDict()
_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0, "errors": 0, "evictions": 0}


def _source_key(source: str, filename: str = "<cached>") -> str:
    """Stable content-addressed key for ``source``.

    We include ``filename`` in the hash so two sources with the same
    text but different ``filename`` arguments stay distinct (the
    filename shows up in SyntaxError messages and stacktraces).
    """
    h = hashlib.sha1()
    h.update(filename.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(source.encode("utf-8", errors="replace"))
    return h.hexdigest()


# ── Public API ────────────────────────────────────────────────────────────


def parse_cached(
    source: str,
    filename: str = "<cached>",
    *,
    raise_on_syntax_error: bool = False,
) -> Optional[ast.Module]:
    """Parse ``source`` with caching, returning the resulting Module.

    Args:
        source:                Python source text.
        filename:              Virtual filename (shows up in
                               SyntaxErrors); part of the cache key.
        raise_on_syntax_error: When True, re-raise the stored
                               SyntaxError on subsequent calls for
                               the same input.  When False (default),
                               return ``None`` on syntax errors so
                               callers can treat them as "skip this
                               file" rather than wrapping in
                               try/except at every call site.

    Returns:
        Parsed :class:`ast.Module`, or ``None`` if the source has a
        syntax error and ``raise_on_syntax_error=False``.
    """
    key = _source_key(source, filename)

    with _lock:
        if key in _cache:
            # Move to end (most recently used).
            _cache.move_to_end(key)
            _stats["hits"] += 1
            return _cache[key]
        if key in _errors:
            _stats["hits"] += 1
            if raise_on_syntax_error:
                raise _errors[key]
            return None

    # Cache miss — parse outside the lock.
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        with _lock:
            _stats["misses"] += 1
            _stats["errors"] += 1
            _errors[key] = exc
            if len(_errors) > MAX_CACHE_ENTRIES:
                _errors.popitem(last=False)
                _stats["evictions"] += 1
        if raise_on_syntax_error:
            raise
        return None

    with _lock:
        _stats["misses"] += 1
        _cache[key] = tree
        if len(_cache) > MAX_CACHE_ENTRIES:
            _cache.popitem(last=False)
            _stats["evictions"] += 1
    return tree


def clear_parse_cache() -> None:
    """Drop every cached tree and reset the stats counters.

    Safe to call from any thread.  Useful between benchmark runs or
    when a long-running grinder process wants to reclaim memory.
    """
    with _lock:
        _cache.clear()
        _errors.clear()
        for k in _stats:
            _stats[k] = 0


def cache_stats() -> dict:
    """Return a snapshot of ``{hits, misses, errors, evictions, size}``."""
    with _lock:
        return {
            **_stats,
            "size": len(_cache),
            "error_size": len(_errors),
        }
