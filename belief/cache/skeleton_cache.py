"""
Skeleton cache — memoise deterministic skeleton generation (Session 17).

Skeleton assembly is deterministic: given the same spec (goal hash,
framework, complexity, nutrient-profile digest) it produces the same
artefact.  Running it every build wastes the slowest part of a
local-model pipeline for no gain.  This module stores each skeleton
as plain JSON keyed by the hash of its spec so repeat invocations
are a dict lookup instead of a 30-60s generation.

Single-file JSON-per-key layout (not a database):

    ~/.belief-engine/skeleton_cache/
        9e3f.../spec.json        {"goal":"…","framework":"…","complexity":3}
        9e3f.../skeleton.json    {"files":{...},"metadata":{...}}

Each key has its own directory so a corrupt write only loses that
one entry.  All writes are atomic via ``os.replace`` on a ``.tmp``
sibling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("belief.cache.skeleton_cache")


DEFAULT_CACHE_DIR = Path("~/.belief-engine/skeleton_cache").expanduser()


# ── Spec fingerprinting ───────────────────────────────────────────────────


def _canonical_json(obj: Any) -> str:
    """JSON dump with sorted keys and no whitespace for stable hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def fingerprint_spec(spec: dict) -> str:
    """Return a stable 16-hex-char hash of a skeleton spec dict.

    Identical dicts (regardless of key order) map to the same hash.
    Non-JSON-serialisable values are coerced through ``str()`` so the
    cache never crashes on unexpected inputs — we prefer "maybe-miss"
    over "blow up the build".
    """
    try:
        payload = _canonical_json(spec)
    except TypeError:
        payload = _canonical_json(
            {str(k): str(v) for k, v in dict(spec).items()}
        )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ── Filesystem helpers ────────────────────────────────────────────────────


def _key_dir(key: str, base_dir: Path) -> Path:
    return Path(base_dir).expanduser() / str(key)


def _atomic_write(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON to ``path`` atomically.

    Writes to ``<path>.tmp`` first, then ``os.replace``'s into place.
    This guarantees readers never observe a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)


# ── Public API ────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """Metadata wrapper returned by :func:`get_cached_skeleton`."""

    key: str
    skeleton: dict
    spec: dict
    created_at: float = 0.0
    hit_count: int = 0


def cache_skeleton(
    spec: dict,
    skeleton: dict,
    *,
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> str:
    """Store ``skeleton`` keyed by the fingerprint of ``spec``.

    Returns the cache key (16-char hex digest).  Filesystem errors
    are logged and swallowed — a cache miss is always recoverable.
    """
    key = fingerprint_spec(spec)
    target = _key_dir(key, base_dir)
    meta = {
        "key": key,
        "created_at": float(time.time()),
        "hit_count": 0,
    }
    try:
        target.mkdir(parents=True, exist_ok=True)
        _atomic_write(target / "spec.json", spec)
        _atomic_write(target / "skeleton.json", skeleton)
        _atomic_write(target / "meta.json", meta)
    except OSError as exc:
        logger.warning(
            f"skeleton_cache: store failed for key={key}: {exc}"
        )
    return key


def get_cached_skeleton(
    spec: dict,
    *,
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> Optional[CacheEntry]:
    """Return the cached :class:`CacheEntry` for *spec*, or ``None``.

    A corrupt entry (missing files, invalid JSON) is silently
    discarded — the caller will treat it as a miss and regenerate.
    Each hit increments the entry's ``hit_count``.
    """
    key = fingerprint_spec(spec)
    target = _key_dir(key, base_dir)
    spec_path = target / "spec.json"
    skeleton_path = target / "skeleton.json"
    meta_path = target / "meta.json"
    if not (spec_path.exists() and skeleton_path.exists()):
        return None

    try:
        cached_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        cached_skeleton = json.loads(
            skeleton_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(
            f"skeleton_cache: corrupt entry {key}, treating as miss: {exc}"
        )
        return None

    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}

    meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
    try:
        _atomic_write(meta_path, meta)
    except OSError:
        pass  # Don't fail the lookup if we can't bump the counter.

    return CacheEntry(
        key=key,
        skeleton=cached_skeleton,
        spec=cached_spec,
        created_at=float(meta.get("created_at", 0.0)),
        hit_count=int(meta.get("hit_count", 1)),
    )


def invalidate(
    spec: dict, *, base_dir: Path = DEFAULT_CACHE_DIR,
) -> bool:
    """Delete the cached entry for *spec*, if any.

    Returns True when an entry was removed, False when no entry
    existed.  Filesystem errors are logged and treated as "nothing
    was removed".
    """
    key = fingerprint_spec(spec)
    target = _key_dir(key, base_dir)
    if not target.exists():
        return False
    removed = False
    for child in target.iterdir():
        try:
            child.unlink()
            removed = True
        except OSError as exc:
            logger.debug(f"skeleton_cache: unlink {child} failed: {exc}")
    try:
        target.rmdir()
    except OSError:
        pass
    return removed


def cache_size(base_dir: Path = DEFAULT_CACHE_DIR) -> int:
    """Return the number of entries currently cached."""
    base = Path(base_dir).expanduser()
    if not base.exists():
        return 0
    count = 0
    for child in base.iterdir():
        if child.is_dir() and (child / "skeleton.json").exists():
            count += 1
    return count


# ── In-process cache layer (validation-phase Session 1) ───────────────────
#
# SkeletonArtifact is a Pydantic model; instances aren't hashable, so we
# can't stamp ``@lru_cache`` on the generator directly. Instead we
# fingerprint the spec once and memoise on the 16-char hex digest — a
# string, which ``@lru_cache`` is happy with.  The memo is per-process;
# it disappears when the process exits, which is fine because the
# filesystem cache beneath it persists across runs.


@lru_cache(maxsize=256)
def _memo_load_from_disk(key: str, base_dir: str) -> Optional[str]:
    """Process-local memo that shadows the on-disk cache.

    Returns the cached skeleton-payload JSON string, or ``None`` on
    miss. We return JSON rather than a dict because lru_cache will hand
    the same object back to every caller; callers parse their own copy
    so they can mutate freely.
    """
    target = _key_dir(key, Path(base_dir))
    skeleton_path = target / "skeleton.json"
    if not skeleton_path.exists():
        return None
    try:
        return skeleton_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _invalidate_memo(key: str, base_dir: Path) -> None:
    """Drop a single key from the in-process memo.

    Used after ``cache_skeleton`` so a fresh write is immediately
    visible. Cheap full-clear if the base_dir combination was never
    seen before (lru_cache has no per-arg eviction, but ``cache_clear``
    is cheap on a 256-entry cache).
    """
    try:
        _memo_load_from_disk.cache_clear()
    except Exception:
        pass


def get_or_generate_skeleton(
    spec: dict,
    generator,
    *,
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[dict, bool]:
    """Return the cached skeleton payload, generating it on miss.

    Args:
        spec:      Dict describing the build.  Hashed via
                   :func:`fingerprint_spec`.  Only fields that actually
                   influence the skeleton output should be in here —
                   extra noise reduces the hit rate.
        generator: Callable with no required args that returns the
                   skeleton-payload dict (e.g. ``{"files": {...},
                   "registry_context": "..."}``).  Only invoked on
                   cache miss.
        base_dir:  Override the default on-disk cache root (tests).

    Returns:
        ``(payload, cache_hit)`` — ``cache_hit`` is True when the
        payload came from cache, False when it was freshly generated.

    Design notes:
      * A miss path that raises propagates normally; we don't poison
        the cache with a partial entry.
      * Filesystem errors during write are swallowed (matches the rest
        of this module) — next call just re-generates.
    """
    key = fingerprint_spec(spec)

    # Tier 1: in-process memo — the cheapest possible hit.
    cached_json = _memo_load_from_disk(key, str(base_dir))
    if cached_json is not None:
        try:
            payload = json.loads(cached_json)
            logger.debug(f"skeleton_cache: in-process memo hit key={key}")
            return payload, True
        except json.JSONDecodeError:
            _invalidate_memo(key, Path(base_dir))

    # Tier 2: regenerate. cache_skeleton atomically writes to disk;
    # we then refresh the memo so subsequent in-process calls are fast.
    payload = generator()
    cache_skeleton(spec, payload, base_dir=base_dir)
    _invalidate_memo(key, Path(base_dir))
    logger.info(f"skeleton_cache: generated + stored key={key}")
    return payload, False


# ── AST parse cache (validation-phase Session 1) ──────────────────────────
#
# Several passes re-parse the same source (covenant enforcement, import
# verification, symbol extraction).  AST parsing is deterministic: same
# bytes in, same tree out.  ``lru_cache`` on the SHA-1 of the source
# gives us one parse per unique file per process — roughly a 3-5x
# speedup on any node that walks ``code_files`` more than once.


@lru_cache(maxsize=512)
def _cached_ast_parse_by_hash(code_hash: str, source: str):
    """Internal: the actual parser, keyed on both hash and source.

    Keying on hash is what makes this efficient (short cache keys); we
    keep ``source`` in the arg list so unrelated calls with different
    code but hash collisions can't return the wrong tree.
    """
    import ast as _ast
    return _ast.parse(source)


def cached_ast_parse(source: str):
    """Parse ``source`` into an ``ast.AST`` with in-process memoisation.

    Returns a *fresh* module on cache miss and the *same* cached module
    object on hits — callers must not mutate the tree in place.  A
    parse error is raised, not cached.
    """
    if not isinstance(source, str):
        raise TypeError("cached_ast_parse expects a str")
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return _cached_ast_parse_by_hash(digest, source)


def clear_caches() -> None:
    """Drop every in-process memo for this module.

    Called in tests, and useful when the on-disk cache is mutated out
    of band (e.g. a developer deletes ``~/.belief-engine/skeleton_cache``).
    """
    try:
        _memo_load_from_disk.cache_clear()
    except Exception:
        pass
    try:
        _cached_ast_parse_by_hash.cache_clear()
    except Exception:
        pass
