"""6-layer package validator — Session 3 (v3.2).

Why a separate validator
------------------------

The overnight logs revealed two failure modes the existing PyPI check
couldn't distinguish:

A) False negative on ``pydantic-settings`` (it IS real).  Root cause:
   the validator passed the import-name ``pydantic_settings``
   (underscore) to PyPI without PEP 503 canonicalisation to hyphens.

B) False positive on LLM-hallucinated packages (``settings_library``,
   ``settings``, ``timeit``).  Root cause: no layer between import-name
   normalisation and PyPI lookup; stdlib names and fictional names
   both slipped through to a 150s ``pip install`` failure.

Spracklen et al. (USENIX Security 2025) measured 21.7% hallucination
rate on open-source coder models, 33%+ on CodeLlama, 14-22% on
Qwen2.5-Coder.  The ``larpexodus`` and ``react-codeshift`` incidents
demonstrate real-world dependency-confusion harm.  Treat every
LLM-emitted ``pip install`` as ~1-in-5 suspect.

Layers (latency-first — cheapest checks first)
----------------------------------------------

1. Canonicalise via ``packaging.utils.canonicalize_name`` (PEP 503).
2. Reject stdlib names (``sys.stdlib_module_names``).
3. Reject known-hallucination blocklist (``known_hallucinations.txt``).
4. Accept if in top-15k PyPI corpus (authoritative, offline).
5. PyPI Simple JSON lookup (``GET /simple/{name}/``, JSON Accept).
6. Levenshtein fuzzy-match suggestion (``rapidfuzz``) on rejection.

Security notes
--------------

* Rejection telemetry is LOCAL-ONLY (``~/.belief-engine/hallucination_log.jsonl``).
  Per Seth Larson's "toxic waste" guidance on slopsquatting, exporting
  404 logs to external telemetry platforms is a slopsquatter's
  wishlist.  Do NOT add external log forwarding for rejection events.
* PyPI Simple lookup uses the `.json` form (``Accept: application/vnd.pypi.simple.v1+json``)
  rather than ``/pypi/{name}/json``; the Simple JSON payload is
  ~10× smaller for pure existence checks.  Do NOT request the
  trailing-slash form bare — warehouse#11535 caused redirect loops.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx  # type retained for signatures (AsyncClient param); transport is core_http

from belief.core.http import (
    DEFAULT_ALLOWED_DOMAINS,
    BreakerAsyncClient,
    get_async_client,
    get_bytes_sync,
)
from belief.validators.import_to_package import resolve_import_to_package

logger = logging.getLogger("belief.validators.package_validator")


# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent
_BLOCKLIST_PATH = _PKG_DIR / "known_hallucinations.txt"
_DEFAULT_CACHE_DIR = Path.home() / ".belief-engine"
_DEFAULT_TOP_PACKAGES_PATH = _DEFAULT_CACHE_DIR / "top-pypi-packages-15k.json"
_DEFAULT_LOOKUP_CACHE_PATH = _DEFAULT_CACHE_DIR / "pypi_lookup_cache.json"
_DEFAULT_REJECTION_LOG_PATH = _DEFAULT_CACHE_DIR / "hallucination_log.jsonl"

# Top-15k source.  hugovk/top-pypi-packages publishes weekly, Apache 2.0.
_TOP_PYPI_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
_TOP_PACKAGES_REFRESH_AFTER_S = 7 * 24 * 3600  # one week

# PyPI Simple JSON endpoint.
_PYPI_SIMPLE_BASE = "https://pypi.org/simple"
_PYPI_SIMPLE_ACCEPT = "application/vnd.pypi.simple.v1+json"
_USER_AGENT = "belief-engine/3.2 (metafiopy@example.com)"

# Cache TTLs (seconds).  Negative cache is intentionally short: new
# PyPI projects register constantly, and we don't want a just-published
# legitimate dep stuck in a negative cache for 24 hours.
_POSITIVE_CACHE_TTL_S = 24 * 3600
_NEGATIVE_CACHE_TTL_S = 3600

# Stdlib name set — authoritative per Python 3.10+.
_STDLIB_NAMES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", frozenset()))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of a 6-layer validation pass.

    ``layer`` records WHICH layer made the decision — used by the
    executor to surface a precise error to the debugger ("rejected
    at stdlib layer" → debugger knows to drop the import, not
    rewrite the package name).
    """

    raw_input: str
    canonical_name: str
    accepted: bool
    layer: str
    reason: str
    suggestion: str | None = None  # fuzzy match when rejected

    def to_log_dict(self) -> dict[str, Any]:
        """JSON-safe dict for the rejection log."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def canonicalize_name(name: str) -> str:
    """Apply PEP 503 canonicalisation without importing ``packaging``
    at module load (it's a heavy transitive).  The PEP 503 rule is:
    lowercase, replace any run of ``_-.`` with a single ``-``.
    """
    if not name:
        return ""
    s = name.strip()
    s = re.sub(r"[-_.]+", "-", s).lower()
    return s


# ---------------------------------------------------------------------------
# Blocklist loading
# ---------------------------------------------------------------------------


def _load_blocklist(path: Path | None = None) -> frozenset[str]:
    """Load known_hallucinations.txt into a frozenset of canonicalised names."""
    target = path or _BLOCKLIST_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("blocklist %s not found; using empty set", target)
        return frozenset()
    names: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.add(canonicalize_name(s))
    return frozenset(names)


# ---------------------------------------------------------------------------
# Top-15k corpus (authoritative positive)
# ---------------------------------------------------------------------------


def refresh_top_packages(
    *,
    path: Path = _DEFAULT_TOP_PACKAGES_PATH,
    timeout_s: float = 30.0,
    force: bool = False,
) -> bool:
    """Download the latest top-pypi-packages corpus if stale.

    Returns True if a download happened, False if the existing file
    was fresh (or the refresh failed silently).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists():
        age = time.time() - path.stat().st_mtime
        if age < _TOP_PACKAGES_REFRESH_AFTER_S:
            return False
    # Session 0.5: route outbound HTTP through belief.core.http so the
    # User-Agent, timeout, and domain-allowlist semantics match every
    # other fetch in the engine.  hugovk.github.io is allow-listed in
    # DEFAULT_ALLOWED_DOMAINS for the top-15k corpus specifically.
    body = get_bytes_sync(
        _TOP_PYPI_URL,
        timeout=timeout_s,
        headers={"User-Agent": _USER_AGENT},
        allowed_domains=DEFAULT_ALLOWED_DOMAINS,
    )
    if body is None:
        logger.warning("top-15k refresh failed (keeping stale corpus if any)")
        return False
    path.write_bytes(body)
    return True


def _load_top_packages(path: Path | None = None) -> frozenset[str]:
    """Load the top-15k corpus into a frozenset of canonicalised names."""
    target = path or _DEFAULT_TOP_PACKAGES_PATH
    if not target.exists():
        return frozenset()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("top-15k corpus unreadable at %s: %s", target, e)
        return frozenset()

    # Schema: hugovk publishes as {"last_update": ..., "rows": [{"project": "...", ...}]}
    raw_names: list[str] = []
    if isinstance(data, dict) and "rows" in data:
        for row in data["rows"]:
            if isinstance(row, dict) and "project" in row:
                raw_names.append(str(row["project"]))
    elif isinstance(data, list):
        # Tolerate a plain-list format too (some mirrors publish that way).
        for entry in data:
            if isinstance(entry, str):
                raw_names.append(entry)
            elif isinstance(entry, dict) and "project" in entry:
                raw_names.append(str(entry["project"]))
    return frozenset(canonicalize_name(n) for n in raw_names if n)


# ---------------------------------------------------------------------------
# PyPI Simple JSON lookup cache
# ---------------------------------------------------------------------------


class _LookupCache:
    """TTL'd positive/negative cache, persisted to a JSON file.

    Schema on disk::

        {
            "canonical-name": {"exists": true, "checked_at": 1713731023.0}
        }
    """

    def __init__(self, path: Path = _DEFAULT_LOOKUP_CACHE_PATH) -> None:
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("lookup cache unreadable (%s): starting empty", e)
            self._data = {}
        self._loaded = True

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data), encoding="utf-8")
        except Exception as e:  # pragma: no cover
            logger.debug("lookup cache save failed (%s)", e)

    def get(self, canonical: str) -> bool | None:
        """None = cache miss (or stale); else cached bool."""
        self._load()
        entry = self._data.get(canonical)
        if entry is None:
            return None
        age = time.time() - float(entry.get("checked_at", 0))
        ttl = _POSITIVE_CACHE_TTL_S if entry.get("exists") else _NEGATIVE_CACHE_TTL_S
        if age > ttl:
            return None
        return bool(entry.get("exists"))

    def put(self, canonical: str, exists: bool) -> None:
        self._load()
        self._data[canonical] = {"exists": exists, "checked_at": time.time()}
        self._save()


async def pypi_simple_exists(
    canonical: str,
    *,
    client: "BreakerAsyncClient | httpx.AsyncClient | None" = None,
    timeout_s: float = 10.0,
) -> bool:
    """Query PyPI's Simple JSON endpoint for a single name.

    Returns True on HTTP 200 (package exists), False on 404, raises
    on transport errors.  Uses the ``.json``-style Accept header so
    the payload is small.

    Session 0.5: when no client is supplied, a
    :class:`belief.core.http.BreakerAsyncClient` is constructed so
    retry / circuit-breaker / domain-allowlist semantics apply.
    Tests can still pass a raw ``httpx.AsyncClient`` (same surface
    for the methods used here) to keep mocking simple.
    """
    url = f"{_PYPI_SIMPLE_BASE}/{canonical}/"
    headers = {"Accept": _PYPI_SIMPLE_ACCEPT, "User-Agent": _USER_AGENT}

    if client is None:
        # Session 0.5: default path goes through BreakerAsyncClient so
        # we inherit the allowlist + tenacity retry + pybreaker breaker.
        async with get_async_client(
            timeout=timeout_s,
            headers=headers,
            allowed_domains=DEFAULT_ALLOWED_DOMAINS,
        ) as c:
            resp = await c.get(url, headers=headers)
    else:
        # Caller-supplied client (httpx.AsyncClient or BreakerAsyncClient
        # — both expose the .get / .request methods used here).
        resp = await client.get(url, headers=headers)

    if resp.status_code == 200:
        return True
    if resp.status_code == 404:
        return False
    # 5xx, 429 — treat as lookup failure but surface to caller; the
    # validator converts these into "soft" accepts (don't block on
    # PyPI being down).
    raise httpx.HTTPStatusError(
        f"PyPI Simple lookup returned {resp.status_code}",
        request=resp.request,
        response=resp,
    )


# ---------------------------------------------------------------------------
# Levenshtein fuzzy match
# ---------------------------------------------------------------------------


def _fuzzy_match_suggestion(
    canonical: str, corpus: Iterable[str], *, max_distance: int = 2
) -> str | None:
    """Return the closest name in ``corpus`` within ``max_distance``
    edits, or None.  Uses rapidfuzz when available for speed.
    """
    canonical = canonical.strip()
    if not canonical:
        return None
    try:
        # Availability probe for the rapidfuzz top-level package; the
        # actual Levenshtein implementation is imported below.
        import rapidfuzz  # noqa: F401
    except ImportError:  # pragma: no cover
        return None

    # Use ``rapidfuzz.distance.Levenshtein.distance`` directly — it's
    # simpler and faster than piping scores through ``fuzz.ratio``.
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:  # pragma: no cover
        return None

    best_name: str | None = None
    best_dist = max_distance + 1
    for candidate in corpus:
        d = Levenshtein.distance(canonical, candidate, score_cutoff=max_distance)
        if d is None:
            continue
        if d < best_dist:
            best_dist = d
            best_name = candidate
    return best_name if best_dist <= max_distance else None


# ---------------------------------------------------------------------------
# Rejection telemetry — LOCAL ONLY
# ---------------------------------------------------------------------------


def _log_rejection(result: ValidationResult, log_path: Path = _DEFAULT_REJECTION_LOG_PATH) -> None:
    """Append a rejection event to the local JSONL log.

    Security constraint: this log MUST stay on-disk and MUST NOT be
    shipped to external telemetry (Sentry, Datadog, structured log
    forwarders).  Publishing the set of hallucinated names is a
    slopsquatter's wishlist — they can register them before the next
    run repeats the query.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **result.to_log_dict(),
            "ts": time.time(),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:  # pragma: no cover
        logger.debug("rejection log write failed: %s", e)


# ---------------------------------------------------------------------------
# PackageValidator — the public 6-layer API
# ---------------------------------------------------------------------------


class PackageValidator:
    """6-layer validator: canonicalize → stdlib → hallucinations →
    top-15k → PyPI Simple → fuzzy-suggest.

    Constructor takes paths so tests can inject fixtures; callers in
    production just instantiate with defaults.

    Usage (async)::

        validator = PackageValidator()
        result = await validator.validate("pydantic_settings")
        if not result.accepted:
            logger.warning("blocked: %s (%s)", result.raw_input, result.reason)
    """

    def __init__(
        self,
        *,
        blocklist_path: Path | None = None,
        top_packages_path: Path | None = None,
        lookup_cache_path: Path | None = None,
        rejection_log_path: Path | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._blocklist = _load_blocklist(blocklist_path)
        self._top15k = _load_top_packages(top_packages_path)
        self._lookup_cache = _LookupCache(lookup_cache_path or _DEFAULT_LOOKUP_CACHE_PATH)
        self._rejection_log_path = rejection_log_path or _DEFAULT_REJECTION_LOG_PATH
        self._http_client = http_client  # injectable for tests

    # -- Layer helpers -----------------------------------------------------

    def _layer_2_stdlib(self, canonical: str) -> ValidationResult | None:
        if canonical in _STDLIB_NAMES:
            return ValidationResult(
                raw_input=canonical,
                canonical_name=canonical,
                accepted=False,
                layer="stdlib",
                reason=(
                    f"{canonical} is a stdlib module, not a pip package. "
                    f"Remove from requirements.txt — `import {canonical}` is all you need."
                ),
            )
        return None

    def _layer_3_hallucination(self, canonical: str) -> ValidationResult | None:
        if canonical in self._blocklist:
            return ValidationResult(
                raw_input=canonical,
                canonical_name=canonical,
                accepted=False,
                layer="hallucination",
                reason=(
                    f"{canonical} is in the known-hallucination blocklist "
                    "(seen to fail on prior builds)."
                ),
            )
        return None

    def _layer_4_top15k(self, canonical: str) -> ValidationResult | None:
        if canonical in self._top15k:
            return ValidationResult(
                raw_input=canonical,
                canonical_name=canonical,
                accepted=True,
                layer="top15k",
                reason="present in top-15k PyPI corpus",
            )
        return None

    async def _layer_5_pypi(self, canonical: str) -> ValidationResult | None:
        # Cache first
        cached = self._lookup_cache.get(canonical)
        if cached is True:
            return ValidationResult(
                raw_input=canonical,
                canonical_name=canonical,
                accepted=True,
                layer="pypi_cache",
                reason="cached positive (PyPI Simple lookup)",
            )
        if cached is False:
            return ValidationResult(
                raw_input=canonical,
                canonical_name=canonical,
                accepted=False,
                layer="pypi_cache",
                reason="cached negative (PyPI 404)",
            )

        try:
            exists = await pypi_simple_exists(canonical, client=self._http_client)
        except Exception as e:
            # Soft-accept on PyPI transport failure — blocking every
            # build because pypi.org is flaky is worse than the small
            # chance of letting a hallucinated name through to pip.
            logger.warning("PyPI Simple lookup error for %r: %s — soft-accepting", canonical, e)
            return ValidationResult(
                raw_input=canonical,
                canonical_name=canonical,
                accepted=True,
                layer="pypi_soft_accept",
                reason=f"PyPI lookup failed ({e}); allowing with warning",
            )

        self._lookup_cache.put(canonical, exists)
        if exists:
            return ValidationResult(
                raw_input=canonical,
                canonical_name=canonical,
                accepted=True,
                layer="pypi",
                reason="PyPI Simple returned 200",
            )
        return ValidationResult(
            raw_input=canonical,
            canonical_name=canonical,
            accepted=False,
            layer="pypi",
            reason="PyPI Simple returned 404",
        )

    def _fuzzy_suggest(self, canonical: str) -> str | None:
        if not self._top15k:
            return None
        return _fuzzy_match_suggestion(canonical, self._top15k, max_distance=2)

    # -- Public API --------------------------------------------------------

    async def validate(self, raw_name: str) -> ValidationResult:
        """Run all 6 layers.  Never raises — transport problems
        become soft-accepts with a ``pypi_soft_accept`` layer tag.

        Also resolves import-name aliases (``cv2`` → ``opencv-python``)
        before canonicalisation, so callers can pass import names or
        package names interchangeably.
        """
        # Pre-step: import-name resolution (cv2 → opencv-python).
        resolved = resolve_import_to_package(raw_name) if raw_name else raw_name

        # Layer 1: canonicalise.
        canonical = canonicalize_name(resolved)

        # Layer 2: stdlib.
        r2 = self._layer_2_stdlib(canonical)
        if r2 is not None:
            r2.raw_input = raw_name
            _log_rejection(r2, self._rejection_log_path)
            return r2

        # Layer 3: hallucination blocklist.
        r3 = self._layer_3_hallucination(canonical)
        if r3 is not None:
            r3.raw_input = raw_name
            r3.suggestion = self._fuzzy_suggest(canonical)
            _log_rejection(r3, self._rejection_log_path)
            return r3

        # Layer 4: top-15k positive.
        r4 = self._layer_4_top15k(canonical)
        if r4 is not None:
            r4.raw_input = raw_name
            return r4

        # Layer 5: PyPI Simple lookup with cache.
        r5 = await self._layer_5_pypi(canonical)
        if r5 is not None:
            r5.raw_input = raw_name
            if not r5.accepted:
                r5.suggestion = self._fuzzy_suggest(canonical)
                _log_rejection(r5, self._rejection_log_path)
            return r5

        # Unreachable — _layer_5_pypi always returns.
        return ValidationResult(
            raw_input=raw_name,
            canonical_name=canonical,
            accepted=False,
            layer="unknown",
            reason="validator fell through all layers — this should never happen",
        )

    def add_hallucination(self, name: str) -> None:
        """Append a new name to the blocklist at runtime.  The
        ``belief validator add-hallucination <name>`` CLI delegates
        here.  Persists to ``known_hallucinations.txt``.
        """
        canonical = canonicalize_name(name)
        if canonical in self._blocklist:
            return
        try:
            with _BLOCKLIST_PATH.open("a", encoding="utf-8") as f:
                f.write(f"{canonical}\n")
        except Exception as e:
            logger.warning("could not persist hallucination %r: %s", canonical, e)
            return
        self._blocklist = frozenset(self._blocklist | {canonical})


__all__ = [
    "PackageValidator",
    "ValidationResult",
    "canonicalize_name",
    "pypi_simple_exists",
    "refresh_top_packages",
]
