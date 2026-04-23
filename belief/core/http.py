"""Shared httpx client wrapped with retry + circuit-breaker semantics.

Spec: "All HTTP goes through tenacity + pybreaker wrappers (importable
from belief.core.http)." This is the thin, subsystem-agnostic glue.

Heavy dependencies (tenacity, pybreaker) are **lazy-imported** so that
`belief.core` stays importable even when the [photosynthesis] extra
isn't installed. Callers that don't need resilience can still use a
plain `httpx.AsyncClient`; callers that want resilience import
`get_async_client()` and eat the one-time import cost.

Defaults match the Photosynthesis design doc:

    tenacity.AsyncRetrying(
        stop_after_attempt=5,
        wait_exponential(multiplier=1, min=1, max=30) + random jitter,
        retry_if_exception_type(httpx.HTTPError, ...),
    )
    pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60,
                             exclude=[_is_client_error])

Notes:
- 4xx client errors are *excluded* from the breaker. If GitHub tells us
  we're unauthenticated, opening the breaker just makes things worse;
  the caller should see the 4xx and either fix auth or back off.
- conditional_get() is a small helper for ETag / If-Modified-Since
  patterns. It does NOT wrap in retry automatically — the caller should
  already be operating under the retry/breaker wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import httpx


# ---------------------------------------------------------------------------
# Domain allowlist
# ---------------------------------------------------------------------------

#: Domains that belief-engine is expected to contact in production.
#: Passed to BreakerAsyncClient(allowed_domains=...) to reject unexpected
#: outbound HTTP before the request is even sent.
DEFAULT_ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        # Anthropic
        "api.anthropic.com",
        # Ollama (local)
        "localhost",
        "127.0.0.1",
        # GitHub
        "api.github.com",
        "github.com",
        "raw.githubusercontent.com",
        # HackerNews (Algolia search)
        "hn.algolia.com",
        # ArXiv
        "arxiv.org",
        "export.arxiv.org",
        # PyPI
        "pypi.org",
        "files.pythonhosted.org",
        # Top-PyPI-packages corpus (hugovk, Apache 2.0 — weekly snapshot)
        "hugovk.github.io",
        # Stack Exchange / Overflow
        "api.stackexchange.com",
        # Telegram (optional notifications)
        "api.telegram.org",
        # Railway (deployment health checks)
        "railway.app",
    }
)


# ---------------------------------------------------------------------------
# Lazy dependency loading
# ---------------------------------------------------------------------------


def _load_tenacity() -> Any:
    """Import tenacity lazily. Raises ImportError with a helpful message."""
    try:
        import tenacity  # noqa: F401
    except ImportError as exc:  # pragma: no cover - install instruction
        raise ImportError(
            "tenacity is required for retry behavior. "
            "Install the [photosynthesis] extra: "
            "pip install -e '.[photosynthesis]'"
        ) from exc
    return tenacity


def _load_pybreaker() -> Any:
    """Import pybreaker lazily."""
    try:
        import pybreaker  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pybreaker is required for circuit-breaker behavior. "
            "Install the [photosynthesis] extra."
        ) from exc
    return pybreaker


# ---------------------------------------------------------------------------
# BreakerAsyncClient
# ---------------------------------------------------------------------------


def _is_client_error(exc: BaseException) -> bool:
    """True for 4xx-ish exceptions — do not count them toward breaker failures."""
    if isinstance(exc, httpx.HTTPStatusError):
        return 400 <= exc.response.status_code < 500
    return False


@dataclass
class RetryConfig:
    """Knobs for the shared retry policy."""

    max_attempts: int = 5
    wait_multiplier: float = 1.0
    wait_min: float = 1.0
    wait_max: float = 30.0


@dataclass
class BreakerConfig:
    """Knobs for the shared circuit breaker."""

    fail_max: int = 5
    reset_timeout: float = 60.0


class BreakerAsyncClient:
    """Thin async wrapper around httpx.AsyncClient with retry + breaker.

    Usage::

        async with get_async_client() as client:
            resp = await client.request("GET", url)
            resp.raise_for_status()

    The breaker is per-instance (and therefore per-client) so that a
    misbehaving source doesn't poison unrelated traffic. Callers that
    want process-wide sharing should create a single client and reuse.
    """

    def __init__(
        self,
        retry: Optional[RetryConfig] = None,
        breaker: Optional[BreakerConfig] = None,
        allowed_domains: Optional[frozenset[str]] = None,
        **httpx_kwargs: Any,
    ) -> None:
        self._retry_cfg = retry or RetryConfig()
        self._breaker_cfg = breaker or BreakerConfig()
        self._allowed_domains = allowed_domains  # None means unrestricted
        self._client = httpx.AsyncClient(**httpx_kwargs)

        # Lazy-build retry + breaker the first time request() is called,
        # so we don't pay the import cost for clients that never fire.
        self._retrying: Any = None
        self._breaker: Any = None

    def _ensure_policies(self) -> None:
        if self._retrying is not None and self._breaker is not None:
            return

        tenacity = _load_tenacity()
        pybreaker = _load_pybreaker()

        self._retrying = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self._retry_cfg.max_attempts),
            wait=tenacity.wait_exponential_jitter(
                initial=self._retry_cfg.wait_min,
                max=self._retry_cfg.wait_max,
                jitter=1.0,
            ),
            retry=tenacity.retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
            reraise=True,
        )

        # Exclude 4xx errors from opening the breaker.
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=self._breaker_cfg.fail_max,
            reset_timeout=self._breaker_cfg.reset_timeout,
            exclude=[_is_client_error],
        )

    async def __aenter__(self) -> "BreakerAsyncClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.__aexit__(*args)

    def _check_domain(self, url: str) -> None:
        """Raise ValueError if the URL's host is not in the allowlist."""
        if self._allowed_domains is None:
            return
        host = urlparse(url).hostname or ""
        # Strip port if present and normalise
        host = host.lower().split(":")[0]
        if host not in self._allowed_domains:
            raise ValueError(
                f"Outbound HTTP blocked: '{host}' is not in the allowed domain list. "
                f"Add it to DEFAULT_ALLOWED_DOMAINS or pass allowed_domains= to override."
            )

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request through retry + breaker."""
        self._check_domain(url)
        self._ensure_policies()

        async def _do() -> httpx.Response:
            # breaker.call_async requires python-circuitbreaker-style coro;
            # pybreaker.CircuitBreakerAsync works with awaitable callables.
            resp = await self._client.request(method, url, **kwargs)
            return resp

        # tenacity wraps the coroutine; pybreaker gates before we even try.
        async def _do_with_breaker() -> httpx.Response:
            return await self._breaker.call_async(_do)  # type: ignore[no-any-return]

        async for attempt in self._retrying:
            with attempt:
                return await _do_with_breaker()
        raise RuntimeError("unreachable: tenacity exited without a result")

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)


def get_async_client(
    *,
    retry: Optional[RetryConfig] = None,
    breaker: Optional[BreakerConfig] = None,
    allowed_domains: Optional[frozenset[str]] = None,
    **httpx_kwargs: Any,
) -> BreakerAsyncClient:
    """Factory for a retry+breaker-wrapped httpx.AsyncClient.

    The spec's default configuration is: 5 attempts, exponential backoff
    with jitter 1-30s, breaker fail_max=5, reset=60s, 4xx excluded.
    Override either dataclass to tune.

    Pass `allowed_domains=DEFAULT_ALLOWED_DOMAINS` to restrict outbound
    HTTP to known-safe hosts. Pass `allowed_domains=None` (default) for
    unrestricted mode (backward-compatible).

    Pass through any httpx kwargs (timeout, headers, http2, etc.).
    """
    return BreakerAsyncClient(
        retry=retry,
        breaker=breaker,
        allowed_domains=allowed_domains,
        **httpx_kwargs,
    )


# ---------------------------------------------------------------------------
# conditional_get — small helper for ETag / If-Modified-Since
# ---------------------------------------------------------------------------


async def conditional_get(
    client: BreakerAsyncClient | httpx.AsyncClient,
    url: str,
    *,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    **kwargs: Any,
) -> tuple[httpx.Response, bool]:
    """GET with ETag / If-Modified-Since headers.

    Returns (response, not_modified). When `not_modified` is True the
    response is a 304 with no body; callers should skip re-parsing.
    """
    headers = dict(kwargs.pop("headers", {}) or {})
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    resp = await client.get(url, headers=headers, **kwargs)
    return resp, resp.status_code == 304


# ---------------------------------------------------------------------------
# Synchronous convenience helpers
# ---------------------------------------------------------------------------


def head_sync(url: str, *, timeout: float = 5.0, headers: Optional[dict] = None) -> int:
    """Synchronous HEAD request. Returns the HTTP status code, or 0 on error.

    Uses httpx.Client (same dependency as the async path).  Centralised here
    so callers (executor.py PyPI checks, etc.) don't scatter urllib calls.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.head(url, headers=headers or {}, follow_redirects=True)
            return resp.status_code
    except Exception:
        return 0


def post_form_sync(
    url: str,
    data: dict,
    *,
    timeout: float = 10.0,
) -> int:
    """Synchronous form-encoded POST. Returns the HTTP status code, or 0 on error.

    Centralised here so callers (notify.py Telegram sends, etc.) don't
    scatter urllib calls and benefit from consistent timeout handling.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, data=data)
            return resp.status_code
    except Exception:
        return 0


def get_bytes_sync(
    url: str,
    *,
    timeout: float = 30.0,
    headers: Optional[dict] = None,
    allowed_domains: Optional[frozenset[str]] = None,
) -> Optional[bytes]:
    """Synchronous GET returning the response body as bytes, or None on error.

    Used for one-shot corpus / blob downloads where the caller just wants
    the raw bytes (e.g. package_validator refreshing the top-15k PyPI
    corpus). Centralised here so one-shot fetch callers don't have to
    construct their own ``httpx.Client`` and re-implement error handling.

    Domain allowlist enforcement mirrors :class:`BreakerAsyncClient` —
    pass ``allowed_domains=DEFAULT_ALLOWED_DOMAINS`` to block unexpected
    outbound HTTP.  ``None`` (default) is unrestricted, matching the
    other sync helpers.
    """
    if allowed_domains is not None:
        host = (urlparse(url).hostname or "").lower().split(":")[0]
        if host not in allowed_domains:
            raise ValueError(f"Outbound HTTP blocked: '{host}' is not in the allowed domain list.")
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers or {}, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
    except Exception:
        return None


__all__ = [
    "BreakerAsyncClient",
    "BreakerConfig",
    "DEFAULT_ALLOWED_DOMAINS",
    "RetryConfig",
    "conditional_get",
    "get_async_client",
    "get_bytes_sync",
    "head_sync",
    "post_form_sync",
]
