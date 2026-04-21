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
from typing import Any, Callable, Optional

import httpx


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
        **httpx_kwargs: Any,
    ) -> None:
        self._retry_cfg = retry or RetryConfig()
        self._breaker_cfg = breaker or BreakerConfig()
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
            retry=tenacity.retry_if_exception_type(
                (httpx.TransportError, httpx.TimeoutException)
            ),
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

    async def request(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        """Send one request through retry + breaker."""
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
    **httpx_kwargs: Any,
) -> BreakerAsyncClient:
    """Factory for a retry+breaker-wrapped httpx.AsyncClient.

    The spec's default configuration is: 5 attempts, exponential backoff
    with jitter 1-30s, breaker fail_max=5, reset=60s, 4xx excluded.
    Override either dataclass to tune.

    Pass through any httpx kwargs (timeout, headers, http2, etc.).
    """
    return BreakerAsyncClient(retry=retry, breaker=breaker, **httpx_kwargs)


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


__all__ = [
    "BreakerAsyncClient",
    "BreakerConfig",
    "RetryConfig",
    "conditional_get",
    "get_async_client",
]
