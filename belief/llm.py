"""LLM Client — the single path to any language model.

All agents call through this. Handles:
  - Model routing via ModelRouter
  - Structured output parsing via Pydantic
  - Token usage tracking
  - Graceful fallback on connection errors

Source: forge/llm.py + belief_call.py covenant gate concept
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextvars import ContextVar
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from belief.config.models import Backend, ModelRole, ModelRouter
from belief.config.local_cost_tracker import LocalCostTracker
from belief.llm_errors import (
    OllamaContextExceeded,
    OllamaError,
    OllamaPermanentError,
    OllamaStreamStall,
    OllamaTransientError,
)
from belief.models.artifacts import TokenUsage, _cost_usd

logger = logging.getLogger("belief.llm")

# ---------------------------------------------------------------------------
# Session 1 (v3.2): bulletproof Ollama — optional dependencies
# ---------------------------------------------------------------------------
# tenacity drives the retry loop; pybreaker drives the per-model circuit
# breaker. Both are declared in pyproject.toml as hard deps in v3.2, but
# we import them defensively so a v3.1 install that predates the upgrade
# still boots (the fallback paths disable retry/breaker gracefully).
try:  # pragma: no cover - import-time toggle
    import tenacity  # type: ignore
    from tenacity import (
        AsyncRetrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TENACITY_AVAILABLE = False

try:  # pragma: no cover - import-time toggle
    import pybreaker  # type: ignore

    _PYBREAKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYBREAKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# ROLE_BUDGETS — per-role wall-clock ceiling (seconds)
# ---------------------------------------------------------------------------
# Replaces the single 300s cap that crashed the architect overnight.  The
# budget is a wall-clock watchdog applied with asyncio.wait_for *outside*
# retry and streaming — it is the worst-case time this role may consume
# before the caller gives up.  Separate from inactivity_s (30s per chunk)
# and tenacity retry (3 attempts).
ROLE_BUDGETS: dict[str, float] = {
    "intake": 60.0,
    "research": 180.0,
    "planner": 120.0,
    "architect": 600.0,  # largest — most complex output
    "builder": 300.0,
    "debugger": 180.0,
    "tester": 120.0,
    "gap_analyst": 90.0,
    "synthesizer": 180.0,
    "validator": 90.0,
    "latios": 60.0,
    "executor": 60.0,
    "default": 180.0,
}


# ---------------------------------------------------------------------------
# GRACEFUL_DEGRADATION_CASCADE — model fallback chain for local mode
# ---------------------------------------------------------------------------
# When the primary local model's circuit breaker is open or its call
# stalls past inactivity_s, the AsyncOllamaClient transparently retries
# against the next resident model.  The cloud-only tier is handled by
# LLMClient._call_with_role (it already tracks the fallback counter).
GRACEFUL_DEGRADATION_CASCADE: list[str] = [
    "qwen2.5-coder:14b",  # primary
    "qwen2.5-coder:7b",  # fallback_1 (resident per OLLAMA_MAX_LOADED_MODELS=2)
    # fallback_2 is cloud Anthropic Haiku — LLMClient handles that path.
]


# ---------------------------------------------------------------------------
# Per-request Ollama option defaults that are NOT tied to a specific model
# ---------------------------------------------------------------------------
# These are the session-1 "prefix-cache enables + throughput" defaults
# from the research report.  Applied in AsyncOllamaClient.generate() AFTER
# per-model config (local_models.py), so a model-specific setting wins
# when it's provided.  Caller-supplied values (max_tokens, temperature)
# override everything.
#
# num_keep=512 is the most important entry — it pins the system-prompt
# KV cache across requests, which combined with a byte-stable system
# prefix gives the ~18x TTFT speedup the research report describes.
# DO NOT lower num_keep below the longest system prompt length in tokens.
_SESSION1_OPTION_DEFAULTS: dict[str, Any] = {
    "num_gpu": 99,  # offload all layers to GPU on Metal
    "num_thread": 6,  # M2 Air has 4 perf + 4 efficiency cores
    "num_batch": 256,  # batch size for prompt eval
    "num_keep": 512,  # KV-cache pin for system-prompt prefix
    "mirostat": 0,  # greedy sampling — predictable throughput
    # repeat_penalty is set by local_models.py per model; 1.05 is the
    # session-1 default only when the model config doesn't specify one.
}


# ---------------------------------------------------------------------------
# Circuit breaker registry — one breaker per model name
# ---------------------------------------------------------------------------
# A wedged 14B instance tripping the breaker must not block a healthy
# 7B instance, so we key by model name rather than by base_url.
_MODEL_BREAKERS: dict[str, Any] = {}


def _get_breaker(model: str) -> Any | None:
    """Lazy-allocate a pybreaker.CircuitBreaker for ``model``.

    fail_max=5, reset_timeout=60s.  OllamaPermanentError and subclasses
    are excluded so context-length-exceeded (which will never succeed
    on retry) doesn't count toward the 5-failure trip threshold.
    """
    if not _PYBREAKER_AVAILABLE:
        return None
    b = _MODEL_BREAKERS.get(model)
    if b is None:
        b = pybreaker.CircuitBreaker(  # type: ignore[attr-defined]
            fail_max=5,
            reset_timeout=60,
            exclude=[OllamaPermanentError, OllamaContextExceeded],
            name=f"ollama:{model}",
        )
        _MODEL_BREAKERS[model] = b
    return b


def _breaker_is_open(model: str) -> bool:
    b = _MODEL_BREAKERS.get(model)
    if b is None:
        return False
    try:
        return bool(getattr(b, "current_state", "closed") == "open")
    except Exception:
        return False


def _breaker_record(model: str, exc: BaseException | None) -> None:
    """Feed pybreaker a synthetic call reflecting the outcome.

    ``exc=None`` → success (resets failure count).
    ``exc`` → failure (may trip the breaker after fail_max).
    Permanent errors are ignored per the breaker's exclude list.

    Lazy-allocates the breaker via :func:`_get_breaker` so the first
    observed failure is actually recorded (previously callers only hit
    ``_MODEL_BREAKERS.get(model)`` which returned ``None`` until
    someone else had already called ``_get_breaker``).
    """
    b = _get_breaker(model)
    if b is None:
        return
    if exc is None:
        try:
            b.call(lambda: None)  # type: ignore[attr-defined]
        except Exception:
            pass
        return
    if isinstance(exc, (OllamaPermanentError, OllamaContextExceeded)):
        return

    def _raiser():
        raise exc  # type: ignore[misc]

    try:
        b.call(_raiser)  # type: ignore[attr-defined]
    except Exception:
        # pybreaker re-raises the underlying error; we only wanted the
        # side-effect on the breaker's state counter.
        pass


# ---------------------------------------------------------------------------
# 4xx error classification helper
# ---------------------------------------------------------------------------
_CONTEXT_EXCEEDED_PHRASES = (
    "context length",
    "context window",
    "context size",
    "exceeds the model's",
    "input is too long",
    "too many tokens",
    "tokens exceed",
)


def _classify_ollama_error(status_code: int, body_text: str) -> OllamaError:
    """Map an HTTP response from Ollama to the right exception class.

    The overnight logs showed a 400 with ``"exceeds context length"`` in
    the body being caught as a generic httpx.HTTPStatusError and then
    retried 3 times.  Classifying it here as :class:`OllamaContextExceeded`
    (a :class:`OllamaPermanentError` subclass) means tenacity's
    ``retry_if_exception_type`` predicate will skip it.
    """
    lower = (body_text or "").lower()
    if any(p in lower for p in _CONTEXT_EXCEEDED_PHRASES):
        return OllamaContextExceeded(
            f"Ollama rejected request (HTTP {status_code}): context length exceeded. "
            f"Body: {body_text[:300]}"
        )
    if status_code in (400, 404, 422):
        return OllamaPermanentError(
            f"Ollama rejected request (HTTP {status_code}): {body_text[:300]}"
        )
    # 5xx, 429, etc. are transient — retry may succeed.
    return OllamaTransientError(f"Ollama transient error (HTTP {status_code}): {body_text[:300]}")


async def health_ok(base_url: str = "http://localhost:11434", timeout_s: float = 5.0) -> bool:
    """5s GET /api/tags preflight.

    Called by the graceful-degradation logic before each agent session
    so a dead Ollama yields a clean "skip and log" path instead of a
    hanging 300s connect.  Kept as a module-level helper so callers
    outside LLMClient (e.g., the ``belief models`` CLI) can reuse it.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
            r = await client.get("/api/tags")
            return r.status_code == 200
    except Exception:
        return False


T = TypeVar("T", bound=BaseModel)

# Context variable for per-agent token accumulation
# BaseAgent.__call__ sets this before run() and reads it after
_usage_ctx: ContextVar[TokenUsage | None] = ContextVar("_usage_ctx", default=None)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# ---------------------------------------------------------------------------
# Session 6: Ollama backend
# ---------------------------------------------------------------------------


# Shared tracker so callers can read fallback count + per-model breakdown.
# Exposed module-level so the CLI `belief models` command can inspect it.
LOCAL_TRACKER = LocalCostTracker()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — Ollama's /api/chat doesn't return usage by default.

    Anthropic tokenization averages ~4 chars/token for English. We use the
    same heuristic for Ollama outputs so the fallback-counter and efficiency
    metrics stay comparable between backends.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class AsyncOllamaClient:
    """Async Ollama backend — Session 1 (v3.2) hardened edition.

    Replaces the fragile single-timeout POST call that crashed the
    architect with httpx.ReadTimeout overnight.  Public surface is
    unchanged so LLMClient._call_with_role doesn't need to learn new
    tricks: ``generate()``, ``is_available()``, ``close()``.

    Three principles from the research report drive the rearchitecture:

    1. **Streaming with per-token inactivity watchdog.**  We POST to
       ``/api/chat`` with ``stream=true`` and iterate NDJSON chunks,
       wrapping each ``__anext__`` in ``asyncio.wait_for(inactivity_s)``.
       A wedged runner that emits no chunks for ``inactivity_s`` raises
       :class:`OllamaStreamStall` (transient → retryable) and POSTs
       ``{"keep_alive": 0}`` to ``/api/generate`` to force-unload the
       runner before the retry sees it.

    2. **Error classification.**  4xx responses whose body mentions
       "context length" or "exceeds" raise
       :class:`OllamaContextExceeded` (permanent → not retried).  Other
       4xx raise :class:`OllamaPermanentError`.  5xx/429 raise
       :class:`OllamaTransientError`.  tenacity's retry predicate only
       fires on transient errors.

    3. **Per-role wall-clock ceiling.**  Instead of a single 300s cap
       for all calls, each call selects a budget from
       :data:`ROLE_BUDGETS` based on the ``role`` arg.  Architect gets
       600s, executor gets 60s.  ``asyncio.wait_for`` is the outer gate
       around retry + streaming.

    Streaming graceful degradation
    ------------------------------
    If the underlying ``httpx.AsyncClient`` doesn't expose ``stream()``
    (e.g., a test's monkeypatched fake that only supports ``post()``),
    :meth:`generate` falls back to a non-streaming POST and returns
    the same Anthropic-shaped dict.  This keeps the existing
    tests/test_local_routing.py fakes working without changes.

    Env var overrides (read on construction):
      BELIEF_OLLAMA_KEEP_ALIVE     — "30m", "1h", "-1" (forever). Default: "30m".
      BELIEF_OLLAMA_NUM_CTX        — integer token window.         Default: 8192.
      BELIEF_OLLAMA_INACTIVITY_S   — per-chunk watchdog in seconds. Default: 30.
      BELIEF_OLLAMA_STREAM         — "0" to force non-streaming.    Default: streaming.
    """

    _DEFAULT_KEEP_ALIVE = "30m"
    _DEFAULT_NUM_CTX = 8192
    _DEFAULT_INACTIVITY_S = 30.0

    def __init__(
        self,
        *,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434",
        timeout: float | httpx.Timeout | None = None,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
        inactivity_s: float | None = None,
        stream: bool | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        # ``timeout=None`` → Session-1 default (connect=10, read=None, write=10,
        # pool=10). read=None delegates stall detection to the per-chunk
        # inactivity watchdog. Passing a float preserves back-compat with
        # callers that want a hard read timeout.
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

        # Pull per-model defaults from belief.config.local_models — when
        # the model name matches a known entry, those defaults apply
        # unless overridden by kwargs or env vars.  Unknown models fall
        # back to the conservative DEFAULT_LOCAL_MODEL_CONFIG.
        from belief.config.local_models import get_model_config

        model_config = get_model_config(model)
        self._model_config = model_config

        # Resolution order: explicit arg > env var > model-table > class default.
        env_keep_alive = os.environ.get("BELIEF_OLLAMA_KEEP_ALIVE", "").strip()
        if keep_alive is not None:
            self.keep_alive = keep_alive
        elif env_keep_alive:
            self.keep_alive = env_keep_alive
        else:
            self.keep_alive = str(model_config.get("keep_alive", self._DEFAULT_KEEP_ALIVE))

        env_num_ctx = os.environ.get("BELIEF_OLLAMA_NUM_CTX", "").strip()
        if num_ctx is not None:
            self.num_ctx = int(num_ctx)
        elif env_num_ctx:
            try:
                self.num_ctx = int(env_num_ctx)
            except ValueError:
                self.num_ctx = int(model_config.get("num_ctx", self._DEFAULT_NUM_CTX))
        else:
            self.num_ctx = int(model_config.get("num_ctx", self._DEFAULT_NUM_CTX))

        # Inactivity watchdog — per-chunk timeout inside the streaming loop.
        env_inactivity = os.environ.get("BELIEF_OLLAMA_INACTIVITY_S", "").strip()
        if inactivity_s is not None:
            self.inactivity_s = float(inactivity_s)
        elif env_inactivity:
            try:
                self.inactivity_s = float(env_inactivity)
            except ValueError:
                self.inactivity_s = self._DEFAULT_INACTIVITY_S
        else:
            self.inactivity_s = self._DEFAULT_INACTIVITY_S

        # Stream toggle — env var BELIEF_OLLAMA_STREAM=0 disables streaming
        # entirely (useful on networks where chunked transfer is flaky).
        env_stream = os.environ.get("BELIEF_OLLAMA_STREAM", "").strip()
        if stream is not None:
            self._stream = bool(stream)
        elif env_stream in {"0", "false", "False", "no"}:
            self._stream = False
        else:
            self._stream = True

        # Additional sampling defaults from the model config — applied
        # in generate() below as Ollama `options`.
        self._default_num_predict = int(model_config.get("num_predict", 4096))
        self._default_temperature = float(model_config.get("temperature", 0.0))
        self._default_repeat_penalty = model_config.get("repeat_penalty")  # optional

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            if self._timeout is None:
                # Session-1 default: unbounded read so the per-chunk
                # watchdog owns stall detection; short connect/write
                # so a dead Ollama fails in <10s instead of 300.
                timeout_arg: Any = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            else:
                timeout_arg = self._timeout
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_arg)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str | None = None,
        role: str = "default",
        budget_s: float | None = None,
    ) -> dict[str, Any]:
        """POST /api/chat with streaming + watchdog + retry + breaker.

        Returns an Anthropic-shaped response dict identical in shape to
        the v3.1 implementation so LLMClient doesn't need to learn new
        tricks::

          {"content": [{"type": "text", "text": "..."}],
           "usage": {"input_tokens": int, "output_tokens": int,
                     "cache_read_input_tokens": 0,
                     "cache_creation_input_tokens": 0},
           "_backend": "ollama"}

        Args:
            system:        system prompt (kept byte-stable for prefix cache)
            user:          user message
            max_tokens:    maps to Ollama ``num_predict``
            temperature:   sampling temperature
            model:         override default model for this call
            role:          selects budget from :data:`ROLE_BUDGETS`
            budget_s:      explicit wall-clock ceiling (overrides role lookup)
        """
        # Wall-clock budget — outer gate.  Per-role defaults via
        # ROLE_BUDGETS; callers may pass an explicit budget_s override.
        if budget_s is None:
            budget_s = ROLE_BUDGETS.get(role, ROLE_BUDGETS["default"])

        target_model = model or self.model

        # Fail fast if the breaker is already open for this model.
        if _breaker_is_open(target_model):
            raise OllamaTransientError(
                f"circuit breaker open for {target_model} (>=5 recent failures)"
            )

        async def _one_attempt() -> dict[str, Any]:
            return await self._generate_once(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                target_model=target_model,
            )

        async def _retry_wrapper() -> dict[str, Any]:
            if not _TENACITY_AVAILABLE:
                # No tenacity — single attempt, with breaker accounting.
                try:
                    out = await _one_attempt()
                    _breaker_record(target_model, None)
                    return out
                except Exception as e:
                    _breaker_record(target_model, e)
                    raise

            # tenacity: retry transient + httpx read/connect errors only.
            retry_predicate = retry_if_exception_type(
                (OllamaTransientError, httpx.ReadError, httpx.ConnectError)
            )
            async_retrying = AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=20),
                retry=retry_predicate,
                reraise=True,
            )
            last_exc: BaseException | None = None
            try:
                async for attempt in async_retrying:
                    with attempt:
                        try:
                            result = await _one_attempt()
                            _breaker_record(target_model, None)
                            return result
                        except Exception as e:
                            last_exc = e
                            _breaker_record(target_model, e)
                            raise
            except tenacity.RetryError as re:  # type: ignore[attr-defined]
                # Shouldn't hit this because reraise=True, but guard anyway.
                if last_exc is not None:
                    raise last_exc
                raise OllamaTransientError("retries exhausted") from re
            # Unreachable; tenacity returns via the attempt block above.
            raise OllamaTransientError("retries exhausted")

        try:
            return await asyncio.wait_for(_retry_wrapper(), timeout=budget_s)
        except asyncio.TimeoutError as e:
            _breaker_record(target_model, OllamaTransientError("role budget exceeded"))
            raise OllamaTransientError(
                f"role={role} budget of {budget_s}s exhausted on model={target_model}"
            ) from e

    async def is_available(self) -> bool:
        """True iff Ollama's HTTP endpoint responds to /api/tags.

        Tolerant of the monkeypatched fakes used in tests/test_local_routing.py:
        we only use ``client.get`` which every fake supports.
        """
        try:
            client = self._get_client()
            r = await client.get("/api/tags")
            return r.status_code == 200
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException):
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal — one-attempt call, streams when possible
    # ------------------------------------------------------------------

    async def _generate_once(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        target_model: str,
    ) -> dict[str, Any]:
        """A single /api/chat call, chosen to stream or POST.

        This is the body of one tenacity attempt.  Raises
        :class:`OllamaTransientError` / :class:`OllamaPermanentError` /
        :class:`OllamaStreamStall` / :class:`OllamaContextExceeded` —
        tenacity's retry predicate decides which propagate.
        """
        client = self._get_client()

        # Build options, layering: session-1 defaults → model config → per-call.
        # Caller's max_tokens and temperature always win.
        options: dict[str, Any] = dict(_SESSION1_OPTION_DEFAULTS)
        options["num_ctx"] = int(self.num_ctx)
        options["num_predict"] = int(max_tokens)
        options["temperature"] = float(temperature)
        if self._default_repeat_penalty is not None:
            options["repeat_penalty"] = float(self._default_repeat_penalty)
        else:
            options.setdefault("repeat_penalty", 1.05)

        payload_messages = [
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user},
        ]

        # Stream-capable httpx.AsyncClient has .stream(); FakeClient in the
        # existing hermetic tests doesn't.  Fall back to .post() when stream
        # is disabled or unavailable.
        use_stream = self._stream and hasattr(client, "stream")
        if not use_stream:
            return await self._generate_post(client, target_model, payload_messages, options)

        return await self._generate_stream(client, target_model, payload_messages, options)

    async def _generate_post(
        self,
        client: Any,
        target_model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Non-streaming POST path — used when streaming is disabled or
        the client (a test fake) doesn't expose ``.stream()``.

        Keeps byte-for-byte compatibility with the v3.1 response shape.
        """
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        try:
            resp = await client.post("/api/chat", json=payload)
        except (httpx.ReadError, httpx.ConnectError) as e:
            # Transient network error — let tenacity retry.
            raise OllamaTransientError(f"network error: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTransientError(f"timeout: {e}") from e

        # Classify HTTP status before trusting the JSON body.
        status_code = getattr(resp, "status_code", 200)
        if 400 <= status_code < 600:
            body_text = ""
            try:
                body_text = resp.text if hasattr(resp, "text") else ""
            except Exception:
                body_text = ""
            raise _classify_ollama_error(status_code, body_text)

        # raise_for_status on fakes is a no-op; on real httpx it's safe
        # because we already handled 4xx/5xx above.
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = getattr(e.response, "text", "") if hasattr(e, "response") else ""
            raise _classify_ollama_error(status_code, body) from e

        data = resp.json()
        text = (data.get("message") or {}).get("content", "") or ""
        system_txt = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_txt = next((m["content"] for m in messages if m.get("role") == "user"), "")
        in_tokens = int(
            data.get("prompt_eval_count") or _estimate_tokens(f"{system_txt} {user_txt}")
        )
        out_tokens = int(data.get("eval_count") or _estimate_tokens(text))
        return {
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "_backend": "ollama",
        }

    async def _generate_stream(
        self,
        client: Any,
        target_model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Streaming NDJSON path with per-chunk inactivity watchdog.

        Ollama's /api/chat with ``stream=true`` emits a stream of
        NDJSON objects.  Each one has ``message.content`` (partial
        delta) plus a ``done: true`` terminator object carrying the
        final token counts.

        We wrap each ``aiter.__anext__()`` in ``asyncio.wait_for`` with
        ``self.inactivity_s`` so a wedged runner that sends no chunks
        for 30 seconds raises :class:`OllamaStreamStall` instead of
        hanging for 300.  On stall we POST ``{"keep_alive": 0}`` to
        ``/api/generate`` with the same model name — that forces Ollama
        to unload the wedged runner so the next retry hits a fresh one.
        """
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": options,
        }

        chunks: list[str] = []
        prompt_eval_count = 0
        eval_count = 0

        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                status_code = response.status_code
                if 400 <= status_code < 600:
                    # Read the body so we can classify it.
                    body_bytes = b""
                    try:
                        async for chunk in response.aiter_bytes():
                            body_bytes += chunk
                            if len(body_bytes) > 4096:
                                break
                    except Exception:
                        pass
                    raise _classify_ollama_error(status_code, body_bytes.decode(errors="replace"))

                aiter = response.aiter_lines()
                while True:
                    try:
                        line = await asyncio.wait_for(
                            aiter.__anext__(),
                            timeout=self.inactivity_s,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as e:
                        logger.warning(
                            "Ollama stream stalled after %.0fs on %s — requesting runner unload",
                            self.inactivity_s,
                            target_model,
                        )
                        # Best-effort runner unload.  We don't let this
                        # block the stall propagation.
                        try:
                            await asyncio.wait_for(
                                client.post(
                                    "/api/generate",
                                    json={"model": target_model, "keep_alive": 0},
                                ),
                                timeout=5.0,
                            )
                        except Exception as unload_err:
                            logger.debug("keep_alive=0 unload failed: %s", unload_err)
                        raise OllamaStreamStall(
                            f"no chunks from {target_model} in {self.inactivity_s}s",
                            inactivity_s=self.inactivity_s,
                            model=target_model,
                        ) from e

                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("skipping non-JSON stream line: %r", line[:80])
                        continue
                    msg = obj.get("message") or {}
                    if msg.get("content"):
                        chunks.append(msg["content"])
                    if obj.get("done"):
                        prompt_eval_count = int(obj.get("prompt_eval_count") or 0)
                        eval_count = int(obj.get("eval_count") or 0)
                        break
        except (httpx.ReadError, httpx.ConnectError) as e:
            raise OllamaTransientError(f"network error during stream: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTransientError(f"timeout during stream: {e}") from e

        text = "".join(chunks)
        system_txt = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_txt = next((m["content"] for m in messages if m.get("role") == "user"), "")
        in_tokens = prompt_eval_count or _estimate_tokens(f"{system_txt} {user_txt}")
        out_tokens = eval_count or _estimate_tokens(text)
        return {
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": int(in_tokens),
                "output_tokens": int(out_tokens),
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "_backend": "ollama",
        }


# ---------------------------------------------------------------------------
# Graceful degradation helper
# ---------------------------------------------------------------------------


async def graceful_degradation_cascade(
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    role: str,
    base_url: str = "http://localhost:11434",
    cascade: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Try models in :data:`GRACEFUL_DEGRADATION_CASCADE` until one works.

    Returns ``(response_dict, model_used)``.  Raises :class:`OllamaError`
    if every model in the cascade fails.  The cloud-Anthropic tier is
    NOT exercised here — that's :class:`LLMClient._call_with_role`'s
    responsibility because only it has access to the router state.
    """
    models = cascade if cascade is not None else GRACEFUL_DEGRADATION_CASCADE
    last_err: BaseException | None = None
    for m in models:
        client = AsyncOllamaClient(model=m, base_url=base_url)
        try:
            out = await client.generate(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                role=role,
            )
            await client.close()
            return out, m
        except OllamaError as e:
            logger.warning(
                "graceful_degradation_cascade: %s failed (%s); advancing",
                m,
                type(e).__name__,
            )
            last_err = e
            try:
                await client.close()
            except Exception:
                pass
    if last_err is not None:
        raise last_err
    raise OllamaError("graceful_degradation_cascade: empty cascade")


class LLMClient:
    """Unified LLM client for the Belief Engine.

    Usage:
        llm = LLMClient(router)
        text = await llm.generate_text(role, system, prompt)
        obj = await llm.generate_structured(role, system, prompt, MyModel)
        await llm.close()
    """

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self._client: httpx.AsyncClient | None = None
        # Session 6: lazy Ollama client. Built only when a call actually
        # needs it, so cloud-mode callers never pay the connection cost.
        self._ollama: AsyncOllamaClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            from belief.config.settings import settings

            self._client = httpx.AsyncClient(
                timeout=120.0,
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": API_VERSION,
                    "anthropic-beta": "prompt-caching-2024-07-31",
                    "content-type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._ollama is not None:
            await self._ollama.close()

    def _get_ollama(self) -> AsyncOllamaClient:
        if self._ollama is None:
            self._ollama = AsyncOllamaClient(
                model=self.router.local_model,
                base_url=self.router.ollama_base_url,
            )
        return self._ollama

    async def _call_with_role(
        self,
        role: ModelRole | str,
        system: str,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        complexity: int,
    ) -> tuple[dict, str, str]:
        """Dispatch to Ollama or Anthropic based on the router's mode.

        Returns (response_dict, model_name_used, backend_used).
        Falls back to cloud with a logged warning if the local backend
        is chosen but isn't available.
        """
        role_str = role.value if isinstance(role, ModelRole) else role
        backend = self.router.backend_for(role)
        cloud_model = self.router.get_model(role, complexity)

        logger.info(
            "Dispatch: role=%s backend=%s mode=%s",
            role_str,
            backend.value,
            self.router.mode.value,
        )

        if backend is Backend.LOCAL:
            ollama = self._get_ollama()
            if not await ollama.is_available():
                logger.warning(
                    "Ollama not available for role=%s; falling back to cloud",
                    role_str,
                )
                self.router.record_fallback()
                LOCAL_TRACKER.record_fallback()
                backend = Backend.CLOUD  # fall through to cloud below
            else:
                user_prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                        break
                try:
                    data = await ollama.generate(
                        system=system,
                        user=user_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        # Session 1: thread the role through so the
                        # per-role budget (ROLE_BUDGETS) fires instead
                        # of the single 300s cap that crashed the
                        # architect overnight.
                        role=role_str,
                    )
                except OllamaError as e:
                    # Classified Ollama failure — log, bump the
                    # fallback counter, and let the cloud path take
                    # over.  Permanent errors (context exceeded) are
                    # logged at WARNING because the cloud call is
                    # unlikely to succeed either, but we still try.
                    log_level = (
                        logging.ERROR if isinstance(e, OllamaPermanentError) else logging.WARNING
                    )
                    logger.log(
                        log_level,
                        "Ollama %s on role=%s: %s; falling back to cloud",
                        type(e).__name__,
                        role_str,
                        e,
                    )
                    self.router.record_fallback()
                    LOCAL_TRACKER.record_fallback()
                    backend = Backend.CLOUD
                else:
                    usage = data.get("usage", {})
                    LOCAL_TRACKER.record_call(
                        model=self.router.local_model,
                        prompt_tokens=int(usage.get("input_tokens", 0)),
                        completion_tokens=int(usage.get("output_tokens", 0)),
                        role=role_str,
                    )
                    return data, self.router.local_model, "local"

        # Cloud path (default or fallback)
        data = await self._call(
            model=cloud_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return data, cloud_model, "cloud"

    def _record_usage(
        self,
        role: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_create_tokens: int = 0,
    ) -> None:
        cost = _cost_usd(
            model, prompt_tokens, completion_tokens, cache_read_tokens, cache_create_tokens
        )
        usage = _usage_ctx.get()
        if usage is not None:
            usage.add_call(role, prompt_tokens, completion_tokens, cost)
        tier = "haiku" if "haiku" in model else "sonnet" if "sonnet" in model else "opus"
        cache_info = ""
        if cache_read_tokens > 0:
            cache_info = f" cache_hit={cache_read_tokens}"
        logger.debug(
            f"LLM call: role={role} tier={tier} "
            f"tokens={prompt_tokens}+{completion_tokens}{cache_info} cost=${cost:.4f}"
        )

    async def _call(
        self,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict:
        """Raw API call. Returns the full response dict.

        Uses prompt caching on system prompts for 90% input cost reduction.
        Cache-eligible system prompts must be >1024 tokens (~4000 chars).
        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        # Prompt caching: wrap system prompt in cache_control block
        # Only cache if system prompt is long enough (>1024 tokens ≈ 4000 chars)
        if system:
            if len(system) > 2000:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system

        try:
            resp = await client.post(API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Log cache performance
            usage = data.get("usage", {})
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            if cache_read > 0:
                logger.debug(f"Prompt cache HIT: {cache_read} tokens read from cache")
            elif cache_create > 0:
                logger.debug(f"Prompt cache MISS: {cache_create} tokens written to cache")

            return data
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot reach Anthropic API: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
            raise

    async def generate_text(
        self,
        role: ModelRole | str,
        system: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        complexity: int = 1,
    ) -> str:
        """Generate free-form text. Returns the response string."""
        role_str = role.value if isinstance(role, ModelRole) else role

        data, model, backend = await self._call_with_role(
            role,
            system,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            complexity=complexity,
        )

        text = data["content"][0]["text"]

        # Local-backend responses are already booked against LOCAL_TRACKER
        # inside _call_with_role; only cloud responses go through
        # the Anthropic-cost accounting path.
        if backend == "cloud":
            usage = data.get("usage", {})
            self._record_usage(
                role_str,
                model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
            )

        return text

    async def generate_structured(
        self,
        role: ModelRole | str,
        system: str,
        prompt: str,
        response_schema: Type[T],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        complexity: int = 1,
    ) -> T:
        """Generate a structured response parsed into a Pydantic model.

        The system prompt is augmented with JSON schema instructions.
        The response is parsed from the first JSON object found.
        """
        role_str = role.value if isinstance(role, ModelRole) else role

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system}\n\n"
            f"Respond ONLY with a valid JSON object matching this schema. "
            f"No markdown fences. No explanation. Just the JSON.\n\n"
            f"Schema:\n{schema_json}"
        )

        data, model, backend = await self._call_with_role(
            role,
            augmented_system,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            complexity=complexity,
        )

        raw_text = data["content"][0]["text"]

        if backend == "cloud":
            usage = data.get("usage", {})
            self._record_usage(
                role_str,
                model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
            )

        return _parse_structured(raw_text, response_schema)


def _parse_structured(raw: str, schema: Type[T]) -> T:
    """Extract and parse JSON from LLM response text.

    Handles several failure modes common to local models (qwen / llama)
    and occasionally Claude under pressure:
      1. Markdown code fences (```json ... ```)
      2. Truncated output (close unclosed brackets/braces)
      3. Trailing commas ({"a": 1,})
      4. Single-quoted strings ({'a': 'b'})
      5. Unquoted keys ({a: 1})
      6. As a last resort, regex-extract the top-level fields that
         match the schema.

    The lenient transforms are only applied on ``json.loads`` failure,
    so valid JSON from Claude stays untouched.
    """
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Find the JSON object
    brace_start = text.find("{")
    if brace_start == -1:
        # No JSON braces at all — still try regex salvage before giving up.
        # Local models occasionally respond with narrative text that
        # happens to contain ``"field": value`` fragments we can recover.
        try:
            salvaged = _regex_extract_fields(text, schema)
            if salvaged:
                logger.info(
                    f"No JSON object found; salvaged {len(salvaged)} fields "
                    f"via regex fallback for {schema.__name__}"
                )
                return schema.model_validate(salvaged)
        except Exception as e:
            logger.debug(f"Brace-less regex salvage failed: {e}")
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    # Try 1: Find matching closing brace (clean JSON)
    depth = 0
    json_str = None
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = text[brace_start : i + 1]
                break

    if json_str:
        try:
            data = json.loads(json_str)
            return schema.model_validate(data)
        except Exception as e:
            logger.debug(f"Initial JSON parse failed, attempting repair: {e}")

    # Try 2: Repair truncated JSON (close unclosed brackets)
    truncated = text[brace_start:]
    repaired = _repair_json(truncated)
    if repaired:
        try:
            data = json.loads(repaired)
            logger.info(f"Parsed repaired JSON ({len(repaired)} chars)")
            return schema.model_validate(data)
        except Exception as e:
            logger.debug(f"Repaired parse still failed, trying lenient clean: {e}")

    # Try 3: Lenient cleanup for local-model-flavoured invalid JSON
    # (trailing commas, single quotes, unquoted keys).
    candidate = repaired if repaired else (json_str or truncated)
    if candidate:
        cleaned = _lenient_json_cleanup(candidate)
        if cleaned and cleaned != candidate:
            try:
                data = json.loads(cleaned)
                logger.info(
                    "Parsed after lenient cleanup (trailing-comma / single-quote / unquoted-key fix)"
                )
                return schema.model_validate(data)
            except Exception as e:
                logger.debug(f"Lenient cleanup failed: {e}")

    # Try 4: Regex fallback — pick top-level string / number fields that
    # the schema expects.  Only handles scalar fields; anything with a
    # nested object or list still fails here.
    try:
        salvaged = _regex_extract_fields(text, schema)
        if salvaged:
            logger.info(
                f"Salvaged {len(salvaged)} field(s) via regex fallback for {schema.__name__}"
            )
            return schema.model_validate(salvaged)
    except Exception as e:
        logger.debug(f"Regex salvage failed: {e}")

    raise ValueError(f"Unclosed JSON object in response: {text[:200]}")


# ── Lenient JSON cleanup for local-model output ────────────────────────────

# These transformations are applied *only* when standard json.loads has
# already failed, so valid JSON is never touched.  They fix the common
# JSON mishaps we see from qwen2.5-coder:14b and friends.


def _lenient_json_cleanup(text: str) -> str:
    """Return ``text`` with common local-model JSON bugs cleaned up.

    Applies, in order:
      - single-quoted string literals → double-quoted
      - unquoted keys → quoted keys
      - trailing commas before ``}`` or ``]`` → removed

    The walk is string-aware so apostrophes inside correctly-quoted
    JSON strings aren't clobbered.  On any processing error we return
    the input unchanged rather than poison the next parse attempt.
    """
    try:
        out = _convert_single_to_double_quotes(text)
        out = _quote_unquoted_keys(out)
        out = _strip_trailing_commas(out)
        return out
    except Exception:
        return text


def _convert_single_to_double_quotes(text: str) -> str:
    """Convert single-quoted string literals to double-quoted.

    Rules:
      - Only strings adjacent to ``:`` or ``[`` / ``,`` / ``{`` contexts
        (i.e., JSON string positions) get converted.
      - Existing double-quoted strings are preserved; apostrophes inside
        them are not touched.
    """
    out: list[str] = []
    i = 0
    in_dq = False  # inside a double-quoted string
    escape = False
    while i < len(text):
        c = text[i]
        if in_dq:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_dq = False
            i += 1
            continue

        if c == '"':
            in_dq = True
            out.append(c)
            i += 1
            continue

        if c == "'":
            # Find the matching closing ' (allow \' escapes)
            j = i + 1
            esc = False
            while j < len(text):
                cj = text[j]
                if esc:
                    esc = False
                elif cj == "\\":
                    esc = True
                elif cj == "'":
                    break
                j += 1
            if j < len(text):
                inner = text[i + 1 : j]
                # Escape any embedded double-quotes
                inner = inner.replace("\\'", "'").replace('"', '\\"')
                out.append('"')
                out.append(inner)
                out.append('"')
                i = j + 1
                continue
            # No closing '; bail — leave as-is to avoid corrupting
            out.append(c)
            i += 1
            continue

        out.append(c)
        i += 1

    return "".join(out)


# Regex for unquoted keys — matches `{ foo: ` or `, bar: ` and adds quotes
_UNQUOTED_KEY_RE = re.compile(r"(?P<sep>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:")


def _quote_unquoted_keys(text: str) -> str:
    """Quote unquoted object keys.  ``{foo: 1}`` → ``{"foo": 1}``.

    Walks the string to skip quoted regions; applies the regex only
    to the outside-string parts so apostrophes / colons inside strings
    aren't mistaken for keys.
    """

    def _process_chunk(chunk: str) -> str:
        return _UNQUOTED_KEY_RE.sub(r'\g<sep>"\g<key>":', chunk)

    # Split into string vs outside-string pieces
    pieces: list[str] = []
    i = 0
    in_str = False
    escape = False
    buf_start = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
                # String just closed; flush from buf_start..i+1 verbatim
                pieces.append(text[buf_start : i + 1])
                buf_start = i + 1
            i += 1
            continue
        if c == '"':
            # Flush outside-string region up to here (processed)
            if i > buf_start:
                pieces.append(_process_chunk(text[buf_start:i]))
            in_str = True
            buf_start = i
            i += 1
            continue
        i += 1

    # Flush tail
    if buf_start < len(text):
        if in_str:
            pieces.append(text[buf_start:])
        else:
            pieces.append(_process_chunk(text[buf_start:]))

    return "".join(pieces)


# Regex for trailing commas before } or ]
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` / ``]``, avoiding string interiors."""
    # Same string-aware walk as _quote_unquoted_keys
    pieces: list[str] = []
    i = 0
    in_str = False
    escape = False
    buf_start = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
                pieces.append(text[buf_start : i + 1])
                buf_start = i + 1
            i += 1
            continue
        if c == '"':
            if i > buf_start:
                pieces.append(_TRAILING_COMMA_RE.sub(r"\1", text[buf_start:i]))
            in_str = True
            buf_start = i
            i += 1
            continue
        i += 1
    if buf_start < len(text):
        if in_str:
            pieces.append(text[buf_start:])
        else:
            pieces.append(_TRAILING_COMMA_RE.sub(r"\1", text[buf_start:]))
    return "".join(pieces)


# ── Regex salvage for the last-resort path ────────────────────────────────


def _regex_extract_fields(text: str, schema: Type[BaseModel]) -> dict | None:
    """Pull top-level scalar fields out of a malformed response.

    Walks the schema's declared field names and, for each one, looks
    for a ``"field": <value>`` pattern in the raw text where ``<value>``
    is a string, number, or boolean.  Fields that aren't found and
    aren't required are omitted (Pydantic will fill defaults).

    Returns ``None`` if no required fields could be salvaged — this
    signals to the caller to raise the original parse error instead
    of hiding it behind a half-built model.
    """
    try:
        fields = schema.model_fields
    except AttributeError:
        return None

    salvaged: dict[str, Any] = {}
    for name, field_info in fields.items():
        # Handle "name": "..." or "name": 123 or "name": true
        # Optional trailing comma/newline. DOTALL for multiline strings.
        patterns = [
            # String value (double-quoted)
            rf'"{re.escape(name)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            # Numeric value
            rf'"{re.escape(name)}"\s*:\s*(-?\d+(?:\.\d+)?)',
            # Boolean
            rf'"{re.escape(name)}"\s*:\s*(true|false)',
            # null
            rf'"{re.escape(name)}"\s*:\s*(null)',
        ]
        for idx, p in enumerate(patterns):
            m = re.search(p, text, re.DOTALL)
            if m:
                raw_val = m.group(1)
                if idx == 0:
                    val: Any = raw_val.encode("utf-8").decode("unicode_escape")
                elif idx == 1:
                    val = float(raw_val) if "." in raw_val else int(raw_val)
                elif idx == 2:
                    val = raw_val == "true"
                else:
                    val = None
                salvaged[name] = val
                break

    # Were any required fields missed? If so, don't return a partial.
    missing_required = [n for n, f in fields.items() if n not in salvaged and f.is_required()]
    if missing_required:
        logger.debug(f"regex salvage missing required fields: {missing_required}")
        return None

    return salvaged if salvaged else None


def _repair_json(text: str) -> str | None:
    """Repair truncated JSON by closing unclosed brackets and braces.

    Strategy: walk the JSON tracking the stack of open delimiters.
    When we find the text is truncated, roll back to the last position
    where the JSON was structurally valid, then close the remaining stack.
    """
    s = text.rstrip()
    if not s:
        return None

    # Walk the string tracking state
    stack: list[str] = []  # Stack of open delimiters: '{' or '['
    in_string = False
    escape = False
    last_complete = 0  # Last position where we completed a value

    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == "\\":
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        # Outside strings
        if c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c in ("}", "]"):
            if stack and stack[-1] == c:
                stack.pop()
            last_complete = i + 1
        elif c == ",":
            last_complete = i
        elif c == ":":
            pass  # After a key, before a value

    # If we ended inside a string, roll back to last complete position
    if in_string and last_complete > 0:
        s = s[:last_complete]
        # Recalculate stack from clean text
        stack = []
        in_str = False
        esc = False
        for c in s:
            if esc:
                esc = False
                continue
            if c == "\\":
                if in_str:
                    esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                stack.append("}")
            elif c == "[":
                stack.append("]")
            elif c in ("}", "]"):
                if stack and stack[-1] == c:
                    stack.pop()

    # Remove trailing comma
    s = s.rstrip()
    if s and s[-1] == ",":
        s = s[:-1]

    # Close the stack in reverse order (innermost first)
    s += "".join(reversed(stack))

    return s if s else None
