"""Hermetic tests for Session 1 (v3.2) Ollama client hardening.

Every test uses a monkeypatched httpx.AsyncClient — no real Ollama,
no real Anthropic, no real network.  Coverage targets:

  1. Inactivity watchdog fires on per-chunk stall (30s default,
     patched to a tiny value for the test).
  2. Context-length-exceeded (4xx + "context length" in body) raises
     OllamaContextExceeded AND does NOT retry (permanent error).
  3. Transient error (5xx or httpx.ReadError) retries exactly 3 times,
     then raises.
  4. Per-model circuit breaker opens after 5 failures and short-
     circuits the 6th call with OllamaTransientError.
  5. Per-role budget is enforced: patching ROLE_BUDGETS for "executor"
     to 0.1s surfaces a budget timeout as OllamaTransientError.

Run with:  pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from belief import llm as llm_module
from belief.llm import (
    ROLE_BUDGETS,
    AsyncOllamaClient,
    LLMClient,
    _MODEL_BREAKERS,
    _classify_ollama_error,
)
from belief.llm_errors import (
    OllamaContextExceeded,
    OllamaPermanentError,
    OllamaStreamStall,
    OllamaTransientError,
)


# ---------------------------------------------------------------------------
# Shared fake httpx transport — supports stream(), post(), get(), aclose()
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Stand-in for httpx.Response.  Supports both the streaming and
    non-streaming code paths in AsyncOllamaClient.  Line iteration and
    body bytes are pre-populated by the constructor.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        lines: list[str] | None = None,
        text: str = "",
        stall_forever: bool = False,
    ) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self.text = text
        self._stall_forever = stall_forever

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://localhost:11434/api/chat"),
                response=self,  # type: ignore[arg-type]
            )

    def json(self) -> Any:
        return json.loads(self.text) if self.text else {}

    def aiter_lines(self) -> Any:
        stall = self._stall_forever
        lines = list(self._lines)

        class _Iter:
            def __aiter__(self_inner) -> "_Iter":
                return self_inner

            async def __anext__(self_inner) -> str:
                if stall:
                    # Park forever — test patches inactivity_s small so
                    # asyncio.wait_for fires on our __anext__ call.
                    await asyncio.Future()
                if not lines:
                    raise StopAsyncIteration
                return lines.pop(0)

        return _Iter()

    async def aiter_bytes(self) -> Any:
        if self.text:
            yield self.text.encode()


class _FakeStreamCtx:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _FakeClient:
    """Replacement for httpx.AsyncClient.  All three methods the
    AsyncOllamaClient touches are supported: stream(), post(), get().
    """

    def __init__(
        self,
        *,
        stream_factory: Any = None,
        post_handler: Any = None,
        get_handler: Any = None,
    ) -> None:
        self._stream_factory = stream_factory
        self._post_handler = post_handler
        self._get_handler = get_handler
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self.stream_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.get_calls: list[str] = []

    @property
    def is_closed(self) -> bool:
        return False

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamCtx:
        self.stream_calls.append((method, url, kwargs))
        assert self._stream_factory is not None, "stream() called but no factory set"
        return _FakeStreamCtx(self._stream_factory())

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.post_calls.append((url, kwargs))
        if self._post_handler is not None:
            return self._post_handler(url, kwargs)
        return _FakeResponse(
            status_code=200, text='{"message":{"role":"assistant","content":"ok"}}'
        )

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.get_calls.append(url)
        if self._get_handler is not None:
            return self._get_handler(url, kwargs)
        return _FakeResponse(status_code=200, text="{}")

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_breakers() -> Any:
    """Clear the per-model breaker registry between tests so a broken
    breaker from one test doesn't leak into the next.  The session-1
    design keys breakers by model name in a module-level dict, so
    cross-test isolation requires clearing that dict.
    """
    yield
    _MODEL_BREAKERS.clear()


# ---------------------------------------------------------------------------
# 4xx classifier — pure function, no network
# ---------------------------------------------------------------------------


class TestClassifyOllamaError:
    def test_context_length_phrase_raises_permanent(self) -> None:
        exc = _classify_ollama_error(400, "prompt exceeds context length of 8192 tokens")
        assert isinstance(exc, OllamaContextExceeded)
        assert isinstance(exc, OllamaPermanentError)

    def test_context_window_phrase_raises_permanent(self) -> None:
        exc = _classify_ollama_error(400, "input exceeds the model's context window")
        assert isinstance(exc, OllamaContextExceeded)

    def test_generic_4xx_raises_permanent(self) -> None:
        exc = _classify_ollama_error(404, "model not found")
        assert isinstance(exc, OllamaPermanentError)
        assert not isinstance(exc, OllamaContextExceeded)

    def test_5xx_raises_transient(self) -> None:
        exc = _classify_ollama_error(503, "service unavailable")
        assert isinstance(exc, OllamaTransientError)


# ---------------------------------------------------------------------------
# 1. Inactivity watchdog
# ---------------------------------------------------------------------------


class TestInactivityWatchdog:
    @pytest.mark.asyncio
    async def test_watchdog_fires_when_stream_stalls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stream that emits no chunks for inactivity_s seconds must
        raise OllamaStreamStall (a transient subtype).  We patch
        inactivity_s to 0.1s and have the fake stream stall forever.
        """

        # Fake stream never yields a line.
        def stream_factory() -> _FakeResponse:
            return _FakeResponse(status_code=200, stall_forever=True)

        fake_client = _FakeClient(
            stream_factory=stream_factory,
            # For the keep_alive=0 unload POST that fires after a stall.
            post_handler=lambda url, kwargs: _FakeResponse(status_code=200, text="{}"),
        )

        monkeypatch.setattr(llm_module.httpx, "AsyncClient", lambda *a, **k: fake_client)

        ollama = AsyncOllamaClient(
            model="qwen2.5-coder:14b",
            base_url="http://localhost:11434",
            inactivity_s=0.1,
        )
        # Force cached client rebuild via the patched factory.
        ollama._client = None

        # Use a generous budget so the watchdog fires before the budget does,
        # and patch ROLE_BUDGETS so "default" gives us enough headroom.
        # tenacity will retry 3x, each attempt stalling 0.1s → ~0.3-5s total
        # with exponential backoff; 30s budget is plenty.
        monkeypatch.setitem(ROLE_BUDGETS, "default", 30.0)

        with pytest.raises(OllamaStreamStall):
            await ollama.generate(
                system="sys",
                user="hi",
                max_tokens=50,
                temperature=0.0,
                role="default",
            )

        # Watchdog fired for EVERY retry attempt — verify the unload POST
        # to /api/generate happened at least once.
        unload_posts = [c for c in fake_client.post_calls if c[0] == "/api/generate"]
        assert len(unload_posts) >= 1, (
            f"Expected keep_alive=0 unload POST on stall, got: {fake_client.post_calls}"
        )
        # And the unload payload must set keep_alive=0 for the exact model.
        assert unload_posts[0][1]["json"]["keep_alive"] == 0
        assert unload_posts[0][1]["json"]["model"] == "qwen2.5-coder:14b"

        await ollama.close()


# ---------------------------------------------------------------------------
# 2. Context-length-exceeded does NOT retry
# ---------------------------------------------------------------------------


class TestContextExceededNoRetry:
    @pytest.mark.asyncio
    async def test_context_exceeded_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 4xx with "context length" in the body must raise
        OllamaContextExceeded on the FIRST attempt and not retry — the
        overnight failure mode this whole session was written to fix.
        """
        call_count = {"n": 0}

        def stream_factory() -> _FakeResponse:
            call_count["n"] += 1
            return _FakeResponse(
                status_code=400,
                text='{"error":"prompt exceeds context length of 8192"}',
            )

        fake_client = _FakeClient(stream_factory=stream_factory)
        monkeypatch.setattr(llm_module.httpx, "AsyncClient", lambda *a, **k: fake_client)

        ollama = AsyncOllamaClient(
            model="ctx-test-model",
            base_url="http://localhost:11434",
            inactivity_s=1.0,
        )
        ollama._client = None

        with pytest.raises(OllamaContextExceeded):
            await ollama.generate(
                system="sys",
                user="huge",
                max_tokens=50,
                temperature=0.0,
                role="default",
            )

        assert call_count["n"] == 1, (
            f"Permanent error retried {call_count['n']} times — it must not retry."
        )

        await ollama.close()


# ---------------------------------------------------------------------------
# 3. Transient error retries exactly 3x then raises
# ---------------------------------------------------------------------------


class TestTransientRetry:
    @pytest.mark.asyncio
    async def test_transient_retries_three_times_then_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 5xx should retry 3 total times (stop_after_attempt(3)) and
        then propagate OllamaTransientError.
        """
        call_count = {"n": 0}

        def stream_factory() -> _FakeResponse:
            call_count["n"] += 1
            return _FakeResponse(status_code=503, text="service unavailable")

        fake_client = _FakeClient(stream_factory=stream_factory)
        monkeypatch.setattr(llm_module.httpx, "AsyncClient", lambda *a, **k: fake_client)

        ollama = AsyncOllamaClient(
            model="transient-test-model",
            base_url="http://localhost:11434",
            inactivity_s=1.0,
        )
        ollama._client = None

        # Keep budgets generous so the retry loop finishes naturally and
        # the failure we see is the transient error, not a budget timeout.
        monkeypatch.setitem(ROLE_BUDGETS, "default", 60.0)
        # Shrink tenacity wait so this test isn't slow — wait_exponential_jitter
        # starts at 1s, which means 3 attempts ≈ 1+2 = 3s; we still finish in time.

        with pytest.raises(OllamaTransientError):
            await ollama.generate(
                system="sys",
                user="hi",
                max_tokens=50,
                temperature=0.0,
                role="default",
            )

        assert call_count["n"] == 3, f"Expected exactly 3 attempts, got {call_count['n']}"

        await ollama.close()


# ---------------------------------------------------------------------------
# 4. Circuit breaker opens after 5 failures
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_breaker_opens_after_five_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drive five transient-error calls through the same model;
        the next call must short-circuit with breaker-open instead of
        hitting the wire again.
        """
        call_count = {"n": 0}

        def stream_factory() -> _FakeResponse:
            call_count["n"] += 1
            return _FakeResponse(status_code=503, text="transient")

        fake_client = _FakeClient(stream_factory=stream_factory)
        monkeypatch.setattr(llm_module.httpx, "AsyncClient", lambda *a, **k: fake_client)

        ollama = AsyncOllamaClient(
            model="breaker-test-model",
            base_url="http://localhost:11434",
            inactivity_s=1.0,
        )
        ollama._client = None
        monkeypatch.setitem(ROLE_BUDGETS, "default", 60.0)

        # Drive transient failures until the breaker trips.  Each call
        # fails 3 times (tenacity) → the breaker sees 3 failures per
        # call.  Two calls = 6 failures, more than fail_max=5, so the
        # breaker opens.
        for _ in range(2):
            with pytest.raises(OllamaTransientError):
                await ollama.generate(
                    system="s",
                    user="u",
                    max_tokens=10,
                    temperature=0.0,
                    role="default",
                )

        calls_before_open = call_count["n"]

        # Next call must short-circuit — the breaker is open.
        with pytest.raises(OllamaTransientError, match="circuit breaker"):
            await ollama.generate(
                system="s",
                user="u",
                max_tokens=10,
                temperature=0.0,
                role="default",
            )

        # No new wire calls on the short-circuit path.
        assert call_count["n"] == calls_before_open, (
            "Breaker should have short-circuited — no new wire calls expected"
        )

        await ollama.close()


# ---------------------------------------------------------------------------
# 5. Per-role budget enforced
# ---------------------------------------------------------------------------


class TestRoleBudget:
    @pytest.mark.asyncio
    async def test_executor_short_budget_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch ROLE_BUDGETS so 'executor' gets 0.2s, and make every
        stream stall until the budget fires.  We expect
        OllamaTransientError with a "budget" message, NOT an
        OllamaStreamStall — because asyncio.wait_for (the budget) wraps
        the retry loop and fires first.
        """

        def stream_factory() -> _FakeResponse:
            return _FakeResponse(status_code=200, stall_forever=True)

        fake_client = _FakeClient(
            stream_factory=stream_factory,
            post_handler=lambda url, kwargs: _FakeResponse(status_code=200, text="{}"),
        )
        monkeypatch.setattr(llm_module.httpx, "AsyncClient", lambda *a, **k: fake_client)

        ollama = AsyncOllamaClient(
            model="budget-test-model",
            base_url="http://localhost:11434",
            inactivity_s=60.0,  # watchdog big, budget small → budget fires first
        )
        ollama._client = None

        monkeypatch.setitem(ROLE_BUDGETS, "executor", 0.2)

        with pytest.raises(OllamaTransientError, match="budget"):
            await ollama.generate(
                system="s",
                user="u",
                max_tokens=10,
                temperature=0.0,
                role="executor",
            )

        await ollama.close()

    def test_role_budgets_dictionary_matches_session_doc(self) -> None:
        """Constant table sanity check — session-1 doc specifies these
        exact numbers.  If someone quietly changes them, this test
        flags it.
        """
        assert ROLE_BUDGETS["intake"] == 60.0
        assert ROLE_BUDGETS["planner"] == 120.0
        assert ROLE_BUDGETS["architect"] == 600.0
        assert ROLE_BUDGETS["builder"] == 300.0
        assert ROLE_BUDGETS["debugger"] == 180.0
        assert ROLE_BUDGETS["synthesizer"] == 180.0
        assert ROLE_BUDGETS["executor"] == 60.0
        assert ROLE_BUDGETS["default"] == 180.0


# ---------------------------------------------------------------------------
# 5b. Cloud path honors the per-role read budget
# ---------------------------------------------------------------------------


class TestCloudRoleBudgetTimeout:
    """Regression for the ReadTimeout that crashed the architect on large
    cloud goals: the cloud `_call` post must use a per-request httpx timeout
    whose `read` equals ROLE_BUDGETS[role], not the old flat 120s.
    """

    @pytest.mark.asyncio
    async def test_architect_gets_600s_read_timeout(self) -> None:
        fake = _FakeClient(
            post_handler=lambda url, kwargs: _FakeResponse(
                status_code=200, text='{"content":[],"usage":{}}'
            )
        )
        llm = LLMClient(router=None)  # router is unused by _call
        llm._client = fake  # inject fake; _get_client returns it (is_closed=False)

        await llm._call(
            model="claude-sonnet-4-6",
            system="x" * 3000,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.0,
            role="architect",
        )

        assert fake.post_calls, "cloud _call never posted"
        url, kwargs = fake.post_calls[0]
        assert url == "https://api.anthropic.com/v1/messages"
        timeout = kwargs.get("timeout")
        assert timeout is not None, "per-request timeout was not passed"
        assert timeout.read == ROLE_BUDGETS["architect"] == 600.0

    @pytest.mark.asyncio
    async def test_unknown_role_falls_back_to_default_budget(self) -> None:
        fake = _FakeClient(
            post_handler=lambda url, kwargs: _FakeResponse(
                status_code=200, text='{"content":[],"usage":{}}'
            )
        )
        llm = LLMClient(router=None)
        llm._client = fake

        await llm._call(
            model="claude-haiku-4-5",
            system="short",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            temperature=0.0,
            role=None,
        )

        _, kwargs = fake.post_calls[0]
        assert kwargs["timeout"].read == ROLE_BUDGETS["default"] == 180.0


# ---------------------------------------------------------------------------
# 5c. generate_long_text — bounded continuation on max_tokens truncation
# ---------------------------------------------------------------------------


class TestGenerateLongText:
    """The builder uses this so a large file is not silently chopped at the
    output-token ceiling. It must auto-continue via assistant-prefill when the
    model stops on ``max_tokens``, and report ``truncated=True`` only if still
    cut after the continuation cap.
    """

    @pytest.mark.asyncio
    async def test_single_call_when_model_finishes(self) -> None:
        llm = LLMClient(router=None)  # router unused once _call_with_role is faked
        calls: list[list[dict]] = []

        async def fake(role, system, messages, *, max_tokens, temperature, complexity):
            calls.append(messages)
            return ({"content": [{"text": "done"}], "stop_reason": "end_turn"}, "m", "local")

        llm._call_with_role = fake  # type: ignore[assignment]
        text, truncated = await llm.generate_long_text("builder", "sys", "p")

        assert text == "done"
        assert truncated is False
        assert len(calls) == 1  # no continuation needed

    @pytest.mark.asyncio
    async def test_continues_then_finishes(self) -> None:
        llm = LLMClient(router=None)
        scripted = [
            ({"content": [{"text": "part1 "}], "stop_reason": "max_tokens"}, "m", "local"),
            ({"content": [{"text": "part2"}], "stop_reason": "end_turn"}, "m", "local"),
        ]
        seen: list[list[dict]] = []

        async def fake(role, system, messages, *, max_tokens, temperature, complexity):
            seen.append(messages)
            return scripted.pop(0)

        llm._call_with_role = fake  # type: ignore[assignment]
        text, truncated = await llm.generate_long_text("builder", "sys", "p")

        # Seed is rstripped before being fed back, so the seam has no double space.
        assert text == "part1part2"
        assert truncated is False
        assert len(seen) == 2
        # Continuation feeds the partial back as assistant history, then a user
        # turn — the conversation MUST end with a user message (assistant
        # prefill 400s on some models; that bug shipped firmware-less builds).
        assert seen[1][0] == {"role": "user", "content": "p"}
        assert seen[1][1] == {"role": "assistant", "content": "part1"}
        assert seen[1][-1]["role"] == "user"
        assert len(seen[1]) == 3

    @pytest.mark.asyncio
    async def test_truncated_flag_set_when_cap_exhausted(self) -> None:
        llm = LLMClient(router=None)
        n = {"calls": 0}

        async def fake(role, system, messages, *, max_tokens, temperature, complexity):
            n["calls"] += 1
            return ({"content": [{"text": "x"}], "stop_reason": "max_tokens"}, "m", "local")

        llm._call_with_role = fake  # type: ignore[assignment]
        text, truncated = await llm.generate_long_text("builder", "sys", "p", max_continuations=2)

        assert truncated is True
        assert n["calls"] == 3  # initial + 2 continuations
        assert text == "xxx"


# ---------------------------------------------------------------------------
# 6. Health check
# ---------------------------------------------------------------------------


class TestHealthOk:
    @pytest.mark.asyncio
    async def test_health_ok_returns_false_on_unroutable_host(self) -> None:
        """Our health_ok() must fail fast (<5s) when Ollama is not
        reachable, not hang on the legacy 300s timeout.
        """
        from belief.llm import health_ok

        ok = await health_ok(base_url="http://127.0.0.1:1", timeout_s=2.0)
        assert ok is False


# ---------------------------------------------------------------------------
# 7. Thermal gate — smoke test (ThermalPressure enum + gate function)
# ---------------------------------------------------------------------------


class TestThermalGate:
    def test_thermal_gate_unknown_is_noop_off_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Linux (CI / sandbox), read_thermal_pressure returns
        UNKNOWN and thermal_gate sleeps 0 seconds.
        """
        from belief.thermal import ThermalPressure, thermal_gate

        # Force the branch: make platform.system return something non-Darwin.
        import platform

        monkeypatch.setattr(platform, "system", lambda: "Linux")

        # Should return quickly without sleeping.
        import time as _time

        t0 = _time.monotonic()
        result = thermal_gate()
        elapsed = _time.monotonic() - t0
        assert result == ThermalPressure.UNKNOWN
        assert elapsed < 1.0, f"off-macOS thermal_gate slept {elapsed:.1f}s (should be ~0)"

    @pytest.mark.asyncio
    async def test_async_thermal_gate_unknown_off_macos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from belief.thermal import ThermalPressure, async_thermal_gate

        import platform

        monkeypatch.setattr(platform, "system", lambda: "Linux")

        import time as _time

        t0 = _time.monotonic()
        result = await async_thermal_gate()
        elapsed = _time.monotonic() - t0
        assert result == ThermalPressure.UNKNOWN
        assert elapsed < 1.0
