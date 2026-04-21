"""Session 6: Ollama backend + hybrid model routing.

Keeps tests hermetic — no real HTTP to Ollama, no real Anthropic calls.
Every network interaction goes through a respx / monkeypatched fake.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest

from belief.config.local_cost_tracker import LocalCostTracker
from belief.config.models import (
    Backend,
    DEFAULT_LOCAL_MODEL,
    HYBRID_ROUTING,
    ModelRole,
    ModelRouter,
    RouteMode,
)


# ---------------------------------------------------------------------------
# LocalCostTracker
# ---------------------------------------------------------------------------


class TestLocalCostTracker:
    def test_record_call_returns_zero_cost(self) -> None:
        t = LocalCostTracker()
        cost = t.record_call("qwen2.5-coder:14b", 500, 100, role="intake")
        assert cost == 0.0
        assert t.total_calls() == 1
        assert t.total_tokens() == 600

    def test_fallback_counter(self) -> None:
        t = LocalCostTracker()
        t.record_fallback()
        t.record_fallback()
        assert t.fallback_count == 2

    def test_by_model_aggregation(self) -> None:
        t = LocalCostTracker()
        t.record_call("qwen2.5-coder:14b", 100, 50, role="intake")
        t.record_call("qwen2.5-coder:14b", 200, 75, role="tester")
        t.record_call("llama3:8b", 300, 120, role="intake")
        by_model = t.by_model()
        assert by_model["qwen2.5-coder:14b"]["calls"] == 2
        assert by_model["qwen2.5-coder:14b"]["prompt_tokens"] == 300
        assert by_model["qwen2.5-coder:14b"]["completion_tokens"] == 125
        assert by_model["llama3:8b"]["calls"] == 1

    def test_by_role_aggregation(self) -> None:
        t = LocalCostTracker()
        t.record_call("qwen2.5-coder:14b", 100, 50, role="intake")
        t.record_call("qwen2.5-coder:14b", 200, 75, role="intake")
        t.record_call("qwen2.5-coder:14b", 300, 120, role="tester")
        by_role = t.by_role()
        assert by_role["intake"]["calls"] == 2
        assert by_role["tester"]["calls"] == 1

    def test_threaded_records_no_lost_updates(self) -> None:
        import threading

        t = LocalCostTracker()

        def worker(n: int) -> None:
            for i in range(n):
                t.record_call("qwen2.5-coder:14b", 10, 5, role=f"t{i%3}")

        threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert t.total_calls() == 200
        assert t.total_tokens() == 200 * 15


# ---------------------------------------------------------------------------
# ModelRouter mode + backend routing
# ---------------------------------------------------------------------------


class TestModelRouterMode:
    def test_default_mode_is_cloud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)
        r = ModelRouter()
        assert r.mode is RouteMode.CLOUD

    def test_env_var_sets_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BELIEF_MODEL_MODE", "hybrid")
        r = ModelRouter()
        assert r.mode is RouteMode.HYBRID

    def test_env_var_invalid_falls_back_to_cloud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BELIEF_MODEL_MODE", "quantum")
        r = ModelRouter()
        assert r.mode is RouteMode.CLOUD

    def test_local_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BELIEF_LOCAL_MODEL", "llama3:8b")
        r = ModelRouter()
        assert r.local_model == "llama3:8b"

    def test_set_mode_resets_fallback_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)
        r = ModelRouter()
        r.record_fallback()
        r.record_fallback()
        assert r.fallback_count == 2
        r.set_mode("hybrid")
        assert r.fallback_count == 0


class TestBackendForRole:
    def _router(self, mode: RouteMode) -> ModelRouter:
        r = ModelRouter()
        r.mode = mode
        return r

    def test_cloud_mode_always_cloud(self) -> None:
        r = self._router(RouteMode.CLOUD)
        for role in ModelRole:
            assert r.backend_for(role) is Backend.CLOUD

    def test_hybrid_follows_hybrid_routing(self) -> None:
        r = self._router(RouteMode.HYBRID)
        assert r.backend_for(ModelRole.INTAKE) is Backend.LOCAL
        assert r.backend_for(ModelRole.PLANNER) is Backend.CLOUD
        assert r.backend_for(ModelRole.BUILDER) is Backend.CLOUD

    def test_local_mode_routes_everything_local(self) -> None:
        r = self._router(RouteMode.LOCAL)
        # Reasoning roles should still route local under mode='local'
        assert r.backend_for(ModelRole.PLANNER) is Backend.LOCAL
        assert r.backend_for(ModelRole.BUILDER) is Backend.LOCAL

    def test_backend_for_accepts_string(self) -> None:
        r = self._router(RouteMode.HYBRID)
        assert r.backend_for("intake") is Backend.LOCAL


class TestRoutingTable:
    def test_every_role_represented(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)
        r = ModelRouter()
        rows = r.routing_table()
        roles = {row[0] for row in rows}
        expected = {role.value for role in ModelRole}
        assert roles == expected

    def test_cloud_mode_shows_cloud_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BELIEF_MODEL_MODE", raising=False)
        r = ModelRouter()
        rows = r.routing_table()
        for role, backend, model in rows:
            assert backend is Backend.CLOUD
            assert "claude" in model

    def test_hybrid_mode_shows_local_model_for_mechanical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BELIEF_MODEL_MODE", "hybrid")
        r = ModelRouter()
        rows = dict((row[0], row) for row in r.routing_table())
        intake = rows["intake"]
        assert intake[1] is Backend.LOCAL
        assert intake[2] == DEFAULT_LOCAL_MODEL
        planner = rows["planner"]
        assert planner[1] is Backend.CLOUD


# ---------------------------------------------------------------------------
# AsyncOllamaClient (hermetic)
# ---------------------------------------------------------------------------


class TestAsyncOllamaClient:
    @pytest.mark.asyncio
    async def test_is_available_false_on_connection_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from belief.llm import AsyncOllamaClient

        ollama = AsyncOllamaClient(base_url="http://127.0.0.1:1")  # unroutable
        assert await ollama.is_available() is False
        await ollama.close()

    @pytest.mark.asyncio
    async def test_generate_produces_anthropic_shaped_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from belief.llm import AsyncOllamaClient

        ollama = AsyncOllamaClient(base_url="http://localhost:11434")

        class FakeResp:
            status_code = 200

            def raise_for_status(self) -> None: ...
            def json(self) -> Any:
                return {
                    "message": {"role": "assistant", "content": "hello world"},
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                }

        class FakeClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None: ...

            @property
            def is_closed(self) -> bool:
                return False

            async def post(self, *args: Any, **kwargs: Any) -> FakeResp:
                return FakeResp()

            async def aclose(self) -> None: ...

        monkeypatch.setattr(
            "belief.llm.httpx.AsyncClient", lambda *a, **k: FakeClient()
        )
        # Force the cached client to be re-created against the patched factory
        ollama._client = None
        resp = await ollama.generate(
            system="sys", user="hi", max_tokens=50, temperature=0.0
        )
        assert resp["content"][0]["text"] == "hello world"
        assert resp["usage"]["input_tokens"] == 10
        assert resp["usage"]["output_tokens"] == 2
        assert resp["_backend"] == "ollama"
        await ollama.close()


# ---------------------------------------------------------------------------
# LLMClient routing integration — fallback path when Ollama unavailable
# ---------------------------------------------------------------------------


class TestLLMClientFallback:
    @pytest.mark.asyncio
    async def test_local_mode_falls_back_to_cloud_when_ollama_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When mode=hybrid and Ollama isn't there, the dispatcher logs a
        warning, bumps the fallback counter, and proceeds to cloud.
        We don't actually issue the cloud request — we assert the
        fallback counter increments before the cloud call happens by
        stubbing _call.
        """
        from belief.llm import LLMClient

        monkeypatch.setenv("BELIEF_MODEL_MODE", "hybrid")
        router = ModelRouter()
        client = LLMClient(router)

        # Stub Ollama availability to False
        async def not_available() -> bool:
            return False

        # Stub _call to capture the cloud fallback call
        captured: dict[str, Any] = {}

        async def fake_cloud_call(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "content": [{"type": "text", "text": "from cloud"}],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }

        # Prime the Ollama client and patch its availability
        ollama = client._get_ollama()
        monkeypatch.setattr(ollama, "is_available", not_available)
        monkeypatch.setattr(client, "_call", fake_cloud_call)

        data, model, backend = await client._call_with_role(
            ModelRole.INTAKE,
            "sys",
            [{"role": "user", "content": "hi"}],
            max_tokens=50,
            temperature=0.0,
            complexity=1,
        )

        assert backend == "cloud"
        assert router.fallback_count == 1
        assert data["content"][0]["text"] == "from cloud"
        await client.close()

    @pytest.mark.asyncio
    async def test_cloud_mode_never_consults_ollama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from belief.llm import LLMClient

        monkeypatch.setenv("BELIEF_MODEL_MODE", "cloud")
        router = ModelRouter()
        client = LLMClient(router)

        async def fake_cloud_call(**kwargs: Any) -> dict[str, Any]:
            return {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 1, "output_tokens": 1,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                },
            }

        monkeypatch.setattr(client, "_call", fake_cloud_call)

        # Assert ollama client is not built on cloud-mode dispatch
        data, model, backend = await client._call_with_role(
            ModelRole.INTAKE,
            "sys",
            [{"role": "user", "content": "hi"}],
            max_tokens=50, temperature=0.0, complexity=1,
        )
        assert backend == "cloud"
        assert client._ollama is None  # never built
        assert router.fallback_count == 0
        await client.close()
