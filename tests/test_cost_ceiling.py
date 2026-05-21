"""Session A (handoff Q4): hard cost ceiling.

These tests are hermetic — no network, no real LLM calls. They exercise the
ceiling primitives, the pre-call gate, the BaseAgent wiring (pre-agent gate +
mid-agent abort), and the CLI verdict resolution.
"""

from __future__ import annotations

import pytest

from belief.agents.base import BaseAgent
from belief.cli import _resolve_verdict
from belief.config.models import ModelRole, ModelRouter
from belief.llm import (
    BudgetExceededError,
    CostCeiling,
    _ceiling_ctx,
    _enforce_cost_ceiling,
    _usage_ctx,
    cost_grace_usd,
)
from belief.models.artifacts import TokenUsage
from belief.models.state import Phase


# ── CostCeiling primitive ────────────────────────────────────────────────────


def test_would_exceed_respects_grace_margin():
    # cap 3.0, grace 0.5 → headroom for committed+pending is 2.5
    ceiling = CostCeiling(max_cost_usd=3.0, committed_usd=0.0, grace_usd=0.5)
    assert ceiling.would_exceed(2.4) is False  # 0 + 2.4 + 0.5 = 2.9 < 3.0
    assert ceiling.would_exceed(2.5) is True  # 0 + 2.5 + 0.5 = 3.0 >= 3.0


def test_would_exceed_committed_below_cap_is_allowed():
    ceiling = CostCeiling(max_cost_usd=3.0, committed_usd=2.0, grace_usd=0.5)
    assert ceiling.would_exceed(0.0) is False  # 2.5 < 3.0
    assert ceiling.would_exceed(0.5) is True  # 3.0 >= 3.0


def test_zero_or_negative_cap_never_blocks():
    assert CostCeiling(max_cost_usd=0.0).would_exceed(1000.0) is False
    assert CostCeiling(max_cost_usd=-1.0).would_exceed(1000.0) is False


def test_check_raises_with_accounting():
    ceiling = CostCeiling(max_cost_usd=1.0, committed_usd=0.8, grace_usd=0.5)
    with pytest.raises(BudgetExceededError) as exc:
        ceiling.check(0.0)
    assert exc.value.max_cost_usd == 1.0
    assert exc.value.committed_usd == pytest.approx(0.8)


def test_check_does_not_raise_when_under():
    CostCeiling(max_cost_usd=10.0, committed_usd=0.0, grace_usd=0.5).check(1.0)  # no raise


# ── cost_grace_usd env override ──────────────────────────────────────────────


def test_cost_grace_default(monkeypatch):
    monkeypatch.delenv("BELIEF_COST_GRACE_USD", raising=False)
    assert cost_grace_usd() == pytest.approx(0.50)


def test_cost_grace_env_override(monkeypatch):
    monkeypatch.setenv("BELIEF_COST_GRACE_USD", "1.25")
    assert cost_grace_usd() == pytest.approx(1.25)


def test_cost_grace_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("BELIEF_COST_GRACE_USD", "not-a-number")
    assert cost_grace_usd() == pytest.approx(0.50)


# ── pre-call gate (_enforce_cost_ceiling) ────────────────────────────────────


def test_enforce_noop_without_ceiling():
    # No ceiling installed → gate must be a no-op even with usage present.
    usage = TokenUsage()
    usage.add_call("builder", 1000, 1000, cost_usd=99.0)
    tok = _usage_ctx.set(usage)
    try:
        _enforce_cost_ceiling()  # must not raise
    finally:
        _usage_ctx.reset(tok)


def test_enforce_raises_when_pending_crosses_ceiling():
    ceiling = CostCeiling(max_cost_usd=1.0, committed_usd=0.0, grace_usd=0.0)
    usage = TokenUsage()
    usage.add_call("builder", 0, 0, cost_usd=1.5)  # pending 1.5 >= cap 1.0
    ct = _ceiling_ctx.set(ceiling)
    ut = _usage_ctx.set(usage)
    try:
        with pytest.raises(BudgetExceededError):
            _enforce_cost_ceiling()
    finally:
        _usage_ctx.reset(ut)
        _ceiling_ctx.reset(ct)


def test_enforce_allows_when_under_ceiling():
    ceiling = CostCeiling(max_cost_usd=10.0, committed_usd=0.0, grace_usd=0.5)
    usage = TokenUsage()
    usage.add_call("builder", 0, 0, cost_usd=0.5)
    ct = _ceiling_ctx.set(ceiling)
    ut = _usage_ctx.set(usage)
    try:
        _enforce_cost_ceiling()  # 0.5 + 0.5 = 1.0 < 10.0 → no raise
    finally:
        _usage_ctx.reset(ut)
        _ceiling_ctx.reset(ct)


# ── BaseAgent wiring ─────────────────────────────────────────────────────────


class _DummyAgent(BaseAgent):
    role = ModelRole.INTAKE
    name = "DummyAgent"

    def __init__(self, router, *, raise_budget=False):
        super().__init__(router)
        self._raise_budget = raise_budget
        self.run_called = False

    async def run(self, state):
        self.run_called = True
        if self._raise_budget:
            raise BudgetExceededError(committed_usd=99.0, max_cost_usd=1.0, grace_usd=0.5)
        state.phase = Phase.COMPLETE
        return state


@pytest.fixture
def router():
    return ModelRouter()


async def test_pre_agent_gate_skips_run_when_over_budget(router):
    agent = _DummyAgent(router)
    state = {
        "run_id": "t",
        "user_goal": "g",
        "max_cost_usd": 1.0,
        "token_usage": {
            "total_cost_usd": 2.0,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 50,
        },
    }
    out = await agent(state)
    assert agent.run_called is False  # never started — bounded spend
    assert out["aborted_budget"] is True
    assert out["phase"] == Phase.FAILED.value
    # It's an abort, not a crash.
    assert not any("crashed" in e for e in out.get("errors", []))


async def test_mid_agent_budget_error_marks_aborted_not_crashed(router):
    agent = _DummyAgent(router, raise_budget=True)
    state = {
        "run_id": "t",
        "user_goal": "g",
        "max_cost_usd": 10.0,  # under budget at start → run() is entered
        "token_usage": {"total_cost_usd": 0.0},
    }
    out = await agent(state)
    assert agent.run_called is True
    assert out["aborted_budget"] is True
    assert out["phase"] == Phase.FAILED.value
    assert not any("crashed" in e for e in out.get("errors", []))
    assert any("aborted" in w for w in out.get("warnings", []))


async def test_under_budget_agent_runs_normally(router):
    agent = _DummyAgent(router)
    state = {
        "run_id": "t",
        "user_goal": "g",
        "max_cost_usd": 10.0,
        "token_usage": {"total_cost_usd": 0.1},
    }
    out = await agent(state)
    assert agent.run_called is True
    assert out.get("aborted_budget", False) is False
    assert out["phase"] == Phase.COMPLETE.value


# ── CLI verdict resolution ───────────────────────────────────────────────────


def test_resolve_verdict_maps_aborted_build():
    assert _resolve_verdict({"aborted_budget": True}, "pass") == "aborted_budget"


def test_resolve_verdict_passes_through_normal_verdict():
    assert _resolve_verdict({"aborted_budget": False}, "pass") == "pass"
    assert _resolve_verdict({}, "fail_fixable") == "fail_fixable"
