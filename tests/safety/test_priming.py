"""Tests for defense-priming propagation (mycorrhizal Stage 6, Area 7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from belief.safety.priming import (
    CheckResult,
    PrimingPropagator,
    Warning,
    WarningKind,
    WarningStore,
    cli_format_warnings,
)


@pytest.fixture
def store(tmp_path: Path) -> WarningStore:
    s = WarningStore(db_path=tmp_path / "warnings.db")
    yield s
    s.close()


@pytest.fixture
def prop(store: WarningStore) -> PrimingPropagator:
    return PrimingPropagator(store=store)


# ── Emit ────────────────────────────────────────────────────────────────────


def test_emit_priming_stores_warning(prop: PrimingPropagator) -> None:
    w = prop.emit_priming("flaky-import-pattern", evidence={"build": "b1"})
    assert w.kind is WarningKind.PRIMING
    assert w.hops_remaining == prop.gossip_ttl
    active = prop.current_warnings()
    assert len(active) == 1
    assert active[0].pattern == "flaky-import-pattern"


def test_emit_covenant_is_eager_zero_hops(prop: PrimingPropagator) -> None:
    w = prop.emit_covenant("bare-except-violation")
    assert w.kind is WarningKind.COVENANT
    assert w.hops_remaining == 0  # eager broadcast, not gossip


def test_warning_rejects_naive_timestamp() -> None:
    with pytest.raises(Exception):
        Warning(
            kind=WarningKind.PRIMING,
            pattern="x",
            expires_at=datetime(2026, 1, 1),  # naive
            originating_agent_id="a",
        )


# ── Decay ─────────────────────────────────────────────────────────────────


def test_expired_warning_pruned_on_read(store: WarningStore) -> None:
    prop = PrimingPropagator(store=store, priming_half_life=timedelta(hours=1))
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    prop.emit_priming("soon-expired", now=t0)
    # Read 2 hours later → expired, pruned.
    later = t0 + timedelta(hours=2)
    assert prop.current_warnings(now=later) == []


def test_active_warning_not_pruned(store: WarningStore) -> None:
    prop = PrimingPropagator(store=store, priming_half_life=timedelta(hours=24))
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    prop.emit_priming("still-active", now=t0)
    soon = t0 + timedelta(hours=1)
    assert len(prop.current_warnings(now=soon)) == 1


def test_retrigger_refreshes_expiry(store: WarningStore) -> None:
    prop = PrimingPropagator(store=store, priming_half_life=timedelta(hours=2))
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    prop.emit_priming("recurring", now=t0)
    # Re-trigger at t0+1h → expiry pushed to t0+3h.
    refreshed = prop.retrigger("recurring", now=t0 + timedelta(hours=1))
    assert refreshed == 1
    # At t0+2.5h the warning would have expired without refresh; still active.
    assert len(prop.current_warnings(now=t0 + timedelta(hours=2, minutes=30))) == 1


# ── check_operation ─────────────────────────────────────────────────────────


def test_priming_match_raises_sentinel_not_block(prop: PrimingPropagator) -> None:
    prop.emit_priming("sql-injection")
    result = prop.check_operation("agent", "running a sql-injection prone query")
    assert isinstance(result, CheckResult)
    assert "sql-injection" in result.primed_patterns
    assert result.blocked is False  # priming never blocks


def test_covenant_match_blocks(prop: PrimingPropagator) -> None:
    prop.emit_covenant("rm -rf /")
    result = prop.check_operation("agent", "about to run rm -rf / on the host")
    assert result.blocked is True
    assert "rm -rf /" in result.blocking_warnings


def test_no_match_clean(prop: PrimingPropagator) -> None:
    prop.emit_priming("some-pattern")
    result = prop.check_operation("agent", "totally unrelated operation")
    assert result.primed_patterns == []
    assert result.blocked is False


def test_check_is_case_insensitive(prop: PrimingPropagator) -> None:
    prop.emit_priming("FlakyTest")
    result = prop.check_operation("agent", "this triggers flakytest behaviour")
    assert "FlakyTest" in result.primed_patterns


# ── Gossip reach ────────────────────────────────────────────────────────────


def test_gossip_reach_grows_with_ttl(prop: PrimingPropagator) -> None:
    """Reach should expand by ~K per carrier per tick, up to population."""
    agents = [f"a{i}" for i in range(100)]
    reach_ttl1 = prop.simulate_gossip_reach(agents, k=3, ttl=1)
    reach_ttl2 = prop.simulate_gossip_reach(agents, k=3, ttl=2)
    # TTL=1: origin + 3 = 4. TTL=2: origin + 3 + 9 = 13.
    assert len(reach_ttl1) == 4
    assert len(reach_ttl2) == 13


def test_gossip_reach_capped_at_population(prop: PrimingPropagator) -> None:
    agents = ["a", "b", "c"]
    reach = prop.simulate_gossip_reach(agents, k=3, ttl=5)
    assert reach == {"a", "b", "c"}  # can't exceed the population


def test_gossip_empty_population(prop: PrimingPropagator) -> None:
    assert prop.simulate_gossip_reach([], k=3, ttl=5) == set()


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_empty(prop: PrimingPropagator) -> None:
    out = cli_format_warnings(prop)
    assert "no active warnings" in out


def test_cli_populated(prop: PrimingPropagator) -> None:
    prop.emit_priming("pattern-a")
    prop.emit_covenant("pattern-b")
    out = cli_format_warnings(prop)
    assert "pattern-a" in out
    assert "pattern-b" in out
    assert "priming" in out
    assert "covenant" in out
