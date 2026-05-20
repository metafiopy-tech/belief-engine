"""Tests for Router — the bypass invariant is the load-bearing one."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.memory.reciprocity import ReciprocityLedger
from belief.routing._store import RoutingStore
from belief.routing.hubs import HubRegistry
from belief.routing.router import (
    RoutingKind,
    Router,
    enforcement_enabled,
)


@pytest.fixture
def recip(tmp_path: Path) -> ReciprocityLedger:
    r = ReciprocityLedger(db_path=tmp_path / "reciprocity.db")
    yield r
    r.close()


@pytest.fixture
def store(tmp_path: Path) -> RoutingStore:
    s = RoutingStore(db_path=tmp_path / "routing.db")
    yield s
    s.close()


@pytest.fixture
def hubs(store: RoutingStore, recip: ReciprocityLedger) -> HubRegistry:
    return HubRegistry(store, recip, lifetime_floor=100.0)


def _promote(recip: ReciprocityLedger, hubs: HubRegistry, agent_id: str) -> None:
    recip.record_request(agent_id, cost=1.0)
    for i in range(150):
        recip.record_contribution(agent_id, 1.0, f"{agent_id}-n{i}")
    hubs.recompute()


# ── Bypass invariant ─────────────────────────────────────────────────────────


def test_no_hubs_routes_direct(store: RoutingStore, hubs: HubRegistry) -> None:
    """THE invariant: with no hubs, every request bypasses straight to the
    engine. This is what keeps the build pipeline unaffected."""
    router = Router(store, hubs)
    d = router.route("belief_engine")
    assert d.kind is RoutingKind.DIRECT
    assert d.is_bypass is True
    assert d.hub_id is None


def test_route_never_raises_on_empty_state(store: RoutingStore, hubs: HubRegistry) -> None:
    router = Router(store, hubs)
    # Many calls, never an exception, always DIRECT.
    for i in range(50):
        d = router.route(f"agent-{i}")
        assert d.kind is RoutingKind.DIRECT


def test_routing_event_recorded(store: RoutingStore, hubs: HubRegistry) -> None:
    router = Router(store, hubs)
    router.route("alice")
    events = store.events_since()
    assert len(events) == 1
    assert events[0]["agent_id"] == "alice"
    assert events[0]["decision_kind"] == "direct"


def test_record_false_skips_event(store: RoutingStore, hubs: HubRegistry) -> None:
    router = Router(store, hubs)
    router.route("alice", record=False)
    assert store.events_since() == []


# ── With hubs ────────────────────────────────────────────────────────────────


def test_hub_sender_routes_direct(
    store: RoutingStore, hubs: HubRegistry, recip: ReciprocityLedger
) -> None:
    _promote(recip, hubs, "alice")
    router = Router(store, hubs)
    d = router.route("alice")  # alice is herself a hub
    assert d.kind is RoutingKind.DIRECT


def test_peripheral_routes_via_hub(
    store: RoutingStore, hubs: HubRegistry, recip: ReciprocityLedger
) -> None:
    _promote(recip, hubs, "alice")
    router = Router(store, hubs)
    d = router.route("carol")  # carol is peripheral
    assert d.kind is RoutingKind.VIA_HUB
    assert d.hub_id == "alice"


def test_cache_hit_when_query_cached(
    store: RoutingStore, hubs: HubRegistry, recip: ReciprocityLedger
) -> None:
    _promote(recip, hubs, "alice")
    router = Router(store, hubs)
    # First call: VIA_HUB (no cache). Then populate cache, second call hits.
    d1 = router.route("carol", query_key="q1")
    assert d1.kind is RoutingKind.VIA_HUB
    router.cache_response("alice", "q1", response="cached-answer")
    d2 = router.route("carol", query_key="q1")
    assert d2.kind is RoutingKind.CACHE_HIT
    assert d2.cached_response == "cached-answer"


def test_lru_cache_evicts_oldest(
    store: RoutingStore, hubs: HubRegistry, recip: ReciprocityLedger
) -> None:
    _promote(recip, hubs, "alice")
    router = Router(store, hubs, cache_capacity=2)
    router.cache_response("alice", "k1", "v1")
    router.cache_response("alice", "k2", "v2")
    router.cache_response("alice", "k3", "v3")  # evicts k1
    assert router.route("carol", query_key="k1").kind is RoutingKind.VIA_HUB  # miss
    assert router.route("carol", query_key="k3").kind is RoutingKind.CACHE_HIT


# ── Enforcement flag ─────────────────────────────────────────────────────────


def test_enforcement_default_off(monkeypatch) -> None:
    monkeypatch.delenv("BELIEF_ROUTING_ENFORCE", raising=False)
    assert enforcement_enabled() is False


def test_enforcement_flag_parsing(monkeypatch) -> None:
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("BELIEF_ROUTING_ENFORCE", truthy)
        assert enforcement_enabled() is True
    for falsy in ("0", "false", "", "no"):
        monkeypatch.setenv("BELIEF_ROUTING_ENFORCE", falsy)
        assert enforcement_enabled() is False


def test_decision_carries_enforced_flag(
    store: RoutingStore, hubs: HubRegistry, monkeypatch
) -> None:
    monkeypatch.setenv("BELIEF_ROUTING_ENFORCE", "1")
    router = Router(store, hubs)
    d = router.route("alice")
    assert d.enforced is True
    # Still DIRECT because no hubs — enforcement doesn't manufacture hubs.
    assert d.kind is RoutingKind.DIRECT
