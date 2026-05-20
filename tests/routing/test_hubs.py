"""Tests for HubRegistry — derivation, promotion, demotion."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.memory.reciprocity import ReciprocityLedger
from belief.routing._store import RoutingStore
from belief.routing.hubs import HubRegistry


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


def _make_hub_eligible(recip: ReciprocityLedger, agent_id: str, nutrients: float) -> None:
    """Give an agent a high exchange rate and lifetime nutrients above floor."""
    recip.record_request(agent_id, cost=1.0)
    # nutrients contributions of value 1.0 each → lifetime nutrients_returned
    for i in range(int(nutrients)):
        recip.record_contribution(agent_id, nutrient_value=1.0, nutrient_id=f"{agent_id}-n{i}")


# ── Bypass: empty ledger → no hubs ──────────────────────────────────────────


def test_empty_ledger_has_no_hubs(store: RoutingStore, recip: ReciprocityLedger) -> None:
    """The load-bearing bypass property: with no agents, no hubs exist."""
    reg = HubRegistry(store, recip)
    assert reg.recompute() == []
    assert reg.current_hubs() == []
    assert reg.is_hub("anyone") is False
    assert reg.nearest_hub("anyone") is None


def test_low_activity_agent_does_not_become_hub(
    store: RoutingStore, recip: ReciprocityLedger
) -> None:
    """A single agent with a few nutrients (below floor) is NOT promoted —
    even though it's trivially 'top decile' of a one-agent population. The
    lifetime floor is what keeps the singular belief_engine agent from
    becoming a hub during normal builds."""
    _make_hub_eligible(recip, "belief_engine", nutrients=5)  # below floor 100
    reg = HubRegistry(store, recip, lifetime_floor=100.0)
    assert reg.recompute() == []


# ── Promotion ────────────────────────────────────────────────────────────────


def test_high_reciprocity_agent_promoted(store: RoutingStore, recip: ReciprocityLedger) -> None:
    _make_hub_eligible(recip, "alice", nutrients=150)  # above floor
    reg = HubRegistry(store, recip, lifetime_floor=100.0)
    hubs = reg.recompute()
    assert "alice" in hubs
    assert reg.is_hub("alice") is True


def test_nearest_hub_picks_highest_exchange(store: RoutingStore, recip: ReciprocityLedger) -> None:
    # Two hubs, alice with higher exchange rate than bob.
    recip.record_request("alice", cost=1.0)
    for i in range(150):
        recip.record_contribution("alice", 1.0, f"a{i}")
    recip.record_request("bob", cost=10.0)
    for i in range(150):
        recip.record_contribution("bob", 1.0, f"b{i}")
    reg = HubRegistry(store, recip, lifetime_floor=100.0)
    reg.recompute()
    # peripheral agent 'carol' routes to the higher-exchange hub (alice:
    # 150/1 vs bob: 150/10).
    assert reg.nearest_hub("carol") == "alice"


def test_hub_does_not_route_to_itself(store: RoutingStore, recip: ReciprocityLedger) -> None:
    _make_hub_eligible(recip, "alice", nutrients=150)
    reg = HubRegistry(store, recip, lifetime_floor=100.0)
    reg.recompute()
    # alice is the only hub → nearest_hub for alice excludes herself → None
    assert reg.nearest_hub("alice") is None


# ── Demotion hysteresis ──────────────────────────────────────────────────────


def test_hub_demoted_after_n_subthreshold_recomputes(
    store: RoutingStore, recip: ReciprocityLedger
) -> None:
    """A hub that stops qualifying is demoted only after demote_after
    consecutive sub-threshold recomputes — not on the first dip."""
    _make_hub_eligible(recip, "alice", nutrients=150)
    reg = HubRegistry(store, recip, lifetime_floor=100.0, demote_after=3)
    reg.recompute()
    assert reg.is_hub("alice") is True

    # Now make alice fail to qualify by raising the floor above her nutrients.
    reg_strict = HubRegistry(store, recip, lifetime_floor=10_000.0, demote_after=3)
    # First two recomputes: still a hub (hysteresis).
    reg_strict.recompute()
    assert reg_strict.is_hub("alice") is True
    reg_strict.recompute()
    assert reg_strict.is_hub("alice") is True
    # Third recompute crosses demote_after → demoted.
    reg_strict.recompute()
    assert reg_strict.is_hub("alice") is False


def test_requalifying_resets_demotion_counter(
    store: RoutingStore, recip: ReciprocityLedger
) -> None:
    _make_hub_eligible(recip, "alice", nutrients=150)
    reg = HubRegistry(store, recip, lifetime_floor=100.0, demote_after=3)
    reg.recompute()

    # One sub-threshold recompute (raise floor), then re-qualify (lower floor).
    HubRegistry(store, recip, lifetime_floor=10_000.0, demote_after=3).recompute()
    # Re-qualify: counter should reset, so alice stays a hub indefinitely.
    reg.recompute()
    reg.recompute()
    reg.recompute()
    assert reg.is_hub("alice") is True


def test_top_fraction_validation(store: RoutingStore, recip: ReciprocityLedger) -> None:
    with pytest.raises(ValueError):
        HubRegistry(store, recip, top_fraction=0.0)
    with pytest.raises(ValueError):
        HubRegistry(store, recip, top_fraction=1.5)
