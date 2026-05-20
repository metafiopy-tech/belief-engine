"""Tests for SanctionsEngine — advisory verdicts + graveyard."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.memory.reciprocity import ReciprocityLedger
from belief.routing._store import RoutingStore
from belief.routing.sanctions import (
    SanctionAction,
    SanctionsEngine,
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
def sanctions(store: RoutingStore, recip: ReciprocityLedger) -> SanctionsEngine:
    return SanctionsEngine(store, recip)


# ── ALLOW cases ──────────────────────────────────────────────────────────────


def test_unknown_agent_allowed(sanctions: SanctionsEngine) -> None:
    """Never sanction an agent we've never charged."""
    d = sanctions.evaluate("ghost")
    assert d.action is SanctionAction.ALLOW


def test_productive_agent_allowed(sanctions: SanctionsEngine, recip: ReciprocityLedger) -> None:
    recip.record_request("alice", cost=1.0)
    recip.record_contribution("alice", nutrient_value=5.0, nutrient_id="n1")
    d = sanctions.evaluate("alice")
    assert d.action is SanctionAction.ALLOW


# ── THROTTLE ─────────────────────────────────────────────────────────────────


def test_low_recent_exchange_throttled(
    sanctions: SanctionsEngine, recip: ReciprocityLedger
) -> None:
    # Lots of compute spent, almost nothing returned → 7d rate well below 0.1.
    recip.record_request("leech", cost=100.0)
    recip.record_contribution("leech", nutrient_value=1.0, nutrient_id="n1")
    d = sanctions.evaluate("leech")
    assert d.action is SanctionAction.THROTTLE
    assert d.backoff_hint_s is not None and d.backoff_hint_s > 0


# ── TERMINATE ────────────────────────────────────────────────────────────────


def test_persistent_parasite_terminated_after_grace(
    store: RoutingStore, recip: ReciprocityLedger
) -> None:
    # 30d rate below terminate threshold AND > grace_period_n requests.
    eng = SanctionsEngine(store, recip, grace_period_n=50)
    for i in range(60):
        recip.record_request("parasite", cost=100.0, idempotency_key=f"r{i}")
    # No contributions → 30d exchange rate ~0.
    d = eng.evaluate("parasite")
    assert d.action is SanctionAction.TERMINATE
    # Terminated agents are archived to the graveyard.
    assert eng.is_archived("parasite") is True


def test_parasite_within_grace_only_throttled(
    store: RoutingStore, recip: ReciprocityLedger
) -> None:
    """Below the terminate threshold but inside the grace period → the
    softer THROTTLE, not TERMINATE. Grace lets new agents establish."""
    eng = SanctionsEngine(store, recip, grace_period_n=50)
    for i in range(10):  # only 10 requests, < grace_period_n
        recip.record_request("newbie", cost=100.0, idempotency_key=f"r{i}")
    d = eng.evaluate("newbie")
    assert d.action is SanctionAction.THROTTLE
    assert eng.is_archived("newbie") is False


def test_terminate_archive_can_be_suppressed(store: RoutingStore, recip: ReciprocityLedger) -> None:
    eng = SanctionsEngine(store, recip, grace_period_n=50)
    for i in range(60):
        recip.record_request("p", cost=100.0, idempotency_key=f"r{i}")
    d = eng.evaluate("p", archive_on_terminate=False)
    assert d.action is SanctionAction.TERMINATE
    assert eng.is_archived("p") is False  # archival suppressed


# ── Soft recovery ────────────────────────────────────────────────────────────


def test_throttled_agent_recovers_when_reciprocity_returns(
    sanctions: SanctionsEngine, recip: ReciprocityLedger
) -> None:
    """Sanctions are soft: an agent that re-establishes reciprocity is
    allowed again on the next evaluate (the decision reads live rate)."""
    recip.record_request("comeback", cost=100.0)
    recip.record_contribution("comeback", nutrient_value=1.0, nutrient_id="n1")
    assert sanctions.evaluate("comeback").action is SanctionAction.THROTTLE
    # Pour in contributions to lift the 7d exchange rate above threshold.
    for i in range(200):
        recip.record_contribution("comeback", nutrient_value=1.0, nutrient_id=f"c{i}")
    assert sanctions.evaluate("comeback").action is SanctionAction.ALLOW
