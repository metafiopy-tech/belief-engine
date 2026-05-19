"""Tests for the trigger registry + example predicates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from belief.signal.alphabet import Signal
from belief.signal.store import SignalStore
from belief.signal.triggers import (
    Trigger,
    TriggerRegistry,
    covenant_warn_predicate,
    stress_request_conjunction_predicate,
    sustained_offer_predicate,
)


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    s = SignalStore(db_path=tmp_path / "sig.db")
    yield s
    s.close()


def test_register_and_evaluate_empty(store: SignalStore) -> None:
    reg = TriggerRegistry(store=store)
    assert reg.evaluate() == []


def test_unknown_trigger_name(store: SignalStore) -> None:
    reg = TriggerRegistry(store=store)
    reg.unregister("nothing")  # no-op
    assert reg.names() == []


def test_register_requires_name(store: SignalStore) -> None:
    reg = TriggerRegistry(store=store)
    with pytest.raises(ValueError):
        reg.register(Trigger(name="", predicate=lambda *_: False))


# ── stress+request conjunction ─────────────────────────────────────────────


def test_stress_request_conjunction_fires_only_on_both(store: SignalStore) -> None:
    """Single high STRESS or single high REQUEST should NOT fire — the
    conjunction is what makes this a high-confidence signal."""
    fired: list[str] = []
    reg = TriggerRegistry(store=store)
    reg.register(
        Trigger(
            name="stress_req",
            predicate=stress_request_conjunction_predicate(
                stress_threshold=0.5,
                request_threshold=0.2,
                window=timedelta(minutes=10),
                half_life=timedelta(minutes=2),
            ),
            action=lambda agent_id: fired.append(agent_id),
        )
    )

    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Step 1: only STRESS, no REQUEST → no fire.
    store.emit(
        Signal(agent_id="alice", token="STRESS", magnitude=1.0, timestamp=t0, idempotency_key="s0")
    )
    events = reg.evaluate(agent_id="alice", now=t0 + timedelta(seconds=1))
    assert events == []

    # Step 2: add REQUEST → fires.
    store.emit(
        Signal(agent_id="alice", token="REQUEST", magnitude=0.5, timestamp=t0, idempotency_key="r0")
    )
    events = reg.evaluate(agent_id="alice", now=t0 + timedelta(seconds=1))
    assert len(events) == 1
    assert events[0].trigger_name == "stress_req"
    assert fired == ["alice"]


def test_predicate_failure_does_not_crash_registry(store: SignalStore) -> None:
    """A buggy predicate shouldn't take down sibling triggers."""

    def buggy(store, agent_id, now):
        raise RuntimeError("predicate exploded")

    def ok_predicate(store, agent_id, now):
        return True

    fired: list[str] = []
    reg = TriggerRegistry(store=store)
    reg.register(Trigger(name="buggy", predicate=buggy))
    reg.register(
        Trigger(
            name="ok",
            predicate=ok_predicate,
            action=lambda a: fired.append(a),
        )
    )
    store.emit(Signal(agent_id="alice", token="STRESS", magnitude=0.5))
    events = reg.evaluate(agent_id="alice")
    # The buggy trigger reports an error event; the OK trigger fires.
    assert any(e.trigger_name == "buggy" and e.error for e in events)
    assert any(e.trigger_name == "ok" and not e.error for e in events)
    assert fired == ["alice"]


def test_action_failure_does_not_crash_registry(store: SignalStore) -> None:
    def boom(_: str) -> None:
        raise RuntimeError("action exploded")

    reg = TriggerRegistry(store=store)
    reg.register(Trigger(name="x", predicate=lambda *_: True, action=boom))
    store.emit(Signal(agent_id="alice", token="STRESS", magnitude=0.5))
    events = reg.evaluate(agent_id="alice")
    assert len(events) == 1
    assert events[0].error  # the action's exception is captured, not raised


# ── sustained_offer ────────────────────────────────────────────────────────


def test_sustained_offer_fires_on_high_offer_concentration(store: SignalStore) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Five back-to-back OFFER emissions at magnitude 1.0 — concentration
    # right after the last one is roughly the sum of magnitudes weighted
    # by decay, ~4–5 with a 15-minute half-life over a few seconds.
    for i in range(5):
        store.emit(
            Signal(
                agent_id="alice",
                token="OFFER",
                magnitude=1.0,
                timestamp=t0 + timedelta(seconds=i),
                idempotency_key=f"o{i}",
            )
        )
    reg = TriggerRegistry(store=store)
    fired: list[str] = []
    reg.register(
        Trigger(
            name="sustained",
            predicate=sustained_offer_predicate(threshold=2.0),
            action=lambda a: fired.append(a),
        )
    )
    events = reg.evaluate(agent_id="alice", now=t0 + timedelta(seconds=5))
    assert fired == ["alice"]
    assert any(e.trigger_name == "sustained" for e in events)


def test_sustained_offer_does_not_fire_on_single_burst(store: SignalStore) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    store.emit(
        Signal(
            agent_id="alice", token="OFFER", magnitude=1.0, timestamp=t0, idempotency_key="single"
        )
    )
    reg = TriggerRegistry(store=store)
    reg.register(
        Trigger(
            name="sustained",
            predicate=sustained_offer_predicate(threshold=2.0),
        )
    )
    events = reg.evaluate(agent_id="alice", now=t0 + timedelta(seconds=1))
    assert events == []


# ── covenant_warn ─────────────────────────────────────────────────────────


def test_covenant_warn_predicate(store: SignalStore) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(3):
        store.emit(
            Signal(
                agent_id="alice",
                token="WARN",
                magnitude=1.0,
                timestamp=t0 + timedelta(seconds=i),
                idempotency_key=f"w{i}",
            )
        )
    reg = TriggerRegistry(store=store)
    reg.register(Trigger(name="warn", predicate=covenant_warn_predicate(threshold=1.5)))
    events = reg.evaluate(agent_id="alice", now=t0 + timedelta(seconds=3))
    assert events and events[0].trigger_name == "warn"


# ── Evaluate-all-agents ────────────────────────────────────────────────────


def test_evaluate_visits_all_known_agents(store: SignalStore) -> None:
    store.emit(Signal(agent_id="a", token="STRESS", magnitude=0.9))
    store.emit(Signal(agent_id="b", token="STRESS", magnitude=0.9))
    fired: list[str] = []
    reg = TriggerRegistry(store=store)
    reg.register(
        Trigger(
            name="always",
            predicate=lambda store, agent_id, now: True,
            action=lambda a: fired.append(a),
        )
    )
    reg.evaluate()
    assert sorted(fired) == ["a", "b"]
