"""Tests for the signal store + temporal integration math."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from belief.signal.alphabet import Signal
from belief.signal.store import (
    DEFAULT_HALF_LIFE,
    DEFAULT_WINDOW,
    SignalStore,
    parse_duration,
)


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    s = SignalStore(db_path=tmp_path / "sig.db", buffer_size=100)
    yield s
    s.close()


# ── parse_duration ─────────────────────────────────────────────────────────


def test_parse_duration_string_units() -> None:
    assert parse_duration("30s") == timedelta(seconds=30)
    assert parse_duration("5m") == timedelta(minutes=5)
    assert parse_duration("2h") == timedelta(hours=2)
    assert parse_duration("1d") == timedelta(days=1)


def test_parse_duration_passthrough_timedelta() -> None:
    td = timedelta(hours=3)
    assert parse_duration(td) is td


def test_parse_duration_seconds_number() -> None:
    assert parse_duration(60) == timedelta(seconds=60)
    assert parse_duration(0.5) == timedelta(seconds=0.5)


def test_parse_duration_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_duration("forever")
    with pytest.raises(ValueError):
        parse_duration(-1)
    with pytest.raises(TypeError):
        parse_duration(None)  # type: ignore[arg-type]


# ── Emit + idempotency ─────────────────────────────────────────────────────


def test_emit_round_trip(store: SignalStore) -> None:
    sig = Signal(agent_id="alice", token="STRESS", magnitude=0.5)
    assert store.emit(sig) is True
    # Idempotent — same effective key returns False on replay.
    assert store.emit(sig) is False
    assert store.count_for("alice", "STRESS") == 1


def test_distinct_signals_both_recorded(store: SignalStore) -> None:
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    a = Signal(agent_id="alice", token="STRESS", magnitude=0.5, timestamp=base)
    b = Signal(
        agent_id="alice",
        token="STRESS",
        magnitude=0.5,
        timestamp=base + timedelta(seconds=1),
    )
    assert store.emit(a) is True
    assert store.emit(b) is True
    assert store.count_for("alice", "STRESS") == 2


def test_buffer_circular_pruning(tmp_path: Path) -> None:
    """When buffer_size is exceeded for an (agent, token), oldest rows
    are pruned on emit so the count never grows past buffer_size."""
    s = SignalStore(db_path=tmp_path / "sig.db", buffer_size=5)
    try:
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(20):
            sig = Signal(
                agent_id="alice",
                token="STRESS",
                magnitude=0.5,
                timestamp=base + timedelta(seconds=i),
                idempotency_key=f"k{i}",
            )
            s.emit(sig)
        assert s.count_for("alice", "STRESS") == 5
        # Pruning preserves the *newest* — verify the surviving rows have
        # the latest timestamps.
        recent = s.recent_signals(
            "alice",
            window=timedelta(hours=1),
            token="STRESS",
            now=base + timedelta(seconds=30),
        )
        assert len(recent) == 5
        seconds = [r.timestamp.second for r in recent]
        assert seconds == sorted(seconds)  # ascending
        assert seconds[-1] == 19  # newest preserved
    finally:
        s.close()


def test_buffer_isolated_per_token(store: SignalStore) -> None:
    """The 100-row buffer is per (agent, token), not global."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(20):
        store.emit(
            Signal(
                agent_id="alice",
                token="STRESS",
                magnitude=0.5,
                timestamp=base + timedelta(seconds=i),
                idempotency_key=f"s{i}",
            )
        )
        store.emit(
            Signal(
                agent_id="alice",
                token="OFFER",
                magnitude=0.5,
                timestamp=base + timedelta(seconds=i),
                idempotency_key=f"o{i}",
            )
        )
    assert store.count_for("alice", "STRESS") == 20
    assert store.count_for("alice", "OFFER") == 20


# ── Decay math — the load-bearing tests ────────────────────────────────────


def test_concentration_decay_at_half_life(store: SignalStore) -> None:
    """Emission of magnitude 1.0 at t0 → concentration 0.5 at t0+half_life,
    0.25 at t0+2*half_life. This is the math the receiver-priming pattern
    depends on; if it's wrong, triggers fire at the wrong times."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    hl = timedelta(seconds=60)
    win = timedelta(minutes=10)
    store.emit(
        Signal(
            agent_id="alice",
            token="STRESS",
            magnitude=1.0,
            timestamp=t0,
            idempotency_key="single",
        )
    )
    at_zero = store.concentration("alice", "STRESS", window=win, half_life=hl, now=t0)
    at_hl = store.concentration("alice", "STRESS", window=win, half_life=hl, now=t0 + hl)
    at_2hl = store.concentration("alice", "STRESS", window=win, half_life=hl, now=t0 + 2 * hl)
    assert at_zero == pytest.approx(1.0)
    assert at_hl == pytest.approx(0.5)
    assert at_2hl == pytest.approx(0.25)


def test_concentration_window_excludes_old_events(store: SignalStore) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    win = timedelta(minutes=1)
    hl = timedelta(seconds=60)
    # Emit 5 minutes ago, far outside a 1-minute window.
    store.emit(
        Signal(
            agent_id="alice",
            token="STRESS",
            magnitude=1.0,
            timestamp=t0 - timedelta(minutes=5),
            idempotency_key="old",
        )
    )
    assert store.concentration("alice", "STRESS", window=win, half_life=hl, now=t0) == 0.0


def test_concentration_unknown_agent_or_token_is_zero(store: SignalStore) -> None:
    assert store.concentration("nobody", "STRESS") == 0.0
    store.emit(Signal(agent_id="alice", token="STRESS", magnitude=0.5))
    assert store.concentration("alice", "WARN") == 0.0


def test_concentration_rejects_zero_half_life(store: SignalStore) -> None:
    with pytest.raises(ValueError):
        store.concentration("alice", "STRESS", half_life=timedelta(0))


# ── Joint concentration ───────────────────────────────────────────────────


def test_joint_concentration_is_product(store: SignalStore) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    hl = timedelta(seconds=60)
    win = timedelta(minutes=10)
    store.emit(
        Signal(agent_id="alice", token="STRESS", magnitude=0.8, timestamp=t0, idempotency_key="s")
    )
    store.emit(
        Signal(agent_id="alice", token="REQUEST", magnitude=0.4, timestamp=t0, idempotency_key="r")
    )
    joint = store.joint_concentration(
        "alice", ("STRESS", "REQUEST"), window=win, half_life=hl, now=t0
    )
    # 0.8 * 0.4 = 0.32 exactly at t0 (no decay applied yet)
    assert joint == pytest.approx(0.8 * 0.4)


def test_joint_concentration_zero_if_one_missing(store: SignalStore) -> None:
    store.emit(Signal(agent_id="a", token="STRESS", magnitude=1.0))
    assert store.joint_concentration("a", ("STRESS", "OFFER")) == 0.0


# ── recent_signals + profile ──────────────────────────────────────────────


def test_recent_signals_filters_by_token(store: SignalStore) -> None:
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    store.emit(Signal(agent_id="a", token="STRESS", magnitude=0.5, timestamp=base))
    store.emit(
        Signal(
            agent_id="a",
            token="OFFER",
            magnitude=0.7,
            timestamp=base + timedelta(seconds=1),
        )
    )
    all_recent = store.recent_signals(
        "a", window=timedelta(hours=1), now=base + timedelta(seconds=5)
    )
    only_stress = store.recent_signals(
        "a",
        window=timedelta(hours=1),
        token="STRESS",
        now=base + timedelta(seconds=5),
    )
    assert len(all_recent) == 2
    assert len(only_stress) == 1
    assert only_stress[0].token == "STRESS"


def test_profile_vector_has_all_five_tokens(store: SignalStore) -> None:
    store.emit(Signal(agent_id="a", token="STRESS", magnitude=0.5))
    profile = store.profile("a")
    assert set(profile.keys()) == {"STRESS", "DISCOVER", "REQUEST", "OFFER", "WARN"}
    assert profile["STRESS"] > 0
    for t in ("DISCOVER", "REQUEST", "OFFER", "WARN"):
        assert profile[t] == 0.0


def test_known_agents_dedup(store: SignalStore) -> None:
    store.emit(Signal(agent_id="alice", token="STRESS", magnitude=0.1, idempotency_key="a1"))
    store.emit(Signal(agent_id="alice", token="OFFER", magnitude=0.1, idempotency_key="a2"))
    store.emit(Signal(agent_id="bob", token="STRESS", magnitude=0.1, idempotency_key="b1"))
    assert store.known_agents() == ["alice", "bob"]


# ── Defaults sanity ───────────────────────────────────────────────────────


def test_defaults_are_reasonable() -> None:
    assert DEFAULT_WINDOW.total_seconds() > 0
    assert DEFAULT_HALF_LIFE.total_seconds() > 0
    # Half-life shorter than the window so decay actually happens inside it.
    assert DEFAULT_HALF_LIFE < DEFAULT_WINDOW
