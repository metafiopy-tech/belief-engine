"""Hermetic tests for belief.memory.reciprocity (mycorrhizal Stage 1).

All tests use tmp_path so nothing touches the real ~/.belief-engine/ tree.
Run with:

    python3 -m pytest tests/memory/test_reciprocity.py -q --timeout=60
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from belief.memory.reciprocity import (
    DEFAULT_WINDOW,
    ReciprocityLedger,
    _parse_window,
    cli_format_ledger,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def ledger(tmp_path: Path) -> ReciprocityLedger:
    """Fresh ledger against tmp_path. Closed automatically at fixture teardown."""
    ledg = ReciprocityLedger(db_path=tmp_path / "reciprocity.db")
    yield ledg
    ledg.close()


# ── 1. Window parsing ──────────────────────────────────────────────────────


def test_parse_window_days() -> None:
    assert _parse_window("7d") == timedelta(days=7)


def test_parse_window_hours() -> None:
    assert _parse_window("24h") == timedelta(hours=24)


def test_parse_window_minutes() -> None:
    assert _parse_window("30m") == timedelta(minutes=30)


def test_parse_window_all_returns_none() -> None:
    assert _parse_window("all") is None
    assert _parse_window("ALL") is None


def test_parse_window_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_window("forever")
    with pytest.raises(ValueError):
        _parse_window("7y")  # unsupported unit
    with pytest.raises(ValueError):
        _parse_window("0d")
    with pytest.raises(ValueError):
        _parse_window("-1d")


# ── 2. Basic round-trip ────────────────────────────────────────────────────


def test_unknown_agent_returns_zero(ledger: ReciprocityLedger) -> None:
    """Per the contract: never seen → 0.0, not undefined, not error."""
    assert ledger.exchange_rate("nobody") == 0.0
    stats = ledger.stats("nobody")
    assert stats.carbon_received == 0.0
    assert stats.nutrients_returned == 0.0
    assert stats.exchange_rate == 0.0
    assert stats.created_at is None


def test_round_trip_request_and_contribution(ledger: ReciprocityLedger) -> None:
    assert ledger.record_request("alice", cost=4.0) is True
    assert ledger.record_contribution("alice", nutrient_value=2.0, nutrient_id="nut-1") is True

    stats = ledger.stats("alice")
    assert stats.carbon_received == pytest.approx(4.0)
    assert stats.nutrients_returned == pytest.approx(2.0)
    assert stats.exchange_rate == pytest.approx(0.5)
    assert stats.request_count == 1
    assert stats.contribution_count == 1
    assert stats.last_seen_at is not None
    assert stats.created_at is not None


def test_contribution_without_requests_uses_epsilon(ledger: ReciprocityLedger) -> None:
    """An agent that contributes but is never charged should still rank
    finite (epsilon floor in the denominator). This matters because Session
    2's downstream-reference credit will arrive before any new request."""
    ledger.record_contribution("bob", nutrient_value=1.0, nutrient_id="nut-2")
    rate = ledger.exchange_rate("bob")
    assert rate > 0.0
    assert rate == pytest.approx(1000.0)  # 1.0 / 1e-3


def test_zero_cost_request_still_touches_agent(ledger: ReciprocityLedger) -> None:
    """Zero-cost requests record liveness without inflating the denominator."""
    ledger.record_request("carol", cost=0.0)
    assert "carol" in ledger.all_agent_ids()
    stats = ledger.stats("carol")
    assert stats.request_count == 1
    assert stats.carbon_received == pytest.approx(0.0)


def test_negative_values_rejected(ledger: ReciprocityLedger) -> None:
    with pytest.raises(ValueError):
        ledger.record_request("alice", cost=-1.0)
    with pytest.raises(ValueError):
        ledger.record_contribution("alice", nutrient_value=-1.0)


def test_empty_agent_id_rejected(ledger: ReciprocityLedger) -> None:
    with pytest.raises(ValueError):
        ledger.record_request("", cost=1.0)
    with pytest.raises(ValueError):
        ledger.record_contribution("", nutrient_value=1.0)


# ── 3. Idempotency ─────────────────────────────────────────────────────────


def test_idempotency_key_skips_duplicates(ledger: ReciprocityLedger) -> None:
    assert ledger.record_request("alice", cost=2.0, idempotency_key="req-1") is True
    assert ledger.record_request("alice", cost=2.0, idempotency_key="req-1") is False
    assert ledger.record_request("alice", cost=2.0, idempotency_key="req-1") is False

    stats = ledger.stats("alice")
    assert stats.carbon_received == pytest.approx(2.0)
    assert stats.request_count == 1


def test_idempotency_key_namespace_isolation(ledger: ReciprocityLedger) -> None:
    """Different agents using the same key collide — that's the intended
    semantics: keys are global across event types and agents (a single
    UNIQUE index). Callers prefix their keys with their domain."""
    assert ledger.record_request("alice", cost=1.0, idempotency_key="shared") is True
    assert ledger.record_request("bob", cost=1.0, idempotency_key="shared") is False
    # Convention: prefix to avoid collisions.
    assert ledger.record_request("alice", cost=1.0, idempotency_key="alice:r1") is True
    assert ledger.record_request("bob", cost=1.0, idempotency_key="bob:r1") is True


def test_null_idempotency_key_never_collides(ledger: ReciprocityLedger) -> None:
    """Partial UNIQUE index only constrains non-null keys."""
    for _ in range(5):
        assert ledger.record_request("alice", cost=1.0) is True
    assert ledger.stats("alice").request_count == 5
    assert ledger.stats("alice").carbon_received == pytest.approx(5.0)


def test_contribution_dedup_by_nutrient_id_pattern(ledger: ReciprocityLedger) -> None:
    """The decomposer's keying convention 'contrib:<nutrient_id>' should
    survive replays of the decomposer for the same build."""
    ledger.record_contribution(
        "engine", nutrient_value=1.0, nutrient_id="abc", idempotency_key="contrib:abc"
    )
    ledger.record_contribution(
        "engine", nutrient_value=1.0, nutrient_id="abc", idempotency_key="contrib:abc"
    )
    assert ledger.stats("engine").nutrients_returned == pytest.approx(1.0)
    assert ledger.stats("engine").contribution_count == 1


# ── 4. Window aggregation ──────────────────────────────────────────────────


def test_window_excludes_old_events(ledger: ReciprocityLedger) -> None:
    """Events older than the window must not appear in aggregation."""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    recent = now - timedelta(days=2)

    ledger.record_request("alice", cost=10.0, ts=old)
    ledger.record_request("alice", cost=2.0, ts=recent)
    ledger.record_contribution("alice", nutrient_value=1.0, nutrient_id="n-old", ts=old)
    ledger.record_contribution("alice", nutrient_value=3.0, nutrient_id="n-recent", ts=recent)

    # 7-day window: only the recent pair.
    stats_7d = ledger.stats("alice", window="7d")
    assert stats_7d.carbon_received == pytest.approx(2.0)
    assert stats_7d.nutrients_returned == pytest.approx(3.0)
    assert stats_7d.exchange_rate == pytest.approx(1.5)

    # All-time: full totals.
    stats_all = ledger.stats("alice", window="all")
    assert stats_all.carbon_received == pytest.approx(12.0)
    assert stats_all.nutrients_returned == pytest.approx(4.0)


def test_window_one_day_isolates_from_two_day_old(ledger: ReciprocityLedger) -> None:
    now = datetime.now(timezone.utc)
    two_days_ago = now - timedelta(days=2)
    one_hour_ago = now - timedelta(hours=1)
    ledger.record_contribution("alice", nutrient_value=5.0, nutrient_id="old", ts=two_days_ago)
    ledger.record_contribution("alice", nutrient_value=2.0, nutrient_id="recent", ts=one_hour_ago)
    assert ledger.stats("alice", window="1d").nutrients_returned == pytest.approx(2.0)
    assert ledger.stats("alice", window="3d").nutrients_returned == pytest.approx(7.0)


# ── 5. Ranking & threshold ─────────────────────────────────────────────────


def test_rank_agents_sorted_descending_by_exchange_rate(
    ledger: ReciprocityLedger,
) -> None:
    # alice: contribution-heavy (high rate)
    ledger.record_request("alice", cost=1.0)
    ledger.record_contribution("alice", nutrient_value=5.0, nutrient_id="a1")
    # bob: parasitic (much lower rate)
    ledger.record_request("bob", cost=10.0)
    ledger.record_contribution("bob", nutrient_value=1.0, nutrient_id="b1")
    # carol: nothing (zero rate)
    ledger.record_request("carol", cost=2.0)

    rows = ledger.rank_agents()
    assert [r.agent_id for r in rows] == ["alice", "bob", "carol"]
    assert rows[0].exchange_rate > rows[1].exchange_rate > rows[2].exchange_rate


def test_agents_below_threshold(ledger: ReciprocityLedger) -> None:
    ledger.record_request("hot", cost=1.0)
    ledger.record_contribution("hot", nutrient_value=10.0, nutrient_id="h")
    ledger.record_request("cold", cost=10.0)
    ledger.record_contribution("cold", nutrient_value=0.1, nutrient_id="c")
    ledger.record_request("dead", cost=5.0)

    below = ledger.agents_below_threshold(threshold=0.1)
    ids = [s.agent_id for s in below]
    assert "dead" in ids
    # 'cold' has rate 0.01 < 0.1 → in the set
    assert "cold" in ids
    # 'hot' has rate 10.0 > 0.1 → not in the set
    assert "hot" not in ids


# ── 6. Durability ──────────────────────────────────────────────────────────


def test_writes_persist_across_reopen(tmp_path: Path) -> None:
    """Closing and reopening the ledger at the same path must show the
    same totals — this is the persistence guarantee."""
    db_path = tmp_path / "reciprocity.db"
    a = ReciprocityLedger(db_path=db_path)
    a.record_request("alice", cost=3.0)
    a.record_contribution("alice", nutrient_value=2.0, nutrient_id="nut-1")
    a.close()

    b = ReciprocityLedger(db_path=db_path)
    stats = b.stats("alice")
    b.close()

    assert stats.carbon_received == pytest.approx(3.0)
    assert stats.nutrients_returned == pytest.approx(2.0)


# ── 7. Concurrency ─────────────────────────────────────────────────────────


def test_concurrent_writes_do_not_corrupt(ledger: ReciprocityLedger) -> None:
    """Many threads writing simultaneously must not corrupt counts or
    crash on SQLite locking. SQLite WAL + the process-local lock should
    serialize cleanly."""
    n_threads = 8
    writes_per_thread = 25
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(writes_per_thread):
                # Unique idempotency_keys so every write counts.
                ledger.record_request(
                    "alice",
                    cost=1.0,
                    idempotency_key=f"t{tid}-r{i}",
                )
                ledger.record_contribution(
                    "alice",
                    nutrient_value=1.0,
                    nutrient_id=f"t{tid}-n{i}",
                    idempotency_key=f"t{tid}-c{i}",
                )
        except BaseException as exc:  # pragma: no cover — diagnostic
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors!r}"
    expected = n_threads * writes_per_thread
    stats = ledger.stats("alice")
    assert stats.request_count == expected
    assert stats.contribution_count == expected
    assert stats.carbon_received == pytest.approx(float(expected))
    assert stats.nutrients_returned == pytest.approx(float(expected))


def test_concurrent_idempotency_no_double_counting(ledger: ReciprocityLedger) -> None:
    """Two threads racing on the same idempotency key both attempt
    insertion; SQLite's unique constraint ensures exactly one wins."""
    barrier = threading.Barrier(2)
    results: list[bool] = []
    lock = threading.Lock()

    def racer() -> None:
        barrier.wait()
        outcome = ledger.record_request("alice", cost=1.0, idempotency_key="race-key")
        with lock:
            results.append(outcome)

    t1 = threading.Thread(target=racer)
    t2 = threading.Thread(target=racer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results.count(True) == 1
    assert results.count(False) == 1
    assert ledger.stats("alice").request_count == 1


# ── 8. CLI rendering ───────────────────────────────────────────────────────


def test_cli_format_empty_ledger(ledger: ReciprocityLedger) -> None:
    out = cli_format_ledger(ledger)
    assert "Reciprocity ledger" in out
    assert "0 known agents" in out
    assert "(no events recorded yet" in out


def test_cli_format_populated(ledger: ReciprocityLedger) -> None:
    ledger.record_request("alice", cost=2.0)
    ledger.record_contribution("alice", nutrient_value=4.0, nutrient_id="x")
    out = cli_format_ledger(ledger)
    assert "alice" in out
    assert "exchange" in out  # header
    # Counts column showed up.
    assert "  1" in out  # request_count rendered


def test_default_window_constant_is_7d() -> None:
    """Sanity: the documented default matches the constant."""
    assert DEFAULT_WINDOW == "7d"


# ── 9. AgentStats dataclass ────────────────────────────────────────────────


def test_agent_stats_to_dict_round_trip(ledger: ReciprocityLedger) -> None:
    ledger.record_request("alice", cost=1.0)
    ledger.record_contribution("alice", nutrient_value=1.0, nutrient_id="x")
    d = ledger.stats("alice").to_dict()
    assert d["agent_id"] == "alice"
    assert d["carbon_received"] == pytest.approx(1.0)
    assert d["nutrients_returned"] == pytest.approx(1.0)
    assert d["exchange_rate"] == pytest.approx(1.0)
    assert d["window"] == DEFAULT_WINDOW
    assert d["last_seen_at"] is not None


# ── 10. SQLite schema sanity ───────────────────────────────────────────────


def test_partial_unique_index_supported_by_sqlite() -> None:
    """If this fails, the host SQLite is < 3.8 and the idempotency
    guarantee is silently weaker. Surface it as a clear test failure."""
    v = sqlite3.sqlite_version_info
    assert v >= (3, 8, 0), f"need sqlite >= 3.8 for partial indexes, got {v}"
