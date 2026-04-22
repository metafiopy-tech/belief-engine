"""CostTracker: pricing, windows, caps, threaded writes."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from belief.photosynthesis.safety.cost_tracker import (
    BreakerAnthropic,
    BudgetExceeded,
    CostTracker,
    Usage,
    price_usd,
)


@pytest.fixture()
def tracker(tmp_path: Path) -> CostTracker:
    return CostTracker(
        db_path=tmp_path / "costs.db",
        daily_cap_usd=1.0,
        weekly_cap_usd=5.0,
        monthly_cap_usd=20.0,
    )


def test_price_usd_known_model() -> None:
    cost = price_usd("claude-haiku-4-5-20251001", Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    # Haiku: $1 input + $5 output per M tokens
    assert cost == pytest.approx(6.0, rel=1e-3)


def test_price_usd_unknown_model_zero() -> None:
    assert price_usd("claude-never-released", Usage(input_tokens=1000, output_tokens=1000)) == 0.0


def test_record_and_total(tracker: CostTracker) -> None:
    tracker.record(
        model="claude-haiku-4-5-20251001",
        input_tokens=100_000,
        output_tokens=50_000,
        tag="test",
    )
    assert tracker.total() > 0
    assert tracker.total(tag="test") == tracker.total()
    assert tracker.total(tag="other") == 0.0


def test_spent_windows_are_narrowing(tracker: CostTracker) -> None:
    tracker.record(model="claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
    day = tracker.spent("1 day")
    week = tracker.spent("7 day")
    month = tracker.spent("30 day")
    # Same call lands in every longer window
    assert day > 0
    assert week >= day
    assert month >= week


def test_check_raises_budget_exceeded_at_cap(tracker: CostTracker) -> None:
    # Burn the whole daily cap in one go (Haiku: 1M input tokens = $1)
    tracker.record(model="claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
    with pytest.raises(BudgetExceeded):
        tracker.check(projected_cost_usd=0.01)


def test_under_budget_mirrors_check(tracker: CostTracker) -> None:
    assert tracker.under_budget(projected_cost_usd=0.10) is True
    tracker.record(model="claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
    assert tracker.under_budget(projected_cost_usd=0.10) is False


def test_threaded_writes_no_race(tmp_path: Path) -> None:
    t = CostTracker(db_path=tmp_path / "costs.db", daily_cap_usd=100.0)

    def worker(i: int) -> None:
        for _ in range(20):
            t.record(
                model="claude-haiku-4-5-20251001",
                input_tokens=1000,
                output_tokens=100,
                tag=f"w{i}",
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # 4 threads x 20 writes = 80 rows total
    with t.conn() as c:
        n = c.execute("SELECT COUNT(*) AS n FROM calls;").fetchone()["n"]
    assert int(n) == 80


# ---------------------------------------------------------------------------
# BreakerAnthropic — duck-typed fake client
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, inp: int, out: int) -> None:
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class FakeResp:
    def __init__(self, usage: FakeUsage) -> None:
        self.usage = usage


class FakeMessages:
    def __init__(self, resp: FakeResp) -> None:
        self.resp = resp
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp


class FakeClient:
    def __init__(self, resp: FakeResp) -> None:
        self.messages = FakeMessages(resp)


def test_breaker_anthropic_records_cost(tmp_path: Path) -> None:
    t = CostTracker(db_path=tmp_path / "costs.db", daily_cap_usd=100.0)
    resp = FakeResp(FakeUsage(500, 100))
    client = FakeClient(resp)
    breaker = BreakerAnthropic(client, tracker=t)

    out = breaker.create(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=50,
        tag="smoke",
    )
    assert out is resp
    assert t.total(tag="smoke") > 0


def test_breaker_anthropic_budget_raises_before_call(tmp_path: Path) -> None:
    t = CostTracker(
        db_path=tmp_path / "costs.db",
        daily_cap_usd=0.0001,  # cap so low that any call would cross it
    )
    resp = FakeResp(FakeUsage(100, 100))
    client = FakeClient(resp)
    breaker = BreakerAnthropic(client, tracker=t)

    with pytest.raises(BudgetExceeded):
        breaker.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=50,
            projected_cost_usd=0.10,
        )
    # No call should have actually been issued
    assert client.messages.calls == []
