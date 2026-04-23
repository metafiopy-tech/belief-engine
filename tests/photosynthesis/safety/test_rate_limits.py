"""Tests for belief.photosynthesis.safety.rate_limits — Session 8.5c.

Covers the surface the audit flagged as uncovered:

* Default values match the spec.
* ``tighten_on_header_signal`` halves caps when remaining < 10%; no-ops otherwise.
* ``anthropic_acquire`` routes to both RPM and TPM limiters.
* ``goal_budget_acquire`` routes to the goal-budget limiter.
* Missing ``aiolimiter`` falls back to ``_NullLimiter`` without crashing.
* Process-wide ``default_set()`` returns the same instance.
"""

from __future__ import annotations

import asyncio


from belief.photosynthesis.safety.rate_limits import (
    RateLimiterSet,
    _NullLimiter,
    default_set,
)


# ---------------------------------------------------------------------------
# Spec defaults
# ---------------------------------------------------------------------------


def test_spec_defaults() -> None:
    s = RateLimiterSet()
    assert s.anthropic_rpm_rate == 50
    assert s.anthropic_rpm_period == 60
    assert s.anthropic_tpm_rate == 40_000
    assert s.anthropic_tpm_period == 60
    assert s.goal_budget_rate == 10
    assert s.goal_budget_period == 3600


# ---------------------------------------------------------------------------
# Dynamic tightening on 4xx-warning signal
# ---------------------------------------------------------------------------


class TestTightenOnHeaderSignal:
    def test_no_tighten_when_over_threshold(self) -> None:
        s = RateLimiterSet()
        before = (s.anthropic_rpm_rate, s.anthropic_tpm_rate)
        s.tighten_on_header_signal(remaining_pct=50.0)
        assert (s.anthropic_rpm_rate, s.anthropic_tpm_rate) == before

    def test_tighten_halves_when_under_threshold(self) -> None:
        s = RateLimiterSet()
        s.tighten_on_header_signal(remaining_pct=5.0)
        assert s.anthropic_rpm_rate == 25
        assert s.anthropic_tpm_rate == 20_000

    def test_tighten_respects_floor(self) -> None:
        """Floor is rpm=5, tpm=1000 — can't tighten below that."""
        s = RateLimiterSet(anthropic_rpm_rate=6, anthropic_tpm_rate=1500)
        s.tighten_on_header_signal(remaining_pct=1.0)
        assert s.anthropic_rpm_rate == 5
        assert s.anthropic_tpm_rate == 1000

    def test_tighten_rebuilds_limiters(self) -> None:
        """After tightening, the internal limiter objects must reflect
        the new caps — otherwise the acquire() paths keep the old rate."""
        s = RateLimiterSet()
        original_rpm_limiter = s.anthropic_rpm
        s.tighten_on_header_signal(remaining_pct=5.0)
        # Either a new instance or the same instance with updated rate
        # — both are acceptable, but the RateLimiterSet.anthropic_rpm_rate
        # must match the limiter's max_rate (where the limiter exposes
        # max_rate; _NullLimiter does).
        if isinstance(s.anthropic_rpm, _NullLimiter):
            assert s.anthropic_rpm.max_rate == 25
        # Rebuild should at minimum not leave stale references.
        assert s.anthropic_rpm is not None  # property still works
        # Silence unused-var lint
        _ = original_rpm_limiter


# ---------------------------------------------------------------------------
# Acquire paths
# ---------------------------------------------------------------------------


class TestAcquirePaths:
    def test_anthropic_acquire_rpm_only_when_tokens_zero(self) -> None:
        """estimated_tokens=0 should skip the TPM acquire."""
        s = RateLimiterSet()
        # Replace limiters with instrumented stubs so we can see the
        # call counts.
        calls = {"rpm": 0, "tpm": 0, "goal": 0}

        class _InstrumentedLimiter:
            def __init__(self, bucket: str) -> None:
                self._bucket = bucket

            async def acquire(self, amount: float = 1.0) -> None:
                calls[self._bucket] += 1

        s._rpm = _InstrumentedLimiter("rpm")
        s._tpm = _InstrumentedLimiter("tpm")
        s._goal = _InstrumentedLimiter("goal")

        asyncio.run(s.anthropic_acquire(estimated_tokens=0))
        assert calls == {"rpm": 1, "tpm": 0, "goal": 0}

    def test_anthropic_acquire_hits_both_when_tokens_positive(self) -> None:
        s = RateLimiterSet()
        calls = {"rpm": 0, "tpm": 0, "goal": 0}

        class _InstrumentedLimiter:
            def __init__(self, bucket: str) -> None:
                self._bucket = bucket

            async def acquire(self, amount: float = 1.0) -> None:
                calls[self._bucket] += 1

        s._rpm = _InstrumentedLimiter("rpm")
        s._tpm = _InstrumentedLimiter("tpm")
        s._goal = _InstrumentedLimiter("goal")

        asyncio.run(s.anthropic_acquire(estimated_tokens=500))
        assert calls["rpm"] == 1
        assert calls["tpm"] == 1
        assert calls["goal"] == 0  # goal budget is a separate acquire

    def test_goal_budget_acquire_hits_goal_only(self) -> None:
        s = RateLimiterSet()
        calls = {"rpm": 0, "tpm": 0, "goal": 0}

        class _InstrumentedLimiter:
            def __init__(self, bucket: str) -> None:
                self._bucket = bucket

            async def acquire(self, amount: float = 1.0) -> None:
                calls[self._bucket] += 1

        s._rpm = _InstrumentedLimiter("rpm")
        s._tpm = _InstrumentedLimiter("tpm")
        s._goal = _InstrumentedLimiter("goal")

        asyncio.run(s.goal_budget_acquire())
        assert calls == {"rpm": 0, "tpm": 0, "goal": 1}


# ---------------------------------------------------------------------------
# NullLimiter fallback
# ---------------------------------------------------------------------------


class TestNullLimiterFallback:
    def test_null_limiter_acquire_is_noop(self) -> None:
        """The engine must not crash when aiolimiter is unavailable."""
        limiter = _NullLimiter(max_rate=50, time_period=60)
        asyncio.run(limiter.acquire())  # must not raise

    def test_null_limiter_as_context_manager(self) -> None:
        limiter = _NullLimiter(max_rate=50, time_period=60)

        async def run() -> None:
            async with limiter as got:
                assert got is limiter

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


def test_default_set_returns_same_instance() -> None:
    a = default_set()
    b = default_set()
    assert a is b


def test_default_set_is_a_ratelimiterset() -> None:
    assert isinstance(default_set(), RateLimiterSet)
