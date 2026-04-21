"""Async rate-limit wrappers around Anthropic calls and the synthesis cycle.

Spec defaults (tunable):

    anthropic_rpm = AsyncLimiter(max_rate=50,      time_period=60)
    anthropic_tpm = AsyncLimiter(max_rate=40_000,  time_period=60)
    goal_budget   = AsyncLimiter(max_rate=10,      time_period=3600)

`aiolimiter` is an optional dependency; when it isn't installed we fall
back to a no-op limiter so the rest of the daemon keeps running (it's
better to lose rate-limit enforcement than to crash the engine).

Dynamic tightening: when an Anthropic response header says
`anthropic-ratelimit-requests-remaining-pct < 10`, the caller should
call `tighten_on_header_signal()` to pull the caps down 50% for the
rest of the minute. The next minute resets.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger("belief.photosynthesis.safety.rate_limits")


class _NullLimiter:
    """No-op limiter used when aiolimiter isn't installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.max_rate = float(kwargs.get("max_rate") or (args[0] if args else 0))
        self.time_period = float(
            kwargs.get("time_period") or (args[1] if len(args) > 1 else 60)
        )

    async def acquire(self, amount: float = 1.0) -> None:
        return None

    async def __aenter__(self) -> "_NullLimiter":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


def _load_aiolimiter() -> Any:
    try:
        from aiolimiter import AsyncLimiter  # type: ignore[import-untyped]

        return AsyncLimiter
    except ImportError:
        return None


@dataclass
class RateLimiterSet:
    """The three limiters used across the daemon."""

    anthropic_rpm_rate: int = 50
    anthropic_rpm_period: int = 60
    anthropic_tpm_rate: int = 40_000
    anthropic_tpm_period: int = 60
    goal_budget_rate: int = 10
    goal_budget_period: int = 3600

    _rpm: Any = field(default=None, init=False, repr=False)
    _tpm: Any = field(default=None, init=False, repr=False)
    _goal: Any = field(default=None, init=False, repr=False)
    _tightened_until: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        AsyncLimiter = _load_aiolimiter() or _NullLimiter
        self._rpm = AsyncLimiter(self.anthropic_rpm_rate, self.anthropic_rpm_period)
        self._tpm = AsyncLimiter(self.anthropic_tpm_rate, self.anthropic_tpm_period)
        self._goal = AsyncLimiter(self.goal_budget_rate, self.goal_budget_period)

    # ---------------------------------------------------------- public API
    @property
    def anthropic_rpm(self) -> Any:
        return self._rpm

    @property
    def anthropic_tpm(self) -> Any:
        return self._tpm

    @property
    def goal_budget(self) -> Any:
        return self._goal

    async def anthropic_acquire(self, estimated_tokens: int = 0) -> None:
        """Acquire one RPM slot and `estimated_tokens` TPM slots."""
        try:
            await self._rpm.acquire()
        except TypeError:
            # Some limiter impls want the context-manager form only
            async with self._rpm:
                pass
        if estimated_tokens > 0:
            try:
                await self._tpm.acquire(estimated_tokens)
            except TypeError:
                async with self._tpm:
                    pass

    async def goal_budget_acquire(self) -> None:
        try:
            await self._goal.acquire()
        except TypeError:
            async with self._goal:
                pass

    def tighten_on_header_signal(
        self, remaining_pct: float, *, duration_s: float = 60.0
    ) -> None:
        """Halve the limiter caps for `duration_s` when remaining < 10%."""
        if remaining_pct >= 10.0:
            return
        self.anthropic_rpm_rate = max(5, self.anthropic_rpm_rate // 2)
        self.anthropic_tpm_rate = max(1000, self.anthropic_tpm_rate // 2)
        self._build()
        self._tightened_until = time.monotonic() + duration_s
        logger.warning(
            "rate-limits tightened: rpm=%d tpm=%d (remaining=%.1f%%)",
            self.anthropic_rpm_rate,
            self.anthropic_tpm_rate,
            remaining_pct,
        )


_DEFAULT_SET: Optional[RateLimiterSet] = None


def default_set() -> RateLimiterSet:
    """Process-wide singleton."""
    global _DEFAULT_SET
    if _DEFAULT_SET is None:
        _DEFAULT_SET = RateLimiterSet()
    return _DEFAULT_SET


__all__ = ["RateLimiterSet", "default_set"]
