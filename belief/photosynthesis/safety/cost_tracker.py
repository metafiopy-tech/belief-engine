"""SQLite-backed cost tracker + BreakerAnthropic wrapper.

**Scope: Photosynthesis daemon only.**

This module is the spend-tracking and circuit-breaker layer for the
*scheduled goal-synthesis* subsystem (``belief/photosynthesis/``).  It
is NOT the build-pipeline LLM dispatcher — main-pipeline agents route
all LLM calls through :mod:`belief.llm`, which has its own retry and
per-role budget semantics.

**Why two LLM dispatchers** (as of 2026-04-23): photosynthesis runs as
a background daemon with a hard daily-budget cap reconciled against
Anthropic's Admin API, and its per-call cost metering has to be
authoritative — any drift makes the cap meaningless.  The main
pipeline's dispatcher optimises for per-role latency and graceful
degradation instead.  The two paths are tracked for eventual
consolidation — see ``docs/architecture/http_boundary.md``.

``BreakerAnthropic`` is structurally typed (``HasMessagesCreate`` Protocol)
so tests can inject fakes without installing ``anthropic``.  The class
is **instantiated only** from ``belief/photosynthesis/`` (it is the
daemon's dispatcher); it may be **passed through** to downstream
helpers such as :func:`belief.memory.library_inductor.promote_eligible`
when the daemon drives them.  Instantiating it from the main pipeline
is a bug.

Source of truth for Photosynthesis spend. Anthropic's Admin API is
eventually consistent; this file is always current.

Schema (WAL, one write-path):

    calls(
        ts          INTEGER  -- unix seconds
        model       TEXT
        in_tok      INTEGER
        out_tok     INTEGER
        cache_r     INTEGER  -- cache read tokens
        cache_w     INTEGER  -- cache write tokens
        cost        REAL     -- USD
        tag         TEXT
    )

BudgetExceeded is raised before a call that would push any window
(daily/weekly/monthly) over its cap. The caller must handle it — the
tracker never silently drops.

BreakerAnthropic is the runtime wrapper around an anthropic.Anthropic
instance. It's deliberately typed structurally (`HasMessagesCreate`
Protocol) so tests can inject a fake without installing anthropic. The
breaker gates before the call; the tracker records after.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Optional, Protocol


logger = logging.getLogger("belief.photosynthesis.safety.cost_tracker")


class BudgetExceeded(RuntimeError):
    """Raised when a call would cross a daily/weekly/monthly cap."""


# ---------------------------------------------------------------------------
# Pricing — USD per million tokens.
# Refresh weekly from LiteLLM's prices JSON; numbers below are the Session 5
# starting point and DO NOT need to match current Anthropic list prices
# byte-for-byte to be useful (BudgetExceeded still fires correctly).
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # Canonical short names used by Session 4 code
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
    "claude-opus-4-6": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
}


def _per_token(price_per_million: float) -> float:
    return price_per_million / 1_000_000.0


SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       INTEGER NOT NULL,
    model    TEXT    NOT NULL,
    in_tok   INTEGER NOT NULL DEFAULT 0,
    out_tok  INTEGER NOT NULL DEFAULT 0,
    cache_r  INTEGER NOT NULL DEFAULT 0,
    cache_w  INTEGER NOT NULL DEFAULT 0,
    cost     REAL    NOT NULL,
    tag      TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls(ts);
CREATE INDEX IF NOT EXISTS idx_calls_tag ON calls(tag);
"""


@dataclass
class Usage:
    """Normalized usage across Anthropic response shapes."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_anthropic(cls, usage: Any) -> "Usage":
        """Pull token counts off an Anthropic response.usage object."""
        if usage is None:
            return cls()
        g = getattr
        return cls(
            input_tokens=int(g(usage, "input_tokens", 0) or 0),
            output_tokens=int(g(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(g(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(g(usage, "cache_creation_input_tokens", 0) or 0),
        )


def price_usd(model: str, usage: Usage) -> float:
    """Return total USD for a single call given token usage."""
    p = PRICING.get(model)
    if p is None:
        logger.warning("no pricing entry for model %r; assuming zero cost", model)
        return 0.0
    cost = (
        _per_token(p["input"]) * usage.input_tokens
        + _per_token(p["output"]) * usage.output_tokens
        + _per_token(p.get("cache_read", 0.0)) * usage.cache_read_input_tokens
        + _per_token(p.get("cache_write", 0.0)) * usage.cache_creation_input_tokens
    )
    return cost


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


DEFAULT_COSTS_DB = Path("/var/lib/photosynthesis/costs.db")
WARN_FRACTION = 0.80


class CostTracker:
    """Session 5 real implementation. Preserves Session 4 stub surface.

    Threading: sqlite3.Connection isn't shared across threads, so every
    method opens its own connection via conn(). The writer lock comes
    from SQLite's BEGIN IMMEDIATE.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_COSTS_DB,
        *,
        daily_cap_usd: float = 5.0,
        weekly_cap_usd: float = 25.0,
        monthly_cap_usd: float = 80.0,
    ) -> None:
        self.db_path = str(db_path)
        self.daily_cap_usd = float(daily_cap_usd)
        self.weekly_cap_usd = float(weekly_cap_usd)
        self.monthly_cap_usd = float(monthly_cap_usd)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("PRAGMA synchronous = NORMAL;")
            c.execute("PRAGMA busy_timeout = 5000;")
            c.row_factory = sqlite3.Row
            yield c
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self.conn() as c:
            c.executescript(SCHEMA)

    # ---------------------------------------------------------------- record
    def record(
        self,
        *,
        model: str,
        usage: Usage | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_r: int = 0,
        cache_w: int = 0,
        cost_usd: Optional[float] = None,
        tag: str = "",
    ) -> float:
        """Insert one call row. Returns the computed cost.

        Accepts either a Usage object or raw token kwargs (for tests).
        If cost_usd is provided, it's trusted; otherwise we compute it
        from the PRICING table.
        """
        if usage is None:
            usage = Usage(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cache_read_input_tokens=int(cache_r),
                cache_creation_input_tokens=int(cache_w),
            )
        cost = price_usd(model, usage) if cost_usd is None else float(cost_usd)

        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "INSERT INTO calls(ts, model, in_tok, out_tok, cache_r, cache_w, cost, tag) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?);",
                (
                    int(time.time()),
                    model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_input_tokens,
                    usage.cache_creation_input_tokens,
                    cost,
                    tag,
                ),
            )
            c.execute("COMMIT;")
        return cost

    # ---------------------------------------------------------------- spend
    def spent(self, window: str = "1 day") -> float:
        """Sum of cost over a trailing window. Supports '1 day', '7 day',
        '30 day' (strftime math under the hood).

        Returns 0.0 on any parse error rather than raising.
        """
        seconds = _window_seconds(window)
        if seconds is None:
            return 0.0
        since = int(time.time()) - seconds
        with self.conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(cost), 0.0) AS s FROM calls WHERE ts >= ?;",
                (since,),
            ).fetchone()
            return float(row["s"]) if row else 0.0

    def check(self, projected_cost_usd: float = 0.0) -> None:
        """Raise BudgetExceeded if any window is at or beyond its cap.

        WARN-log at 80% of any cap. Callers should invoke check()
        BEFORE an LLM call, using `projected_cost_usd` when they have
        a reasonable estimate (Sonnet: ~$0.05 for a 1k-prompt spec gen).
        """
        day = self.spent("1 day")
        week = self.spent("7 day")
        month = self.spent("30 day")

        # Absolute check first
        checks = (
            ("daily", day + projected_cost_usd, self.daily_cap_usd),
            ("weekly", week + projected_cost_usd, self.weekly_cap_usd),
            ("monthly", month + projected_cost_usd, self.monthly_cap_usd),
        )
        for label, projected, cap in checks:
            if cap <= 0:
                continue
            if projected >= cap:
                raise BudgetExceeded(
                    f"{label} cap ${cap:.2f} would be crossed by a "
                    f"${projected_cost_usd:.4f} call "
                    f"(current {label} total ${projected - projected_cost_usd:.4f})"
                )
            if projected >= cap * WARN_FRACTION:
                logger.warning(
                    "cost tracker: %s spend at %.1f%% of $%.2f cap",
                    label,
                    100.0 * projected / cap,
                    cap,
                )

    # Preserve the Session-4-stub method name.
    def under_budget(self, projected_cost_usd: float = 0.0, tag: str = "") -> bool:
        try:
            self.check(projected_cost_usd)
        except BudgetExceeded:
            return False
        return True

    # Preserve stub's `total` method name for any caller that used it.
    def total(self, tag: Optional[str] = None) -> float:
        with self.conn() as c:
            if tag is None:
                row = c.execute("SELECT COALESCE(SUM(cost), 0.0) AS s FROM calls;").fetchone()
            else:
                row = c.execute(
                    "SELECT COALESCE(SUM(cost), 0.0) AS s FROM calls WHERE tag = ?;",
                    (tag,),
                ).fetchone()
            return float(row["s"]) if row else 0.0


def _window_seconds(window: str) -> Optional[int]:
    parts = window.strip().split()
    if len(parts) != 2:
        return None
    try:
        n = int(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower().rstrip("s")
    if unit in {"day", "d"}:
        return n * 86400
    if unit in {"hour", "hr", "h"}:
        return n * 3600
    if unit in {"minute", "min", "m"}:
        return n * 60
    return None


# ---------------------------------------------------------------------------
# BreakerAnthropic — Protocol-typed wrapper around anthropic.Anthropic
# ---------------------------------------------------------------------------


class HasMessagesCreate(Protocol):
    """Structural type matching anthropic.Anthropic's messages.create interface."""

    class _MessagesNS(Protocol):
        def create(self, **kwargs: Any) -> Any: ...  # noqa: E704

    messages: "_MessagesNS"


@dataclass
class BreakerConfig:
    fail_max: int = 5
    reset_timeout: float = 60.0


class BreakerAnthropic:
    """Wrap an anthropic client so every call is gated by budget + breaker.

    Usage:

        from anthropic import Anthropic
        client = Anthropic()
        breaker = BreakerAnthropic(client, tracker=cost_tracker)
        resp = breaker.create(
            model="claude-haiku-4-5-20251001",
            messages=[...],
            max_tokens=500,
            tag="novelty_judge",
        )

    Tests pass a fake client that quacks like Anthropic. The breaker
    counts 5xx / timeout / connection errors; 4xx and auth errors are
    excluded.
    """

    def __init__(
        self,
        client: HasMessagesCreate,
        *,
        tracker: CostTracker,
        breaker_config: Optional[BreakerConfig] = None,
    ) -> None:
        self.client = client
        self.tracker = tracker
        self._breaker_config = breaker_config or BreakerConfig()
        self._breaker: Any = None

    def _ensure_breaker(self) -> None:
        if self._breaker is not None:
            return
        try:
            import pybreaker  # type: ignore[import-untyped]
        except ImportError:
            # Degrade gracefully — no breaker, still tracks cost.
            self._breaker = _PassthroughBreaker()
            return

        # Exclude anthropic client errors that shouldn't trip the breaker.
        excluded: list[type[BaseException]] = []
        for name in (
            "BadRequestError",
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
        ):
            try:
                mod = __import__("anthropic", fromlist=[name])
                excluded.append(getattr(mod, name))
            except (ImportError, AttributeError):
                pass

        self._breaker = pybreaker.CircuitBreaker(
            fail_max=self._breaker_config.fail_max,
            reset_timeout=self._breaker_config.reset_timeout,
            exclude=excluded or None,
        )

    def create(
        self,
        *,
        model: str,
        tag: str = "",
        projected_cost_usd: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """Synchronous messages.create, budget-checked + breaker-gated."""
        self._ensure_breaker()
        self.tracker.check(projected_cost_usd=projected_cost_usd)

        def _call() -> Any:
            return self.client.messages.create(model=model, **kwargs)

        resp = self._breaker.call(_call)
        usage = Usage.from_anthropic(getattr(resp, "usage", None))
        self.tracker.record(model=model, usage=usage, tag=tag)
        return resp


class _PassthroughBreaker:
    """Fallback when pybreaker isn't installed — behaves as no-op."""

    def call(self, fn: Callable[[], Any]) -> Any:
        return fn()

    async def call_async(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        return await fn()


__all__ = [
    "BreakerAnthropic",
    "BreakerConfig",
    "BudgetExceeded",
    "CostTracker",
    "DEFAULT_COSTS_DB",
    "HasMessagesCreate",
    "PRICING",
    "Usage",
    "price_usd",
]
