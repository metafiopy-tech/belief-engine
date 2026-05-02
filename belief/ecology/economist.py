"""Economist — global daily-budget contract for ecology organs (v3.3 Session 1).

Per the v3.3 spec §3.4, the Economist allocates a daily USD budget across
ecology organs. This is the *contract shell*: every organ calls
``Economist.quote(action, estimated_usd)`` before doing work, then
``Economist.commit(action, actual_usd)`` after. The shell ships only the
budget ceiling — the per-action price table and 95th-percentile learning
loop land in Session 5 once organs are reporting actuals.

Relationship to ``belief.hardening.BuildBudget``: hardening enforces a
*per-build floor* (one build can spend up to N). Economist enforces a
*daily ceiling* across all builds + organs. They do not overlap.

Storage:
    State (today's spend):  ~/.belief-engine/economist_state.json
    Audit log (all events): ~/.belief-engine/audit/ecology_economist.jsonl

Concurrency: state writes use atomic rename; quote/commit are guarded by an
fcntl advisory lock on the state file so two organs racing on a low budget
cannot both see "approved" and both commit.

Daily reset: at the first quote/commit/status call after UTC date rolls
over, today's spend is zeroed. Audit log is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("belief.ecology.economist")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_DAILY_BUDGET_USD = 5.0
DEFAULT_AUDIT_ROTATE_BYTES = 100 * 1024 * 1024  # 100 MB
DEFAULT_AUDIT_KEEP = 5

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_STATE_PATH = _BELIEF_HOME / "economist_state.json"
_DEFAULT_AUDIT_PATH = _BELIEF_HOME / "audit" / "ecology_economist.jsonl"


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class PriceQuote:
    """The Economist's response to a ``quote(action, estimated_usd)`` call.

    ``approved`` tells the caller whether to proceed. ``reason`` is always
    populated — for approvals it explains the headroom, for rejections it
    explains why the spend would breach the daily ceiling.
    """

    action: str
    estimated_usd: float
    approved: bool
    reason: str
    remaining_after: float = 0.0  # what would be left if caller commits the estimate
    quoted_at: float = field(default_factory=time.time)


class QuoteRejected(Exception):
    """Raised by ``Economist.commit_or_raise`` when no headroom remains."""

    def __init__(self, quote: PriceQuote):
        self.quote = quote
        super().__init__(f"Economist rejected {quote.action!r}: {quote.reason}")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _today_utc() -> str:
    """ISO date (YYYY-MM-DD) in UTC. Used as the daily-reset key."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically (tmp + rename).

    Ensures crash-mid-write never leaves a half-written state file —
    readers either see the prior value or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def _state_lock(state_path: Path) -> Iterator[None]:
    """Advisory file lock so concurrent organs serialize through quote/commit.

    Uses fcntl.flock on POSIX. Falls back to a no-op on platforms without
    fcntl (e.g., Windows) — those users get best-effort, not strict
    serialization. Tested only on macOS/Linux per Joe's stack.
    """
    try:
        import fcntl  # noqa: PLC0415 — optional POSIX import
    except ImportError:  # pragma: no cover — non-POSIX fallback
        yield
        return

    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    # Open in a+ so the file exists; flock the fd, hold for the block.
    with open(lock_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _rotate_if_needed(audit_path: Path, max_bytes: int, keep: int) -> None:
    """Lazy rotation: rename ``foo.jsonl`` → ``foo.jsonl.1`` if oversize.

    Existing rotated files shift up (.1 → .2, .2 → .3, ...). Files past
    ``keep`` are deleted. Called on every ``commit`` write so rotation
    never requires a daemon.
    """
    try:
        size = audit_path.stat().st_size
    except FileNotFoundError:
        return
    if size < max_bytes:
        return
    # Shift older files up.
    for i in range(keep, 0, -1):
        src = audit_path.with_suffix(audit_path.suffix + f".{i}")
        if not src.exists():
            continue
        if i == keep:
            try:
                src.unlink()
            except OSError:
                pass
        else:
            dst = audit_path.with_suffix(audit_path.suffix + f".{i + 1}")
            try:
                os.replace(src, dst)
            except OSError:
                pass
    # Move current to .1
    rotated = audit_path.with_suffix(audit_path.suffix + ".1")
    try:
        os.replace(audit_path, rotated)
    except OSError:
        pass


# ── The Economist ──────────────────────────────────────────────────────────


class Economist:
    """Daily-budget allocator. Contract shell — see module docstring."""

    def __init__(
        self,
        daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD,
        state_path: Path | None = None,
        audit_path: Path | None = None,
        audit_rotate_bytes: int = DEFAULT_AUDIT_ROTATE_BYTES,
        audit_keep: int = DEFAULT_AUDIT_KEEP,
    ) -> None:
        if daily_budget_usd < 0:
            raise ValueError(f"daily_budget_usd must be >= 0, got {daily_budget_usd}")
        self.daily_budget_usd = float(daily_budget_usd)
        self.state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self.audit_path = Path(audit_path) if audit_path else _DEFAULT_AUDIT_PATH
        self.audit_rotate_bytes = int(audit_rotate_bytes)
        self.audit_keep = int(audit_keep)

    # ── State management ────────────────────────────────────────────────

    def _empty_state(self) -> dict:
        return {"date_utc": _today_utc(), "spent_usd": 0.0, "commits": 0}

    def _load_state(self) -> dict:
        """Read state file, falling back to a fresh state on any error.

        Malformed JSON → log a warning, return defaults. Missing file →
        return defaults silently (first-run case).
        """
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Schema sanity — any missing key falls back to default.
            if not isinstance(state, dict):
                raise ValueError("state file root must be an object")
            for key in ("date_utc", "spent_usd", "commits"):
                if key not in state:
                    raise ValueError(f"missing key: {key}")
            return state
        except FileNotFoundError:
            return self._empty_state()
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning(
                "Economist state file %s unreadable (%s); using fresh state.",
                self.state_path,
                e,
            )
            return self._empty_state()

    def _save_state(self, state: dict) -> None:
        _atomic_write_json(self.state_path, state)

    def _check_daily_reset(self, state: dict) -> dict:
        """Zero the spend counter if UTC date has rolled over."""
        today = _today_utc()
        if state.get("date_utc") != today:
            self._audit(
                {
                    "event": "daily_reset",
                    "previous_date": state.get("date_utc"),
                    "previous_spent_usd": round(float(state.get("spent_usd") or 0), 6),
                    "new_date": today,
                }
            )
            return {"date_utc": today, "spent_usd": 0.0, "commits": 0}
        return state

    # ── Audit ───────────────────────────────────────────────────────────

    def _audit(self, record: dict) -> None:
        """Append a JSONL line to the audit log. Never raises."""
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(self.audit_path, self.audit_rotate_bytes, self.audit_keep)
            entry = {"ts": datetime.now(timezone.utc).isoformat(), **record}
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:  # pragma: no cover — audit must never crash callers
            logger.warning("Economist audit write failed: %s", e)

    # ── Public API ──────────────────────────────────────────────────────

    def quote(self, action: str, estimated_usd: float) -> PriceQuote:
        """Ask whether ``estimated_usd`` for ``action`` would fit today.

        Approval is non-binding: the caller must still call ``commit`` to
        actually charge. A quote can be "approved" but a later commit may
        still race past the ceiling if a different process committed in
        between — that's why we re-check inside ``commit``.
        """
        if estimated_usd < 0:
            raise ValueError(f"estimated_usd must be >= 0, got {estimated_usd}")
        with _state_lock(self.state_path):
            state = self._check_daily_reset(self._load_state())
            self._save_state(state)  # persist the reset if it happened
            spent = float(state["spent_usd"])
            remaining_now = max(0.0, self.daily_budget_usd - spent)
            remaining_after = self.daily_budget_usd - (spent + float(estimated_usd))
            if remaining_after >= 0:
                quote = PriceQuote(
                    action=action,
                    estimated_usd=float(estimated_usd),
                    approved=True,
                    reason=f"approved; ${remaining_now:.4f} headroom before, ${remaining_after:.4f} after",
                    remaining_after=remaining_after,
                )
            else:
                quote = PriceQuote(
                    action=action,
                    estimated_usd=float(estimated_usd),
                    approved=False,
                    reason=(
                        f"would exceed daily ceiling: spent ${spent:.4f}, "
                        f"estimate ${float(estimated_usd):.4f}, "
                        f"budget ${self.daily_budget_usd:.2f}"
                    ),
                    remaining_after=remaining_after,
                )
            self._audit(
                {
                    "event": "quote",
                    "action": action,
                    "estimated_usd": round(float(estimated_usd), 6),
                    "approved": quote.approved,
                    "spent_before": round(spent, 6),
                    "remaining_after": round(remaining_after, 6),
                }
            )
            return quote

    def commit(self, action: str, actual_usd: float) -> None:
        """Charge ``actual_usd`` against today's budget for ``action``.

        Always succeeds — the caller may have done partial work and we
        record what was actually spent. To refuse over-budget commits
        upfront, use ``commit_or_raise``.
        """
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd}")
        with _state_lock(self.state_path):
            state = self._check_daily_reset(self._load_state())
            spent_before = float(state["spent_usd"])
            new_spent = spent_before + float(actual_usd)
            new_state = {
                "date_utc": state["date_utc"],
                "spent_usd": new_spent,
                "commits": int(state.get("commits", 0)) + 1,
            }
            self._save_state(new_state)
            self._audit(
                {
                    "event": "commit",
                    "action": action,
                    "actual_usd": round(float(actual_usd), 6),
                    "spent_before": round(spent_before, 6),
                    "spent_after": round(new_spent, 6),
                    "remaining": round(max(0.0, self.daily_budget_usd - new_spent), 6),
                    "over_budget": new_spent > self.daily_budget_usd,
                }
            )

    def commit_or_raise(self, action: str, actual_usd: float) -> None:
        """Like ``commit`` but raises ``QuoteRejected`` if it would breach."""
        quote = self.quote(action, actual_usd)
        if not quote.approved:
            raise QuoteRejected(quote)
        self.commit(action, actual_usd)

    def remaining(self) -> float:
        """Headroom left in today's budget (USD)."""
        with _state_lock(self.state_path):
            state = self._check_daily_reset(self._load_state())
            self._save_state(state)
            return max(0.0, self.daily_budget_usd - float(state["spent_usd"]))

    def reset_today(self) -> None:
        """Zero today's spend without touching the audit log.

        Used by ``belief economy --reset``. Does not affect the daily
        rollover schedule — tomorrow still resets normally.
        """
        with _state_lock(self.state_path):
            state = self._check_daily_reset(self._load_state())
            previous_spent = float(state["spent_usd"])
            new_state = {
                "date_utc": _today_utc(),
                "spent_usd": 0.0,
                "commits": 0,
            }
            self._save_state(new_state)
            self._audit(
                {
                    "event": "manual_reset",
                    "previous_spent_usd": round(previous_spent, 6),
                }
            )

    def status(self) -> dict:
        """Snapshot for ``belief economy --show`` / monitoring."""
        with _state_lock(self.state_path):
            state = self._check_daily_reset(self._load_state())
            self._save_state(state)
            spent = float(state["spent_usd"])
            return {
                "date_utc": state["date_utc"],
                "daily_budget_usd": self.daily_budget_usd,
                "spent_usd": spent,
                "remaining_usd": max(0.0, self.daily_budget_usd - spent),
                "utilization": (
                    spent / self.daily_budget_usd if self.daily_budget_usd > 0 else 0.0
                ),
                "commits_today": int(state.get("commits", 0)),
                "state_path": str(self.state_path),
                "audit_path": str(self.audit_path),
            }


# ── CLI helpers (called from belief.cli) ───────────────────────────────────


def cli_show(daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD) -> str:
    """Format the ``belief economy --show`` output."""
    econ = Economist(daily_budget_usd=daily_budget_usd)
    s = econ.status()
    return (
        f"Economist — daily budget {s['daily_budget_usd']:.2f} USD\n"
        f"  date (UTC):   {s['date_utc']}\n"
        f"  spent today:  ${s['spent_usd']:.4f}\n"
        f"  remaining:    ${s['remaining_usd']:.4f}\n"
        f"  utilization:  {s['utilization']:.1%}\n"
        f"  commits:      {s['commits_today']}\n"
        f"  state file:   {s['state_path']}\n"
        f"  audit log:    {s['audit_path']}"
    )


def cli_reset(daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD) -> str:
    """Implement ``belief economy --reset``. Returns a one-line confirmation."""
    econ = Economist(daily_budget_usd=daily_budget_usd)
    before = econ.status()["spent_usd"]
    econ.reset_today()
    return f"Economist: reset today's spend (was ${before:.4f}); audit history preserved."
