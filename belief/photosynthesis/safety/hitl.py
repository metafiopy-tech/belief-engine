"""Human-in-the-loop approval flow.

Auto-approve below `auto_threshold`, hard-block above `hard_threshold`,
ask the human in between. Fail-closed on timeout.

The transport (Telegram via python-telegram-bot v21) lives behind an
`ApprovalClient` protocol so tests can swap in a null client. Real
wiring sits in a callers that the ops person runs separately; this
module only owns the approval DB and the decision logic.

Spec constraints honored:

  - auto-approve floor 0.01, ceiling 0.20
  - fail-closed on timeout (reject)
  - hard-block at >= 1.00 — no approval path; caller must edit config
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol


logger = logging.getLogger("belief.photosynthesis.safety.hitl")


DEFAULT_HITL_DB = Path("/var/lib/photosynthesis/hitl.db")

# Spec: hard floor / ceiling on the configurable auto-approve threshold
AUTO_THRESHOLD_FLOOR = 0.01
AUTO_THRESHOLD_CEILING = 0.20
HARD_BLOCK_THRESHOLD = 1.00


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_REJECTED = "auto_rejected"
    HARD_BLOCKED = "hard_blocked"
    TIMEOUT = "timeout"


SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         INTEGER NOT NULL,
    app        TEXT    NOT NULL,
    payload    TEXT    NOT NULL,
    est_cost   REAL    NOT NULL,
    status     TEXT    NOT NULL,
    decided_at INTEGER,
    decided_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, ts);
"""


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------


class ApprovalClient(Protocol):
    """Any transport that can deliver an approval request and return a decision.

    Implementations (Telegram, Slack, pager) should resolve the future
    with True/False. Callers never construct this directly — the daemon
    wires in the concrete client at startup.
    """

    async def request(
        self,
        *,
        approval_id: int,
        app: str,
        payload: dict[str, Any],
        est_cost: float,
        timeout: float,
    ) -> Optional[bool]: ...  # noqa: E704


class NullApprovalClient:
    """No-op transport used in tests and degraded mode.

    Always returns None (== timeout), which the decision path translates
    into fail-closed rejection. Safe default: if nothing is wired up,
    no human-gated call ever goes through.
    """

    async def request(
        self,
        *,
        approval_id: int,
        app: str,
        payload: dict[str, Any],
        est_cost: float,
        timeout: float,
    ) -> Optional[bool]:
        return None


class AutoApproveTestClient:
    """Test helper: answer `decision` immediately without any wait."""

    def __init__(self, decision: bool) -> None:
        self.decision = decision
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        *,
        approval_id: int,
        app: str,
        payload: dict[str, Any],
        est_cost: float,
        timeout: float,
    ) -> Optional[bool]:
        self.calls.append(
            {
                "approval_id": approval_id,
                "app": app,
                "est_cost": est_cost,
                "timeout": timeout,
            }
        )
        return self.decision


# ---------------------------------------------------------------------------
# Approval store
# ---------------------------------------------------------------------------


@dataclass
class ApprovalStore:
    db_path: Path = DEFAULT_HITL_DB

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("PRAGMA busy_timeout = 5000;")
            c.row_factory = sqlite3.Row
            yield c
        finally:
            c.close()

    def insert_pending(
        self, *, app: str, payload: dict[str, Any], est_cost: float
    ) -> int:
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            cur = c.execute(
                "INSERT INTO approvals(ts, app, payload, est_cost, status) "
                "VALUES(?, ?, ?, ?, ?);",
                (
                    int(time.time()),
                    app,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    float(est_cost),
                    ApprovalStatus.PENDING.value,
                ),
            )
            c.execute("COMMIT;")
            return int(cur.lastrowid or 0)

    def resolve(
        self, approval_id: int, status: ApprovalStatus, decided_by: str = ""
    ) -> None:
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ? "
                "WHERE id = ?;",
                (status.value, int(time.time()), decided_by, approval_id),
            )
            c.execute("COMMIT;")

    def by_status(self, status: ApprovalStatus, limit: int = 100) -> list[sqlite3.Row]:
        with self._conn() as c:
            return list(
                c.execute(
                    "SELECT * FROM approvals WHERE status = ? ORDER BY ts DESC LIMIT ?;",
                    (status.value, limit),
                )
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class ApprovalDecision:
    granted: bool
    status: ApprovalStatus
    approval_id: int


async def request_approval(
    app: str,
    payload: dict[str, Any],
    est_cost: float,
    *,
    client: ApprovalClient,
    store: ApprovalStore,
    auto_threshold: float = 0.05,
    hard_threshold: float = HARD_BLOCK_THRESHOLD,
    timeout: float = 900.0,
) -> ApprovalDecision:
    """Gate a costly call behind auto-approve / ask / hard-block policy.

    Returns an ApprovalDecision — the caller should never see the
    'timeout' status as "granted" and the daemon's call sites treat
    granted=False as "abort this cycle."
    """
    auto = max(
        AUTO_THRESHOLD_FLOOR, min(AUTO_THRESHOLD_CEILING, float(auto_threshold))
    )
    hard = float(hard_threshold)
    cost = float(est_cost)

    # Hard block (spec: >= 1.00 by default; no approval path)
    if cost >= hard:
        aid = store.insert_pending(app=app, payload=payload, est_cost=cost)
        store.resolve(aid, ApprovalStatus.HARD_BLOCKED, decided_by="policy")
        logger.warning(
            "hitl: est_cost $%.2f >= hard_threshold $%.2f — hard blocked",
            cost,
            hard,
        )
        return ApprovalDecision(
            granted=False, status=ApprovalStatus.HARD_BLOCKED, approval_id=aid
        )

    # Auto-approve
    if cost < auto:
        aid = store.insert_pending(app=app, payload=payload, est_cost=cost)
        store.resolve(aid, ApprovalStatus.AUTO_APPROVED, decided_by="policy")
        return ApprovalDecision(
            granted=True, status=ApprovalStatus.AUTO_APPROVED, approval_id=aid
        )

    # Ask the human
    aid = store.insert_pending(app=app, payload=payload, est_cost=cost)
    try:
        decision = await asyncio.wait_for(
            client.request(
                approval_id=aid,
                app=app,
                payload=payload,
                est_cost=cost,
                timeout=timeout,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        store.resolve(aid, ApprovalStatus.TIMEOUT, decided_by="timeout")
        return ApprovalDecision(
            granted=False, status=ApprovalStatus.TIMEOUT, approval_id=aid
        )

    if decision is None:
        # Transport returned "no answer" — fail closed.
        store.resolve(aid, ApprovalStatus.AUTO_REJECTED, decided_by="transport_none")
        return ApprovalDecision(
            granted=False, status=ApprovalStatus.AUTO_REJECTED, approval_id=aid
        )
    if decision:
        store.resolve(aid, ApprovalStatus.APPROVED, decided_by="human")
        return ApprovalDecision(
            granted=True, status=ApprovalStatus.APPROVED, approval_id=aid
        )
    store.resolve(aid, ApprovalStatus.REJECTED, decided_by="human")
    return ApprovalDecision(
        granted=False, status=ApprovalStatus.REJECTED, approval_id=aid
    )


__all__ = [
    "AUTO_THRESHOLD_CEILING",
    "AUTO_THRESHOLD_FLOOR",
    "ApprovalClient",
    "ApprovalDecision",
    "ApprovalStatus",
    "ApprovalStore",
    "AutoApproveTestClient",
    "HARD_BLOCK_THRESHOLD",
    "NullApprovalClient",
    "request_approval",
]
