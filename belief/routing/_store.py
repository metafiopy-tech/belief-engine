"""Shared SQLite store for the routing subsystem (mycorrhizal Stage 5).

Centralises the connection + DDL for the three routing tables so
``HubRegistry``, ``Router``, ``SanctionsEngine``, and ``TopologyDiagnostics``
all read/write a consistent schema without each re-deriving it.

Tables:
  hub_status      — derived hub membership + demotion counters
  routing_events  — one row per routed request (for diagnostics)
  graveyard       — archived agents (sanction terminations)

Same connection conventions as the Stage 1/2/4 ledgers: WAL mode,
process-local lock, ``check_same_thread=False``.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_DB_PATH = _BELIEF_HOME / "routing.db"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class RoutingStore:
    """Owns the routing.db connection + schema."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._create_tables()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _create_tables(self) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hub_status (
                    agent_id       TEXT PRIMARY KEY,
                    is_hub         INTEGER NOT NULL DEFAULT 0,
                    below_count    INTEGER NOT NULL DEFAULT 0,
                    promoted_at    TEXT,
                    updated_at     TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id      TEXT NOT NULL,
                    decision_kind TEXT NOT NULL,
                    hub_id        TEXT,
                    ts            TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_routing_events_ts
                ON routing_events(ts)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS graveyard (
                    agent_id     TEXT PRIMARY KEY,
                    archived_at  TEXT NOT NULL,
                    reason       TEXT NOT NULL DEFAULT ''
                )
                """
            )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ── hub_status ──────────────────────────────────────────────────────

    def get_hub_status(self, agent_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM hub_status WHERE agent_id = ?", (agent_id,)
        ).fetchone()

    def all_hub_status(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM hub_status").fetchall()

    def current_hub_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT agent_id FROM hub_status WHERE is_hub = 1 ORDER BY agent_id"
        ).fetchall()
        return [r["agent_id"] for r in rows]

    def upsert_hub_status(
        self,
        agent_id: str,
        is_hub: bool,
        below_count: int,
        promoted_at: Optional[str],
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO hub_status(agent_id, is_hub, below_count, promoted_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    is_hub = excluded.is_hub,
                    below_count = excluded.below_count,
                    promoted_at = excluded.promoted_at,
                    updated_at = excluded.updated_at
                """,
                (
                    agent_id,
                    1 if is_hub else 0,
                    int(below_count),
                    promoted_at,
                    _iso(_utcnow()),
                ),
            )

    # ── routing_events ──────────────────────────────────────────────────

    def record_event(
        self,
        agent_id: str,
        decision_kind: str,
        hub_id: Optional[str],
        ts: Optional[datetime] = None,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO routing_events(agent_id, decision_kind, hub_id, ts)
                VALUES (?, ?, ?, ?)
                """,
                (agent_id, decision_kind, hub_id, _iso(ts or _utcnow())),
            )

    def events_since(self, cutoff_iso: Optional[str] = None) -> list[sqlite3.Row]:
        if cutoff_iso is None:
            return self._conn.execute("SELECT * FROM routing_events ORDER BY ts ASC").fetchall()
        return self._conn.execute(
            "SELECT * FROM routing_events WHERE ts >= ? ORDER BY ts ASC",
            (cutoff_iso,),
        ).fetchall()

    # ── graveyard ───────────────────────────────────────────────────────

    def archive_agent(self, agent_id: str, reason: str = "") -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO graveyard(agent_id, archived_at, reason)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    archived_at = excluded.archived_at,
                    reason = excluded.reason
                """,
                (agent_id, _iso(_utcnow()), reason),
            )

    def is_archived(self, agent_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM graveyard WHERE agent_id = ? LIMIT 1", (agent_id,)
        ).fetchone()
        return row is not None

    def graveyard_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT agent_id FROM graveyard ORDER BY archived_at DESC"
        ).fetchall()
        return [r["agent_id"] for r in rows]
