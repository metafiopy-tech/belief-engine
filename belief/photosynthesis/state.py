"""SQLite (WAL-mode) state store for the Photosynthesis daemon.

Three tables own the full ingest bookkeeping:

    raw_signals      — normalized CandidateSeed rows (one per harvested item)
    seen             — (source, item_id) uniqueness index for O(1) dedup
    watermarks       — per-source last-seen timestamp/cursor so each
                       harvester does strictly incremental fetches

WAL mode is mandatory (the Grinder shares the same SQLite file for
jobs.sqlite; we must not block it during long reads). PRAGMA settings
follow the spec verbatim:

    journal_mode = WAL
    synchronous  = NORMAL
    busy_timeout = 5000 (ms)

All write helpers use BEGIN IMMEDIATE so that concurrent writers
serialize on the writer lock instead of deadlocking on the first
`INSERT`. See SQLite docs for WAL concurrency.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


DEFAULT_DB_PATH = "/var/lib/photosynthesis/signals.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,
    source_id     TEXT    NOT NULL,
    title         TEXT    NOT NULL DEFAULT '',
    summary       TEXT    NOT NULL DEFAULT '',
    raw_excerpt   TEXT    NOT NULL DEFAULT '',
    captured_at   INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'raw',
    filter_score  REAL,
    stage_reached INTEGER NOT NULL DEFAULT 0,
    UNIQUE(source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_signals_status
    ON raw_signals(status, captured_at);

CREATE TABLE IF NOT EXISTS seen (
    source     TEXT    NOT NULL,
    item_id    TEXT    NOT NULL,
    first_seen INTEGER NOT NULL,
    PRIMARY KEY(source, item_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS watermarks (
    source      TEXT    PRIMARY KEY,
    last_ts     INTEGER,
    last_cursor TEXT,
    updated_at  INTEGER NOT NULL
) WITHOUT ROWID;

-- Session 4: ACCEL-style bounded priority heap for synthesis candidates.
-- Capacity is enforced by the heap class, not a SQL constraint. Each row
-- is a single pushed seed with its current computed value; the generator
-- pops the top and writes a goal spec.
CREATE TABLE IF NOT EXISTS synthesis_heap (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    seed_json TEXT    NOT NULL,
    value     REAL    NOT NULL,
    added_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_synthesis_heap_value
    ON synthesis_heap(value);

-- Session 4: rolling log of per-cycle min-value samples used by the heap's
-- saturation detector. 3 consecutive cycles with min(recent 20) > 0.70 is
-- the signal to pause upstream.
CREATE TABLE IF NOT EXISTS synthesis_cycle_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_ts     INTEGER NOT NULL,
    min_value    REAL,
    mean_value   REAL,
    pushed_count INTEGER NOT NULL DEFAULT 0,
    saturation   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_synthesis_cycle_log_ts
    ON synthesis_cycle_log(cycle_ts);
"""


@dataclass
class CandidateSeed:
    """A uniform, normalized view of one harvested item.

    Every source normalizes its native shape into this dataclass before
    touching the DB. `raw_excerpt` preserves enough context for later
    filter stages (up to ~2000 chars); `title` and `summary` are what
    the cascade scores.
    """

    source: str
    source_id: str
    title: str = ""
    summary: str = ""
    raw_excerpt: str = ""
    captured_at: int = field(default_factory=lambda: int(time.time()))


class PhotosynthesisState:
    """Thin helper around the SQLite store."""

    def __init__(self, db_path: str | os.PathLike[str] = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ connection
    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        """Yield a sqlite3.Connection with WAL pragmas applied."""
        c = sqlite3.connect(
            self.db_path,
            timeout=5.0,
            isolation_level=None,  # autocommit — we do explicit BEGIN
        )
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("PRAGMA synchronous = NORMAL;")
            c.execute("PRAGMA busy_timeout = 5000;")
            c.execute("PRAGMA foreign_keys = ON;")
            c.row_factory = sqlite3.Row
            yield c
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self.conn() as c:
            c.executescript(SCHEMA)

    # ------------------------------------------------------------------ seen / dedup
    def mark_if_new(self, source: str, item_id: str) -> bool:
        """Return True iff (source, item_id) was not previously seen.

        Uses INSERT OR IGNORE so the caller doesn't need to pre-check.
        The rowcount distinguishes: 1 => newly inserted, 0 => duplicate.
        Atomic under WAL with BEGIN IMMEDIATE.
        """
        now = int(time.time())
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            cur = c.execute(
                "INSERT OR IGNORE INTO seen(source, item_id, first_seen) VALUES(?, ?, ?);",
                (source, item_id, now),
            )
            c.execute("COMMIT;")
            return cur.rowcount == 1

    # ------------------------------------------------------------------ watermark
    def get_watermark(self, source: str) -> tuple[Optional[int], Optional[str]]:
        """Return (last_ts, last_cursor) for a source, or (None, None)."""
        with self.conn() as c:
            row = c.execute(
                "SELECT last_ts, last_cursor FROM watermarks WHERE source = ?;",
                (source,),
            ).fetchone()
            if row is None:
                return (None, None)
            return (row["last_ts"], row["last_cursor"])

    def set_watermark(
        self,
        source: str,
        *,
        last_ts: Optional[int] = None,
        last_cursor: Optional[str] = None,
    ) -> None:
        """Upsert the watermark row for this source."""
        now = int(time.time())
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "INSERT INTO watermarks(source, last_ts, last_cursor, updated_at) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(source) DO UPDATE SET "
                "  last_ts     = COALESCE(excluded.last_ts,     last_ts), "
                "  last_cursor = COALESCE(excluded.last_cursor, last_cursor), "
                "  updated_at  = excluded.updated_at;",
                (source, last_ts, last_cursor, now),
            )
            c.execute("COMMIT;")

    # ------------------------------------------------------------------ raw_signals
    def insert_signal(self, seed: CandidateSeed) -> Optional[int]:
        """Insert one signal. Returns the rowid, or None if it was a duplicate."""
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            cur = c.execute(
                "INSERT OR IGNORE INTO raw_signals"
                "(source, source_id, title, summary, raw_excerpt, captured_at) "
                "VALUES(?, ?, ?, ?, ?, ?);",
                (
                    seed.source,
                    seed.source_id,
                    seed.title,
                    seed.summary,
                    seed.raw_excerpt,
                    seed.captured_at,
                ),
            )
            c.execute("COMMIT;")
            if cur.rowcount == 0:
                return None
            return cur.lastrowid

    def pending_signals(self, limit: int = 1000) -> list[sqlite3.Row]:
        """Return raw signals still needing filter evaluation."""
        with self.conn() as c:
            return list(
                c.execute(
                    "SELECT id, source, title, summary, raw_excerpt "
                    "FROM raw_signals WHERE status = 'raw' "
                    "ORDER BY captured_at ASC LIMIT ?;",
                    (limit,),
                )
            )

    def update_filter_result(
        self,
        signal_id: int,
        *,
        stage_reached: int,
        filter_score: float,
        status: str,
    ) -> None:
        """Record the outcome of a cascade filter pass."""
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "UPDATE raw_signals "
                "SET stage_reached = ?, filter_score = ?, status = ? "
                "WHERE id = ?;",
                (stage_reached, filter_score, status, signal_id),
            )
            c.execute("COMMIT;")

    def set_signal_status(self, signal_id: int, status: str) -> None:
        """Transition a raw_signal row to a new terminal status.

        Used by the synthesis cycle to mark rows 'promoted' (written to
        pending_sessions) or 'rejected' (failed novelty/ZPD gates).
        """
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "UPDATE raw_signals SET status = ? WHERE id = ?;",
                (status, signal_id),
            )
            c.execute("COMMIT;")

    def survivors_for_synthesis(self, limit: int = 20) -> list[sqlite3.Row]:
        """Top-k stage-3 survivors ranked by filter_score, newest first.

        Only rows that are still 'kept' (haven't been promoted or rejected)
        are returned — makes the synthesis cycle idempotent: re-running
        on the same raw_signals won't re-produce duplicate goals.
        """
        with self.conn() as c:
            return list(
                c.execute(
                    "SELECT id, source, source_id, title, summary, raw_excerpt, "
                    "       filter_score, captured_at "
                    "FROM raw_signals "
                    "WHERE status = 'kept' AND stage_reached = 3 "
                    "ORDER BY filter_score DESC, captured_at DESC "
                    "LIMIT ?;",
                    (limit,),
                )
            )

    # ------------------------------------------------------------------ diagnostics
    def count_by_source(self) -> dict[str, int]:
        """SELECT source, COUNT(*) GROUP BY source — used by the daemon logs."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT source, COUNT(*) AS n FROM raw_signals GROUP BY source;"
            ).fetchall()
            return {row["source"]: row["n"] for row in rows}

    def duplicates_probe(self) -> int:
        """Returns the number of (source, source_id) pairs with >1 row.

        Used by an acceptance-criteria check — should always be 0 because
        the UNIQUE constraint on raw_signals enforces it.
        """
        with self.conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM ("
                "  SELECT source, source_id FROM raw_signals "
                "  GROUP BY source, source_id HAVING COUNT(*) > 1"
                ");"
            ).fetchone()
            return int(row["n"]) if row else 0


__all__ = [
    "CandidateSeed",
    "DEFAULT_DB_PATH",
    "PhotosynthesisState",
    "SCHEMA",
]
