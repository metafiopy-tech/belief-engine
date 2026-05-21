"""Quarantine collection (mycorrhizal Stage 7, Area 4).

Some failures must NOT be decomposed back into the general soil — a build
that produced clearly broken output (a security violation, an infinite loop,
a touch on a protected resource) is a toxic byproduct. Decomposing it would
risk seeding future builds with the very pattern that made it dangerous.
Mirror the biological handling of phenolic intermediates: quarantine, don't
distribute, and require manual review before anything from it becomes
eligible for the primitive library.

Storage: SQLite at ``~/.belief-engine/quarantine.db``. ``belief quarantine
review`` lists pending items and supports approve / reject / delete.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("belief.memory.quarantine")

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_DB_PATH = _BELIEF_HOME / "quarantine.db"

# Substrings in an error/verdict trail that mark a build as quarantine-worthy
# rather than safely decomposable.
TOXIC_MARKERS = (
    "security",
    "vulnerability",
    "injection",
    "rm -rf",
    "infinite loop",
    "fork bomb",
    "privilege",
    "exfiltrat",
    "secret",
    "credential leak",
)


class QuarantineStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class QuarantineItem:
    build_id: str
    reason: str
    status: QuarantineStatus
    quarantined_at: datetime
    evidence: dict


def is_toxic(errors: list[str], verdict: str = "", exec_error: str = "") -> Optional[str]:
    """Return the matched toxic marker if a build's output is quarantine-
    worthy, else None. Case-insensitive substring scan over the error trail."""
    blob = " ".join([*(errors or []), verdict or "", exec_error or ""]).lower()
    for marker in TOXIC_MARKERS:
        if marker in blob:
            return marker
    return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class QuarantineCollection:
    """SQLite-backed quarantine with manual-review gating."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False, isolation_level="DEFERRED"
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS quarantine (
                    build_id        TEXT PRIMARY KEY,
                    reason          TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    quarantined_at  TEXT NOT NULL,
                    evidence_json   TEXT
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

    def quarantine(self, build_id: str, reason: str, evidence: Optional[dict] = None) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO quarantine(build_id, reason, status, quarantined_at, evidence_json)
                VALUES (?, ?, 'pending', ?, ?)
                ON CONFLICT(build_id) DO UPDATE SET
                    reason = excluded.reason,
                    evidence_json = excluded.evidence_json
                """,
                (build_id, reason, _iso(_utcnow()), json.dumps(evidence or {}, default=str)),
            )

    def _row(self, row: sqlite3.Row) -> QuarantineItem:
        return QuarantineItem(
            build_id=row["build_id"],
            reason=row["reason"],
            status=QuarantineStatus(row["status"]),
            quarantined_at=datetime.fromisoformat(row["quarantined_at"]),
            evidence=json.loads(row["evidence_json"]) if row["evidence_json"] else {},
        )

    def pending(self) -> list[QuarantineItem]:
        rows = self._conn.execute(
            "SELECT * FROM quarantine WHERE status = 'pending' ORDER BY quarantined_at DESC"
        ).fetchall()
        return [self._row(r) for r in rows]

    def all_items(self) -> list[QuarantineItem]:
        rows = self._conn.execute(
            "SELECT * FROM quarantine ORDER BY quarantined_at DESC"
        ).fetchall()
        return [self._row(r) for r in rows]

    def is_quarantined(self, build_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM quarantine WHERE build_id = ? LIMIT 1", (build_id,)
        ).fetchone()
        return row is not None

    def set_status(self, build_id: str, status: QuarantineStatus) -> bool:
        with self._tx() as cur:
            cur.execute(
                "UPDATE quarantine SET status = ? WHERE build_id = ?",
                (status.value, build_id),
            )
            return cur.rowcount > 0

    def approve(self, build_id: str) -> bool:
        return self.set_status(build_id, QuarantineStatus.APPROVED)

    def reject(self, build_id: str) -> bool:
        return self.set_status(build_id, QuarantineStatus.REJECTED)

    def delete(self, build_id: str) -> bool:
        with self._tx() as cur:
            cur.execute("DELETE FROM quarantine WHERE build_id = ?", (build_id,))
            return cur.rowcount > 0


_default_collection: Optional[QuarantineCollection] = None
_default_lock = threading.Lock()


def get_default_collection() -> QuarantineCollection:
    global _default_collection
    with _default_lock:
        if _default_collection is None:
            _default_collection = QuarantineCollection()
        return _default_collection


def _reset_default_collection_for_tests() -> None:
    global _default_collection
    with _default_lock:
        if _default_collection is not None:
            _default_collection.close()
            _default_collection = None


def cli_review() -> str:
    coll = get_default_collection()
    items = coll.pending()
    header = f"Quarantine — {len(items)} pending"
    if not items:
        return header + "\n  (nothing pending review)"
    lines = [header, ""]
    for it in items:
        lines.append(
            f"  {it.build_id:<24} {it.reason}  ({it.quarantined_at.strftime('%Y-%m-%d %H:%M:%S')})"
        )
    lines.append("")
    lines.append("Approve/reject/delete via the QuarantineCollection API.")
    return "\n".join(lines)
