"""SHA-256 hash-chain audit log.

Tamper-evident append-only log. Every event's hash is
`sha256(canonical_json(payload) || prev_hash)`; flipping a single byte
anywhere in the history breaks every subsequent hash.

Canonical JSON means `json.dumps(payload, sort_keys=True,
separators=(',', ':'))` — no whitespace, deterministic key order. Both
the write path and the verify path MUST produce the exact same bytes
for the chain to be meaningful.

The daily `audit_anchor` job posts the head hash to an external sink
(spec: Discord). This module only exposes `head_hash()`; a caller
wires the webhook post separately — that way Discord being down doesn't
block new appends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


logger = logging.getLogger("belief.photosynthesis.safety.audit")


DEFAULT_AUDIT_DB = Path("/var/lib/photosynthesis/audit.db")
GENESIS_PREV_HASH = "0" * 64  # before the first event


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        INTEGER NOT NULL,
    payload   TEXT    NOT NULL,
    prev_hash TEXT    NOT NULL,
    hash      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON bytes for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_hash(payload_json: str, prev_hash: str) -> str:
    """Spec: sha256(canonical_json(payload) || prev_hash)."""
    h = hashlib.sha256()
    h.update(payload_json.encode("utf-8"))
    h.update(prev_hash.encode("utf-8"))
    return h.hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    break_seq: Optional[int]
    reason: str

    def as_tuple(self) -> tuple[bool, Optional[int], str]:
        return self.ok, self.break_seq, self.reason


class AuditLog:
    """Hash-chained event log."""

    def __init__(self, db_path: Path | str = DEFAULT_AUDIT_DB) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("PRAGMA busy_timeout = 5000;")
            c.row_factory = sqlite3.Row
            yield c
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # --------------------------------------------------------------- append
    def append(self, event: dict[str, Any]) -> str:
        """Append one event; return its hash."""
        payload_json = canonical_json(event)
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            last = c.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1;").fetchone()
            prev = last["hash"] if last is not None else GENESIS_PREV_HASH
            new_hash = compute_hash(payload_json, prev)
            c.execute(
                "INSERT INTO events(ts, payload, prev_hash, hash) VALUES(?, ?, ?, ?);",
                (int(time.time()), payload_json, prev, new_hash),
            )
            c.execute("COMMIT;")
        return new_hash

    # --------------------------------------------------------------- head
    def head_hash(self) -> str:
        with self._conn() as c:
            row = c.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1;").fetchone()
            return row["hash"] if row else GENESIS_PREV_HASH

    def count(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) AS n FROM events;").fetchone()
            return int(row["n"]) if row else 0

    # --------------------------------------------------------------- verify
    def verify(self) -> VerifyResult:
        """Walk the chain from seq=1, recomputing each hash.

        Returns a VerifyResult with .ok=True and reason='ok' when every
        link validates. Otherwise .break_seq identifies the first
        tampered event and .reason is 'hash mismatch' or
        'prev_hash mismatch'.
        """
        prev = GENESIS_PREV_HASH
        with self._conn() as c:
            cur = c.execute("SELECT seq, payload, prev_hash, hash FROM events ORDER BY seq ASC;")
            for row in cur:
                seq = int(row["seq"])
                payload_json = row["payload"]
                expected_prev = prev
                stored_prev = row["prev_hash"]
                if stored_prev != expected_prev:
                    return VerifyResult(
                        ok=False,
                        break_seq=seq,
                        reason="prev_hash mismatch",
                    )
                recomputed = compute_hash(payload_json, stored_prev)
                if recomputed != row["hash"]:
                    return VerifyResult(ok=False, break_seq=seq, reason="hash mismatch")
                prev = row["hash"]
        return VerifyResult(ok=True, break_seq=None, reason="ok")


__all__ = [
    "AuditLog",
    "DEFAULT_AUDIT_DB",
    "GENESIS_PREV_HASH",
    "VerifyResult",
    "canonical_json",
    "compute_hash",
]
