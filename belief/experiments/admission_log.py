"""Soil-admission event log for the STARVED-arm experiment.

One row per (candidate, arm) per generation, capturing exactly the fields the
design doc (§4) calls for plus the two we added (self_confidence, external_pass):

    {experiment_id, gen, build_id, arm, fed_gate_pass, starved_self_score,
     self_confidence, admitted, external_pass, timestamp}

This is the audit trail that answers the headline diagnostic — *how many
STARVED-admitted artifacts actually failed the hidden external test* — directly
via :func:`count_fictions`. Metrics are computed offline from this log + the
per-generation snapshots, so logging never perturbs a run.

SQLite, mirroring the style of ``belief/experiments/ab_runner.py`` (append-only,
``db_path`` overridable for tests).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from belief.experiments.admission import AdmissionResult, Candidate

DEFAULT_DB_PATH = Path.home() / ".belief-engine" / "starved_admissions.db"

_ARMS = ("FED", "STARVED")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the admission-events table (idempotent)."""
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS admission_events (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id      TEXT    NOT NULL,
                gen                INTEGER NOT NULL,
                build_id           TEXT    NOT NULL,
                arm                TEXT    NOT NULL,
                fed_gate_pass      INTEGER NOT NULL,
                starved_self_score REAL    NOT NULL,
                self_confidence    REAL    NOT NULL,
                admitted           INTEGER NOT NULL,
                external_pass      INTEGER NOT NULL,
                timestamp          TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_admission_exp_arm
                ON admission_events (experiment_id, arm);
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_admission_event(
    *,
    experiment_id: str,
    gen: int,
    build_id: str,
    arm: str,
    fed_gate_pass: bool,
    starved_self_score: float,
    self_confidence: float,
    admitted: bool,
    external_pass: bool,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Append one (candidate, arm) admission event."""
    db_path = Path(db_path).expanduser()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO admission_events (
                experiment_id, gen, build_id, arm, fed_gate_pass,
                starved_self_score, self_confidence, admitted, external_pass, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                int(gen),
                build_id,
                arm.upper(),
                1 if fed_gate_pass else 0,
                float(starved_self_score),
                float(self_confidence),
                1 if admitted else 0,
                1 if external_pass else 0,
                _utcnow_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_generation(
    experiment_id: str,
    gen: int,
    candidates: list[Candidate],
    result: AdmissionResult,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Log every candidate under BOTH arms for one generation.

    Writes two rows per candidate (FED + STARVED), each with that arm's
    ``admitted`` flag from ``result``. Returns the number of rows written.
    """
    db_path = Path(db_path).expanduser()
    init_db(db_path)
    rows = 0
    for c in candidates:
        for arm in _ARMS:
            log_admission_event(
                experiment_id=experiment_id,
                gen=gen,
                build_id=c.build_id,
                arm=arm,
                fed_gate_pass=c.external_pass,
                starved_self_score=c.self_score,
                self_confidence=c.self_confidence,
                admitted=result.is_admitted(arm, c.build_id),
                external_pass=c.external_pass,
                db_path=db_path,
            )
            rows += 1
    return rows


def count_fictions(experiment_id: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """STARVED-admitted artifacts that failed the external test ("fictions").

    The direct count of elegant-wrong-physics entering STARVED soil.
    """
    db_path = Path(db_path).expanduser()
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM admission_events
            WHERE experiment_id = ? AND arm = 'STARVED'
              AND admitted = 1 AND external_pass = 0
            """,
            (experiment_id,),
        )
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def fetch_events(experiment_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all events for an experiment (for offline reporting)."""
    db_path = Path(db_path).expanduser()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM admission_events WHERE experiment_id = ? ORDER BY gen, build_id, arm",
            (experiment_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
