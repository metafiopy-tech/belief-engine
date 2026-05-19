"""Signal store + temporal integration (mycorrhizal Stage 4, Areas 1 + 10).

Stores per-agent, per-token signal emissions in a SQLite-backed circular
buffer and answers concentration queries that integrate the buffer over a
time window with exponential decay. This is the priming-pattern math from
Babikova et al. 2013 and the cell-signaling dynamics literature (Cheong
2011; Selimkhanov 2014): receivers respond to the *integral* of recent
emissions, not the most recent event.

Concentration formula::

    c(agent, token, window, half_life)
        = Σ over events e in window of:
              e.magnitude * (1/2) ** ((now - e.timestamp) / half_life)

This is an exponentially-weighted sum, not a normalized average — that
way summing two streams with the same window/half-life yields the joint
"concentration in either channel" by construction, which makes the
``joint_concentration`` semantic natural.

Storage: SQLite at ``~/.belief-engine/signals.db``. Same schema pattern
as the reciprocity / niche ledgers (WAL mode, partial UNIQUE index on
``idempotency_key``). A per-(agent, token) circular buffer of N (default
1000) is enforced by pruning oldest rows on every emit so disk usage
stays bounded under sustained traffic.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

from belief.signal.alphabet import SIGNAL_TOKENS, Signal, SignalToken

logger = logging.getLogger("belief.signal.store")


# ── Defaults ────────────────────────────────────────────────────────────────

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_DB_PATH = _BELIEF_HOME / "signals.db"

#: Default circular-buffer size per (agent, token). 1000 emissions at one
#: per second is ~17 minutes of high-frequency history per channel — more
#: than enough to inform a 5-minute concentration window.
DEFAULT_BUFFER_SIZE = 1000

#: Default decay half-life. Matches the priming-pattern receiver-response
#: timescale (Babikova 2013 reports peak receiver activation 48–100h
#: post-donor-stress, but the *signal* itself decays much faster — minutes
#: in cell signaling).
DEFAULT_HALF_LIFE = timedelta(minutes=2)

#: Default integration window for concentration queries.
DEFAULT_WINDOW = timedelta(minutes=5)


# ── Helpers ─────────────────────────────────────────────────────────────────


_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$", re.IGNORECASE)


def parse_duration(value) -> timedelta:
    """Accept a ``timedelta``, a number of seconds, or a string like
    ``'5m'``, ``'2h'``, ``'1d'``, ``'30s'``.

    Designed to keep call sites readable (``concentration(window="5m")``)
    without forcing every caller to import ``timedelta``.
    """
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"duration must be non-negative, got {value}")
        return timedelta(seconds=float(value))
    if isinstance(value, str):
        m = _DURATION_RE.match(value.strip().lower())
        if not m:
            raise ValueError(f"unparseable duration: {value!r} (try '5m', '2h', '30s', '1d')")
        n = float(m.group(1))
        unit = m.group(2)
        if unit == "s":
            return timedelta(seconds=n)
        if unit == "m":
            return timedelta(minutes=n)
        if unit == "h":
            return timedelta(hours=n)
        if unit == "d":
            return timedelta(days=n)
    raise TypeError(f"unsupported duration type: {type(value).__name__}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ── The store ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmittedSignal:
    """A frozen view of one stored signal row, returned by ``recent_signals``."""

    agent_id: str
    token: SignalToken
    magnitude: float
    timestamp: datetime
    payload: Optional[dict]


class SignalStore:
    """SQLite-backed circular buffer + temporal integration math.

    Thread-safe (single connection, process-local lock + WAL). Cross-
    process safety relies on SQLite file locking + the partial UNIQUE
    index on ``idempotency_key`` — two processes racing on the same
    derived idempotency key can both attempt INSERT, exactly one wins.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB_PATH,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got {buffer_size}")
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = int(buffer_size)
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

    def _create_tables(self) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id        TEXT NOT NULL,
                    token           TEXT NOT NULL,
                    magnitude       REAL NOT NULL,
                    ts              TEXT NOT NULL,
                    payload_json    TEXT,
                    idempotency_key TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_idem
                ON signals(idempotency_key)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_signals_agent_token_ts
                ON signals(agent_id, token, ts)
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

    # ── Write ───────────────────────────────────────────────────────────

    def emit(self, signal: Signal) -> bool:
        """Record ``signal``. Returns True if newly stored, False if the
        idempotency key already exists.

        Side effect: enforces the per-(agent, token) circular buffer by
        deleting the oldest rows beyond ``self.buffer_size`` for this
        (agent, token) pair. Pruning happens on every emit so the DB
        size stays bounded under sustained streams.
        """
        import json

        key = signal.effective_idempotency_key()
        payload_blob = (
            json.dumps(signal.payload, default=str, sort_keys=True)
            if signal.payload is not None
            else None
        )
        with self._tx() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO signals(
                        agent_id, token, magnitude, ts,
                        payload_json, idempotency_key
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.agent_id,
                        signal.token,
                        float(signal.magnitude),
                        _iso(signal.timestamp),
                        payload_blob,
                        key,
                    ),
                )
            except sqlite3.IntegrityError:
                return False
            # Circular buffer: prune oldest rows for this (agent, token)
            # past the configured size.
            cur.execute(
                """
                DELETE FROM signals
                WHERE id IN (
                    SELECT id FROM signals
                    WHERE agent_id = ? AND token = ?
                    ORDER BY ts DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (signal.agent_id, signal.token, self.buffer_size),
            )
        return True

    # ── Read ────────────────────────────────────────────────────────────

    def concentration(
        self,
        agent_id: str,
        token: SignalToken,
        window=DEFAULT_WINDOW,
        half_life=DEFAULT_HALF_LIFE,
        now: Optional[datetime] = None,
    ) -> float:
        """Time-weighted exponential-decay integral of ``(agent_id, token)``
        magnitudes within the past ``window``.

        ``half_life`` controls how quickly past emissions matter less.
        At ``t == half_life`` after an emission, its weight is exactly
        0.5; at ``2 * half_life``, exactly 0.25; and so on.
        """
        win = parse_duration(window)
        hl = parse_duration(half_life)
        if hl.total_seconds() <= 0:
            raise ValueError("half_life must be > 0")
        now = now or _utcnow()
        cutoff = now - win
        rows = self._conn.execute(
            """
            SELECT magnitude, ts FROM signals
            WHERE agent_id = ? AND token = ? AND ts >= ?
            """,
            (agent_id, token, _iso(cutoff)),
        ).fetchall()
        if not rows:
            return 0.0
        hl_seconds = hl.total_seconds()
        total = 0.0
        for r in rows:
            ts = _parse_iso(r["ts"])
            age_s = (now - ts).total_seconds()
            if age_s < 0:
                # Future-dated signal — treat as if at now (weight 1.0).
                age_s = 0.0
            weight = math.pow(0.5, age_s / hl_seconds)
            total += float(r["magnitude"]) * weight
        return total

    def joint_concentration(
        self,
        agent_id: str,
        token_pair: tuple[SignalToken, SignalToken],
        window=DEFAULT_WINDOW,
        half_life=DEFAULT_HALF_LIFE,
        now: Optional[datetime] = None,
    ) -> float:
        """Product of concentrations across two tokens — the HIPV-blend
        semantic. Useful for triggers like "agent is stressed AND
        requesting help" where the conjunction is what matters."""
        a, b = token_pair
        ca = self.concentration(agent_id, a, window, half_life, now)
        cb = self.concentration(agent_id, b, window, half_life, now)
        return ca * cb

    def recent_signals(
        self,
        agent_id: str,
        window=DEFAULT_WINDOW,
        token: Optional[SignalToken] = None,
        now: Optional[datetime] = None,
    ) -> list[EmittedSignal]:
        """Raw events for an agent inside a window. Useful for debugging
        and for the capacity harness which needs sample paths, not just
        integrals."""
        import json

        win = parse_duration(window)
        now = now or _utcnow()
        cutoff = now - win
        if token is None:
            rows = self._conn.execute(
                """
                SELECT agent_id, token, magnitude, ts, payload_json
                FROM signals
                WHERE agent_id = ? AND ts >= ?
                ORDER BY ts ASC, id ASC
                """,
                (agent_id, _iso(cutoff)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT agent_id, token, magnitude, ts, payload_json
                FROM signals
                WHERE agent_id = ? AND token = ? AND ts >= ?
                ORDER BY ts ASC, id ASC
                """,
                (agent_id, token, _iso(cutoff)),
            ).fetchall()
        out: list[EmittedSignal] = []
        for r in rows:
            payload = json.loads(r["payload_json"]) if r["payload_json"] else None
            out.append(
                EmittedSignal(
                    agent_id=r["agent_id"],
                    token=r["token"],  # type: ignore[arg-type]
                    magnitude=float(r["magnitude"]),
                    timestamp=_parse_iso(r["ts"]),
                    payload=payload,
                )
            )
        return out

    def known_agents(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT agent_id FROM signals ORDER BY agent_id"
        ).fetchall()
        return [r["agent_id"] for r in rows]

    def count_for(self, agent_id: str, token: Optional[SignalToken] = None) -> int:
        if token is None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE agent_id = ? AND token = ?",
                (agent_id, token),
            ).fetchone()
        return int(row["n"])

    def profile(
        self,
        agent_id: str,
        window=DEFAULT_WINDOW,
        half_life=DEFAULT_HALF_LIFE,
        now: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Concentration vector across all five tokens — used by
        triggers + CLI for a one-shot view of an agent's state."""
        return {t: self.concentration(agent_id, t, window, half_life, now) for t in SIGNAL_TOKENS}


# ── Singleton accessor ─────────────────────────────────────────────────────


_default_store: Optional[SignalStore] = None
_default_lock = threading.Lock()


def get_default_store() -> SignalStore:
    """Return (and lazily construct) the shared store at the default path."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = SignalStore()
        return _default_store


def _reset_default_store_for_tests() -> None:
    global _default_store
    with _default_lock:
        if _default_store is not None:
            _default_store.close()
            _default_store = None
