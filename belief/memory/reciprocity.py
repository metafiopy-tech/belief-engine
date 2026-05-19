"""Reciprocity Ledger — per-agent contribution accounting (mycorrhizal Stage 1).

The Belief Engine has historically treated every request as anonymous. There
was no notion of which caller produced value versus which consumed it. The
biological-market literature (Kiers et al. 2011 *Science*; Wyatt, Kiers,
Gardner & West 2014 *Evolution*) is unambiguous: mutualism is stable when
partners measure each other's contributions and allocate rewards in
proportion to observed performance. This module is the foundation that every
downstream mycorrhizal stage (hubs, sanctions, niche credit) builds on.

For each ``agent_id`` we track two streams of events:

* **Requests** (``carbon_received``) — compute / token / tool-call cost the
  engine spent on this agent's behalf. Updated by request entry points.
* **Contributions** (``nutrients_returned``) — validated outputs the agent
  put back into the soil layer. Updated by the decomposer hook and by any
  future soil-write path that wants to credit a constructor.

The derived metric is ``exchange_rate(window)`` —
``nutrients_returned / max(carbon_received, epsilon)`` over a rolling
window. Routing in Session 5 will consume this to bias allocation toward
high-exchange-rate agents (Wyatt et al. 2014's "in direct relation to the
relative amount of resources received").

Storage: SQLite at ``~/.belief-engine/reciprocity.db``. Same pattern as
``belief.evolution.archive`` — persistent connection, WAL mode for safe
concurrent readers, atomic-write semantics for individual events.

Idempotency: every write accepts an optional ``idempotency_key``. Duplicate
keys are silently dropped via a partial UNIQUE index. Two sessions racing
on the same event will not double-count.

This module does **not** make routing or sanction decisions — it only
observes and tallies. Session 5 (hubs + sanctions) consumes these reads.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("belief.memory.reciprocity")

# ── Defaults ────────────────────────────────────────────────────────────────

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_DB_PATH = _BELIEF_HOME / "reciprocity.db"

# Small epsilon prevents division-by-zero for agents that have produced
# contributions but never been charged. The value (0.001 "carbon units") is
# small enough that any non-trivial cost dominates it.
_EPSILON = 1e-3

# Default window for read APIs. Matches the biological "recent behavior over
# lifetime" framing — Whiteside et al. 2019 fungal exchange rates respond to
# local conditions, not lifetime averages.
DEFAULT_WINDOW = "7d"


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentStats:
    """A read-only snapshot of one agent's ledger state over a window."""

    agent_id: str
    carbon_received: float
    nutrients_returned: float
    exchange_rate: float
    request_count: int
    contribution_count: int
    last_seen_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    window: str = DEFAULT_WINDOW

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "carbon_received": round(self.carbon_received, 6),
            "nutrients_returned": round(self.nutrients_returned, 6),
            "exchange_rate": round(self.exchange_rate, 6),
            "request_count": self.request_count,
            "contribution_count": self.contribution_count,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "window": self.window,
        }


# ── Window parsing ─────────────────────────────────────────────────────────


_WINDOW_RE = re.compile(r"^(\d+)([dhm])$|^all$", re.IGNORECASE)


def _parse_window(window: str) -> Optional[timedelta]:
    """Parse a window spec into a ``timedelta`` (or ``None`` for ``"all"``).

    Accepts ``"7d"``, ``"24h"``, ``"30m"``, ``"all"``. Raises ``ValueError``
    on malformed input rather than silently treating it as "all" — caller
    bugs should surface, not propagate.
    """
    if window is None:
        raise ValueError("window must be a string, got None")
    s = window.strip().lower()
    if s == "all":
        return None
    m = _WINDOW_RE.match(s)
    if not m or m.group(1) is None:
        raise ValueError(f"unrecognized window spec: {window!r} (try '7d', '24h', '30m', 'all')")
    n = int(m.group(1))
    unit = m.group(2)
    if n <= 0:
        raise ValueError(f"window quantity must be positive, got {n}")
    if unit == "d":
        return timedelta(days=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "m":
        return timedelta(minutes=n)
    # _WINDOW_RE only matches d/h/m, so this is unreachable.
    raise ValueError(f"unsupported window unit: {unit!r}")  # pragma: no cover


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# ── The ledger ──────────────────────────────────────────────────────────────


class ReciprocityLedger:
    """SQLite-backed event log + read-side aggregator.

    Thread-safe: the connection is opened with ``check_same_thread=False``
    and writes are serialized through a process-local lock. WAL mode lets
    readers proceed without blocking writers. For multi-process safety we
    rely on SQLite's file-level locking — the partial UNIQUE index on
    ``idempotency_key`` is what guarantees duplicate-event safety across
    processes, not the in-process lock.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        # WAL = concurrent readers don't block the writer; safe for
        # CLI + daemon both touching this file. NORMAL sync trades a
        # vanishingly small crash-recovery window for substantial
        # write throughput, matching the soil-layer trade-off.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id     TEXT PRIMARY KEY,
                    created_at   TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id        TEXT NOT NULL,
                    event_type      TEXT NOT NULL
                                    CHECK(event_type IN ('request','contribution')),
                    value           REAL NOT NULL,
                    nutrient_id     TEXT,
                    idempotency_key TEXT,
                    ts              TEXT NOT NULL
                )
                """
            )
            # Partial UNIQUE index: idempotency keys collide only when
            # both are non-null. Allows nullable column with strict
            # dedup on real keys.
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idem
                ON events(idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_agent_ts
                ON events(agent_id, ts)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_type_ts
                ON events(event_type, ts)
                """
            )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        """Lock-guarded write transaction.

        Acquires the process-local lock and yields a cursor inside an
        IMMEDIATE transaction. On exit, commits unless an exception was
        raised (in which case the connection's ``__exit__`` rolls back).
        """
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
        """Close the underlying connection. Tests that re-open the DB should
        call this first to release the WAL files cleanly."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ── Write API ───────────────────────────────────────────────────────

    def _touch_agent(self, cur: sqlite3.Cursor, agent_id: str, now: datetime) -> None:
        cur.execute(
            """
            INSERT INTO agents(agent_id, created_at, last_seen_at)
            VALUES (?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (agent_id, _iso(now), _iso(now)),
        )

    def record_request(
        self,
        agent_id: str,
        cost: float,
        idempotency_key: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> bool:
        """Charge ``cost`` units of carbon (compute) against ``agent_id``.

        Returns ``True`` if the event was recorded, ``False`` if the
        idempotency key matched a prior event and the write was skipped.
        Never raises on duplicates — duplicates are an expected condition
        in a system where multiple subsystems may try to credit the same
        request.

        ``cost`` must be non-negative. A zero-cost request is valid (it
        records the touch without inflating denominator) and useful for
        marking liveness.
        """
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if cost < 0:
            raise ValueError(f"cost must be >= 0, got {cost}")
        now = ts or _utcnow()
        with self._tx() as cur:
            self._touch_agent(cur, agent_id, now)
            try:
                cur.execute(
                    """
                    INSERT INTO events(agent_id, event_type, value,
                                       nutrient_id, idempotency_key, ts)
                    VALUES (?, 'request', ?, NULL, ?, ?)
                    """,
                    (agent_id, float(cost), idempotency_key, _iso(now)),
                )
                return True
            except sqlite3.IntegrityError:
                # idempotency_key already present — silently skip.
                return False

    def record_contribution(
        self,
        agent_id: str,
        nutrient_value: float,
        nutrient_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> bool:
        """Credit ``agent_id`` for a validated soil deposit.

        ``nutrient_value`` is the per-event credit. The decomposer typically
        passes 1.0 (one nutrient = one unit of credit); Session 2's niche
        ledger will pass smaller values (0.1) for downstream reference
        credit to keep widely-used niches from dominating the ledger from a
        single big tool.

        Returns ``True`` if recorded, ``False`` if the idempotency key
        already existed.
        """
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if nutrient_value < 0:
            raise ValueError(f"nutrient_value must be >= 0, got {nutrient_value}")
        now = ts or _utcnow()
        with self._tx() as cur:
            self._touch_agent(cur, agent_id, now)
            try:
                cur.execute(
                    """
                    INSERT INTO events(agent_id, event_type, value,
                                       nutrient_id, idempotency_key, ts)
                    VALUES (?, 'contribution', ?, ?, ?, ?)
                    """,
                    (
                        agent_id,
                        float(nutrient_value),
                        nutrient_id,
                        idempotency_key,
                        _iso(now),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    # ── Read API ────────────────────────────────────────────────────────

    def _window_cutoff(self, window: str) -> Optional[str]:
        delta = _parse_window(window)
        if delta is None:
            return None
        return _iso(_utcnow() - delta)

    def _aggregate(self, agent_id: str, window: str) -> AgentStats:
        cutoff = self._window_cutoff(window)
        if cutoff is None:
            row = self._conn.execute(
                """
                SELECT
                    SUM(CASE WHEN event_type='request'      THEN value ELSE 0 END) AS carbon,
                    SUM(CASE WHEN event_type='contribution' THEN value ELSE 0 END) AS nutrients,
                    SUM(CASE WHEN event_type='request'      THEN 1 ELSE 0 END) AS req_n,
                    SUM(CASE WHEN event_type='contribution' THEN 1 ELSE 0 END) AS con_n
                FROM events WHERE agent_id = ?
                """,
                (agent_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT
                    SUM(CASE WHEN event_type='request'      THEN value ELSE 0 END) AS carbon,
                    SUM(CASE WHEN event_type='contribution' THEN value ELSE 0 END) AS nutrients,
                    SUM(CASE WHEN event_type='request'      THEN 1 ELSE 0 END) AS req_n,
                    SUM(CASE WHEN event_type='contribution' THEN 1 ELSE 0 END) AS con_n
                FROM events WHERE agent_id = ? AND ts >= ?
                """,
                (agent_id, cutoff),
            ).fetchone()
        meta = self._conn.execute(
            "SELECT created_at, last_seen_at FROM agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        carbon = float(row["carbon"] or 0.0)
        nutrients = float(row["nutrients"] or 0.0)
        return AgentStats(
            agent_id=agent_id,
            carbon_received=carbon,
            nutrients_returned=nutrients,
            exchange_rate=nutrients / max(carbon, _EPSILON),
            request_count=int(row["req_n"] or 0),
            contribution_count=int(row["con_n"] or 0),
            last_seen_at=_parse_iso(meta["last_seen_at"]) if meta else None,
            created_at=_parse_iso(meta["created_at"]) if meta else None,
            window=window,
        )

    def exchange_rate(self, agent_id: str, window: str = DEFAULT_WINDOW) -> float:
        """Return ``nutrients_returned / max(carbon_received, eps)`` over the
        window. Unknown agents return 0.0 (not undefined, not error)."""
        return self._aggregate(agent_id, window).exchange_rate if self._exists(agent_id) else 0.0

    def stats(self, agent_id: str, window: str = DEFAULT_WINDOW) -> AgentStats:
        """Full stats snapshot for one agent. Unknown agents return a zeroed
        ``AgentStats`` (created_at/last_seen_at None) so callers can render
        a row instead of branching on missing."""
        if not self._exists(agent_id):
            return AgentStats(
                agent_id=agent_id,
                carbon_received=0.0,
                nutrients_returned=0.0,
                exchange_rate=0.0,
                request_count=0,
                contribution_count=0,
                window=window,
            )
        return self._aggregate(agent_id, window)

    def _exists(self, agent_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM agents WHERE agent_id = ? LIMIT 1",
            (agent_id,),
        ).fetchone()
        return row is not None

    def all_agent_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT agent_id FROM agents ORDER BY agent_id").fetchall()
        return [r["agent_id"] for r in rows]

    def rank_agents(self, window: str = DEFAULT_WINDOW) -> list[AgentStats]:
        """Every known agent's stats, sorted by exchange rate descending.

        Agents with zero activity in the window still appear at the bottom
        (their exchange_rate is 0.0 from the epsilon floor). Use
        ``agents_below_threshold`` to filter for sanctions in Session 5.
        """
        return sorted(
            (self._aggregate(aid, window) for aid in self.all_agent_ids()),
            key=lambda s: (-s.exchange_rate, s.agent_id),
        )

    def agents_below_threshold(
        self, threshold: float, window: str = DEFAULT_WINDOW
    ) -> list[AgentStats]:
        """Agents whose exchange rate is strictly below ``threshold``.

        Session 5's ``SanctionsEngine`` consumes this. Returned in
        ascending exchange-rate order so the worst offenders surface first.
        """
        return sorted(
            (s for s in self.rank_agents(window) if s.exchange_rate < threshold),
            key=lambda s: (s.exchange_rate, s.agent_id),
        )


# ── Process-wide singleton accessor ────────────────────────────────────────


_default_ledger: Optional[ReciprocityLedger] = None
_default_lock = threading.Lock()


def get_default_ledger() -> ReciprocityLedger:
    """Return (and lazily construct) the shared ledger at the default path.

    Use this from hook sites (decomposer, future routing layer) that need
    a singleton ledger without each call site managing lifecycle. Tests
    construct ``ReciprocityLedger`` directly against ``tmp_path``.
    """
    global _default_ledger
    with _default_lock:
        if _default_ledger is None:
            _default_ledger = ReciprocityLedger()
        return _default_ledger


def _reset_default_ledger_for_tests() -> None:
    """Test helper — closes and clears the singleton so tests don't bleed
    state across cases. Not part of the public API."""
    global _default_ledger
    with _default_lock:
        if _default_ledger is not None:
            _default_ledger.close()
            _default_ledger = None


# ── CLI rendering ──────────────────────────────────────────────────────────


def cli_format_ledger(ledger: ReciprocityLedger, window: str = DEFAULT_WINDOW) -> str:
    """Format a ranked table for ``belief reciprocity``.

    Output is plain text — no rich formatting — so it tees cleanly into
    pipelines and the existing ecology-organ output style.
    """
    rows = ledger.rank_agents(window=window)
    header = (
        f"Reciprocity ledger — window={window} db={ledger._db_path}\n"
        f"  {len(rows)} known agent{'s' if len(rows) != 1 else ''}"
    )
    if not rows:
        return header + "\n  (no events recorded yet; the ledger is empty)"
    lines = [
        header,
        "",
        f"  {'agent_id':<28} {'carbon':>10} {'nutrients':>12} "
        f"{'exchange':>10} {'req':>5} {'con':>5}  last_seen",
        f"  {'-' * 28} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 5} {'-' * 5}  {'-' * 19}",
    ]
    for s in rows:
        last = s.last_seen_at.strftime("%Y-%m-%d %H:%M:%S") if s.last_seen_at else "—"
        lines.append(
            f"  {s.agent_id[:28]:<28} "
            f"{s.carbon_received:>10.4f} "
            f"{s.nutrients_returned:>12.4f} "
            f"{s.exchange_rate:>10.4f} "
            f"{s.request_count:>5d} "
            f"{s.contribution_count:>5d}  "
            f"{last}"
        )
    return "\n".join(lines)


def cli_show(window: str = DEFAULT_WINDOW) -> str:
    """Implementation of ``belief reciprocity`` — renders the default
    ledger over the requested window."""
    return cli_format_ledger(get_default_ledger(), window=window)
