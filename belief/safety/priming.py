"""Defense-priming propagation (mycorrhizal Stage 6, Area 7).

Babikova et al. 2013 (*Ecology Letters*) showed plants connected by a
mycorrhizal network warn neighbours of aphid attack — and crucially the
receivers don't fully induce defenses, they enter a *primed* state: defense
machinery poised but not deployed. Priming is cheap; full induction is
expensive (Heil & Karban 2009). The evolved middle is to commit to costly
defense only on confirmed attack.

The Belief Engine analogue, two warning classes:

* **Priming-class** (gossip with TTL): an unconfirmed pattern that *might* be
  problematic. Propagates by gossip — each receiver forwards to up to K
  connections, decrementing TTL. Receivers raise a sentinel threshold on the
  pattern; they do NOT refuse operations.
* **Covenant-class** (eager via hubs): a confirmed safety/covenant violation.
  Propagates eagerly through hubs. Receivers refuse the matching operation
  until the warning is explicitly cleared.

Both decay. Priming warnings have a 24h half-life, covenant 7d. A re-triggered
warning refreshes its expiry. Expired warnings are pruned on read.

**Build-path safety.** The build pipeline does NOT call ``check_operation`` —
there are no autonomous agents to gate yet. This module ships the propagation
+ decay machinery and the ``check_operation`` consumer API; wiring it into a
real operation-gating path waits for autonomous agents. Until then warnings
are observability-only (the ``belief warnings`` CLI) and never block a build.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("belief.safety.priming")

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_DB_PATH = _BELIEF_HOME / "warnings.db"

DEFAULT_PRIMING_HALF_LIFE = timedelta(hours=24)
DEFAULT_COVENANT_HALF_LIFE = timedelta(days=7)
DEFAULT_GOSSIP_K = 3
DEFAULT_GOSSIP_TTL = 5


class WarningKind(str, Enum):
    PRIMING = "priming"
    COVENANT = "covenant"


class Warning(BaseModel):
    """One propagated warning. ``hops_remaining`` is meaningful only for
    priming-class (gossip); covenant-class broadcasts eagerly."""

    warning_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    kind: WarningKind
    pattern: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    originating_agent_id: str = Field(..., min_length=1)
    hops_remaining: int = 0
    evidence: dict = Field(default_factory=dict)

    @field_validator("created_at", "expires_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v


@dataclass(frozen=True)
class CheckResult:
    """Result of ``check_operation`` for one agent + operation.

    ``primed_patterns`` lists priming-class warnings whose pattern matched
    the operation (the agent should raise its sentinel but proceed).
    ``blocked`` is True iff a covenant-class warning matched (the operation
    should be refused until the warning clears)."""

    agent_id: str
    operation: str
    primed_patterns: list[str] = field(default_factory=list)
    blocking_warnings: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.blocking_warnings)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


class WarningStore:
    """SQLite-backed warning store with decay-on-read pruning."""

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
                CREATE TABLE IF NOT EXISTS warnings (
                    warning_id           TEXT PRIMARY KEY,
                    kind                 TEXT NOT NULL,
                    pattern              TEXT NOT NULL,
                    created_at           TEXT NOT NULL,
                    expires_at           TEXT NOT NULL,
                    originating_agent_id TEXT NOT NULL,
                    hops_remaining       INTEGER NOT NULL DEFAULT 0,
                    evidence_json        TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_warnings_kind ON warnings(kind)")

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

    def upsert(self, w: Warning) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO warnings(warning_id, kind, pattern, created_at,
                                     expires_at, originating_agent_id,
                                     hops_remaining, evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(warning_id) DO UPDATE SET
                    expires_at = excluded.expires_at,
                    hops_remaining = excluded.hops_remaining,
                    evidence_json = excluded.evidence_json
                """,
                (
                    w.warning_id,
                    w.kind.value,
                    w.pattern,
                    _iso(w.created_at),
                    _iso(w.expires_at),
                    w.originating_agent_id,
                    int(w.hops_remaining),
                    json.dumps(w.evidence, default=str),
                ),
            )

    def _prune_expired(self, now: datetime) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM warnings WHERE expires_at < ?", (_iso(now),))

    def _row_to_warning(self, row: sqlite3.Row) -> Warning:
        return Warning(
            warning_id=row["warning_id"],
            kind=WarningKind(row["kind"]),
            pattern=row["pattern"],
            created_at=_parse_iso(row["created_at"]),
            expires_at=_parse_iso(row["expires_at"]),
            originating_agent_id=row["originating_agent_id"],
            hops_remaining=int(row["hops_remaining"]),
            evidence=json.loads(row["evidence_json"]) if row["evidence_json"] else {},
        )

    def active_warnings(self, now: Optional[datetime] = None) -> list[Warning]:
        now = now or _utcnow()
        self._prune_expired(now)
        rows = self._conn.execute(
            "SELECT * FROM warnings WHERE expires_at >= ? ORDER BY created_at DESC",
            (_iso(now),),
        ).fetchall()
        return [self._row_to_warning(r) for r in rows]

    def get(self, warning_id: str) -> Optional[Warning]:
        row = self._conn.execute(
            "SELECT * FROM warnings WHERE warning_id = ?", (warning_id,)
        ).fetchone()
        return self._row_to_warning(row) if row else None

    def refresh(self, warning_id: str, new_expiry: datetime) -> bool:
        with self._tx() as cur:
            cur.execute(
                "UPDATE warnings SET expires_at = ? WHERE warning_id = ?",
                (_iso(new_expiry), warning_id),
            )
            return cur.rowcount > 0


class PrimingPropagator:
    """Emits + propagates warnings and answers operation checks."""

    def __init__(
        self,
        store: Optional[WarningStore] = None,
        priming_half_life: timedelta = DEFAULT_PRIMING_HALF_LIFE,
        covenant_half_life: timedelta = DEFAULT_COVENANT_HALF_LIFE,
        gossip_k: int = DEFAULT_GOSSIP_K,
        gossip_ttl: int = DEFAULT_GOSSIP_TTL,
    ) -> None:
        self._store = store if store is not None else WarningStore()
        self.priming_half_life = priming_half_life
        self.covenant_half_life = covenant_half_life
        self.gossip_k = int(gossip_k)
        self.gossip_ttl = int(gossip_ttl)

    # ── Emit ────────────────────────────────────────────────────────────

    def emit_priming(
        self,
        pattern: str,
        evidence: Optional[dict] = None,
        originating_agent_id: str = "belief_engine",
        now: Optional[datetime] = None,
    ) -> Warning:
        now = now or _utcnow()
        w = Warning(
            kind=WarningKind.PRIMING,
            pattern=pattern,
            created_at=now,
            expires_at=now + self.priming_half_life,
            originating_agent_id=originating_agent_id,
            hops_remaining=self.gossip_ttl,
            evidence=evidence or {},
        )
        self._store.upsert(w)
        return w

    def emit_covenant(
        self,
        pattern: str,
        evidence: Optional[dict] = None,
        originating_agent_id: str = "belief_engine",
        now: Optional[datetime] = None,
    ) -> Warning:
        now = now or _utcnow()
        w = Warning(
            kind=WarningKind.COVENANT,
            pattern=pattern,
            created_at=now,
            expires_at=now + self.covenant_half_life,
            originating_agent_id=originating_agent_id,
            hops_remaining=0,  # eager broadcast, not gossip
            evidence=evidence or {},
        )
        self._store.upsert(w)
        return w

    # ── Gossip simulation ───────────────────────────────────────────────

    def simulate_gossip_reach(
        self, agent_ids: list[str], k: Optional[int] = None, ttl: Optional[int] = None
    ) -> set[str]:
        """Deterministically simulate how far a priming warning spreads over
        a set of agents: starting from one origin, each tick up to K new
        agents are reached, for TTL ticks. Returns the reached set.

        Faithful to the gossip-with-TTL pattern without needing a live
        agent network — the brief's reach claim is ``K^TTL`` agents within
        TTL ticks (capped at the population size)."""
        k = self.gossip_k if k is None else k
        ttl = self.gossip_ttl if ttl is None else ttl
        if not agent_ids:
            return set()
        reached: list[str] = [agent_ids[0]]
        frontier = [agent_ids[0]]
        idx = 1
        for _ in range(ttl):
            new_frontier: list[str] = []
            for _carrier in frontier:
                for _ in range(k):
                    if idx >= len(agent_ids):
                        break
                    nxt = agent_ids[idx]
                    idx += 1
                    reached.append(nxt)
                    new_frontier.append(nxt)
            frontier = new_frontier
            if not frontier:
                break
        return set(reached)

    # ── Read / check ────────────────────────────────────────────────────

    def current_warnings(
        self, agent_id: Optional[str] = None, now: Optional[datetime] = None
    ) -> list[Warning]:
        """All active (non-expired) warnings. ``agent_id`` is accepted for
        a future per-agent priming-set view; at this stage warnings are
        engine-global so the param is informational."""
        return self._store.active_warnings(now=now)

    def check_operation(
        self, agent_id: str, operation_description: str, now: Optional[datetime] = None
    ) -> CheckResult:
        """Check an operation against active warnings.

        Priming-class warnings whose pattern is a substring of the
        operation raise a sentinel (returned in ``primed_patterns``).
        Covenant-class matches block the operation (``blocking_warnings``).
        Matching is case-insensitive substring — a deliberately simple
        first cut; semantic matching can come later."""
        op = operation_description.lower()
        primed: list[str] = []
        blocking: list[str] = []
        for w in self.current_warnings(now=now):
            if w.pattern.lower() in op:
                if w.kind is WarningKind.COVENANT:
                    blocking.append(w.pattern)
                else:
                    primed.append(w.pattern)
        return CheckResult(
            agent_id=agent_id,
            operation=operation_description,
            primed_patterns=primed,
            blocking_warnings=blocking,
        )

    def retrigger(self, pattern: str, now: Optional[datetime] = None) -> int:
        """Re-observing a pattern refreshes the expiry of every active
        warning with that pattern. Returns how many were refreshed."""
        now = now or _utcnow()
        count = 0
        for w in self.current_warnings(now=now):
            if w.pattern == pattern:
                half_life = (
                    self.covenant_half_life
                    if w.kind is WarningKind.COVENANT
                    else self.priming_half_life
                )
                if self._store.refresh(w.warning_id, now + half_life):
                    count += 1
        return count


# ── Singleton + CLI ─────────────────────────────────────────────────────────

_default_propagator: Optional[PrimingPropagator] = None
_default_lock = threading.Lock()


def get_default_propagator() -> PrimingPropagator:
    global _default_propagator
    with _default_lock:
        if _default_propagator is None:
            _default_propagator = PrimingPropagator()
        return _default_propagator


def _reset_default_propagator_for_tests() -> None:
    global _default_propagator
    with _default_lock:
        if _default_propagator is not None:
            _default_propagator._store.close()
            _default_propagator = None


def cli_format_warnings(prop: PrimingPropagator) -> str:
    warnings = prop.current_warnings()
    header = f"Active warnings — {len(warnings)}"
    if not warnings:
        return header + "\n  (no active warnings)"
    lines = [
        header,
        "",
        f"  {'kind':<9} {'pattern':<32} {'hops':>4}  expires_at",
        f"  {'-' * 9} {'-' * 32} {'-' * 4}  {'-' * 19}",
    ]
    for w in warnings:
        lines.append(
            f"  {w.kind.value:<9} {w.pattern[:32]:<32} {w.hops_remaining:>4}  "
            f"{w.expires_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    return "\n".join(lines)


def cli_show() -> str:
    return cli_format_warnings(get_default_propagator())
