"""Niche-Modification Ledger — capability bootstrap accounting (mycorrhizal Stage 2).

Niche construction theory (Odling-Smee, Laland & Feldman 2003): organisms
modify environments in ways that change selection pressures on themselves
and their descendants. The Belief Engine's autocatalytic loop is a software
analog — every new tool registered via ``NEW_TOOL``, every novel primitive
extracted from a build, every newly-promoted covenant *adds capability* to
the system. Without a ledger that tracks these additions and *who built
them*, an agent that builds widely-useful primitives is credited identically
to one whose work is never reused.

This module is the substrate for downstream-reference credit assignment:
when a later build consumes a previously-constructed niche, the original
constructor's reciprocity ledger gets a small fixed credit. Widely-used
niches accumulate substantial credit over time; unused ones don't. The
emergent incentive is: build things others can use.

Storage: SQLite at ``~/.belief-engine/niches.db`` (separate file from the
reciprocity ledger so Stage 3 snapshot semantics stay independent and
tests isolate cleanly). WAL mode, partial UNIQUE indexes for idempotent
record_modification and record_reference paths.

Integration with the reciprocity ledger: ``record_reference`` calls into
``ReciprocityLedger.record_contribution`` for the *original* constructor
(not the referring build's agent). Per the session spec the credit is a
small fixed value (0.1 per reference) so widely-used niches earn at
sub-linear rate from any one reuse but accumulate substantial weight from
heavy use. Idempotency keys prevent double-credit on replays.

This module does **not** decide routing or scoring weights — Session 5's
router will read ``top_constructors`` and apply policy. Stage 2 only
observes and credits.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("belief.memory.niche_ledger")


# ── Defaults ────────────────────────────────────────────────────────────────

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_DB_PATH = _BELIEF_HOME / "niches.db"

# Per-reference credit awarded to the original constructor. Small enough
# that a single high-traffic niche doesn't dominate the reciprocity table
# from one tool; large enough that ~10 references match the credit of a
# fresh nutrient deposit (which is 1.0). Session 5 will tune this if the
# emergent ranking is too top-heavy or too flat.
DEFAULT_REFERENCE_CREDIT = 0.1

# The set of niche kinds is closed — bumping it requires a migration.
# Keeping it tight prevents the ledger from sprawling into a general
# event log; this is specifically about *capability additions*.
NICHE_KINDS: tuple[str, ...] = ("tool", "primitive", "pattern", "covenant")


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NicheRecord:
    """A read-only snapshot of one niche-modification entry."""

    niche_id: str
    constructing_agent_id: str
    kind: str
    soil_reference: str
    pre_state_description: str
    post_state_description: str
    created_at: datetime
    last_referenced_at: Optional[datetime]
    reference_count: int

    def to_dict(self) -> dict:
        return {
            "niche_id": self.niche_id,
            "constructing_agent_id": self.constructing_agent_id,
            "kind": self.kind,
            "soil_reference": self.soil_reference,
            "pre_state_description": self.pre_state_description,
            "post_state_description": self.post_state_description,
            "created_at": self.created_at.isoformat(),
            "last_referenced_at": (
                self.last_referenced_at.isoformat() if self.last_referenced_at else None
            ),
            "reference_count": self.reference_count,
        }


@dataclass(frozen=True)
class ConstructorStats:
    """A read-only snapshot of one constructor's downstream impact."""

    agent_id: str
    niche_count: int
    total_references: int

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "niche_count": self.niche_count,
            "total_references": self.total_references,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────


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


def _row_to_niche(row: sqlite3.Row) -> NicheRecord:
    return NicheRecord(
        niche_id=row["niche_id"],
        constructing_agent_id=row["constructing_agent_id"],
        kind=row["kind"],
        soil_reference=row["soil_reference"],
        pre_state_description=row["pre_state_description"] or "",
        post_state_description=row["post_state_description"] or "",
        created_at=_parse_iso(row["created_at"]) or _utcnow(),
        last_referenced_at=_parse_iso(row["last_referenced_at"]),
        reference_count=int(row["reference_count"] or 0),
    )


# ── The ledger ──────────────────────────────────────────────────────────────


class NicheLedger:
    """SQLite-backed niche-modification + reference event log.

    Two tables: ``niches`` (one row per capability addition, deduplicated
    by (kind, soil_reference)) and ``niche_references`` (one row per
    consumption event, idempotency-keyed). On every ``record_reference``,
    the constructor's reciprocity ledger gets a fixed credit so this
    module is the *bridge* from raw consumption to incentive shaping.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB_PATH,
        reference_credit: float = DEFAULT_REFERENCE_CREDIT,
        reciprocity_ledger=None,
    ) -> None:
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
        if reference_credit < 0:
            raise ValueError(f"reference_credit must be >= 0, got {reference_credit}")
        self.reference_credit = float(reference_credit)
        # Injected for tests; production uses the singleton. We accept a
        # ledger-like duck rather than the concrete class so a no-op
        # stand-in can be passed in suites that exercise the niche
        # ledger in isolation.
        self._reciprocity_ledger = reciprocity_ledger
        self._create_tables()

    def _create_tables(self) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS niches (
                    niche_id                 TEXT PRIMARY KEY,
                    constructing_agent_id    TEXT NOT NULL,
                    kind                     TEXT NOT NULL
                                             CHECK(kind IN
                                                ('tool','primitive','pattern','covenant')),
                    soil_reference           TEXT NOT NULL,
                    pre_state_description    TEXT NOT NULL DEFAULT '',
                    post_state_description   TEXT NOT NULL DEFAULT '',
                    created_at               TEXT NOT NULL,
                    last_referenced_at       TEXT,
                    reference_count          INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # (kind, soil_reference) is the natural dedup key — the same
            # ChromaDB tool id should never produce two distinct niche
            # rows. Partial-UNIQUE is overkill here because both fields
            # are NOT NULL, but a plain UNIQUE constraint suffices.
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_niches_soil_kind
                ON niches(kind, soil_reference)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_niches_constructor
                ON niches(constructing_agent_id)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS niche_references (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    niche_id            TEXT NOT NULL,
                    referring_build_id  TEXT NOT NULL,
                    idempotency_key     TEXT,
                    ts                  TEXT NOT NULL,
                    FOREIGN KEY(niche_id) REFERENCES niches(niche_id)
                )
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_niche_refs_idem
                ON niche_references(idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_niche_refs_niche
                ON niche_references(niche_id, ts)
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

    # ── Write API ───────────────────────────────────────────────────────

    def record_modification(
        self,
        constructing_agent_id: str,
        kind: str,
        soil_reference: str,
        pre_state_description: str = "",
        post_state_description: str = "",
        niche_id: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> str:
        """Register a new niche, or return the existing one's id.

        Dedup is by ``(kind, soil_reference)`` — registering the same tool
        twice from two code paths produces one row, and the returned
        ``niche_id`` is stable across calls. The pre/post descriptions
        are not updated on duplicate calls (first-write-wins) because
        re-derivation drift would otherwise overwrite known-good prose.

        Raises ``ValueError`` on empty agent_id or invalid kind. Returns
        the niche_id string in either case (insert or existing match).
        """
        if not constructing_agent_id:
            raise ValueError("constructing_agent_id must be a non-empty string")
        if kind not in NICHE_KINDS:
            raise ValueError(f"kind must be one of {NICHE_KINDS}, got {kind!r}")
        if not soil_reference:
            raise ValueError("soil_reference must be a non-empty string")

        now = ts or _utcnow()
        nid = niche_id or str(uuid.uuid4())
        with self._tx() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO niches(
                        niche_id, constructing_agent_id, kind, soil_reference,
                        pre_state_description, post_state_description,
                        created_at, last_referenced_at, reference_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0)
                    """,
                    (
                        nid,
                        constructing_agent_id,
                        kind,
                        soil_reference,
                        pre_state_description,
                        post_state_description,
                        _iso(now),
                    ),
                )
                return nid
            except sqlite3.IntegrityError:
                # (kind, soil_reference) already present — return the
                # existing niche_id rather than letting the caller think
                # they got a fresh registration.
                row = cur.execute(
                    """
                    SELECT niche_id FROM niches
                    WHERE kind = ? AND soil_reference = ?
                    """,
                    (kind, soil_reference),
                ).fetchone()
                if row is None:  # pragma: no cover — shouldn't happen
                    raise
                return row["niche_id"]

    def record_reference(
        self,
        niche_id: str,
        referring_build_id: str,
        idempotency_key: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> bool:
        """Record that ``referring_build_id`` used ``niche_id``.

        Side effect: the original constructor receives
        ``self.reference_credit`` credit in the reciprocity ledger, keyed
        ``niche-ref:<niche_id>:<referring_build_id>`` so replays of the
        recomposer for the same build never double-credit.

        Returns ``True`` if a new reference was recorded, ``False`` if
        the idempotency key matched a prior reference. Unknown niche ids
        are silently ignored (returns ``False``) rather than raising —
        the recomposer is a hot path and shouldn't crash on a stale id.
        """
        if not niche_id or not referring_build_id:
            raise ValueError("niche_id and referring_build_id are required")
        now = ts or _utcnow()
        idem = idempotency_key or f"niche-ref:{niche_id}:{referring_build_id}"
        constructor: Optional[str] = None
        with self._tx() as cur:
            niche_row = cur.execute(
                "SELECT constructing_agent_id FROM niches WHERE niche_id = ?",
                (niche_id,),
            ).fetchone()
            if niche_row is None:
                logger.debug(f"record_reference: unknown niche_id {niche_id!r}, skipping")
                return False
            constructor = niche_row["constructing_agent_id"]
            try:
                cur.execute(
                    """
                    INSERT INTO niche_references(
                        niche_id, referring_build_id, idempotency_key, ts
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (niche_id, referring_build_id, idem, _iso(now)),
                )
            except sqlite3.IntegrityError:
                # Replay — already recorded. Do not increment counters
                # and do not credit the constructor again.
                return False
            cur.execute(
                """
                UPDATE niches
                SET reference_count    = reference_count + 1,
                    last_referenced_at = ?
                WHERE niche_id = ?
                """,
                (_iso(now), niche_id),
            )

        # Cross-ledger credit propagation. Done outside the niche-ledger
        # transaction so a reciprocity-ledger hiccup never rolls back
        # the niche reference itself. Best-effort; logs and moves on.
        if constructor:
            try:
                ledger = self._reciprocity_ledger
                if ledger is None:
                    from belief.memory.reciprocity import get_default_ledger

                    ledger = get_default_ledger()
                ledger.record_contribution(
                    agent_id=constructor,
                    nutrient_value=self.reference_credit,
                    nutrient_id=f"niche-ref:{niche_id}",
                    idempotency_key=idem,
                )
            except Exception as e:  # pragma: no cover — best-effort
                logger.debug(f"Reciprocity downstream-credit skipped: {e}")
        return True

    # ── Read API ────────────────────────────────────────────────────────

    def get_niche(self, niche_id: str) -> Optional[NicheRecord]:
        row = self._conn.execute("SELECT * FROM niches WHERE niche_id = ?", (niche_id,)).fetchone()
        return _row_to_niche(row) if row else None

    def lookup_by_soil_reference(self, kind: str, soil_reference: str) -> Optional[NicheRecord]:
        """Reverse lookup — given a tool id or nutrient id, find its niche.

        Used by the recomposer/decomposer hooks to translate a soil-layer
        identifier into the niche_id needed for ``record_reference``.
        """
        row = self._conn.execute(
            "SELECT * FROM niches WHERE kind = ? AND soil_reference = ?",
            (kind, soil_reference),
        ).fetchone()
        return _row_to_niche(row) if row else None

    def niches_by_agent(self, agent_id: str) -> list[NicheRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM niches
            WHERE constructing_agent_id = ?
            ORDER BY reference_count DESC, created_at DESC
            """,
            (agent_id,),
        ).fetchall()
        return [_row_to_niche(r) for r in rows]

    def query_niches(self, text_search: str, kind: Optional[str] = None) -> list[NicheRecord]:
        """Discovery API. Case-insensitive substring match across the
        pre/post-state descriptions and the soil_reference field.

        Session 5 may replace this with an embedding-based search once
        we have descriptions consistently populated. For now, LIKE is
        adequate and avoids dragging chromadb into a SQLite module.
        """
        if kind is not None and kind not in NICHE_KINDS:
            raise ValueError(f"kind must be one of {NICHE_KINDS}, got {kind!r}")
        like = f"%{text_search.lower()}%"
        params: list = [like, like, like]
        sql = (
            "SELECT * FROM niches WHERE ("
            " LOWER(pre_state_description)  LIKE ?"
            " OR LOWER(post_state_description) LIKE ?"
            " OR LOWER(soil_reference) LIKE ?"
            ")"
        )
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY reference_count DESC, created_at DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_niche(r) for r in rows]

    def top_referenced(self, limit: int = 10) -> list[NicheRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM niches
            WHERE reference_count > 0
            ORDER BY reference_count DESC, last_referenced_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [_row_to_niche(r) for r in rows]

    def top_constructors(
        self, window: Optional[str] = None, limit: int = 10
    ) -> list[ConstructorStats]:
        """Agents ranked by total downstream references.

        If ``window`` is provided (e.g. ``"30d"``), only references whose
        timestamp falls inside the window count toward each agent's total
        — matching biological-market "recent reciprocity" semantics. If
        omitted, lifetime totals are returned.
        """
        if window is None:
            rows = self._conn.execute(
                """
                SELECT n.constructing_agent_id   AS agent_id,
                       COUNT(*)                  AS niche_count,
                       COALESCE(SUM(n.reference_count), 0) AS total_refs
                FROM niches n
                GROUP BY n.constructing_agent_id
                ORDER BY total_refs DESC, niche_count DESC, agent_id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                ConstructorStats(
                    agent_id=r["agent_id"],
                    niche_count=int(r["niche_count"]),
                    total_references=int(r["total_refs"]),
                )
                for r in rows
            ]
        # Windowed: count references inside the window, joined to niches.
        from belief.memory.reciprocity import _parse_window  # local import

        delta = _parse_window(window)
        if delta is None:
            return self.top_constructors(window=None, limit=limit)
        cutoff = _iso(_utcnow() - delta)
        rows = self._conn.execute(
            """
            SELECT n.constructing_agent_id   AS agent_id,
                   COUNT(DISTINCT n.niche_id) AS niche_count,
                   COUNT(r.id)               AS total_refs
            FROM niches n
            LEFT JOIN niche_references r
              ON r.niche_id = n.niche_id AND r.ts >= ?
            GROUP BY n.constructing_agent_id
            HAVING total_refs > 0
            ORDER BY total_refs DESC, niche_count DESC, agent_id
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
        return [
            ConstructorStats(
                agent_id=r["agent_id"],
                niche_count=int(r["niche_count"]),
                total_references=int(r["total_refs"]),
            )
            for r in rows
        ]

    def count_niches(self, kind: Optional[str] = None) -> int:
        if kind is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM niches").fetchone()
        else:
            if kind not in NICHE_KINDS:
                raise ValueError(f"kind must be one of {NICHE_KINDS}, got {kind!r}")
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM niches WHERE kind = ?", (kind,)
            ).fetchone()
        return int(row["n"])


# ── Process-wide singleton ─────────────────────────────────────────────────

_default_ledger: Optional[NicheLedger] = None
_default_lock = threading.Lock()


def get_default_ledger() -> NicheLedger:
    """Return (and lazily construct) the shared niche ledger at the default
    path. Hook sites (tool_registry, decomposer, recomposer) use this;
    tests construct ``NicheLedger`` directly against ``tmp_path``."""
    global _default_ledger
    with _default_lock:
        if _default_ledger is None:
            _default_ledger = NicheLedger()
        return _default_ledger


def _reset_default_ledger_for_tests() -> None:
    """Close + clear singleton so test cases don't bleed state."""
    global _default_ledger
    with _default_lock:
        if _default_ledger is not None:
            _default_ledger.close()
            _default_ledger = None


# ── CLI rendering ──────────────────────────────────────────────────────────


def cli_format_top(ledger: NicheLedger, limit: int = 10) -> str:
    """Render the ``belief niches`` default view — top-N referenced."""
    rows = ledger.top_referenced(limit=limit)
    total = ledger.count_niches()
    header = (
        f"Niche ledger — db={ledger._db_path}\n"
        f"  {total} total niche{'s' if total != 1 else ''}; "
        f"showing top {len(rows)} by reference count"
    )
    if not rows:
        return (
            header + "\n  (no referenced niches yet; either nothing constructed "
            "or nothing consumed)"
        )
    lines = [
        header,
        "",
        f"  {'kind':<10} {'refs':>5} {'agent':<24} {'soil_ref':<24} last_referenced",
        f"  {'-' * 10} {'-' * 5} {'-' * 24} {'-' * 24} {'-' * 19}",
    ]
    for n in rows:
        last = n.last_referenced_at.strftime("%Y-%m-%d %H:%M:%S") if n.last_referenced_at else "—"
        lines.append(
            f"  {n.kind:<10} "
            f"{n.reference_count:>5d} "
            f"{n.constructing_agent_id[:24]:<24} "
            f"{n.soil_reference[:24]:<24} "
            f"{last}"
        )
    return "\n".join(lines)


def cli_format_by_agent(ledger: NicheLedger, agent_id: str) -> str:
    rows = ledger.niches_by_agent(agent_id)
    header = (
        f"Niches constructed by {agent_id!r} — db={ledger._db_path}\n"
        f"  {len(rows)} niche{'s' if len(rows) != 1 else ''}"
    )
    if not rows:
        return header + "\n  (this agent has not been credited with any niches)"
    lines = [
        header,
        "",
        f"  {'kind':<10} {'refs':>5} {'soil_ref':<28} created_at",
        f"  {'-' * 10} {'-' * 5} {'-' * 28} {'-' * 19}",
    ]
    for n in rows:
        created = n.created_at.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(
            f"  {n.kind:<10} {n.reference_count:>5d} {n.soil_reference[:28]:<28} {created}"
        )
        if n.post_state_description:
            lines.append(f"      → {n.post_state_description[:80]}")
    return "\n".join(lines)


def cli_format_query(ledger: NicheLedger, text: str, kind: Optional[str] = None) -> str:
    rows = ledger.query_niches(text, kind=kind)
    header = (
        f"Niche query {text!r}" + (f" kind={kind}" if kind else "") + "\n"
        f"  {len(rows)} match{'es' if len(rows) != 1 else ''}"
    )
    if not rows:
        return header + "\n  (no niches match)"
    lines = [header, ""]
    for n in rows:
        lines.append(
            f"  [{n.kind}] {n.constructing_agent_id} — {n.soil_reference}"
            f" (refs={n.reference_count})"
        )
        if n.post_state_description:
            lines.append(f"      → {n.post_state_description[:120]}")
    return "\n".join(lines)


def cli_show(
    agent: Optional[str] = None,
    query: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Implementation of ``belief niches``. Routes to the right view based
    on which flag is set; ``--agent`` and ``--query`` are mutually
    exclusive but if both are supplied ``--agent`` wins."""
    ledger = get_default_ledger()
    if agent:
        return cli_format_by_agent(ledger, agent)
    if query:
        return cli_format_query(ledger, query, kind=kind)
    return cli_format_top(ledger, limit=limit)
