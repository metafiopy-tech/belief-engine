"""ACCEL-style bounded priority heap.

Backed by the synthesis_heap table (created in state.py). Capacity is
enforced at push-time: if the heap is full, the new seed evicts the
current minimum only if its value is higher; otherwise the new seed is
dropped. This is ACCEL's discipline for preventing BabyAGI's
unbounded-queue failure mode.

Saturation detector:

    For each cycle, record (cycle_ts, min_value, pushed_count) into
    synthesis_cycle_log. When the min-value of the 20 most recent heap
    entries stays above 0.70 for 3 consecutive cycles, raise
    NoveltySaturation — the caller pauses upstream harvesting because
    we're drowning in highly-valued candidates we aren't getting to.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger("belief.photosynthesis.synthesis.heap")


DEFAULT_CAPACITY = 64
SATURATION_MIN_THRESHOLD = 0.70
SATURATION_RECENT_WINDOW = 20
SATURATION_CYCLES_REQUIRED = 3


class NoveltySaturation(RuntimeError):
    """Raised when the heap signals upstream to pause."""


@dataclass
class HeapEntry:
    id: int
    seed: dict[str, Any]
    value: float
    added_at: int


class BoundedPriorityHeap:
    """SQLite-backed bounded max-priority heap.

    Thin wrapper around the synthesis_heap table — we don't keep an
    in-memory heap because APScheduler runs jobs in a threadpool and a
    shared SQLite table is cheaper than holding a lock over a heapq.
    All operations take a `conn()` context from a PhotosynthesisState.
    """

    def __init__(
        self,
        state: Any,  # PhotosynthesisState; avoid circular import
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        self.state = state
        self.capacity = capacity

    # ---------------------------------------------------------- size / peek
    def size(self) -> int:
        with self.state.conn() as c:
            row = c.execute("SELECT COUNT(*) AS n FROM synthesis_heap;").fetchone()
            return int(row["n"]) if row else 0

    def peek_min(self) -> Optional[HeapEntry]:
        """Lowest-value entry currently in the heap, or None if empty."""
        with self.state.conn() as c:
            row = c.execute(
                "SELECT id, seed_json, value, added_at "
                "FROM synthesis_heap ORDER BY value ASC LIMIT 1;"
            ).fetchone()
            return _row_to_entry(row) if row else None

    def peek_top(self) -> Optional[HeapEntry]:
        with self.state.conn() as c:
            row = c.execute(
                "SELECT id, seed_json, value, added_at "
                "FROM synthesis_heap ORDER BY value DESC LIMIT 1;"
            ).fetchone()
            return _row_to_entry(row) if row else None

    # ---------------------------------------------------------- push / pop
    def push(self, seed: dict[str, Any], value: float) -> bool:
        """Insert a seed at the given value. Returns True iff it was stored.

        Capacity rule (spec): if heap is full and new_value > current min,
        evict the min and insert the new entry; otherwise drop the new
        entry. Equal values — we drop to avoid churn.
        """
        now = int(time.time())
        blob = json.dumps(seed, separators=(",", ":"))

        with self.state.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            count_row = c.execute(
                "SELECT COUNT(*) AS n FROM synthesis_heap;"
            ).fetchone()
            count = int(count_row["n"]) if count_row else 0

            if count < self.capacity:
                c.execute(
                    "INSERT INTO synthesis_heap(seed_json, value, added_at) "
                    "VALUES(?, ?, ?);",
                    (blob, float(value), now),
                )
                c.execute("COMMIT;")
                return True

            # Heap full — check the current min
            min_row = c.execute(
                "SELECT id, value FROM synthesis_heap "
                "ORDER BY value ASC, id ASC LIMIT 1;"
            ).fetchone()
            if min_row is None:
                # Shouldn't happen given count >= capacity > 0
                c.execute("ROLLBACK;")
                return False
            if float(value) <= float(min_row["value"]):
                c.execute("ROLLBACK;")
                return False

            c.execute(
                "DELETE FROM synthesis_heap WHERE id = ?;", (int(min_row["id"]),)
            )
            c.execute(
                "INSERT INTO synthesis_heap(seed_json, value, added_at) "
                "VALUES(?, ?, ?);",
                (blob, float(value), now),
            )
            c.execute("COMMIT;")
            return True

    def pop_top(self) -> Optional[HeapEntry]:
        """Remove and return the highest-value entry (or None if empty)."""
        with self.state.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            row = c.execute(
                "SELECT id, seed_json, value, added_at "
                "FROM synthesis_heap ORDER BY value DESC, id ASC LIMIT 1;"
            ).fetchone()
            if row is None:
                c.execute("COMMIT;")
                return None
            c.execute("DELETE FROM synthesis_heap WHERE id = ?;", (int(row["id"]),))
            c.execute("COMMIT;")
            return _row_to_entry(row)

    # ---------------------------------------------------------- saturation
    def record_cycle(self, pushed_count: int = 0) -> None:
        """Append one row to synthesis_cycle_log from current heap state."""
        now = int(time.time())
        with self.state.conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            agg = c.execute(
                "SELECT MIN(value) AS mn, AVG(value) AS mu "
                "FROM synthesis_heap;"
            ).fetchone()
            mn = float(agg["mn"]) if agg and agg["mn"] is not None else None
            mu = float(agg["mu"]) if agg and agg["mu"] is not None else None
            saturated = int(self._is_saturated_after_insert(c, mn))
            c.execute(
                "INSERT INTO synthesis_cycle_log"
                "(cycle_ts, min_value, mean_value, pushed_count, saturation) "
                "VALUES(?, ?, ?, ?, ?);",
                (now, mn, mu, int(pushed_count), saturated),
            )
            c.execute("COMMIT;")

    def check_saturation(self) -> bool:
        """True iff the last N cycles meet the saturation criterion."""
        with self.state.conn() as c:
            rows = c.execute(
                "SELECT saturation FROM synthesis_cycle_log "
                "ORDER BY id DESC LIMIT ?;",
                (SATURATION_CYCLES_REQUIRED,),
            ).fetchall()
        if len(rows) < SATURATION_CYCLES_REQUIRED:
            return False
        return all(int(r["saturation"]) for r in rows)

    def raise_if_saturated(self) -> None:
        if self.check_saturation():
            raise NoveltySaturation(
                f"heap min stayed > {SATURATION_MIN_THRESHOLD} for "
                f"{SATURATION_CYCLES_REQUIRED} consecutive cycles"
            )

    # ---------------------------------------------------------- internals
    def _is_saturated_after_insert(
        self, c: sqlite3.Connection, current_min: Optional[float]
    ) -> bool:
        """Current min across the N most-recent rows above threshold?"""
        window_row = c.execute(
            "SELECT MIN(value) AS mn FROM ("
            "  SELECT value FROM synthesis_heap ORDER BY added_at DESC LIMIT ?"
            ");",
            (SATURATION_RECENT_WINDOW,),
        ).fetchone()
        if not window_row or window_row["mn"] is None:
            return False
        return float(window_row["mn"]) > SATURATION_MIN_THRESHOLD


def _row_to_entry(row: Any) -> HeapEntry:
    seed = json.loads(row["seed_json"])
    return HeapEntry(
        id=int(row["id"]),
        seed=seed,
        value=float(row["value"]),
        added_at=int(row["added_at"]),
    )


__all__ = [
    "BoundedPriorityHeap",
    "DEFAULT_CAPACITY",
    "HeapEntry",
    "NoveltySaturation",
    "SATURATION_CYCLES_REQUIRED",
    "SATURATION_MIN_THRESHOLD",
    "SATURATION_RECENT_WINDOW",
]
