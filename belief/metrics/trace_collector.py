"""Per-step build-trace collector — training data for the confidence probe.

Every agent execution in the LangGraph pipeline (builder, tester,
debugger, etc.) emits one StepTrace row. At build end, the outcome
(pass / fail) is backfilled onto every row of that build. Session 10
then trains a GradientBoostedClassifier over the resulting dataset to
predict build success from per-step features, and the graph can
circuit-break out of doomed runs.

Design:

  - **SQLite, not ChromaDB** — tabular data. WAL mode for concurrent
    reads during writes.
  - **Async writes** — a background daemon thread drains a queue of
    writes, so agents never block on disk IO. The write queue has a
    soft cap (default 10_000) to prevent unbounded memory in a runaway
    loop; overflow entries are dropped with a warning.
  - **Idempotent finalize_build** — safe to call multiple times; only
    rows that still have build_passed IS NULL are updated.
  - **close() is deterministic** — flushes the queue and joins the
    writer. Safe to call multiple times.

Schema:

    traces(
        id           INTEGER PK AUTOINCREMENT,
        build_id     TEXT    NOT NULL,
        step_index   INTEGER NOT NULL,
        agent_name   TEXT    NOT NULL,
        output_summary TEXT  NOT NULL DEFAULT '',
        edge_decision  TEXT  NOT NULL DEFAULT '',
        cost_so_far  REAL    NOT NULL DEFAULT 0.0,
        iteration    INTEGER NOT NULL DEFAULT 0,
        build_passed INTEGER,      -- NULL until finalize_build
        timestamp    REAL    NOT NULL
    )

Bulk training queries read rows where build_passed IS NOT NULL —
unfinished builds are ignored.
"""

from __future__ import annotations

import csv
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("belief.metrics.trace_collector")


DEFAULT_TRACE_DB = Path("~/.belief-engine/traces.db").expanduser()
OUTPUT_SUMMARY_LIMIT = 500
QUEUE_SOFT_CAP = 10_000
WRITER_BATCH_SIZE = 50
WRITER_BATCH_TIMEOUT_S = 0.5


SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id       TEXT    NOT NULL,
    step_index     INTEGER NOT NULL,
    agent_name     TEXT    NOT NULL,
    output_summary TEXT    NOT NULL DEFAULT '',
    edge_decision  TEXT    NOT NULL DEFAULT '',
    cost_so_far    REAL    NOT NULL DEFAULT 0.0,
    iteration      INTEGER NOT NULL DEFAULT 0,
    build_passed   INTEGER,
    timestamp      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_build_id ON traces(build_id);
CREATE INDEX IF NOT EXISTS idx_traces_outcome  ON traces(build_passed);
"""


# ---------------------------------------------------------------------------
# StepTrace
# ---------------------------------------------------------------------------


@dataclass
class StepTrace:
    build_id: str
    step_index: int
    agent_name: str
    output_summary: str = ""
    edge_decision: str = ""
    cost_so_far: float = 0.0
    iteration: int = 0
    build_passed: Optional[bool] = None
    timestamp: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        if self.output_summary and len(self.output_summary) > OUTPUT_SUMMARY_LIMIT:
            self.output_summary = self.output_summary[:OUTPUT_SUMMARY_LIMIT]

    def to_row(self) -> tuple:
        """Positional tuple matching INSERT column order."""
        passed = None if self.build_passed is None else (1 if self.build_passed else 0)
        return (
            self.build_id,
            int(self.step_index),
            self.agent_name,
            self.output_summary or "",
            self.edge_decision or "",
            float(self.cost_so_far),
            int(self.iteration),
            passed,
            float(self.timestamp),
        )


# ---------------------------------------------------------------------------
# TraceCollector
# ---------------------------------------------------------------------------


_SENTINEL = object()


class TraceCollector:
    """Async SQLite-backed trace log.

    One instance per process is expected. The background writer thread
    is spawned lazily on the first record_step() call; callers don't
    need to set it up explicitly.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_TRACE_DB,
        *,
        start_writer: bool = True,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=QUEUE_SOFT_CAP)
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_should_stop = threading.Event()
        self._closed = False
        self._dropped = 0
        self._dropped_lock = threading.Lock()

        if start_writer:
            self._start_writer()

    # ---------------------------------------------------------- infrastructure
    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None)
        c.execute("PRAGMA journal_mode = WAL;")
        c.execute("PRAGMA synchronous = NORMAL;")
        c.execute("PRAGMA busy_timeout = 5000;")
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        c = self._connect()
        try:
            c.executescript(SCHEMA)
        finally:
            c.close()

    def _start_writer(self) -> None:
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        self._writer_should_stop.clear()
        t = threading.Thread(
            target=self._writer_loop,
            name="trace-collector-writer",
            daemon=True,
        )
        t.start()
        self._writer_thread = t

    def _writer_loop(self) -> None:
        c = self._connect()
        try:
            while not self._writer_should_stop.is_set() or not self._queue.empty():
                batch: list[tuple] = []
                # Block on the first item; drain remaining without blocking.
                try:
                    first = self._queue.get(timeout=WRITER_BATCH_TIMEOUT_S)
                except queue.Empty:
                    continue
                if first is _SENTINEL:
                    break
                batch.append(first)
                while len(batch) < WRITER_BATCH_SIZE:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _SENTINEL:
                        self._flush_batch(c, batch)
                        return
                    batch.append(nxt)
                self._flush_batch(c, batch)
        finally:
            c.close()

    def _flush_batch(self, c: sqlite3.Connection, batch: list[tuple]) -> None:
        if not batch:
            return
        try:
            c.execute("BEGIN IMMEDIATE;")
            c.executemany(
                "INSERT INTO traces"
                "(build_id, step_index, agent_name, output_summary, "
                " edge_decision, cost_so_far, iteration, build_passed, timestamp) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);",
                batch,
            )
            c.execute("COMMIT;")
        except Exception as exc:
            logger.warning("trace writer failed on a batch of %d: %s", len(batch), exc)
            try:
                c.execute("ROLLBACK;")
            except Exception:
                pass

    # ---------------------------------------------------------- public API
    def record_step(self, trace: StepTrace) -> bool:
        """Enqueue one step for async write. Returns False iff dropped."""
        if self._closed:
            return False
        self._start_writer()
        try:
            self._queue.put_nowait(trace.to_row())
            return True
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.warning("trace queue full; %d step(s) dropped so far", self._dropped)
            return False

    def flush(self) -> None:
        """Block until the write queue is empty (bounded wait)."""
        # Simple poll — SQLite batched writes finish quickly.
        t0 = time.monotonic()
        while not self._queue.empty():
            if time.monotonic() - t0 > 10.0:
                logger.warning("flush timed out with %d items left", self._queue.qsize())
                break
            time.sleep(0.01)

    def close(self) -> None:
        """Flush and stop the writer thread. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._writer_should_stop.set()
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)

    # ---------------------------------------------------------- queries
    def finalize_build(self, build_id: str, passed: bool) -> int:
        """Backfill outcome onto every open row for this build.

        Returns the number of rows updated. Idempotent — rerunning with
        the same (build_id, passed) is a no-op.
        """
        # Ensure pending writes are persisted before we update.
        self.flush()
        c = self._connect()
        try:
            c.execute("BEGIN IMMEDIATE;")
            cur = c.execute(
                "UPDATE traces SET build_passed = ? WHERE build_id = ? AND build_passed IS NULL;",
                (1 if passed else 0, build_id),
            )
            count = cur.rowcount
            c.execute("COMMIT;")
            return int(count)
        finally:
            c.close()

    def dropped_count(self) -> int:
        with self._dropped_lock:
            return self._dropped

    def row_count(self) -> int:
        c = self._connect()
        try:
            row = c.execute("SELECT COUNT(*) AS n FROM traces;").fetchone()
            return int(row["n"]) if row else 0
        finally:
            c.close()

    def build_count(self) -> int:
        c = self._connect()
        try:
            row = c.execute(
                "SELECT COUNT(DISTINCT build_id) AS n FROM traces WHERE build_passed IS NOT NULL;"
            ).fetchone()
            return int(row["n"]) if row else 0
        finally:
            c.close()

    def get_training_data(self, *, min_builds: int = 50) -> list[dict]:
        """Return per-step rows with a resolved build outcome.

        Rows are dropped when fewer than `min_builds` distinct builds
        have completed — a too-small sample trains a useless probe.
        Returns an empty list in that case (caller gets the silent
        "not enough data" signal and waits for more builds).
        """
        if self.build_count() < int(min_builds):
            return []
        c = self._connect()
        try:
            rows = c.execute(
                "SELECT build_id, step_index, agent_name, output_summary, "
                "       edge_decision, cost_so_far, iteration, "
                "       build_passed, timestamp "
                "FROM traces WHERE build_passed IS NOT NULL "
                "ORDER BY build_id, step_index;"
            ).fetchall()
            return [
                {
                    **dict(r),
                    "build_passed": bool(r["build_passed"]),
                }
                for r in rows
            ]
        finally:
            c.close()

    def export_for_probe_training(self, path: Path | str) -> int:
        """Write a CSV with every finalized step trace. Returns row count.

        The CSV is the input format for Session 10's `belief probe train`
        command. Columns match get_training_data keys; outputs are
        emitted as UTF-8 with newline=''.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.get_training_data(min_builds=0)
        if not rows:
            # Still write an empty file with just the header so downstream
            # tools don't error on "missing file."
            headers = [
                "build_id",
                "step_index",
                "agent_name",
                "output_summary",
                "edge_decision",
                "cost_so_far",
                "iteration",
                "build_passed",
                "timestamp",
            ]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return 0
        headers = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return len(rows)


# ---------------------------------------------------------------------------
# Singleton (lazy) for graph.py hook
# ---------------------------------------------------------------------------


_global: Optional[TraceCollector] = None
_global_lock = threading.Lock()


def get_default_collector() -> TraceCollector:
    """Return the process-wide collector, creating it lazily."""
    global _global
    with _global_lock:
        if _global is None:
            _global = TraceCollector(DEFAULT_TRACE_DB)
        return _global


def set_default_collector(collector: Optional[TraceCollector]) -> None:
    """Override the process-wide collector — tests only."""
    global _global
    with _global_lock:
        _global = collector


def record_step_from_state(
    state: Any,
    *,
    agent_name: str,
    step_index: Optional[int] = None,
    edge_decision: str = "",
    output_summary: Optional[str] = None,
) -> None:
    """Record one step trace from a pipeline state dict.

    The state dict is expected to carry:
      - build_id: unique id for this build (we populate one if absent)
      - iteration: debugger-loop iteration number (defaults to 0)
      - budget: BuildBudget with .spent_usd (defaults to 0.0)

    Agents call this helper from inside their node function; the graph
    wrapper from graph.py wires it in without requiring every agent to
    import belief.metrics.trace_collector.
    """
    if not isinstance(state, dict):
        return
    build_id = state.get("build_id") or ""
    if not build_id:
        import uuid as _uuid

        build_id = f"b-{_uuid.uuid4().hex[:12]}"
        state["build_id"] = build_id
    if step_index is None:
        step_index = int(state.get("_step_index", 0))
        state["_step_index"] = step_index + 1

    summary = output_summary
    if summary is None:
        # Best-effort: pick a representative field from state.
        summary = ""
        for key in ("last_agent_output", "agent_output", "message", "output"):
            val = state.get(key)
            if isinstance(val, str):
                summary = val
                break

    cost = 0.0
    budget = state.get("budget")
    if budget is not None and hasattr(budget, "spent_usd"):
        cost = float(budget.spent_usd)

    try:
        trace = StepTrace(
            build_id=build_id,
            step_index=int(step_index),
            agent_name=str(agent_name),
            output_summary=summary,
            edge_decision=str(edge_decision),
            cost_so_far=cost,
            iteration=int(state.get("iteration", 0) or 0),
        )
        get_default_collector().record_step(trace)
    except Exception:
        # Tracing must never fail the build. Swallow all errors.
        logger.debug("record_step_from_state failed", exc_info=True)


def is_tracing_enabled() -> bool:
    """Honor BELIEF_ENABLE_TRACE env var. Default OFF (no-op in tests)."""
    return os.environ.get("BELIEF_ENABLE_TRACE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


__all__ = [
    "DEFAULT_TRACE_DB",
    "OUTPUT_SUMMARY_LIMIT",
    "StepTrace",
    "TraceCollector",
    "get_default_collector",
    "is_tracing_enabled",
    "record_step_from_state",
    "set_default_collector",
]
