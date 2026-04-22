"""SWE-Bench Verified + Polyglot task mirror.

Weekly-refreshed local cache of the task distributions that SN62
validators score on. We store the bare minimum per task
(problem_statement, repo, ecosystem) so downstream code can embed and
average without re-downloading.

HuggingFace `datasets` is an optional heavy dep. When unavailable, the
mirror can still load an explicit fixture list (used in tests and any
cold-start deployment without internet). Never crashes the daemon on
import.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence


logger = logging.getLogger("belief.photosynthesis.bittensor.swebench_mirror")


DEFAULT_MIRROR_DB = Path("/var/lib/photosynthesis/bittensor_tasks.db")
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
POLYGLOT_DATASET = "paul-gauthier/aider-polyglot-benchmark"


SCHEMA = """
CREATE TABLE IF NOT EXISTS bittensor_tasks (
    id                 TEXT PRIMARY KEY,
    problem_statement  TEXT NOT NULL,
    repo               TEXT NOT NULL,
    ecosystem          TEXT NOT NULL,
    added_at           INTEGER NOT NULL,
    dataset            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bittensor_tasks_ecosystem
    ON bittensor_tasks(ecosystem);
"""


@dataclass
class BittensorTask:
    id: str
    problem_statement: str
    repo: str = ""
    ecosystem: str = ""
    dataset: str = ""


class SwebenchMirror:
    """Local SQLite-backed cache of SWE-Bench + Polyglot tasks."""

    def __init__(self, db_path: Path | str = DEFAULT_MIRROR_DB) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.row_factory = sqlite3.Row
            yield c
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # -------------------------------------------------------------- loaders
    def ingest_fixture(
        self,
        tasks: Sequence[BittensorTask],
        *,
        dataset: str = "fixture",
    ) -> int:
        """Insert a static list of tasks. Useful in tests / air-gapped setups.

        Returns the number of newly-inserted rows (duplicates ignored).
        """
        now = int(time.time())
        inserted = 0
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            for t in tasks:
                cur = c.execute(
                    "INSERT OR IGNORE INTO bittensor_tasks"
                    "(id, problem_statement, repo, ecosystem, added_at, dataset) "
                    "VALUES(?, ?, ?, ?, ?, ?);",
                    (t.id, t.problem_statement, t.repo, t.ecosystem, now, dataset),
                )
                if cur.rowcount:
                    inserted += 1
            c.execute("COMMIT;")
        return inserted

    def ingest_swebench_verified(
        self,
        *,
        split: str = "test",
        limit: Optional[int] = None,
    ) -> int:
        """Pull SWE-Bench Verified via the HuggingFace datasets library.

        Returns 0 if datasets isn't installed (caller gets a log line,
        not an exception). No internet? Same.
        """
        rows = _try_load_hf_rows(
            SWEBENCH_DATASET,
            split=split,
            limit=limit,
        )
        if not rows:
            return 0
        tasks = [
            BittensorTask(
                id=str(r.get("instance_id") or r.get("id") or ""),
                problem_statement=str(r.get("problem_statement") or ""),
                repo=str(r.get("repo") or ""),
                ecosystem="python",
            )
            for r in rows
            if r.get("problem_statement")
        ]
        return self.ingest_fixture(tasks, dataset=SWEBENCH_DATASET)

    def ingest_polyglot(
        self,
        *,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> int:
        """Best-effort Polyglot loader. Same graceful-fallback semantics."""
        rows = _try_load_hf_rows(POLYGLOT_DATASET, split=split, limit=limit)
        if not rows:
            return 0
        tasks = []
        for r in rows:
            if not r.get("problem_statement") and not r.get("prompt"):
                continue
            tasks.append(
                BittensorTask(
                    id=str(r.get("instance_id") or r.get("id") or ""),
                    problem_statement=str(
                        r.get("problem_statement") or r.get("prompt") or ""
                    ),
                    repo=str(r.get("repo") or ""),
                    ecosystem=str(r.get("language") or "polyglot"),
                )
            )
        return self.ingest_fixture(tasks, dataset=POLYGLOT_DATASET)

    # -------------------------------------------------------------- query
    def count(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) AS n FROM bittensor_tasks;").fetchone()
            return int(row["n"]) if row else 0

    def sample(self, n: int, ecosystem: Optional[str] = None) -> list[BittensorTask]:
        """Return up to `n` random tasks, optionally filtered by ecosystem."""
        with self._conn() as c:
            if ecosystem:
                rows = c.execute(
                    "SELECT * FROM bittensor_tasks WHERE ecosystem = ? "
                    "ORDER BY RANDOM() LIMIT ?;",
                    (ecosystem, int(n)),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM bittensor_tasks ORDER BY RANDOM() LIMIT ?;",
                    (int(n),),
                ).fetchall()
        return [_row_to_task(r) for r in rows]


def _row_to_task(row: Any) -> BittensorTask:
    return BittensorTask(
        id=row["id"],
        problem_statement=row["problem_statement"],
        repo=row["repo"] or "",
        ecosystem=row["ecosystem"] or "",
        dataset=row["dataset"] or "",
    )


def _try_load_hf_rows(
    dataset: str, *, split: str, limit: Optional[int]
) -> list[dict[str, Any]]:
    """Attempt to pull rows via HuggingFace `datasets`. Return [] on failure."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.info(
            "datasets not installed; skipping HF pull for %s. "
            "Install with: pip install datasets",
            dataset,
        )
        return []
    try:
        ds = load_dataset(dataset, split=split)
    except Exception as exc:  # network, auth, shape mismatch, etc.
        logger.warning("HF load failed for %s: %s", dataset, exc)
        return []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        if hasattr(row, "asdict"):
            out.append(row.asdict())
        elif isinstance(row, dict):
            out.append(row)
        else:
            try:
                out.append(dict(row))
            except Exception:
                continue
    return out


__all__ = [
    "BittensorTask",
    "DEFAULT_MIRROR_DB",
    "POLYGLOT_DATASET",
    "SWEBENCH_DATASET",
    "SwebenchMirror",
]
