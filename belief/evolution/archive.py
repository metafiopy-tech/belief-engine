"""
Evolutionary Archive — versioned DAG of agent configurations with SQLite storage.

Replaces SICA's "keep best, discard rest" strategy with a full archive that
preserves every variant.  Inspired by Darwin Godel Machine (arXiv:2505.22954):
the best agent variant often descends through performance dips — discarding
regressions kills stepping stones.

Features:
  - Parent-pointer DAG: every AgentVersion knows its parent
  - MAP-Elites niche tracking: best version per behavioral niche
  - DGM-style parent selection: biased toward high-utility + low-children
  - Full benchmark result storage per version
  - Seed version creation from current v2.6.0 configuration

Storage: SQLite at ~/.belief-engine/archive.db
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Data models ─────────────────────────────────────────────────────────────


@dataclass
class AgentVersion:
    """One version of the full agent configuration in the evolutionary DAG."""

    id: str                                       # UUID4
    parent_id: Optional[str]                      # None for seed
    created_at: datetime
    system_prompts: dict[str, str]                # agent_name -> prompt text hash
    tool_ids: list[str]                           # ChromaDB tool IDs active
    principle_ids: list[str]                      # ChromaDB principle IDs active
    covenant_ids: list[str]                       # Covenant IDs active
    model_config: dict[str, str]                  # agent_name -> model string
    diff_from_parent: str                         # What changed
    proposal_rationale: str                       # Why
    utility: float                                # Composite score
    children_count: int = 0
    niche_descriptor: tuple = ()                  # MAP-Elites niche
    canary_passed: bool = False


@dataclass
class BenchmarkResult:
    """Result of running one benchmark challenge against a version."""

    version_id: str
    challenge_id: str
    passed: bool
    score: float
    cost_usd: float
    time_seconds: float
    error_summary: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Archive ─────────────────────────────────────────────────────────────────


class Archive:
    """SQLite-backed evolutionary archive of agent versions.

    Stores the full DAG of agent configurations and their benchmark results.
    Supports DGM-style parent selection and MAP-Elites niche tracking.

    Usage:
        archive = Archive()
        create_seed_version(archive)
        parent = archive.select_parent()
    """

    def __init__(self, db_path: str = "~/.belief-engine/archive.db") -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                created_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                diff TEXT NOT NULL DEFAULT '',
                rationale TEXT NOT NULL DEFAULT '',
                utility REAL NOT NULL DEFAULT 0.0,
                children_count INTEGER NOT NULL DEFAULT 0,
                niche TEXT NOT NULL DEFAULT '()',
                canary INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                score REAL NOT NULL DEFAULT 0.0,
                cost REAL NOT NULL DEFAULT 0.0,
                time REAL NOT NULL DEFAULT 0.0,
                error TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Save / Load ─────────────────────────────────────────────────────────

    def save_version(self, version: AgentVersion) -> None:
        """INSERT a version into the archive."""
        config = {
            "system_prompts": version.system_prompts,
            "tool_ids": version.tool_ids,
            "principle_ids": version.principle_ids,
            "covenant_ids": version.covenant_ids,
            "model_config": version.model_config,
        }
        niche_str = repr(version.niche_descriptor)

        self._conn.execute(
            """INSERT OR REPLACE INTO versions
               (id, parent_id, created_at, config_json, diff, rationale,
                utility, children_count, niche, canary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version.id,
                version.parent_id,
                version.created_at.isoformat(),
                json.dumps(config),
                version.diff_from_parent,
                version.proposal_rationale,
                version.utility,
                version.children_count,
                niche_str,
                1 if version.canary_passed else 0,
            ),
        )
        self._conn.commit()

    def save_result(self, result: BenchmarkResult) -> None:
        """INSERT a benchmark result."""
        self._conn.execute(
            """INSERT INTO benchmark_results
               (version_id, challenge_id, passed, score, cost, time, error, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.version_id,
                result.challenge_id,
                1 if result.passed else 0,
                result.score,
                result.cost_usd,
                result.time_seconds,
                result.error_summary,
                result.timestamp.isoformat(),
            ),
        )
        self._conn.commit()

    def get_version(self, version_id: str) -> AgentVersion:
        """Retrieve a single version by ID."""
        row = self._conn.execute(
            "SELECT * FROM versions WHERE id = ?", (version_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Version {version_id} not found")
        return self._row_to_version(row)

    def get_all_versions(self) -> list[AgentVersion]:
        """Retrieve all versions ordered by creation time."""
        rows = self._conn.execute(
            "SELECT * FROM versions ORDER BY created_at"
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def get_results(self, version_id: str) -> list[BenchmarkResult]:
        """Retrieve all benchmark results for a version."""
        rows = self._conn.execute(
            "SELECT * FROM benchmark_results WHERE version_id = ? ORDER BY timestamp",
            (version_id,),
        ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def get_all_results_recent(self, n: int = 10) -> list[BenchmarkResult]:
        """Retrieve the N most recent benchmark results across all versions."""
        rows = self._conn.execute(
            "SELECT * FROM benchmark_results ORDER BY timestamp DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def get_children(self, version_id: str) -> list[AgentVersion]:
        """Get all direct children of a version."""
        rows = self._conn.execute(
            "SELECT * FROM versions WHERE parent_id = ? ORDER BY created_at",
            (version_id,),
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def get_lineage(self, version_id: str) -> list[AgentVersion]:
        """Walk parent pointers from *version_id* to the root seed.

        Returns a list from the root (seed) to *version_id* inclusive.
        """
        chain: list[AgentVersion] = []
        current_id: Optional[str] = version_id
        visited: set[str] = set()

        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            try:
                version = self.get_version(current_id)
            except KeyError:
                break
            chain.append(version)
            current_id = version.parent_id

        chain.reverse()  # root first
        return chain

    # ── Parent Selection (DGM sampling) ─────────────────────────────────────

    def select_parent(self) -> AgentVersion:
        """Select a parent for the next mutation using DGM sampling.

        Selection weight for version *v*:
            w(v) = sigmoid(10 * (utility - 0.5)) * 1 / (1 + children_count)

        This biases toward high-utility versions that haven't been
        over-explored (low children_count).  The sigmoid keeps low-utility
        versions alive at reduced probability — crucial for DGM's
        stepping-stone preservation.
        """
        versions = self.get_all_versions()
        if not versions:
            raise ValueError("Archive is empty — create a seed version first")
        if len(versions) == 1:
            return versions[0]

        weights: list[float] = []
        for v in versions:
            sig = 1.0 / (1.0 + math.exp(-10.0 * (v.utility - 0.5)))
            exploration = 1.0 / (1.0 + v.children_count)
            weights.append(sig * exploration)

        total = sum(weights)
        if total <= 0:
            return random.choice(versions)

        probs = [w / total for w in weights]
        selected = random.choices(versions, weights=probs, k=1)[0]
        return selected

    # ── MAP-Elites Niche Tracking ───────────────────────────────────────────

    def get_best_in_niche(self, niche: tuple) -> Optional[AgentVersion]:
        """Return the highest-utility version in *niche*, or None."""
        niche_str = repr(niche)
        row = self._conn.execute(
            "SELECT * FROM versions WHERE niche = ? ORDER BY utility DESC LIMIT 1",
            (niche_str,),
        ).fetchone()
        return self._row_to_version(row) if row else None

    def get_niche_map(self) -> dict[tuple, AgentVersion]:
        """Return a dict mapping each filled niche to its best version."""
        rows = self._conn.execute(
            "SELECT * FROM versions ORDER BY utility DESC"
        ).fetchall()

        niche_map: dict[tuple, AgentVersion] = {}
        for row in rows:
            version = self._row_to_version(row)
            if version.niche_descriptor not in niche_map:
                niche_map[version.niche_descriptor] = version
        return niche_map

    # ── Utility Computation ─────────────────────────────────────────────────

    @staticmethod
    def compute_utility(results: list[BenchmarkResult]) -> float:
        """Compute composite utility from benchmark results.

        U = 0.5 * avg_score + 0.25 * (1 - total_cost / 10) + 0.25 * (1 - total_time / 300)

        Clamped to [0, 1].
        """
        if not results:
            return 0.0

        avg_score = sum(r.score for r in results) / len(results)
        total_cost = sum(r.cost_usd for r in results)
        total_time = sum(r.time_seconds for r in results)

        score_term = 0.5 * avg_score
        cost_term = 0.25 * max(0.0, 1.0 - total_cost / 10.0)
        time_term = 0.25 * max(0.0, 1.0 - total_time / 300.0)

        return max(0.0, min(1.0, score_term + cost_term + time_term))

    # ── Children count ──────────────────────────────────────────────────────

    def increment_children(self, version_id: str) -> None:
        """Increment the children_count for a version."""
        self._conn.execute(
            "UPDATE versions SET children_count = children_count + 1 WHERE id = ?",
            (version_id,),
        )
        self._conn.commit()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _row_to_version(self, row: sqlite3.Row) -> AgentVersion:
        config = json.loads(row["config_json"])
        niche_str = row["niche"]
        try:
            niche = eval(niche_str) if niche_str else ()  # noqa: S307
        except Exception:
            niche = ()

        return AgentVersion(
            id=row["id"],
            parent_id=row["parent_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            system_prompts=config.get("system_prompts", {}),
            tool_ids=config.get("tool_ids", []),
            principle_ids=config.get("principle_ids", []),
            covenant_ids=config.get("covenant_ids", []),
            model_config=config.get("model_config", {}),
            diff_from_parent=row["diff"],
            proposal_rationale=row["rationale"],
            utility=row["utility"],
            children_count=row["children_count"],
            niche_descriptor=niche if isinstance(niche, tuple) else (),
            canary_passed=bool(row["canary"]),
        )

    def _row_to_result(self, row: sqlite3.Row) -> BenchmarkResult:
        return BenchmarkResult(
            version_id=row["version_id"],
            challenge_id=row["challenge_id"],
            passed=bool(row["passed"]),
            score=row["score"],
            cost_usd=row["cost"],
            time_seconds=row["time"],
            error_summary=row["error"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )


# ── Seed version ────────────────────────────────────────────────────────────


def create_seed_version(archive: Archive) -> AgentVersion:
    """Create the initial AgentVersion from current v2.6.0 configuration.

    Reads system prompts from belief/prompts/__init__.py, model config from
    belief/config/models.py, and saves as the root of the archive DAG.
    """
    # Check if seed already exists
    existing = archive.get_all_versions()
    for v in existing:
        if v.parent_id is None:
            return v  # Seed already exists

    # Collect system prompts
    from belief.prompts import (
        ARCHITECT_SYSTEM,
        BUILDER_SYSTEM,
        DEBUGGER_SYSTEM,
        GAP_ANALYST_SYSTEM,
        INTAKE_SYSTEM,
        LATIOS_SYSTEM,
        PLANNER_SYSTEM,
        RESEARCH_SYSTEM,
        SYNTHESIZER_SYSTEM,
        TESTER_SYSTEM,
        VALIDATOR_SYSTEM,
    )

    prompts = {
        "intake": hashlib.sha256(INTAKE_SYSTEM.encode()).hexdigest()[:16],
        "research": hashlib.sha256(RESEARCH_SYSTEM.encode()).hexdigest()[:16],
        "planner": hashlib.sha256(PLANNER_SYSTEM.encode()).hexdigest()[:16],
        "architect": hashlib.sha256(ARCHITECT_SYSTEM.encode()).hexdigest()[:16],
        "builder": hashlib.sha256(BUILDER_SYSTEM.encode()).hexdigest()[:16],
        "tester": hashlib.sha256(TESTER_SYSTEM.encode()).hexdigest()[:16],
        "debugger": hashlib.sha256(DEBUGGER_SYSTEM.encode()).hexdigest()[:16],
        "gap_analyst": hashlib.sha256(GAP_ANALYST_SYSTEM.encode()).hexdigest()[:16],
        "synthesizer": hashlib.sha256(SYNTHESIZER_SYSTEM.encode()).hexdigest()[:16],
        "validator": hashlib.sha256(VALIDATOR_SYSTEM.encode()).hexdigest()[:16],
        "latios": hashlib.sha256(LATIOS_SYSTEM.encode()).hexdigest()[:16],
    }

    # Collect model config
    from belief.config.models import _DEFAULTS

    model_config = {role.value: model for role, model in _DEFAULTS.items()}

    seed = AgentVersion(
        id=str(uuid.uuid4()),
        parent_id=None,
        created_at=datetime.now(timezone.utc),
        system_prompts=prompts,
        tool_ids=[],
        principle_ids=[],
        covenant_ids=[],
        model_config=model_config,
        diff_from_parent="seed — initial v2.6.0 configuration",
        proposal_rationale="Root of the evolutionary archive",
        utility=0.5,  # Neutral starting utility
        niche_descriptor=(0, 0, 0),  # Default niche
        canary_passed=True,
    )

    archive.save_version(seed)
    return seed
