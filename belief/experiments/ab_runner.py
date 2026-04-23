"""Run controlled A/B experiments comparing engine vs raw model.

Three conditions:
  engine_cloud — Belief Engine + Claude (Anthropic cloud)
  engine_local — Belief Engine + Ollama (local)
  raw_local    — Raw Ollama call, no engine (control group)

Results are stored in SQLite for longitudinal analysis. The db_path
parameter on public functions can be overridden for tests.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from belief.experiments.raw_runner import RawRunResult, run_raw

DEFAULT_DB_PATH = Path.home() / ".belief-engine" / "experiments.db"

_ALL_CONDITIONS = ("engine_cloud", "engine_local", "raw_local")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    experiment_id: str
    challenge_id: str
    goal: str
    condition: str  # "engine_cloud" | "engine_local" | "raw_local"
    model: str
    passed: bool
    tests_passed: int
    tests_total: int
    weighted_score: float
    cost_usd: float
    time_seconds: float
    soil_size: int  # builds in soil at time of run
    covenant_count: int
    tool_count: int
    error: Optional[str] = None
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the experiments database (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id   TEXT    NOT NULL,
                challenge_id    TEXT    NOT NULL,
                goal            TEXT    NOT NULL,
                condition       TEXT    NOT NULL,
                model           TEXT    NOT NULL,
                passed          BOOLEAN NOT NULL,
                tests_passed    INTEGER NOT NULL,
                tests_total     INTEGER NOT NULL,
                weighted_score  REAL    NOT NULL,
                cost_usd        REAL    NOT NULL,
                time_seconds    REAL    NOT NULL,
                soil_size       INTEGER NOT NULL,
                covenant_count  INTEGER NOT NULL,
                tool_count      INTEGER NOT NULL,
                error           TEXT,
                timestamp       TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiment_meta (
                experiment_id   TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                description     TEXT,
                challenges_json TEXT,
                status          TEXT DEFAULT 'running'
            );
        """)
    finally:
        conn.close()


def store_result(
    result: ExperimentResult,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Append a single result row (never updates existing rows)."""
    conn = sqlite3.connect(str(db_path))
    try:
        d = asdict(result)
        cols = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        conn.execute(
            f"INSERT INTO results ({cols}) VALUES ({placeholders})",
            list(d.values()),
        )
        conn.commit()
    finally:
        conn.close()


def _set_experiment_status(
    experiment_id: str,
    status: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE experiment_meta SET status=? WHERE experiment_id=?",
            (status, experiment_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Engine-side runner
# ---------------------------------------------------------------------------


async def run_engine_build(
    goal: str,
    mode: str,
    timeout: int = 2400,
) -> dict:
    """Run a goal through the Belief Engine as a subprocess.

    Sets BELIEF_MODEL_MODE=<mode> in the child environment so the
    routing is isolated from the parent process.

    Returns a dict: passed, tests_passed, tests_total,
    weighted_score, cost, time_seconds, error.
    """
    import asyncio

    env = os.environ.copy()
    env["BELIEF_MODEL_MODE"] = mode

    start = time.time()

    loop = asyncio.get_event_loop()
    proc_result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            ["belief", "--goal", goal, "--json-output"],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        ),
    )

    elapsed = time.time() - start
    stdout = proc_result.stdout or ""
    stderr = proc_result.stderr or ""

    # Parse the JSON result line emitted by the engine with --json-output
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "verdict" in line:
            try:
                data = json.loads(line)
                return {
                    "passed": data.get("verdict") == "pass",
                    "tests_passed": data.get("tests_passed", 0),
                    "tests_total": data.get("tests_total", 0),
                    "weighted_score": data.get("weighted_score", 0.0),
                    "cost": data.get("cost_usd", data.get("cost", 0.0)),
                    "time_seconds": elapsed,
                    "error": None,
                }
            except (json.JSONDecodeError, KeyError):
                pass

    # Fallback: parse the human-readable output
    passed = "Verdict: pass" in stdout or "\nVerdict: pass" in stdout
    cost = 0.0
    cost_m = re.search(r"Cost:\s*\$?([\d.]+)", stdout)
    if cost_m:
        cost = float(cost_m.group(1))

    error = None
    if proc_result.returncode not in (0, 1):
        error = (stderr or stdout)[:300] or f"exit code {proc_result.returncode}"

    return {
        "passed": passed,
        "tests_passed": 0,
        "tests_total": 0,
        "weighted_score": 1.0 if passed else 0.0,
        "cost": cost,
        "time_seconds": elapsed,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Engine state snapshot
# ---------------------------------------------------------------------------


def get_engine_state() -> tuple[int, int, int]:
    """Return (soil_size, covenant_count, tool_count) at the moment of the call.

    Reads from the real BuildStore / Soil / ToolRegistry. Returns (0,0,0)
    gracefully if any component is unavailable.
    """
    soil_size = 0
    covenant_count = 0
    tool_count = 0

    try:
        from belief.config.settings import settings
        from belief.memory.store import BuildStore

        store = BuildStore(Path(settings.db_path).expanduser())
        soil_size = store.count()
        store.close()
    except Exception:
        pass

    try:
        from belief.memory.soil import Soil
        from belief.memory.collections import CollectionName

        soil = Soil()
        # Count covenant entries in the belief_covenants collection
        col = soil.get_collection(CollectionName.COVENANTS)
        if col is not None:
            covenant_count = col.count()
    except Exception:
        pass

    try:
        from belief.memory.soil import Soil
        from belief.memory.tool_registry import ToolRegistry

        soil = Soil()
        reg = ToolRegistry(soil)
        tool_count = len(reg.get_active_tools())
    except Exception:
        pass

    return soil_size, covenant_count, tool_count


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------


async def run_experiment(
    challenges: list[dict],
    experiment_id: Optional[str] = None,
    conditions: Optional[list[str]] = None,
    model: str = "qwen2.5-coder:14b",
    db_path: Path = DEFAULT_DB_PATH,
) -> str:
    """Run a full A/B experiment across all conditions.

    Args:
        challenges:     list of {"id": str, "goal": str}
        experiment_id:  auto-generated timestamp ID if omitted
        conditions:     subset of ("engine_cloud","engine_local","raw_local")
        model:          Ollama model for local conditions
        db_path:        SQLite path (override in tests)

    Returns:
        experiment_id
    """
    if experiment_id is None:
        experiment_id = "exp-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if conditions is None:
        conditions = list(_ALL_CONDITIONS)

    init_db(db_path)

    # Store metadata
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO experiment_meta "
            "(experiment_id, created_at, description, challenges_json, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                f"A/B test: {len(challenges)} challenge(s), conditions: {', '.join(conditions)}",
                json.dumps([c["id"] for c in challenges]),
                "running",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    soil_size, covenant_count, tool_count = get_engine_state()

    for challenge in challenges:
        cid = challenge["id"]
        goal = challenge["goal"]
        print(f"\n{'=' * 60}")
        print(f"  Challenge: {cid}")
        print(f"  Goal: {goal[:80]}{'...' if len(goal) > 80 else ''}")
        print(f"{'=' * 60}")

        for condition in conditions:
            print(f"\n  ── {condition} ──")

            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                if condition == "raw_local":
                    raw: RawRunResult = await run_raw(goal, model=model)
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        challenge_id=cid,
                        goal=goal,
                        condition=condition,
                        model=model,
                        passed=raw.weighted_score >= 0.5,
                        tests_passed=raw.tests_passed,
                        tests_total=raw.tests_total,
                        weighted_score=raw.weighted_score,
                        cost_usd=0.0,
                        time_seconds=raw.time_seconds,
                        soil_size=0,
                        covenant_count=0,
                        tool_count=0,
                        error=raw.error,
                        timestamp=timestamp,
                    )

                else:
                    mode = "cloud" if condition == "engine_cloud" else "local"
                    cloud_model = "claude-sonnet-4-6"
                    data = await run_engine_build(goal, mode=mode)
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        challenge_id=cid,
                        goal=goal,
                        condition=condition,
                        model=cloud_model if condition == "engine_cloud" else model,
                        passed=data["passed"],
                        tests_passed=data["tests_passed"],
                        tests_total=data["tests_total"],
                        weighted_score=data["weighted_score"],
                        cost_usd=data["cost"],
                        time_seconds=data["time_seconds"],
                        soil_size=soil_size,
                        covenant_count=covenant_count,
                        tool_count=tool_count,
                        error=data.get("error"),
                        timestamp=timestamp,
                    )

                store_result(result, db_path)
                status = "PASS" if result.passed else "FAIL"
                print(
                    f"    {status} | score={result.weighted_score:.2f} | "
                    f"time={result.time_seconds:.0f}s | "
                    f"cost=${result.cost_usd:.4f}"
                )

            except Exception as exc:
                print(f"    ERROR: {exc}")
                store_result(
                    ExperimentResult(
                        experiment_id=experiment_id,
                        challenge_id=cid,
                        goal=goal,
                        condition=condition,
                        model=model,
                        passed=False,
                        tests_passed=0,
                        tests_total=0,
                        weighted_score=0.0,
                        cost_usd=0.0,
                        time_seconds=0.0,
                        soil_size=soil_size,
                        covenant_count=covenant_count,
                        tool_count=tool_count,
                        error=str(exc),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    db_path,
                )

    _set_experiment_status(experiment_id, "complete", db_path)
    return experiment_id
