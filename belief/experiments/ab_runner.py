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
    condition: str  # "engine_cloud" | "engine_local" | "raw_local" | "soil_only" | "full"
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
    # Substrate-transfer experiment fields (defaults preserve backward compat
    # with the existing A/B harness — old rows store experiment_type="ab",
    # build_seq=0, measurement_point=True).
    experiment_type: str = "ab"  # "ab" | "substrate_transfer"
    build_seq: int = 0  # 0 = not applicable; 1/5/15 for substrate_transfer
    measurement_point: bool = True  # True for rows we score in the report


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

        # Idempotent column additions for the substrate-transfer experiment.
        # SQLite has no native "ADD COLUMN IF NOT EXISTS"; we read the
        # existing columns and add only what's missing. Existing rows get
        # the default values (preserves backward compat with prior data).
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)").fetchall()}
        if "experiment_type" not in existing_cols:
            conn.execute(
                "ALTER TABLE results ADD COLUMN experiment_type TEXT NOT NULL DEFAULT 'ab'"
            )
        if "build_seq" not in existing_cols:
            conn.execute("ALTER TABLE results ADD COLUMN build_seq INTEGER NOT NULL DEFAULT 0")
        if "measurement_point" not in existing_cols:
            conn.execute(
                "ALTER TABLE results ADD COLUMN measurement_point INTEGER NOT NULL DEFAULT 1"
            )
        conn.commit()
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

    # Belt + suspenders: the env var alone doesn't fully prevent the engine
    # from hitting cloud roles. The shakedown run on 2026-05-28 (subxfer-
    # 20260528-034212) showed ~$0.01-0.02 spend per soil_only/full cell
    # despite BELIEF_MODEL_MODE=local. Pass explicit --mode and
    # --local-model CLI flags so the cost surface is fully closed.
    cmd: list[str] = ["belief"]
    if mode == "local":
        cmd.extend(
            [
                "--mode",
                "local",
                "--local-model",
                os.environ.get("BELIEF_LOCAL_MODEL", "qwen2.5-coder:14b"),
            ]
        )
    cmd.extend(["--goal", goal, "--json-output"])

    start = time.time()

    loop = asyncio.get_event_loop()
    proc_result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            cmd,
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


# ---------------------------------------------------------------------------
# Substrate-transfer experiment driver
# ---------------------------------------------------------------------------


# Public so callers can name conditions consistently.
SUBSTRATE_TRANSFER_CONDITIONS: tuple[str, ...] = ("raw_local", "soil_only", "full")
DEFAULT_BUILD_SEQ_POINTS: tuple[int, ...] = (1, 5, 15)


async def run_substrate_transfer_experiment(
    challenges: list[dict],
    baseline_snapshots: dict[tuple[str, int], Path],
    experiment_id: Optional[str] = None,
    conditions: Optional[list[str]] = None,
    build_seq_points: tuple[int, ...] = DEFAULT_BUILD_SEQ_POINTS,
    model: str = "qwen2.5-coder:14b",
    db_path: Path = DEFAULT_DB_PATH,
    snapshot_taker=None,
) -> str:
    """Run the 3-condition substrate-transfer experiment.

    For each (condition, build_seq, challenge) cell where the condition uses
    soil, the baseline snapshot ``baseline_snapshots[(condition, build_seq)]``
    is restored before the build runs. This prevents soil from one cell
    leaking into another. ``raw_local`` skips snapshot restoration entirely
    because it bypasses the engine.

    Args:
        challenges:
            List of ``{"id": str, "goal": str}``.
        baseline_snapshots:
            Map from ``(condition, build_seq)`` → Path to a SoilSnapshot.
            Must be present for every (condition, build_seq) cell where the
            condition uses soil. Missing entries cause that cell to record
            an error rather than crash the run.
        experiment_id:
            Auto-generated timestamp if omitted.
        conditions:
            Subset of ``SUBSTRATE_TRANSFER_CONDITIONS``.
        build_seq_points:
            Which build-seq numbers to measure at. Default (1, 5, 15) per
            the reduced experiment design.
        model:
            Ollama model identifier (passed to ``run_raw`` / engine).
        db_path:
            Override for tests.
        snapshot_taker:
            Optional ``SoilSnapshot`` instance. Default-constructed on first
            need. Tests inject a fake to avoid touching disk.

    Returns:
        ``experiment_id``.
    """
    if experiment_id is None:
        experiment_id = "subxfer-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if conditions is None:
        conditions = list(SUBSTRATE_TRANSFER_CONDITIONS)

    init_db(db_path)

    # Record metadata
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO experiment_meta "
            "(experiment_id, created_at, description, challenges_json, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                (
                    f"substrate_transfer: {len(challenges)} challenge(s), "
                    f"conditions={','.join(conditions)}, "
                    f"build_seq={','.join(str(b) for b in build_seq_points)}"
                ),
                json.dumps([c["id"] for c in challenges]),
                "running",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Lazy snapshot taker — only constructed if we actually need to restore.
    def _snap():
        nonlocal snapshot_taker
        if snapshot_taker is None:
            from belief.memory.snapshot import SoilSnapshot

            snapshot_taker = SoilSnapshot()
        return snapshot_taker

    def _cell_uses_soil(cond: str) -> bool:
        return cond in ("soil_only", "full")

    for condition in conditions:
        # raw_local has no build_seq axis — score once per challenge with build_seq=0.
        cond_build_points = build_seq_points if _cell_uses_soil(condition) else (0,)

        for build_seq in cond_build_points:
            for challenge in challenges:
                cid = challenge["id"]
                goal = challenge["goal"]
                timestamp = datetime.now(timezone.utc).isoformat()
                print(f"\n  ── {condition} | build_seq={build_seq} | {cid} ──")

                # Restore baseline snapshot for soil-using cells.
                restore_error: Optional[str] = None
                if _cell_uses_soil(condition):
                    key = (condition, build_seq)
                    snap_path = baseline_snapshots.get(key)
                    if snap_path is None:
                        restore_error = (
                            f"no baseline snapshot configured for "
                            f"condition={condition} build_seq={build_seq}"
                        )
                    else:
                        try:
                            _snap().restore_snapshot(Path(snap_path))
                        except Exception as exc:
                            restore_error = f"snapshot restore failed: {exc}"

                if restore_error is not None:
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
                            soil_size=0,
                            covenant_count=0,
                            tool_count=0,
                            error=restore_error,
                            timestamp=timestamp,
                            experiment_type="substrate_transfer",
                            build_seq=build_seq,
                            measurement_point=True,
                        ),
                        db_path,
                    )
                    print(f"    ERROR: {restore_error}")
                    continue

                # Snapshot engine state AFTER restore so we record the
                # actual soil_size etc. observed by the build.
                soil_size, covenant_count, tool_count = get_engine_state()

                # Novel-artifact challenges (challenge_id starts with "novel-")
                # override the build's pytest-based weighted_score with a
                # mechanical validator over the produced artifact files.
                # See belief/experiments/novel_artifact_challenges.py.
                from belief.experiments.novel_artifact_challenges import (
                    apply_novel_artifact_validation,
                    is_novel_artifact_id,
                )

                is_novel = is_novel_artifact_id(cid)

                try:
                    build_start = time.time()
                    if condition == "raw_local":
                        raw: RawRunResult = await run_raw(goal, model=model)
                        # For novel-artifact challenges, override the score
                        # using the validator over raw_runner's in-memory
                        # code_files dict.
                        if is_novel:
                            na_passed, na_msg, na_score = apply_novel_artifact_validation(
                                challenge_id=cid,
                                code_files=raw.code_files,
                            )
                            raw_passed = na_passed
                            raw_score = na_score
                            raw_error = f"novel-artifact: {na_msg}" if not na_passed else None
                        else:
                            raw_passed = raw.weighted_score >= 0.5
                            raw_score = raw.weighted_score
                            raw_error = raw.error
                        result = ExperimentResult(
                            experiment_id=experiment_id,
                            challenge_id=cid,
                            goal=goal,
                            condition=condition,
                            model=model,
                            passed=raw_passed,
                            tests_passed=raw.tests_passed,
                            tests_total=raw.tests_total,
                            weighted_score=raw_score,
                            cost_usd=0.0,
                            time_seconds=raw.time_seconds,
                            soil_size=0,
                            covenant_count=0,
                            tool_count=0,
                            error=raw_error,
                            timestamp=timestamp,
                            experiment_type="substrate_transfer",
                            build_seq=build_seq,
                            measurement_point=True,
                        )
                    else:
                        # Set BELIEF_EXPERIMENT_CONDITION for the subprocess —
                        # this is what activates the soil_only toggle wired
                        # in task #3 (see belief/experiments/conditions.py).
                        prior_cond = os.environ.get("BELIEF_EXPERIMENT_CONDITION")
                        os.environ["BELIEF_EXPERIMENT_CONDITION"] = condition
                        try:
                            data = await run_engine_build(goal, mode="local")
                        finally:
                            if prior_cond is None:
                                os.environ.pop("BELIEF_EXPERIMENT_CONDITION", None)
                            else:
                                os.environ["BELIEF_EXPERIMENT_CONDITION"] = prior_cond

                        # For novel-artifact challenges, override the score
                        # using the validator over the engine's output dir.
                        if is_novel:
                            na_passed, na_msg, na_score = apply_novel_artifact_validation(
                                challenge_id=cid,
                                build_start_time=build_start,
                            )
                            data_passed = na_passed
                            data_score = na_score
                            data_error = (
                                f"novel-artifact: {na_msg}" if not na_passed else data.get("error")
                            )
                        else:
                            data_passed = data["passed"]
                            data_score = data["weighted_score"]
                            data_error = data.get("error")
                        result = ExperimentResult(
                            experiment_id=experiment_id,
                            challenge_id=cid,
                            goal=goal,
                            condition=condition,
                            model=model,
                            passed=data_passed,
                            tests_passed=data["tests_passed"],
                            tests_total=data["tests_total"],
                            weighted_score=data_score,
                            cost_usd=data["cost"],
                            time_seconds=data["time_seconds"],
                            soil_size=soil_size,
                            covenant_count=covenant_count,
                            tool_count=tool_count,
                            error=data_error,
                            timestamp=timestamp,
                            experiment_type="substrate_transfer",
                            build_seq=build_seq,
                            measurement_point=True,
                        )
                    store_result(result, db_path)
                    status = "PASS" if result.passed else "FAIL"
                    print(
                        f"    {status} | score={result.weighted_score:.2f} | "
                        f"time={result.time_seconds:.0f}s | cost=${result.cost_usd:.4f}"
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
                            experiment_type="substrate_transfer",
                            build_seq=build_seq,
                            measurement_point=True,
                        ),
                        db_path,
                    )

    _set_experiment_status(experiment_id, "complete", db_path)
    return experiment_id
