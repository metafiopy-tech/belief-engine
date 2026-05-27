"""Tests for the substrate-transfer experiment runner additions.

Focuses on the parts testable without real LLM builds:

- Schema migration (new columns added idempotently)
- ExperimentResult dataclass defaults (existing-code backward compat)
- Snapshot dispatch logic — restore is called for soil-using conditions and
  skipped for raw_local
- Missing-snapshot graceful failure (records error, doesn't crash the run)
- BELIEF_EXPERIMENT_CONDITION env var is set in the subprocess's scope

End-to-end with real LLM is exercised at the shakedown stage (task #5).
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_init_db_creates_substrate_transfer_columns(tmp_path: Path) -> None:
    from belief.experiments.ab_runner import init_db

    db_path = tmp_path / "exp.db"
    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(results)").fetchall()}
    finally:
        conn.close()

    assert "experiment_type" in cols
    assert "build_seq" in cols
    assert "measurement_point" in cols


def test_init_db_migration_is_idempotent(tmp_path: Path) -> None:
    """Calling init_db on an already-migrated DB should be safe."""
    from belief.experiments.ab_runner import init_db

    db_path = tmp_path / "exp.db"
    init_db(db_path)
    init_db(db_path)  # should not raise
    init_db(db_path)  # third time for good measure

    conn = sqlite3.connect(str(db_path))
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(results)").fetchall()]
    finally:
        conn.close()
    # Each column should appear exactly once (not duplicated by re-ALTER)
    assert cols.count("experiment_type") == 1
    assert cols.count("build_seq") == 1
    assert cols.count("measurement_point") == 1


def test_init_db_migration_preserves_existing_rows(tmp_path: Path) -> None:
    """Existing rows from a pre-migration DB must get the column defaults."""
    from belief.experiments.ab_runner import init_db

    db_path = tmp_path / "exp.db"

    # Simulate a pre-migration DB by creating just the old schema.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript("""
            CREATE TABLE results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id   TEXT NOT NULL,
                challenge_id    TEXT NOT NULL,
                goal            TEXT NOT NULL,
                condition       TEXT NOT NULL,
                model           TEXT NOT NULL,
                passed          BOOLEAN NOT NULL,
                tests_passed    INTEGER NOT NULL,
                tests_total     INTEGER NOT NULL,
                weighted_score  REAL NOT NULL,
                cost_usd        REAL NOT NULL,
                time_seconds    REAL NOT NULL,
                soil_size       INTEGER NOT NULL,
                covenant_count  INTEGER NOT NULL,
                tool_count      INTEGER NOT NULL,
                error           TEXT,
                timestamp       TEXT NOT NULL
            );
            CREATE TABLE experiment_meta (
                experiment_id   TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                description     TEXT,
                challenges_json TEXT,
                status          TEXT DEFAULT 'running'
            );
        """)
        conn.execute(
            "INSERT INTO results "
            "(experiment_id, challenge_id, goal, condition, model, passed, "
            " tests_passed, tests_total, weighted_score, cost_usd, time_seconds, "
            " soil_size, covenant_count, tool_count, error, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-exp",
                "t1-fizzbuzz",
                "goal",
                "engine_local",
                "qwen",
                1,
                3,
                3,
                1.0,
                0.0,
                12.5,
                0,
                0,
                0,
                None,
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Now apply the migration.
    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT experiment_type, build_seq, measurement_point "
            "FROM results WHERE experiment_id = 'legacy-exp'"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("ab", 0, 1), (
        "legacy rows must get experiment_type='ab', build_seq=0, "
        "measurement_point=1 from the column defaults"
    )


# ---------------------------------------------------------------------------
# ExperimentResult dataclass defaults
# ---------------------------------------------------------------------------


def test_experiment_result_defaults_preserve_ab_compat() -> None:
    """Existing A/B harness call sites construct ExperimentResult without
    passing the substrate-transfer fields. The defaults must keep those
    calls valid AND tag the resulting row as type 'ab'.
    """
    from belief.experiments.ab_runner import ExperimentResult

    r = ExperimentResult(
        experiment_id="x",
        challenge_id="c",
        goal="g",
        condition="engine_local",
        model="qwen",
        passed=True,
        tests_passed=3,
        tests_total=3,
        weighted_score=1.0,
        cost_usd=0.0,
        time_seconds=10.0,
        soil_size=0,
        covenant_count=0,
        tool_count=0,
    )
    assert r.experiment_type == "ab"
    assert r.build_seq == 0
    assert r.measurement_point is True


# ---------------------------------------------------------------------------
# Snapshot dispatch & env-var setup
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    """Records every restore call so we can assert dispatch order/coverage."""

    def __init__(self) -> None:
        self.restores: list[Path] = []

    def restore_snapshot(self, path: Path) -> None:
        self.restores.append(Path(path))


def test_runner_restores_snapshots_only_for_soil_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """raw_local must NOT trigger a restore; soil_only and full must."""
    from belief.experiments import ab_runner as mod
    from belief.experiments.ab_runner import run_substrate_transfer_experiment

    # Stub out the actual build calls — we only care about dispatch here.
    async def _fake_run_raw(goal: str, model: str = "qwen"):
        from belief.experiments.raw_runner import RawRunResult

        return RawRunResult(
            goal=goal,
            model=model,
            tests_passed=1,
            tests_total=1,
            weighted_score=1.0,
            time_seconds=0.01,
            error=None,
        )

    async def _fake_run_engine_build(goal: str, mode: str, timeout: int = 2400):
        return {
            "passed": True,
            "tests_passed": 1,
            "tests_total": 1,
            "weighted_score": 1.0,
            "cost": 0.0,
            "time_seconds": 0.01,
            "error": None,
        }

    monkeypatch.setattr(mod, "run_raw", _fake_run_raw)
    monkeypatch.setattr(mod, "run_engine_build", _fake_run_engine_build)
    monkeypatch.setattr(mod, "get_engine_state", lambda: (0, 0, 0))

    fake_snap = _FakeSnapshot()
    db_path = tmp_path / "exp.db"

    snapshot_path = tmp_path / "baseline_b1"
    snapshot_path.mkdir()
    baseline_snapshots = {
        ("soil_only", 1): snapshot_path,
        ("full", 1): snapshot_path,
    }

    challenges = [{"id": "c1", "goal": "do thing"}]

    asyncio.run(
        run_substrate_transfer_experiment(
            challenges=challenges,
            baseline_snapshots=baseline_snapshots,
            conditions=["raw_local", "soil_only", "full"],
            build_seq_points=(1,),
            db_path=db_path,
            snapshot_taker=fake_snap,
        )
    )

    # raw_local contributes 0 restores; soil_only and full contribute 1 each.
    assert len(fake_snap.restores) == 2
    assert all(p == snapshot_path for p in fake_snap.restores)


def test_runner_records_missing_snapshot_as_error_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing baseline → store an error row and continue, don't raise."""
    from belief.experiments import ab_runner as mod
    from belief.experiments.ab_runner import run_substrate_transfer_experiment

    async def _fake_run_raw(goal, model="qwen"):
        from belief.experiments.raw_runner import RawRunResult

        return RawRunResult(
            goal=goal,
            model=model,
            tests_passed=0,
            tests_total=0,
            weighted_score=0.0,
            time_seconds=0.01,
            error=None,
        )

    async def _fake_run_engine_build(goal, mode, timeout=2400):
        return {
            "passed": True,
            "tests_passed": 1,
            "tests_total": 1,
            "weighted_score": 1.0,
            "cost": 0.0,
            "time_seconds": 0.01,
            "error": None,
        }

    monkeypatch.setattr(mod, "run_raw", _fake_run_raw)
    monkeypatch.setattr(mod, "run_engine_build", _fake_run_engine_build)
    monkeypatch.setattr(mod, "get_engine_state", lambda: (0, 0, 0))

    db_path = tmp_path / "exp.db"
    challenges = [{"id": "c1", "goal": "do thing"}]

    exp_id = asyncio.run(
        run_substrate_transfer_experiment(
            challenges=challenges,
            baseline_snapshots={},  # deliberately empty
            conditions=["soil_only"],
            build_seq_points=(1,),
            db_path=db_path,
            snapshot_taker=_FakeSnapshot(),
        )
    )

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT condition, build_seq, passed, error FROM results WHERE experiment_id = ?",
            (exp_id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    cond, bseq, passed, err = rows[0]
    assert cond == "soil_only"
    assert bseq == 1
    assert passed == 0
    assert "snapshot" in (err or "").lower()


def test_runner_sets_belief_experiment_condition_env_for_engine_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """During run_engine_build the env var must hold the condition string."""
    from belief.experiments import ab_runner as mod
    from belief.experiments.ab_runner import run_substrate_transfer_experiment
    import os as _os

    observed: list[str] = []

    async def _fake_run_engine_build(goal, mode, timeout=2400):
        observed.append(_os.environ.get("BELIEF_EXPERIMENT_CONDITION", ""))
        return {
            "passed": True,
            "tests_passed": 1,
            "tests_total": 1,
            "weighted_score": 1.0,
            "cost": 0.0,
            "time_seconds": 0.01,
            "error": None,
        }

    async def _fake_run_raw(goal, model="qwen"):
        from belief.experiments.raw_runner import RawRunResult

        return RawRunResult(
            goal=goal,
            model=model,
            tests_passed=0,
            tests_total=0,
            weighted_score=0.0,
            time_seconds=0.01,
            error=None,
        )

    monkeypatch.setattr(mod, "run_engine_build", _fake_run_engine_build)
    monkeypatch.setattr(mod, "run_raw", _fake_run_raw)
    monkeypatch.setattr(mod, "get_engine_state", lambda: (0, 0, 0))

    snap = tmp_path / "snap"
    snap.mkdir()

    asyncio.run(
        run_substrate_transfer_experiment(
            challenges=[{"id": "c1", "goal": "g"}],
            baseline_snapshots={("soil_only", 1): snap, ("full", 1): snap},
            conditions=["soil_only", "full"],
            build_seq_points=(1,),
            db_path=tmp_path / "exp.db",
            snapshot_taker=_FakeSnapshot(),
        )
    )

    assert observed == ["soil_only", "full"], (
        f"env var must reflect each cell's condition; got {observed}"
    )

    # And after the runner returns, the env var should be restored to its
    # prior value (None / unset in this test).
    assert "BELIEF_EXPERIMENT_CONDITION" not in _os.environ


def test_runner_writes_substrate_transfer_metadata_to_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All rows from this runner must have experiment_type='substrate_transfer'."""
    from belief.experiments import ab_runner as mod
    from belief.experiments.ab_runner import run_substrate_transfer_experiment

    async def _fake_run_engine_build(goal, mode, timeout=2400):
        return {
            "passed": True,
            "tests_passed": 1,
            "tests_total": 1,
            "weighted_score": 1.0,
            "cost": 0.0,
            "time_seconds": 0.01,
            "error": None,
        }

    async def _fake_run_raw(goal, model="qwen"):
        from belief.experiments.raw_runner import RawRunResult

        return RawRunResult(
            goal=goal,
            model=model,
            tests_passed=1,
            tests_total=1,
            weighted_score=1.0,
            time_seconds=0.01,
            error=None,
        )

    monkeypatch.setattr(mod, "run_engine_build", _fake_run_engine_build)
    monkeypatch.setattr(mod, "run_raw", _fake_run_raw)
    monkeypatch.setattr(mod, "get_engine_state", lambda: (0, 0, 0))

    db_path = tmp_path / "exp.db"
    snap = tmp_path / "snap"
    snap.mkdir()

    exp_id = asyncio.run(
        run_substrate_transfer_experiment(
            challenges=[{"id": "c1", "goal": "g"}],
            baseline_snapshots={("soil_only", 1): snap, ("full", 1): snap},
            conditions=["raw_local", "soil_only", "full"],
            build_seq_points=(1,),
            db_path=db_path,
            snapshot_taker=_FakeSnapshot(),
        )
    )

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT condition, build_seq, experiment_type, measurement_point "
            "FROM results WHERE experiment_id = ?",
            (exp_id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 3, f"expected 3 rows (one per condition), got {len(rows)}"
    for cond, bseq, etype, mp in rows:
        assert etype == "substrate_transfer"
        assert mp == 1  # all rows are measurement points
        if cond == "raw_local":
            assert bseq == 0
        else:
            assert bseq == 1
