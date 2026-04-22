"""Tests for belief.experiments.ab_runner.

No network calls, no subprocess — engine builds are mocked.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from belief.experiments.ab_runner import (
    DEFAULT_DB_PATH,
    ExperimentResult,
    init_db,
    run_experiment,
    store_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    path = tmp_path / "test_experiments.db"
    return path


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:

    def test_creates_tables(self, tmp_db: Path):
        init_db(tmp_db)
        assert tmp_db.exists()

        conn = sqlite3.connect(str(tmp_db))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "results" in tables
        assert "experiment_meta" in tables

    def test_idempotent(self, tmp_db: Path):
        init_db(tmp_db)
        init_db(tmp_db)  # second call must not raise

    def test_creates_parent_dirs(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c" / "exp.db"
        init_db(deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# store_result / append-only invariant
# ---------------------------------------------------------------------------


class TestStoreResult:

    def _make_result(self, **overrides) -> ExperimentResult:
        defaults = dict(
            experiment_id="exp-test",
            challenge_id="t1-fizzbuzz",
            goal="Build FizzBuzz",
            condition="raw_local",
            model="qwen2.5-coder:14b",
            passed=True,
            tests_passed=3,
            tests_total=3,
            weighted_score=1.0,
            cost_usd=0.0,
            time_seconds=5.2,
            soil_size=0,
            covenant_count=0,
            tool_count=0,
            error=None,
            timestamp="2026-04-21T00:00:00+00:00",
        )
        defaults.update(overrides)
        return ExperimentResult(**defaults)

    def test_store_and_retrieve(self, tmp_db: Path):
        init_db(tmp_db)
        result = self._make_result()
        store_result(result, tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM results").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["challenge_id"] == "t1-fizzbuzz"
        assert rows[0]["passed"] == 1
        assert rows[0]["weighted_score"] == pytest.approx(1.0)

    def test_append_only_multiple_rows(self, tmp_db: Path):
        init_db(tmp_db)
        for i in range(5):
            store_result(self._make_result(experiment_id=f"exp-{i}"), tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        conn.close()
        assert count == 5

    def test_stores_error_field(self, tmp_db: Path):
        init_db(tmp_db)
        store_result(self._make_result(error="Model timed out"), tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT error FROM results").fetchone()
        conn.close()
        assert row[0] == "Model timed out"

    def test_null_error_stored(self, tmp_db: Path):
        init_db(tmp_db)
        store_result(self._make_result(error=None), tmp_db)

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT error FROM results").fetchone()
        conn.close()
        assert row[0] is None


# ---------------------------------------------------------------------------
# run_experiment (mocked engine + raw runner)
# ---------------------------------------------------------------------------


def _mock_raw_result(goal: str, **kwargs):
    from belief.experiments.raw_runner import RawRunResult
    return RawRunResult(
        goal=goal,
        model=kwargs.get("model", "test-model"),
        tests_passed=2,
        tests_total=3,
        weighted_score=0.67,
        time_seconds=1.0,
    )


async def _async_mock_raw(**kwargs):
    return _mock_raw_result(**kwargs)


class TestRunExperiment:

    @pytest.mark.asyncio
    async def test_stores_all_conditions(self, tmp_db: Path):
        challenges = [{"id": "t1-fizzbuzz", "goal": "Build FizzBuzz"}]

        engine_data = {
            "passed": True, "tests_passed": 3, "tests_total": 3,
            "weighted_score": 1.0, "cost": 0.05, "time_seconds": 30.0,
            "error": None,
        }

        with (
            patch(
                "belief.experiments.ab_runner.run_raw",
                new_callable=AsyncMock,
                return_value=_mock_raw_result("Build FizzBuzz"),
            ),
            patch(
                "belief.experiments.ab_runner.run_engine_build",
                new_callable=AsyncMock,
                return_value=engine_data,
            ),
            patch(
                "belief.experiments.ab_runner.get_engine_state",
                return_value=(10, 3, 2),
            ),
        ):
            exp_id = await run_experiment(
                challenges,
                conditions=["engine_cloud", "engine_local", "raw_local"],
                db_path=tmp_db,
            )

        conn = sqlite3.connect(str(tmp_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT condition FROM results WHERE experiment_id=?",
            (exp_id,),
        ).fetchall()
        conn.close()

        conditions_stored = {r["condition"] for r in rows}
        assert conditions_stored == {"engine_cloud", "engine_local", "raw_local"}

    @pytest.mark.asyncio
    async def test_experiment_id_in_meta(self, tmp_db: Path):
        challenges = [{"id": "t1-fizzbuzz", "goal": "Build FizzBuzz"}]

        engine_data = {
            "passed": False, "tests_passed": 0, "tests_total": 0,
            "weighted_score": 0.0, "cost": 0.0, "time_seconds": 5.0,
            "error": "timeout",
        }

        with (
            patch(
                "belief.experiments.ab_runner.run_raw",
                new_callable=AsyncMock,
                return_value=_mock_raw_result("Build FizzBuzz"),
            ),
            patch(
                "belief.experiments.ab_runner.run_engine_build",
                new_callable=AsyncMock,
                return_value=engine_data,
            ),
            patch(
                "belief.experiments.ab_runner.get_engine_state",
                return_value=(0, 0, 0),
            ),
        ):
            exp_id = await run_experiment(
                challenges,
                experiment_id="exp-unit-test",
                conditions=["raw_local"],
                db_path=tmp_db,
            )

        assert exp_id == "exp-unit-test"

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT status FROM experiment_meta WHERE experiment_id=?",
            ("exp-unit-test",),
        ).fetchone()
        conn.close()
        assert row[0] == "complete"

    @pytest.mark.asyncio
    async def test_engine_error_stored_gracefully(self, tmp_db: Path):
        """If run_engine_build raises, the error is stored and run continues."""
        challenges = [{"id": "t1-fizzbuzz", "goal": "Build FizzBuzz"}]

        with (
            patch(
                "belief.experiments.ab_runner.run_engine_build",
                new_callable=AsyncMock,
                side_effect=RuntimeError("belief not in PATH"),
            ),
            patch(
                "belief.experiments.ab_runner.get_engine_state",
                return_value=(0, 0, 0),
            ),
        ):
            exp_id = await run_experiment(
                challenges,
                conditions=["engine_local"],
                db_path=tmp_db,
            )

        conn = sqlite3.connect(str(tmp_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT passed, error FROM results WHERE experiment_id=?",
            (exp_id,),
        ).fetchone()
        conn.close()
        assert row["passed"] == 0
        assert "belief not in PATH" in (row["error"] or "")

    @pytest.mark.asyncio
    async def test_raw_only_no_engine_calls(self, tmp_db: Path):
        challenges = [{"id": "c1", "goal": "Hello world"}]

        engine_mock = AsyncMock()
        raw_mock = AsyncMock(return_value=_mock_raw_result("Hello world"))

        with (
            patch("belief.experiments.ab_runner.run_raw", raw_mock),
            patch("belief.experiments.ab_runner.run_engine_build", engine_mock),
            patch("belief.experiments.ab_runner.get_engine_state", return_value=(0, 0, 0)),
        ):
            await run_experiment(
                challenges,
                conditions=["raw_local"],
                db_path=tmp_db,
            )

        engine_mock.assert_not_called()
        raw_mock.assert_called_once()
