"""Tests for belief.experiments.reporter.

Uses SQLite fixture data — no network, no engine.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from belief.experiments.ab_runner import ExperimentResult, init_db, store_result
from belief.experiments.reporter import comparison_table, longitudinal_report


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _row(
    experiment_id: str,
    challenge_id: str,
    condition: str,
    passed: bool,
    score: float = 0.0,
    cost: float = 0.0,
    soil: int = 0,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        challenge_id=challenge_id,
        goal=f"Goal for {challenge_id}",
        condition=condition,
        model="test-model",
        passed=passed,
        tests_passed=3 if passed else 0,
        tests_total=3,
        weighted_score=score if score else (1.0 if passed else 0.0),
        cost_usd=cost,
        time_seconds=10.0,
        soil_size=soil,
        covenant_count=0,
        tool_count=0,
        timestamp="2026-04-21T00:00:00+00:00",
    )


def _seed_db(db_path: Path) -> None:
    """Insert a full three-condition experiment into the DB."""
    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO experiment_meta "
        "(experiment_id, created_at, description, challenges_json, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("exp-001", "2026-04-21T10:00:00", "seed", '["c1","c2"]', "complete"),
    )
    conn.commit()
    conn.close()

    for cid in ("c1", "c2"):
        store_result(_row("exp-001", cid, "engine_cloud", True, 1.0, 0.05, 10), db_path)
        store_result(_row("exp-001", cid, "engine_local", True, 1.0, 0.0, 10), db_path)
        store_result(_row("exp-001", cid, "raw_local", False, 0.0, 0.0, 0), db_path)


# ---------------------------------------------------------------------------
# comparison_table
# ---------------------------------------------------------------------------


class TestComparisonTable:

    def test_returns_string(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        _seed_db(db)
        out = comparison_table(db_path=db)
        assert isinstance(out, str)

    def test_contains_experiment_id(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        _seed_db(db)
        out = comparison_table(db_path=db)
        assert "exp-001" in out

    def test_contains_challenge_ids(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        _seed_db(db)
        out = comparison_table(db_path=db)
        assert "c1" in out
        assert "c2" in out

    def test_pass_fail_markers_present(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        _seed_db(db)
        out = comparison_table(db_path=db)
        # engine conditions passed, raw_local failed
        assert "✓" in out
        assert "✗" in out

    def test_soil_lift_shown(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        _seed_db(db)
        out = comparison_table(db_path=db)
        assert "Soil Lift" in out
        # engine_local 2/2 = 100%, raw_local 0/2 = 0% → lift +100%
        assert "+100" in out or "+1" in out

    def test_no_db_returns_message(self, tmp_path: Path):
        db = tmp_path / "nonexistent.db"
        out = comparison_table(db_path=db)
        assert "No experiments database" in out

    def test_latest_experiment_used_when_id_omitted(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        _seed_db(db)
        # Add a second experiment
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO experiment_meta "
            "(experiment_id, created_at, description, challenges_json, status) "
            "VALUES (?, ?, ?, ?, ?)",
            ("exp-002", "2026-04-22T10:00:00", "second", '["c1"]', "complete"),
        )
        conn.commit()
        conn.close()
        store_result(_row("exp-002", "c1", "raw_local", True), db)

        out = comparison_table(db_path=db)
        assert "exp-002" in out

    def test_explicit_experiment_id(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        _seed_db(db)
        out = comparison_table(experiment_id="exp-001", db_path=db)
        assert "exp-001" in out

    def test_missing_condition_shows_dash(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO experiment_meta "
            "(experiment_id, created_at, description, challenges_json, status) "
            "VALUES (?, ?, ?, ?, ?)",
            ("exp-partial", "2026-04-21T10:00:00", "partial", '["c1"]', "complete"),
        )
        conn.commit()
        conn.close()
        # Only store raw_local — engine conditions missing
        store_result(_row("exp-partial", "c1", "raw_local", True), db)

        out = comparison_table(experiment_id="exp-partial", db_path=db)
        assert "—" in out  # missing conditions show dash


# ---------------------------------------------------------------------------
# longitudinal_report
# ---------------------------------------------------------------------------


class TestLongitudinalReport:

    def _seed_two_experiments(self, db: Path) -> None:
        init_db(db)
        for exp_id, date, soil in [
            ("exp-a", "2026-04-01T00:00:00", 5),
            ("exp-b", "2026-04-15T00:00:00", 50),
        ]:
            conn = sqlite3.connect(str(db))
            conn.execute(
                "INSERT INTO experiment_meta "
                "(experiment_id, created_at, description, challenges_json, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (exp_id, date, "test", '["c1"]', "complete"),
            )
            conn.commit()
            conn.close()
            store_result(
                _row(exp_id, "c1", "engine_local",
                     passed=(exp_id == "exp-b"),  # second experiment passes
                     soil=soil),
                db,
            )
            store_result(
                _row(exp_id, "c1", "raw_local", passed=False, soil=soil),
                db,
            )

    def test_requires_two_experiments(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO experiment_meta "
            "(experiment_id, created_at, description, challenges_json, status) "
            "VALUES (?, ?, ?, ?, ?)",
            ("exp-only", "2026-04-01T00:00:00", "solo", '["c1"]', "complete"),
        )
        conn.commit()
        conn.close()
        out = longitudinal_report(db_path=db)
        assert "at least 2" in out.lower() or "Need" in out

    def test_shows_lift_trend(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        self._seed_two_experiments(db)
        out = longitudinal_report(db_path=db)
        assert "exp-a" in out
        assert "exp-b" in out
        # exp-b engine_local passed, raw_local failed → positive lift
        assert "+" in out

    def test_no_db_returns_message(self, tmp_path: Path):
        db = tmp_path / "nonexistent.db"
        out = longitudinal_report(db_path=db)
        assert "No experiments database" in out

    def test_contains_header(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        self._seed_two_experiments(db)
        out = longitudinal_report(db_path=db)
        assert "LONGITUDINAL" in out

    def test_contains_interpretation_note(self, tmp_path: Path):
        db = tmp_path / "exp.db"
        self._seed_two_experiments(db)
        out = longitudinal_report(db_path=db)
        assert "learning" in out.lower() or "overhead" in out.lower()
