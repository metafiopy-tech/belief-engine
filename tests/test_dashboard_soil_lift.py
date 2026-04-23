"""Session 7: dashboard soil_lift field + tolerant load_all."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belief.metrics.dashboard import IterationMetrics, MetricsDashboard


@pytest.fixture()
def dashboard(tmp_path: Path) -> MetricsDashboard:
    return MetricsDashboard(db_path=str(tmp_path / "metrics.jsonl"))


def test_soil_lift_defaults_to_zero() -> None:
    m = IterationMetrics(iteration=1, timestamp="2026-04-21", benchmark_score=0.83)
    assert m.soil_lift == 0.0


def test_soil_lift_roundtrip(dashboard: MetricsDashboard) -> None:
    m = IterationMetrics(
        iteration=1,
        timestamp="2026-04-21",
        benchmark_score=0.83,
        soil_lift=0.15,
    )
    dashboard.record(m)
    loaded = dashboard.load_all()
    assert len(loaded) == 1
    assert loaded[0].soil_lift == pytest.approx(0.15)


def test_load_all_tolerates_unknown_keys(dashboard: MetricsDashboard) -> None:
    """A future install adds a field we don't know about yet — don't drop the row."""
    row = {
        "iteration": 1,
        "timestamp": "2026-04-21",
        "benchmark_score": 0.5,
        "some_future_field": "ignored",
    }
    with open(dashboard.db_path, "w") as f:
        f.write(json.dumps(row) + "\n")
    loaded = dashboard.load_all()
    assert len(loaded) == 1
    assert loaded[0].iteration == 1


def test_load_all_tolerates_missing_fields(dashboard: MetricsDashboard) -> None:
    """An older install's rows lack soil_lift — default kicks in."""
    row = {
        "iteration": 1,
        "timestamp": "2026-04-21",
        "benchmark_score": 0.5,
    }
    with open(dashboard.db_path, "w") as f:
        f.write(json.dumps(row) + "\n")
    loaded = dashboard.load_all()
    assert len(loaded) == 1
    assert loaded[0].soil_lift == 0.0


def test_load_all_skips_malformed_json(dashboard: MetricsDashboard) -> None:
    with open(dashboard.db_path, "w") as f:
        f.write("{valid: no}\n")
        f.write(
            json.dumps(
                {
                    "iteration": 1,
                    "timestamp": "x",
                    "benchmark_score": 0.7,
                }
            )
            + "\n"
        )
    loaded = dashboard.load_all()
    assert len(loaded) == 1
    assert loaded[0].iteration == 1
