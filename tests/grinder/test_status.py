"""GrinderStatus: atomic roundtrip + tolerant reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belief.grinder.status import (
    GrinderStatus,
    format_status,
    read_status,
    write_status,
)


@pytest.fixture()
def status_path(tmp_path: Path) -> Path:
    return tmp_path / "status.json"


def test_write_then_read_roundtrip(status_path: Path) -> None:
    s = GrinderStatus(
        state="building",
        builds_completed=3,
        builds_failed=1,
        current_goal_id="g-1",
        current_goal_text="Build a FastAPI bookmark API",
        queue_depth=5,
        last_result="pass",
        last_cost_usd=0.42,
        last_duration_s=42.0,
    )
    write_status(s, path=status_path)
    loaded = read_status(path=status_path)
    assert loaded is not None
    assert loaded.state == "building"
    assert loaded.builds_completed == 3
    assert loaded.current_goal_id == "g-1"


def test_read_missing_file_is_none(status_path: Path) -> None:
    assert read_status(path=status_path) is None


def test_read_tolerates_extra_keys(status_path: Path) -> None:
    status_path.write_text(
        json.dumps(
            {
                "state": "idle",
                "builds_completed": 0,
                "future_field": "ignored",
            }
        )
    )
    s = read_status(path=status_path)
    assert s is not None
    assert s.state == "idle"


def test_read_tolerates_malformed_json(status_path: Path) -> None:
    status_path.write_text("{not json")
    assert read_status(path=status_path) is None


def test_format_status_handles_none() -> None:
    assert "no status" in format_status(None)


def test_format_status_with_data() -> None:
    s = GrinderStatus(
        state="paused",
        builds_completed=2,
        builds_failed=0,
        current_goal_id="x",
    )
    out = format_status(s)
    assert "paused" in out
    assert "2 completed" in out
    assert "current: x" in out
