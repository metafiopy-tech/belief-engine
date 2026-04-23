"""Unit tests for belief.photosynthesis.state.

These tests rely only on the stdlib (sqlite3 + dataclasses). They run
without the [photosynthesis] extra installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from belief.photosynthesis.state import CandidateSeed, PhotosynthesisState


@pytest.fixture()
def state(tmp_path: Path) -> PhotosynthesisState:
    db = tmp_path / "signals.sqlite"
    return PhotosynthesisState(str(db))


def test_wal_mode_is_enabled(state: PhotosynthesisState) -> None:
    """PRAGMA journal_mode must return 'wal'."""
    with state.conn() as c:
        mode = c.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"


def test_schema_tables_exist(state: PhotosynthesisState) -> None:
    with state.conn() as c:
        rows = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {"raw_signals", "seen", "watermarks"}.issubset(names)


def test_mark_if_new_is_idempotent(state: PhotosynthesisState) -> None:
    """First mark returns True; any subsequent mark returns False."""
    assert state.mark_if_new("src", "1") is True
    assert state.mark_if_new("src", "1") is False
    assert state.mark_if_new("src", "1") is False
    assert state.mark_if_new("src", "2") is True


def test_watermark_round_trip(state: PhotosynthesisState) -> None:
    assert state.get_watermark("src") == (None, None)
    state.set_watermark("src", last_ts=12345, last_cursor="abc")
    assert state.get_watermark("src") == (12345, "abc")
    # Partial update preserves the other field
    state.set_watermark("src", last_cursor="xyz")
    ts, cur = state.get_watermark("src")
    assert ts == 12345
    assert cur == "xyz"


def test_insert_signal_dedups_via_unique_constraint(
    state: PhotosynthesisState,
) -> None:
    seed = CandidateSeed(source="s", source_id="42", title="t", summary="x")
    first = state.insert_signal(seed)
    second = state.insert_signal(seed)
    assert first is not None
    assert second is None

    # And the duplicates probe reports 0 — the UNIQUE index enforces it.
    assert state.duplicates_probe() == 0


def test_pending_and_update_cycle(state: PhotosynthesisState) -> None:
    seed = CandidateSeed(source="s", source_id="1", title="hello", summary="some fastapi text")
    sid = state.insert_signal(seed)
    assert sid is not None

    pending = state.pending_signals()
    assert len(pending) == 1
    assert pending[0]["source"] == "s"

    state.update_filter_result(sid, stage_reached=2, filter_score=0.42, status="kept")

    # Now it's no longer in 'raw'
    assert state.pending_signals() == []


def test_count_by_source(state: PhotosynthesisState) -> None:
    for i in range(3):
        state.insert_signal(CandidateSeed(source="a", source_id=str(i)))
    for i in range(2):
        state.insert_signal(CandidateSeed(source="b", source_id=str(i)))
    assert state.count_by_source() == {"a": 3, "b": 2}


def test_state_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "signals.sqlite"
    state = PhotosynthesisState(str(nested))
    assert os.path.exists(nested)
    # Basic functionality still works
    assert state.mark_if_new("s", "x") is True
