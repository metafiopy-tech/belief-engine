"""Tests for belief.photosynthesis.bittensor.subnet_watcher — Session 8.5c.

The watcher is a "nice-to-have" daemon job that snapshots bittensor
subnet state into local SQLite.  Its spec says "never crashes the
daemon" — every SDK call is wrapped in try/except.  These tests pin
that contract + the persistence path.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from belief.photosynthesis.bittensor.subnet_watcher import (
    SubnetSnapshot,
    SubnetWatcher,
    _safe_len,
    _safe_sum,
)


# ---------------------------------------------------------------------------
# Persistence — SQLite schema + insert
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_record_round_trips_minimal_snapshot(self, tmp_path: Path) -> None:
        w = SubnetWatcher(db_path=tmp_path / "snap.db")
        snap = SubnetSnapshot(netuid=1, n_miners=256, tao_staked=12345.6)
        w.record(snap)

        import sqlite3

        with sqlite3.connect(tmp_path / "snap.db") as c:
            rows = c.execute("SELECT netuid, n_miners, tao_staked FROM subnet_snapshots").fetchall()
        assert rows == [(1, 256, 12345.6)]

    def test_record_stores_timestamp(self, tmp_path: Path) -> None:
        w = SubnetWatcher(db_path=tmp_path / "snap.db")
        before = int(time.time())
        w.record(SubnetSnapshot(netuid=62))
        after = int(time.time())

        import sqlite3

        with sqlite3.connect(tmp_path / "snap.db") as c:
            (ts,) = c.execute("SELECT snapshot_ts FROM subnet_snapshots").fetchone()
        assert before <= ts <= after

    def test_raw_json_round_trips(self, tmp_path: Path) -> None:
        w = SubnetWatcher(db_path=tmp_path / "snap.db")
        w.record(SubnetSnapshot(netuid=1, raw={"network": "finney", "netuid": 1}))

        import json
        import sqlite3

        with sqlite3.connect(tmp_path / "snap.db") as c:
            (raw,) = c.execute("SELECT raw_json FROM subnet_snapshots").fetchone()
        assert json.loads(raw) == {"netuid": 1, "network": "finney"}

    def test_schema_idempotent(self, tmp_path: Path) -> None:
        """Constructing two watchers on the same DB must not error."""
        db = tmp_path / "snap.db"
        SubnetWatcher(db_path=db)
        SubnetWatcher(db_path=db)  # must not raise


# ---------------------------------------------------------------------------
# Graceful degradation — bittensor SDK missing or failing
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_snapshot_once_empty_when_bittensor_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When bittensor isn't installed, snapshot_once returns [] and
        does not raise.  This is the spec's "never crashes the daemon"
        contract."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # noqa: ANN001
            if name == "bittensor":
                raise ImportError("bittensor not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        w = SubnetWatcher(db_path=tmp_path / "snap.db")
        out = w.snapshot_once()
        assert out == []


# ---------------------------------------------------------------------------
# _safe_len / _safe_sum — resilient against SDK surface drift
# ---------------------------------------------------------------------------


class TestSafeHelpers:
    def test_safe_len_none(self) -> None:
        assert _safe_len(None) is None

    def test_safe_len_list(self) -> None:
        assert _safe_len([1, 2, 3]) == 3

    def test_safe_len_no_len(self) -> None:
        """_safe_len must return None rather than raise for objects
        without __len__ (common when the SDK renames a field)."""
        assert _safe_len(42) is None

    def test_safe_sum_none(self) -> None:
        assert _safe_sum(None) is None

    def test_safe_sum_iterable(self) -> None:
        assert _safe_sum([1.0, 2.5, 0.5]) == 4.0

    def test_safe_sum_nonnumeric(self) -> None:
        """Non-numeric iterables must return None rather than raise."""
        assert _safe_sum(["a", "b"]) is None
