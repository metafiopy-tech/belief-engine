"""Hermetic tests for belief.memory.snapshot (mycorrhizal Stage 3).

The snapshot module touches three on-disk stores: ChromaDB's PersistentClient
directory (covered by integration tests that don't run in this hermetic
suite — chromadb is a heavy import and not the unit under test), and two
SQLite ledger files (covered here). The tests stand up a fake "soil" tree
as a plain directory with mock files; the snapshot module's directory-copy
semantics don't care that it isn't real ChromaDB content, and the SQLite
files are real (via the reciprocity + niche ledgers).

Atomic-restore semantics are exercised end-to-end: populate state, snapshot,
mutate state, restore, verify the mutations were rolled back.

Run with:

    python3 -m pytest tests/memory/test_snapshot.py -q --timeout=60
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from belief.memory.niche_ledger import NicheLedger
from belief.memory.reciprocity import ReciprocityLedger
from belief.memory.snapshot import (
    DEFAULT_KEEP_DAILY,
    DEFAULT_KEEP_RECENT,
    DEFAULT_KEEP_WEEKLY,
    SCHEMA_VERSION,
    SnapshotInfo,
    SnapshotManifest,
    SnapshotPaths,
    SoilSnapshot,
    cli_format_health,
    cli_format_list,
    health_summary,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def state(tmp_path: Path) -> dict:
    """Stand up a fake live state under tmp_path/state with the same
    layout the real ~/.belief-engine has: soil dir + two SQLite files.

    Returns the SnapshotPaths plus convenience handles to the ledgers
    so individual tests can mutate them.
    """
    state_root = tmp_path / "state"
    soil_dir = state_root / "soil"
    soil_dir.mkdir(parents=True)
    # Fake chromadb content — just opaque files so the copy semantics work.
    (soil_dir / "chroma.sqlite3").write_bytes(b"FAKE CHROMA")
    (soil_dir / "uuid").mkdir()
    (soil_dir / "uuid" / "data_level0.bin").write_bytes(b"FAKE INDEX")

    recip_db = state_root / "reciprocity.db"
    niches_db = state_root / "niches.db"
    paths = SnapshotPaths(
        soil_dir=soil_dir,
        reciprocity_db=recip_db,
        niches_db=niches_db,
    )
    # Populate both ledgers.
    rl = ReciprocityLedger(db_path=recip_db)
    rl.record_request("alice", cost=1.0, idempotency_key="r1")
    rl.record_contribution("alice", nutrient_value=2.0, nutrient_id="n1")
    rl.close()

    nl = NicheLedger(db_path=niches_db)
    nl.record_modification("alice", "tool", "tool-1", post_state_description="hello")
    nl.close()

    return {
        "paths": paths,
        "state_root": state_root,
        "soil_dir": soil_dir,
        "recip_db": recip_db,
        "niches_db": niches_db,
    }


@pytest.fixture
def snap(tmp_path: Path, state: dict) -> SoilSnapshot:
    return SoilSnapshot(
        paths=state["paths"],
        snapshots_dir=tmp_path / "snapshots",
        audit_path=tmp_path / "audit.jsonl",
    )


# ── 1. Take ─────────────────────────────────────────────────────────────────


def test_take_snapshot_produces_complete_directory(snap: SoilSnapshot) -> None:
    dest = snap.take_snapshot()
    assert dest.exists() and dest.is_dir()
    assert (dest / "manifest.json").exists()
    assert (dest / "soil" / "chroma.sqlite3").exists()
    assert (dest / "soil" / "uuid" / "data_level0.bin").exists()
    assert (dest / "reciprocity.db").exists()
    assert (dest / "niches.db").exists()


def test_take_snapshot_manifest_fields(snap: SoilSnapshot) -> None:
    dest = snap.take_snapshot(label="post-stage3")
    with open(dest / "manifest.json", "r") as f:
        m = SnapshotManifest.from_dict(json.load(f))
    assert m.schema_version == SCHEMA_VERSION
    assert m.soil_present is True
    assert m.reciprocity_present is True
    assert m.niches_present is True
    assert m.soil_file_count == 2  # chroma.sqlite3 + data_level0.bin
    assert m.extra.get("label") == "post-stage3"
    # taken_at parses back to a UTC datetime
    parsed = datetime.fromisoformat(m.taken_at)
    assert parsed.tzinfo is not None


def test_take_snapshot_fails_if_destination_exists(snap: SoilSnapshot, tmp_path: Path) -> None:
    """Never overwrite a prior snapshot silently."""
    explicit = tmp_path / "explicit-dest"
    snap.take_snapshot(dest_path=explicit)
    with pytest.raises(FileExistsError):
        snap.take_snapshot(dest_path=explicit)


def test_staging_cleanup_after_prior_crash(snap: SoilSnapshot, tmp_path: Path) -> None:
    """Simulate a crashed prior `take` that left a `.staging` dir, then
    take a new snapshot to the same final path — the new run must
    clean the stale staging instead of crashing on it."""
    explicit = tmp_path / "post-crash"
    stale = explicit.with_name(explicit.name + ".staging")
    stale.mkdir(parents=True)
    (stale / "leftover").write_text("crashed mid-write")
    # New take should succeed.
    snap.take_snapshot(dest_path=explicit)
    assert explicit.exists()
    assert not stale.exists()


def test_take_with_only_some_components(tmp_path: Path) -> None:
    """A missing reciprocity.db (e.g. fresh machine) doesn't fail the
    snapshot — the manifest reflects what's actually present."""
    soil_dir = tmp_path / "soil"
    soil_dir.mkdir()
    (soil_dir / "x").write_text("a")
    paths = SnapshotPaths(
        soil_dir=soil_dir,
        reciprocity_db=tmp_path / "missing-r.db",
        niches_db=tmp_path / "missing-n.db",
    )
    snap = SoilSnapshot(
        paths=paths,
        snapshots_dir=tmp_path / "snapshots",
        audit_path=tmp_path / "audit.jsonl",
    )
    dest = snap.take_snapshot()
    with open(dest / "manifest.json", "r") as f:
        m = SnapshotManifest.from_dict(json.load(f))
    assert m.soil_present is True
    assert m.reciprocity_present is False
    assert m.niches_present is False


# ── 2. Verify ───────────────────────────────────────────────────────────────


def test_verify_fresh_snapshot_is_ok(snap: SoilSnapshot) -> None:
    dest = snap.take_snapshot()
    assert snap.verify_snapshot(dest) is True


def test_verify_missing_manifest(snap: SoilSnapshot, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert snap.verify_snapshot(empty) is False


def test_verify_corrupted_manifest(snap: SoilSnapshot) -> None:
    dest = snap.take_snapshot()
    (dest / "manifest.json").write_text("{not valid json")
    assert snap.verify_snapshot(dest) is False


def test_verify_schema_mismatch(snap: SoilSnapshot) -> None:
    dest = snap.take_snapshot()
    with open(dest / "manifest.json", "r") as f:
        m = json.load(f)
    m["schema_version"] = 99
    with open(dest / "manifest.json", "w") as f:
        json.dump(m, f)
    assert snap.verify_snapshot(dest) is False


def test_verify_presence_mismatch(snap: SoilSnapshot) -> None:
    """Manifest claims a soil dir exists, but it doesn't on disk."""
    dest = snap.take_snapshot()
    import shutil

    shutil.rmtree(dest / "soil")
    assert snap.verify_snapshot(dest) is False


# ── 3. Restore (atomic) ────────────────────────────────────────────────────


def test_restore_rolls_back_subsequent_mutations(snap: SoilSnapshot, state: dict) -> None:
    """Take snapshot → mutate live ledgers → restore → confirm originals."""
    dest = snap.take_snapshot()

    # Mutate reciprocity after the snapshot.
    rl = ReciprocityLedger(db_path=state["recip_db"])
    rl.record_request("bob", cost=999.0)
    assert rl.stats("bob").carbon_received > 0
    rl.close()

    # Restore.
    snap.restore_snapshot(dest)

    # Bob's post-snapshot mutation is gone.
    rl = ReciprocityLedger(db_path=state["recip_db"])
    try:
        assert rl.exchange_rate("bob") == 0.0
        # Alice is back to her pre-snapshot state.
        s = rl.stats("alice")
        assert s.carbon_received == pytest.approx(1.0)
        assert s.nutrients_returned == pytest.approx(2.0)
    finally:
        rl.close()


def test_restore_preserves_prior_state(snap: SoilSnapshot, state: dict) -> None:
    """A restore must leave the previous state in a *.preserve-<ts>
    directory so the operator can recover from a botched restore by hand."""
    dest = snap.take_snapshot()

    # Mutate the live soil so the preserve dir has something distinctive.
    (state["soil_dir"] / "post-snapshot-marker").write_text("present at restore time")

    snap.restore_snapshot(dest)

    # The post-snapshot-marker shouldn't be in the live soil anymore
    # (restore replaced it) but it must be findable in some sibling
    # preserve directory.
    live_marker = state["soil_dir"] / "post-snapshot-marker"
    assert not live_marker.exists()

    parent = state["soil_dir"].parent
    preserve_candidates = list(parent.glob("soil.preserve-*"))
    assert preserve_candidates, "expected at least one preserve dir"
    found = any((p / "post-snapshot-marker").exists() for p in preserve_candidates)
    assert found, "the live state at restore time was not preserved"


def test_restore_rejects_invalid_snapshot(snap: SoilSnapshot, tmp_path: Path) -> None:
    """If verify_snapshot fails, restore must not touch the live tree."""
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    with pytest.raises(ValueError):
        snap.restore_snapshot(bogus)


def test_restore_with_missing_components(tmp_path: Path) -> None:
    """A snapshot taken from a fresh machine (no ledgers yet) restores
    cleanly without leaving stray staging files."""
    state_root = tmp_path / "state"
    soil_dir = state_root / "soil"
    soil_dir.mkdir(parents=True)
    (soil_dir / "x").write_text("a")
    paths = SnapshotPaths(
        soil_dir=soil_dir,
        reciprocity_db=state_root / "r.db",
        niches_db=state_root / "n.db",
    )
    snap = SoilSnapshot(
        paths=paths,
        snapshots_dir=tmp_path / "snapshots",
        audit_path=tmp_path / "audit.jsonl",
    )
    dest = snap.take_snapshot()
    snap.restore_snapshot(dest)
    # Live state still consists of just the soil.
    assert soil_dir.exists()
    assert not (state_root / "r.db").exists()
    # No leftover staging
    assert not (state_root / "r.db.staging-restore").exists()
    assert not soil_dir.with_name(soil_dir.name + ".staging-restore").exists()


# ── 4. List ─────────────────────────────────────────────────────────────────


def test_list_empty(snap: SoilSnapshot) -> None:
    assert snap.list_snapshots() == []
    out = cli_format_list(snap)
    assert "0 snapshots" in out


def test_list_returns_newest_first(snap: SoilSnapshot, tmp_path: Path) -> None:
    # Manually create three snapshots with synthetic timestamps in the
    # manifest. Easier than time-travelling for the test.
    snap_root = snap.snapshots_dir
    for i, ts in enumerate(
        ["2026-01-01T00-00-00Z", "2026-05-01T00-00-00Z", "2026-03-01T00-00-00Z"]
    ):
        d = snap_root / ts
        (d / "soil").mkdir(parents=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "taken_at": ts.replace("T", "T").replace("-00-00Z", ":00:00+00:00"),
            "git_sha": None,
            "soil_present": True,
            "reciprocity_present": False,
            "niches_present": False,
            "soil_file_count": 0,
            "extra": {},
        }
        # Convert our slug back to ISO for the manifest content.
        manifest["taken_at"] = (
            "2026-01-01T00:00:00+00:00"
            if i == 0
            else "2026-05-01T00:00:00+00:00"
            if i == 1
            else "2026-03-01T00:00:00+00:00"
        )
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f)
    rows = snap.list_snapshots()
    iso_dates = [r.taken_at.strftime("%Y-%m-%d") for r in rows]
    assert iso_dates == ["2026-05-01", "2026-03-01", "2026-01-01"]


def test_list_skips_staging_dirs(snap: SoilSnapshot) -> None:
    """A `.staging` directory left over from a crashed take must not be
    reported as a real snapshot."""
    stale = snap.snapshots_dir / "in-flight.staging"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text("{}")  # would be ignored anyway
    snap.take_snapshot()
    rows = snap.list_snapshots()
    assert all(".staging" not in r.path.name for r in rows)


# ── 5. Rotation ─────────────────────────────────────────────────────────────


def _seed_snapshot(snap: SoilSnapshot, taken_at: datetime, label: str) -> Path:
    """Write a synthetic snapshot directory whose manifest claims a
    specific taken_at. Used to construct rotation scenarios without
    waiting in real time."""
    name = taken_at.strftime("%Y-%m-%dT%H-%M-%SZ") + f"_{label}"
    d = snap.snapshots_dir / name
    (d / "soil").mkdir(parents=True)
    (d / "soil" / "f").write_text(label)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "taken_at": taken_at.isoformat(),
        "git_sha": None,
        "soil_present": True,
        "reciprocity_present": False,
        "niches_present": False,
        "soil_file_count": 1,
        "extra": {},
    }
    with open(d / "manifest.json", "w") as f:
        json.dump(manifest, f)
    return d


def test_rotation_keeps_most_recent(snap: SoilSnapshot) -> None:
    """With default keep_recent=10, the 10 newest survive even if the
    overall count is much higher."""
    now = datetime.now(timezone.utc)
    paths = [_seed_snapshot(snap, now - timedelta(hours=i), f"h{i}") for i in range(20)]
    deleted = snap.rotate_snapshots()
    survivors = {s.path for s in snap.list_snapshots()}
    # Top-10 (newest) are kept.
    for keeper in paths[:10]:
        assert keeper in survivors
    # The rest may also survive if they fall into daily/weekly buckets,
    # but at minimum every deleted path is one of the older ones.
    for d in deleted:
        assert d in paths
        # Sanity: deleted paths must NOT be in the top-10 newest.
        assert d not in paths[:DEFAULT_KEEP_RECENT]


def test_rotation_keeps_daily_buckets(snap: SoilSnapshot) -> None:
    """One snapshot per UTC date is retained for keep_daily distinct days
    even after the recent window slides past them."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Build 5 days × 4 snapshots/day = 20 snapshots.
    for day in range(5):
        for hour in range(4):
            _seed_snapshot(snap, base + timedelta(days=day, hours=hour * 3), f"d{day}h{hour}")
    snap.rotate_snapshots(keep_recent=2, keep_daily=5, keep_weekly=0)
    survivors = snap.list_snapshots()
    # At least one survivor per UTC date.
    days_present = {s.taken_at.strftime("%Y-%m-%d") for s in survivors}
    assert len(days_present) == 5


def test_rotation_audit_record(snap: SoilSnapshot) -> None:
    """Rotation logs to the audit file so the operator can see what got
    deleted between manual list inspections."""
    now = datetime.now(timezone.utc)
    for i in range(15):
        _seed_snapshot(snap, now - timedelta(hours=i), f"h{i}")
    snap.rotate_snapshots(keep_recent=3, keep_daily=0, keep_weekly=0)
    assert snap.audit_path.exists()
    last = None
    with open(snap.audit_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    assert last is not None and last["event"] == "snapshot_rotated"
    assert last["kept"] == 3


# ── 6. Health summary ──────────────────────────────────────────────────────


def test_health_summary_counts(state: dict) -> None:
    """The cold-start health summary should reflect the live ledger
    contents without needing chromadb installed for the SQLite side."""
    summary = health_summary(paths=state["paths"])
    # ChromaDB isn't real here so soil_collections may be empty/error
    # but the SQLite side must work.
    assert summary["reciprocity_agents"] == 1
    # 1 request + 1 contribution = 2 events
    assert summary["reciprocity_events"] == 2
    assert summary["niches_total"] == 1
    assert summary["niches_referenced"] == 0


def test_cli_format_health_renders(state: dict) -> None:
    summary = health_summary(paths=state["paths"])
    out = cli_format_health(summary)
    assert "agents:" in out
    assert "total niches" in out


# ── 7. Module sanity ───────────────────────────────────────────────────────


def test_defaults_are_sensible() -> None:
    assert DEFAULT_KEEP_RECENT > 0
    assert DEFAULT_KEEP_DAILY > 0
    assert DEFAULT_KEEP_WEEKLY > 0
    assert SCHEMA_VERSION >= 1


def test_snapshot_info_is_frozen() -> None:
    """Mutating one field by accident would silently desync rotation
    decisions from the audit log. The frozen dataclass blocks that."""
    s = SnapshotInfo(
        path=Path("/tmp/x"),
        taken_at=datetime.now(timezone.utc),
        schema_version=1,
        soil_present=True,
        reciprocity_present=False,
        niches_present=False,
        size_bytes=1,
    )
    with pytest.raises((AttributeError, TypeError)):
        s.size_bytes = 99  # type: ignore[misc]
