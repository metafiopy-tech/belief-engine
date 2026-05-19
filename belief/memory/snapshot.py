"""Soil-Layer Snapshot — durable institutional memory (mycorrhizal Stage 3).

The biological literature is unambiguous on resilience: forest mycorrhizal
communities survive fire on decadal timescales (Dove & Hart 2017 meta-analysis),
spore banks persist through disturbance (Glassman et al. 2016), and legacy
retention dramatically accelerates post-disturbance recovery. The architectural
translation: the soil layer is the system's *institutional memory*, and its
persistence must be guaranteed independently of any single process, agent, or
even the Belief Engine itself.

This module produces durable snapshots of the soil layer plus the two
mycorrhizal ledgers (reciprocity, niche) into self-contained directories
that can be:
- Restored on a fresh machine to bring up a functional engine.
- Inspected with raw SQLite + ChromaDB clients without the engine running.
- Auto-rotated so disk usage stays bounded.

A snapshot directory contains:

    <snapshot_dir>/
        manifest.json          schema_version, taken_at, git_sha, source paths
        soil/                  full copy of the ChromaDB PersistentClient dir
        reciprocity.db         copy of the reciprocity ledger
        reciprocity.db-wal     (optional, if present)
        reciprocity.db-shm     (optional, if present)
        niches.db              copy of the niche ledger
        niches.db-wal          (optional, if present)
        niches.db-shm          (optional, if present)

Restore is atomic: contents are staged into a sibling directory, then the
live tree is moved aside and the staged tree promoted via a single rename.
A crash mid-restore leaves the original tree intact (in a ``*.preserve``
backup) so the operator can recover by hand.

Rotation: ``rotate_snapshots`` keeps the N most recent hourly snapshots,
the M most recent daily snapshots (one per UTC date), and the K most
recent weekly snapshots (one per ISO week). Defaults: 10/10/10. Buckets
are computed from each snapshot's ``taken_at`` so the policy is stable
across host clocks.

Scoping note for Stage 3: a "cold-start" mode in the Belief Engine context
isn't bringing up a daemon — the engine is a one-shot LangGraph per
``belief build``. Cold-start means: a fresh machine restores from a
snapshot, runs the health summary, and is ready to serve a build. The
``belief cold-start`` CLI is thin glue around ``restore_snapshot`` plus
the soil-health probe.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("belief.memory.snapshot")


# ── Defaults ────────────────────────────────────────────────────────────────

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_SOIL_DIR = _BELIEF_HOME / "soil"
_DEFAULT_RECIPROCITY_DB = _BELIEF_HOME / "reciprocity.db"
_DEFAULT_NICHES_DB = _BELIEF_HOME / "niches.db"
_DEFAULT_SNAPSHOTS_DIR = _BELIEF_HOME / "snapshots"
_DEFAULT_AUDIT_PATH = _BELIEF_HOME / "audit" / "snapshot.jsonl"

SCHEMA_VERSION = 1

# Default rotation buckets — keep last N at each cadence. Tuned for ~6h
# snapshot cadence: 10 hourly (~2.5 days), 10 daily (~10 days), 10 weekly
# (~10 weeks). Adjustable via env or kwargs on the rotation function.
DEFAULT_KEEP_RECENT = 10
DEFAULT_KEEP_DAILY = 10
DEFAULT_KEEP_WEEKLY = 10


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class SnapshotManifest:
    """The header written into ``manifest.json`` inside each snapshot dir."""

    schema_version: int
    taken_at: str  # ISO UTC
    git_sha: Optional[str]
    soil_present: bool
    reciprocity_present: bool
    niches_present: bool
    soil_file_count: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "taken_at": self.taken_at,
            "git_sha": self.git_sha,
            "soil_present": self.soil_present,
            "reciprocity_present": self.reciprocity_present,
            "niches_present": self.niches_present,
            "soil_file_count": self.soil_file_count,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SnapshotManifest:
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            taken_at=str(d.get("taken_at", "")),
            git_sha=d.get("git_sha"),
            soil_present=bool(d.get("soil_present", False)),
            reciprocity_present=bool(d.get("reciprocity_present", False)),
            niches_present=bool(d.get("niches_present", False)),
            soil_file_count=int(d.get("soil_file_count", 0)),
            extra=dict(d.get("extra", {})),
        )


@dataclass(frozen=True)
class SnapshotInfo:
    """A row in the ``list_snapshots`` output."""

    path: Path
    taken_at: datetime
    schema_version: int
    soil_present: bool
    reciprocity_present: bool
    niches_present: bool
    size_bytes: int


# ── Helpers ─────────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _current_git_sha(repo_path: Optional[Path] = None) -> Optional[str]:
    """Return the short git SHA for the engine's source tree, or None
    if git isn't available or this isn't a working repo."""
    cwd = str(repo_path) if repo_path else None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        sha = out.decode("ascii", errors="replace").strip()
        return sha or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _slugify_timestamp(dt: datetime) -> str:
    """ISO-ish timestamp safe for use in a directory name (no colons)."""
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def _dir_file_count(p: Path) -> int:
    """Total file count under p, recursively. Returns 0 for missing path."""
    if not p.exists():
        return 0
    count = 0
    for _root, _dirs, files in os.walk(p):
        count += len(files)
    return count


def _dir_size_bytes(p: Path) -> int:
    """Total byte size of a directory tree. Best-effort; missing files
    are skipped rather than raised."""
    if not p.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _copy_dir(src: Path, dst: Path) -> None:
    """Recursively copy src to dst. dst must not exist (we never overwrite
    silently). Uses shutil.copytree's symlink=False semantics."""
    shutil.copytree(src, dst, symlinks=False)


def _copy_sqlite(src: Path, dst: Path) -> None:
    """Snapshot a SQLite file using the backup API rather than a raw
    file copy. WAL-mode databases have writes in -wal that a plain copy
    can miss; SQLite's Backup API merges them transparently into a single
    self-contained dst file. Falls back to plain copy if the backup API
    isn't available for some reason (e.g. file is locked uncontrollably).
    """
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src_conn = sqlite3.connect(str(src))
        try:
            dst_conn = sqlite3.connect(str(dst))
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()
    except sqlite3.Error as e:
        logger.warning(
            "snapshot: SQLite backup of %s failed (%s); falling back to file copy",
            src,
            e,
        )
        shutil.copy2(src, dst)


def _audit(audit_path: Path, record: dict) -> None:
    """Append a JSONL audit line. Never raises."""
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": _iso(_utcnow()), **record}
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:  # pragma: no cover — audit must never crash callers
        logger.warning("snapshot audit write failed: %s", e)


# ── Main API ────────────────────────────────────────────────────────────────


@dataclass
class SnapshotPaths:
    """The set of source paths included in a snapshot. Centralised so
    tests can override the whole bundle via one constructor argument."""

    soil_dir: Path = _DEFAULT_SOIL_DIR
    reciprocity_db: Path = _DEFAULT_RECIPROCITY_DB
    niches_db: Path = _DEFAULT_NICHES_DB

    def expanded(self) -> SnapshotPaths:
        return SnapshotPaths(
            soil_dir=Path(self.soil_dir).expanduser(),
            reciprocity_db=Path(self.reciprocity_db).expanduser(),
            niches_db=Path(self.niches_db).expanduser(),
        )


class SoilSnapshot:
    """Pack, restore, list, and verify durable snapshots of the engine state.

    All paths are configurable for testability — production callers use
    the defaults rooted at ``~/.belief-engine/``.
    """

    def __init__(
        self,
        paths: Optional[SnapshotPaths] = None,
        snapshots_dir: Path = _DEFAULT_SNAPSHOTS_DIR,
        audit_path: Path = _DEFAULT_AUDIT_PATH,
        repo_path: Optional[Path] = None,
    ) -> None:
        self.paths = (paths or SnapshotPaths()).expanded()
        self.snapshots_dir = Path(snapshots_dir).expanduser()
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = Path(audit_path).expanduser()
        self.repo_path = repo_path

    # ── take ────────────────────────────────────────────────────────────

    def take_snapshot(self, dest_path: Optional[Path] = None, label: Optional[str] = None) -> Path:
        """Pack the live state into a new snapshot directory.

        Returns the path to the completed snapshot. Writes are staged
        into ``<dest>.staging`` and atomically promoted with ``rename``
        so a crash mid-take never leaves a half-written snapshot
        directory in the rotation set.
        """
        now = _utcnow()
        if dest_path is None:
            slug = _slugify_timestamp(now)
            if label:
                slug = f"{slug}_{label}"
            dest = self.snapshots_dir / slug
        else:
            dest = Path(dest_path).expanduser()
        staging = dest.with_name(dest.name + ".staging")

        # Clean stale staging dir from a prior crash before reusing it.
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=False)

        # Soil dir — full recursive copy (ChromaDB's PersistentClient
        # writes the entire collection state under one root, so a tree
        # copy snapshots vectors + metadata + FSRS state in one shot).
        soil_dst = staging / "soil"
        soil_present = self.paths.soil_dir.exists()
        if soil_present:
            _copy_dir(self.paths.soil_dir, soil_dst)

        # SQLite ledgers — use the backup API so WAL contents merge in.
        recip_present = self.paths.reciprocity_db.exists()
        if recip_present:
            _copy_sqlite(self.paths.reciprocity_db, staging / "reciprocity.db")
        niches_present = self.paths.niches_db.exists()
        if niches_present:
            _copy_sqlite(self.paths.niches_db, staging / "niches.db")

        manifest = SnapshotManifest(
            schema_version=SCHEMA_VERSION,
            taken_at=_iso(now),
            git_sha=_current_git_sha(self.repo_path),
            soil_present=soil_present,
            reciprocity_present=recip_present,
            niches_present=niches_present,
            soil_file_count=_dir_file_count(soil_dst),
            extra={"label": label} if label else {},
        )
        with open(staging / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)

        # Atomic promote: rename staging → final destination.
        if dest.exists():
            raise FileExistsError(f"snapshot dest already exists: {dest}")
        os.replace(staging, dest)

        _audit(
            self.audit_path,
            {
                "event": "snapshot_taken",
                "path": str(dest),
                "size_bytes": _dir_size_bytes(dest),
                "soil_files": manifest.soil_file_count,
                "git_sha": manifest.git_sha,
            },
        )
        logger.info(f"Snapshot taken: {dest}")
        return dest

    # ── verify ──────────────────────────────────────────────────────────

    def verify_snapshot(self, path: Path) -> bool:
        """Sanity-check a snapshot directory before restoring from it.

        Checks: manifest.json parses, schema_version matches, declared
        component presence matches what's on disk. Does NOT validate
        the contents of ChromaDB or SQLite files — that's
        ChromaDB/sqlite's own concern at open time. The point here is
        to catch obvious truncation / wrong-directory cases before we
        clobber the live tree on restore.
        """
        path = Path(path).expanduser()
        mfile = path / "manifest.json"
        if not mfile.exists():
            logger.warning(f"verify_snapshot: missing manifest at {mfile}")
            return False
        try:
            with open(mfile, "r", encoding="utf-8") as f:
                m = SnapshotManifest.from_dict(json.load(f))
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning(f"verify_snapshot: manifest unreadable ({e})")
            return False
        if m.schema_version != SCHEMA_VERSION:
            logger.warning(
                f"verify_snapshot: schema {m.schema_version} != current {SCHEMA_VERSION}"
            )
            return False
        # Declared presence must match disk reality.
        if m.soil_present != (path / "soil").exists():
            logger.warning("verify_snapshot: soil presence/manifest mismatch")
            return False
        if m.reciprocity_present != (path / "reciprocity.db").exists():
            logger.warning("verify_snapshot: reciprocity.db presence/manifest mismatch")
            return False
        if m.niches_present != (path / "niches.db").exists():
            logger.warning("verify_snapshot: niches.db presence/manifest mismatch")
            return False
        return True

    # ── restore ─────────────────────────────────────────────────────────

    def restore_snapshot(self, path: Path) -> None:
        """Atomically replace live state with the snapshot's contents.

        Strategy:
          1. Verify the snapshot before touching anything live.
          2. Stage the new live state in ``<live>.staging-restore``.
          3. Atomically rename live → ``<live>.preserve-<ts>``.
          4. Atomically rename staging → live.
          5. On success, log the preserve path for the operator.

        If any step before the live rename fails, the live tree is
        untouched. If the second rename fails (between live → preserve
        and staging → live), the preserve dir holds the prior state
        for manual recovery.
        """
        path = Path(path).expanduser()
        if not self.verify_snapshot(path):
            raise ValueError(f"snapshot at {path} failed verification")

        # Stage the new live state in tempdirs adjacent to each target.
        ts = _slugify_timestamp(_utcnow())

        # --- Stage soil ---
        soil_src = path / "soil"
        soil_live = self.paths.soil_dir
        soil_staging = soil_live.with_name(soil_live.name + ".staging-restore")
        if soil_staging.exists():
            shutil.rmtree(soil_staging, ignore_errors=True)
        if soil_src.exists():
            _copy_dir(soil_src, soil_staging)

        # --- Stage ledgers ---
        # SQLite files copy faster than directories and don't need a
        # staging directory of their own; we stage them as ``.staging``
        # files alongside their live targets.
        def _stage_sqlite(src: Path, live: Path) -> Optional[Path]:
            if not src.exists():
                return None
            stage_path = live.with_name(live.name + ".staging-restore")
            if stage_path.exists():
                stage_path.unlink()
            shutil.copy2(src, stage_path)
            return stage_path

        recip_stage = _stage_sqlite(path / "reciprocity.db", self.paths.reciprocity_db)
        niches_stage = _stage_sqlite(path / "niches.db", self.paths.niches_db)

        # --- Promote: live → preserve, staging → live ---
        # Preserve dir holds the prior state for one rotation so a
        # botched restore can always be undone by hand.
        moved_pairs: list[tuple[Path, Path]] = []

        def _move(src: Path, dst: Path) -> None:
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
                os.replace(src, dst)
                moved_pairs.append((src, dst))

        try:
            # Soil
            if soil_staging.exists():
                preserve_soil = soil_live.with_name(soil_live.name + f".preserve-{ts}")
                if soil_live.exists():
                    os.replace(soil_live, preserve_soil)
                os.replace(soil_staging, soil_live)
            # Ledgers
            if recip_stage is not None:
                live = self.paths.reciprocity_db
                if live.exists():
                    preserve = live.with_name(live.name + f".preserve-{ts}")
                    os.replace(live, preserve)
                os.replace(recip_stage, live)
            if niches_stage is not None:
                live = self.paths.niches_db
                if live.exists():
                    preserve = live.with_name(live.name + f".preserve-{ts}")
                    os.replace(live, preserve)
                os.replace(niches_stage, live)
        except Exception:
            # Clean up any staging that wasn't promoted; original is
            # already preserved on disk under .preserve-<ts>.
            for stale in (soil_staging, recip_stage, niches_stage):
                if stale is not None and stale.exists():
                    if stale.is_dir():
                        shutil.rmtree(stale, ignore_errors=True)
                    else:
                        try:
                            stale.unlink()
                        except OSError:
                            pass
            raise

        _audit(
            self.audit_path,
            {
                "event": "snapshot_restored",
                "source_path": str(path),
                "preserve_suffix": f".preserve-{ts}",
                "moved_count": len(moved_pairs),
            },
        )
        logger.info(f"Restored snapshot from {path}; prior state preserved at *.preserve-{ts}")

    # ── list ────────────────────────────────────────────────────────────

    def list_snapshots(self) -> list[SnapshotInfo]:
        """Every valid snapshot directory under ``self.snapshots_dir``,
        sorted newest-first. Invalid or partial directories are skipped.
        """
        out: list[SnapshotInfo] = []
        if not self.snapshots_dir.exists():
            return out
        for child in self.snapshots_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.endswith(".staging"):
                continue  # crashed take, skip
            mfile = child / "manifest.json"
            if not mfile.exists():
                continue
            try:
                with open(mfile, "r", encoding="utf-8") as f:
                    m = SnapshotManifest.from_dict(json.load(f))
            except (json.JSONDecodeError, ValueError, OSError):
                continue
            taken = _parse_iso(m.taken_at) or _utcnow()
            out.append(
                SnapshotInfo(
                    path=child,
                    taken_at=taken,
                    schema_version=m.schema_version,
                    soil_present=m.soil_present,
                    reciprocity_present=m.reciprocity_present,
                    niches_present=m.niches_present,
                    size_bytes=_dir_size_bytes(child),
                )
            )
        out.sort(key=lambda s: s.taken_at, reverse=True)
        return out

    # ── rotate ──────────────────────────────────────────────────────────

    def rotate_snapshots(
        self,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        keep_daily: int = DEFAULT_KEEP_DAILY,
        keep_weekly: int = DEFAULT_KEEP_WEEKLY,
    ) -> list[Path]:
        """Apply the GFS retention policy and delete what falls out.

        The selected set is the union of:
          * the ``keep_recent`` most-recent snapshots (any cadence)
          * one snapshot per UTC date (the newest of that day),
            up to ``keep_daily`` distinct days
          * one snapshot per ISO week, up to ``keep_weekly`` weeks

        Returns the list of paths that were deleted.
        """
        snaps = self.list_snapshots()
        keep: set[Path] = set()
        # Recent
        keep.update(s.path for s in snaps[:keep_recent])
        # Daily — newest per UTC date.
        per_day: dict[str, SnapshotInfo] = {}
        for s in snaps:
            key = s.taken_at.strftime("%Y-%m-%d")
            if key not in per_day or s.taken_at > per_day[key].taken_at:
                per_day[key] = s
        for key in sorted(per_day.keys(), reverse=True)[:keep_daily]:
            keep.add(per_day[key].path)
        # Weekly — newest per ISO week.
        per_week: dict[str, SnapshotInfo] = {}
        for s in snaps:
            iso_year, iso_week, _ = s.taken_at.isocalendar()
            key = f"{iso_year:04d}-W{iso_week:02d}"
            if key not in per_week or s.taken_at > per_week[key].taken_at:
                per_week[key] = s
        for key in sorted(per_week.keys(), reverse=True)[:keep_weekly]:
            keep.add(per_week[key].path)

        deleted: list[Path] = []
        for s in snaps:
            if s.path not in keep:
                shutil.rmtree(s.path, ignore_errors=True)
                deleted.append(s.path)

        _audit(
            self.audit_path,
            {
                "event": "snapshot_rotated",
                "kept": len(keep),
                "deleted": len(deleted),
                "policy": {
                    "keep_recent": keep_recent,
                    "keep_daily": keep_daily,
                    "keep_weekly": keep_weekly,
                },
            },
        )
        return deleted


# ── Cold-start health summary ──────────────────────────────────────────────


def health_summary(paths: Optional[SnapshotPaths] = None) -> dict:
    """Smoke-check the live state and return a one-shot summary dict.

    Used by ``belief cold-start`` after a restore. Lightweight — opens
    each ledger briefly, counts rows, asks ChromaDB how big each
    collection is. Never raises; missing components surface as
    ``None``/``0`` in the report so the operator sees what's there.
    """
    p = (paths or SnapshotPaths()).expanded()
    summary: dict = {
        "soil_dir": str(p.soil_dir),
        "soil_present": p.soil_dir.exists(),
        "soil_collections": {},
        "reciprocity_db": str(p.reciprocity_db),
        "reciprocity_agents": 0,
        "reciprocity_events": 0,
        "niches_db": str(p.niches_db),
        "niches_total": 0,
        "niches_referenced": 0,
    }

    # ChromaDB collection counts. Importing chromadb is heavier than the
    # rest of this module so we defer until the summary actually runs.
    if p.soil_dir.exists():
        try:
            import chromadb  # noqa: PLC0415

            client = chromadb.PersistentClient(path=str(p.soil_dir))
            for col in client.list_collections():
                try:
                    summary["soil_collections"][col.name] = col.count()
                except Exception:  # pragma: no cover — collection-specific
                    summary["soil_collections"][col.name] = None
        except Exception as e:  # pragma: no cover — chromadb optional in tests
            summary["soil_error"] = str(e)

    # Reciprocity ledger row counts.
    if p.reciprocity_db.exists():
        try:
            conn = sqlite3.connect(str(p.reciprocity_db))
            try:
                summary["reciprocity_agents"] = int(
                    conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
                )
                summary["reciprocity_events"] = int(
                    conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                )
            finally:
                conn.close()
        except sqlite3.Error as e:  # pragma: no cover
            summary["reciprocity_error"] = str(e)

    # Niche ledger row counts.
    if p.niches_db.exists():
        try:
            conn = sqlite3.connect(str(p.niches_db))
            try:
                summary["niches_total"] = int(
                    conn.execute("SELECT COUNT(*) FROM niches").fetchone()[0]
                )
                summary["niches_referenced"] = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM niches WHERE reference_count > 0"
                    ).fetchone()[0]
                )
            finally:
                conn.close()
        except sqlite3.Error as e:  # pragma: no cover
            summary["niches_error"] = str(e)

    return summary


# ── CLI rendering ──────────────────────────────────────────────────────────


def cli_format_list(snap: SoilSnapshot) -> str:
    rows = snap.list_snapshots()
    header = (
        f"Snapshots — root={snap.snapshots_dir}\n"
        f"  {len(rows)} snapshot{'s' if len(rows) != 1 else ''}"
    )
    if not rows:
        return header + "\n  (none — run `belief snapshot take` to create one)"
    lines = [
        header,
        "",
        f"  {'taken_at (UTC)':<22} {'size':>10} {'soil':>5} {'recip':>6} {'niches':>7}  path",
        f"  {'-' * 22} {'-' * 10} {'-' * 5} {'-' * 6} {'-' * 7}  {'-' * 19}",
    ]
    for r in rows:
        size_kb = r.size_bytes / 1024.0
        if size_kb < 1024:
            size_str = f"{size_kb:>8.1f}KB"
        else:
            size_str = f"{size_kb / 1024:>8.1f}MB"
        lines.append(
            f"  {r.taken_at.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{size_str:>10} "
            f"{'yes' if r.soil_present else 'no':>5} "
            f"{'yes' if r.reciprocity_present else 'no':>6} "
            f"{'yes' if r.niches_present else 'no':>7}  "
            f"{r.path}"
        )
    return "\n".join(lines)


def cli_format_health(summary: dict) -> str:
    lines = [
        "Soil-layer health summary",
        f"  soil dir:           {summary['soil_dir']}",
        f"  soil present:       {summary['soil_present']}",
    ]
    if summary.get("soil_collections"):
        lines.append("  collections:")
        for name, count in sorted(summary["soil_collections"].items()):
            lines.append(f"    {name:<28} {count}")
    if summary.get("soil_error"):
        lines.append(f"  soil error:         {summary['soil_error']}")
    lines.extend(
        [
            f"  reciprocity db:     {summary['reciprocity_db']}",
            f"    agents:           {summary['reciprocity_agents']}",
            f"    events:           {summary['reciprocity_events']}",
            f"  niches db:          {summary['niches_db']}",
            f"    total niches:     {summary['niches_total']}",
            f"    referenced:       {summary['niches_referenced']}",
        ]
    )
    if summary.get("reciprocity_error"):
        lines.append(f"  reciprocity error:  {summary['reciprocity_error']}")
    if summary.get("niches_error"):
        lines.append(f"  niches error:       {summary['niches_error']}")
    return "\n".join(lines)


# ── CLI entry points (called from belief.cli) ──────────────────────────────


def cli_take(label: Optional[str] = None) -> str:
    snap = SoilSnapshot()
    dest = snap.take_snapshot(label=label)
    rotated = snap.rotate_snapshots()
    msg = f"Snapshot taken: {dest}"
    if rotated:
        msg += f"\nRotation deleted {len(rotated)} older snapshot(s)."
    return msg


def cli_restore(path: str) -> str:
    snap = SoilSnapshot()
    snap.restore_snapshot(Path(path))
    return (
        f"Restored from {path}.\n"
        "Prior state preserved under *.preserve-<timestamp> beside each "
        "live target. Delete those after confirming the restore worked."
    )


def cli_list() -> str:
    return cli_format_list(SoilSnapshot())


def cli_verify(path: str) -> str:
    snap = SoilSnapshot()
    ok = snap.verify_snapshot(Path(path))
    return f"verify {path}: {'OK' if ok else 'FAIL — see warnings in log'}"


def cli_cold_start(snapshot_path: Optional[str] = None) -> str:
    """``belief cold-start`` — optional restore, then health summary.

    The engine is one-shot, not a daemon; cold-start is the read-side
    verification that a restored soil + ledger bundle is operational.
    Returns a non-empty health summary string on success.
    """
    if snapshot_path:
        snap = SoilSnapshot()
        snap.restore_snapshot(Path(snapshot_path))
    return cli_format_health(health_summary())
