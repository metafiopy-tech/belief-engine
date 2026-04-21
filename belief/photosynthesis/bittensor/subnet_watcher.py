"""Hourly, read-only snapshot of bittensor subnets.

Watched subnets default to [1, 62, 120]. Every snapshot records
(netuid, n_miners, n_validators, tao_staked, total_emission, snapshot_ts)
into a local SQLite table for trend analysis. No wallet operations; no
registrations; no outbound calls beyond the bittensor SDK's read paths.

The spec stresses "never crashes the daemon" — we wrap every SDK call
in a try/except and log-and-continue on error. The watcher is a nice-
to-have, not a critical path.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence


logger = logging.getLogger("belief.photosynthesis.bittensor.subnet_watcher")


DEFAULT_WATCHER_DB = Path("/var/lib/photosynthesis/subnet_snapshots.db")
DEFAULT_NETUIDS: tuple[int, ...] = (1, 62, 120)


SCHEMA = """
CREATE TABLE IF NOT EXISTS subnet_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    netuid          INTEGER NOT NULL,
    snapshot_ts     INTEGER NOT NULL,
    n_miners        INTEGER,
    n_validators    INTEGER,
    tao_staked      REAL,
    total_emission  REAL,
    raw_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_subnet_snapshots_netuid
    ON subnet_snapshots(netuid, snapshot_ts);
"""


@dataclass
class SubnetSnapshot:
    netuid: int
    n_miners: Optional[int] = None
    n_validators: Optional[int] = None
    tao_staked: Optional[float] = None
    total_emission: Optional[float] = None
    raw: dict[str, Any] = field(default_factory=dict)


class SubnetWatcher:
    def __init__(self, db_path: Path | str = DEFAULT_WATCHER_DB) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.row_factory = sqlite3.Row
            yield c
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    def record(self, snapshot: SubnetSnapshot) -> None:
        import json

        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "INSERT INTO subnet_snapshots"
                "(netuid, snapshot_ts, n_miners, n_validators, "
                " tao_staked, total_emission, raw_json) "
                "VALUES(?, ?, ?, ?, ?, ?, ?);",
                (
                    snapshot.netuid,
                    int(time.time()),
                    snapshot.n_miners,
                    snapshot.n_validators,
                    snapshot.tao_staked,
                    snapshot.total_emission,
                    json.dumps(snapshot.raw, sort_keys=True, separators=(",", ":"))
                    if snapshot.raw
                    else None,
                ),
            )
            c.execute("COMMIT;")

    def snapshot_once(
        self,
        netuids: Sequence[int] = DEFAULT_NETUIDS,
        *,
        network: str = "finney",
    ) -> list[SubnetSnapshot]:
        """Snapshot each netuid. Silently skips any that fail."""
        try:
            import bittensor as bt  # type: ignore[import-untyped]
        except ImportError:
            logger.info(
                "bittensor not installed; skipping subnet snapshot. "
                "Install the [photosynthesis-bittensor] extra."
            )
            return []

        try:
            sub = bt.Subtensor(network)
        except Exception as exc:
            logger.warning("bt.Subtensor(%s) failed: %s", network, exc)
            return []

        out: list[SubnetSnapshot] = []
        for netuid in netuids:
            try:
                mg = bt.Metagraph(netuid=netuid, sync=True, lite=True)
            except Exception as exc:
                logger.warning("metagraph fetch failed for netuid=%d: %s", netuid, exc)
                continue

            # Fields available vary across bittensor SDK versions; guard every access.
            snap = SubnetSnapshot(
                netuid=netuid,
                n_miners=_safe_len(getattr(mg, "uids", None)),
                n_validators=_safe_len(getattr(mg, "validator_trust", None)),
                tao_staked=_safe_sum(getattr(mg, "total_stake", None)),
                total_emission=_safe_sum(getattr(mg, "emission", None)),
                raw={"netuid": netuid, "network": network},
            )
            try:
                self.record(snap)
                out.append(snap)
            except Exception as exc:
                logger.warning("recording snapshot failed: %s", exc)
        return out


def _safe_len(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    try:
        return int(len(obj))
    except Exception:
        return None


def _safe_sum(obj: Any) -> Optional[float]:
    if obj is None:
        return None
    try:
        # torch tensor, numpy array, or list
        total = float(sum(float(x) for x in obj))
        return total
    except Exception:
        try:
            return float(obj)
        except Exception:
            return None


__all__ = [
    "DEFAULT_NETUIDS",
    "DEFAULT_WATCHER_DB",
    "SubnetSnapshot",
    "SubnetWatcher",
]
