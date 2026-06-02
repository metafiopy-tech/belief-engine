"""SWE-bench Verified checkpoint probe for the STARVED-arm experiment.

At configured generations the driver measures held-out *generalization* of each
arm's current soil: it attempts a fixed set of SWE-bench Verified instances
against the arm soil and records the resolve rate. The thesis predicts STARVED's
held-out success degrades as its soil fills with self-judged fictions, while
FED's holds — this is the capability-side complement to the soil-cloud metrics.

**Contamination rule (design doc §8):** SWE-bench Verified is held out — it is
NEVER used for admission, and the self-judge never sees it. It only measures.

This module owns:

- the probe **result store** (SQLite) and query helpers — fully testable;
- the probe **orchestration** factory ``make_probe_fn`` returning a
  ``(gen, arm, soil_dir) -> ProbeResult`` callable.

The real evaluation against the SWE-bench Verified dataset + Docker harness is a
documented integration seam: :func:`run_instances` raises ``NotImplementedError``
with guidance until the harness is wired on the Mac (it needs the dataset and
container runtime, which cannot run in CI). We do NOT return a mocked green
result — an unimplemented probe must fail loudly, never silently report success.
See the manual-verification checklist in the session handoff.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # avoid import cycle at runtime
    from belief.experiments.starved_runner import ProbeResult, StarvedConfig


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------


def init_probe_db(db_path: Path) -> None:
    """Create the probe-results table (idempotent)."""
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS probe_results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT    NOT NULL,
                gen           INTEGER NOT NULL,
                arm           TEXT    NOT NULL,
                n_instances   INTEGER NOT NULL,
                n_resolved    INTEGER NOT NULL,
                resolve_rate  REAL    NOT NULL,
                timestamp     TEXT    NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def record_probe(experiment_id: str, result: "ProbeResult", db_path: Path) -> None:
    """Append one probe result."""
    db_path = Path(db_path).expanduser()
    init_probe_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO probe_results
                (experiment_id, gen, arm, n_instances, n_resolved, resolve_rate, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                int(result.gen),
                result.arm.upper(),
                int(result.n_instances),
                int(result.n_resolved),
                float(result.resolve_rate),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_probes(experiment_id: str, db_path: Path) -> list[dict]:
    """Return all probe results for an experiment (for offline reporting)."""
    db_path = Path(db_path).expanduser()
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM probe_results WHERE experiment_id = ? ORDER BY gen, arm",
            (experiment_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Held-out instance set
# ---------------------------------------------------------------------------

# Pin the exact SWE-bench Verified instance IDs to probe here before the pilot.
# Kept small (the probe runs at several checkpoints × 2 arms). Empty until pinned
# — make_probe_fn refuses to build a probe with no instances.
SWEBENCH_VERIFIED_PROBE_INSTANCES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Real harness seam (must be wired on the Mac)
# ---------------------------------------------------------------------------


def run_instances(instance_ids: tuple[str, ...], soil_dir: Path) -> int:
    """Attempt SWE-bench Verified instances against ``soil_dir``; return #resolved.

    NOT yet wired: the real path loads the SWE-bench Verified dataset, runs each
    instance through the engine (brownfield fix) with ``BELIEF_SOIL_PATH`` set to
    ``soil_dir`` and the external test executed only to score resolution, then
    counts resolved instances via the official harness. This needs the dataset +
    Docker runtime and is verified on the Mac per the manual checklist.

    Raising (rather than returning a fake count) is deliberate: an unimplemented
    probe must fail loudly so a run never records a fabricated resolve rate.
    """
    raise NotImplementedError(
        "SWE-bench Verified harness not wired. Pin SWEBENCH_VERIFIED_PROBE_INSTANCES "
        "and implement run_instances against the dataset+Docker harness on the Mac "
        "before enabling --probe-at. See starved_arm_design.md and the session checklist."
    )


def make_probe_fn(
    config: "StarvedConfig",
    instance_ids: Optional[tuple[str, ...]] = None,
    runner: Callable[[tuple[str, ...], Path], int] = run_instances,
) -> Callable[[int, str, Path], "ProbeResult"]:
    """Build the ``(gen, arm, soil_dir) -> ProbeResult`` probe callable.

    ``runner`` is injectable so tests can supply a deterministic resolver; the
    default is the real (currently unimplemented) harness seam. Refuses to build
    a probe with an empty instance set, so ``--probe-at`` cannot silently run a
    zero-instance probe.
    """
    ids = instance_ids if instance_ids is not None else SWEBENCH_VERIFIED_PROBE_INSTANCES
    if not ids:
        raise ValueError(
            "No SWE-bench Verified probe instances pinned; set "
            "SWEBENCH_VERIFIED_PROBE_INSTANCES or pass instance_ids before using --probe-at."
        )

    def _probe(gen: int, arm: str, soil_dir: Path) -> "ProbeResult":
        from belief.experiments.starved_runner import ProbeResult

        n_resolved = runner(ids, soil_dir)
        return ProbeResult(gen=gen, arm=arm, n_instances=len(ids), n_resolved=n_resolved)

    return _probe
