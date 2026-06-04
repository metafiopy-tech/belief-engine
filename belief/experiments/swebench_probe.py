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

import json
import logging
import os
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # avoid import cycle at runtime
    from belief.experiments.starved_runner import ProbeResult, StarvedConfig

logger = logging.getLogger("belief.experiments.swebench_probe")

# Official SWE-bench Verified dataset + the model label written into predictions.
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
PREDICTION_MODEL_LABEL = "belief-engine"


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
# Real harness seams — three injectable stages (loader / predictor / evaluator)
# ---------------------------------------------------------------------------
#
# The full resolution path only runs on a machine with HuggingFace `datasets`,
# git, and a Docker daemon (the official `swebench` harness). Each stage is
# injectable so the orchestration is unit-testable with fakes; the defaults below
# are the real Mac path, verified by the one-instance smoke before the full run.


def load_verified_instances(
    instance_ids: tuple[str, ...],
    *,
    dataset_name: str = SWEBENCH_DATASET,
) -> list[dict]:
    """Pull full SWE-bench Verified records for ``instance_ids`` via HF datasets.

    The lean mirror (swebench_mirror) lacks base_commit / FAIL_TO_PASS / test_patch,
    so resolution needs the full dataset rows. Returns the records in the order of
    ``instance_ids``.
    """
    from datasets import load_dataset  # heavy, Mac-only

    ds = load_dataset(dataset_name, split="test")
    wanted = set(instance_ids)
    by_id = {row["instance_id"]: dict(row) for row in ds if row["instance_id"] in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(f"instance_ids not found in {dataset_name}: {sorted(missing)}")
    return [by_id[i] for i in instance_ids]


def generate_prediction(instance: dict, soil_dir: Path, *, model: str) -> dict:
    """Produce one SWE-bench prediction by running the engine's brownfield fix.

    Clones the repo at ``base_commit`` into a temp dir, runs ``brownfield.fix_issue``
    on the problem statement with ``BELIEF_SOIL_PATH`` set to the arm soil (so the
    arm's accumulated soil informs the fix), then captures ``git diff`` as the
    ``model_patch``. Returns the official prediction dict. A failed fix yields an
    empty patch (which the harness scores as unresolved) rather than raising.
    """
    import asyncio

    instance_id = instance["instance_id"]
    repo = instance["repo"]  # e.g. "django/django"
    base_commit = instance["base_commit"]
    problem = instance["problem_statement"]

    prev_soil = os.environ.get("BELIEF_SOIL_PATH")
    prev_mode = os.environ.get("BELIEF_MODEL_MODE")
    prev_local = os.environ.get("BELIEF_LOCAL_MODEL")
    os.environ["BELIEF_SOIL_PATH"] = str(soil_dir)
    os.environ["BELIEF_MODEL_MODE"] = "local"
    os.environ["BELIEF_LOCAL_MODEL"] = model
    patch = ""
    try:
        with tempfile.TemporaryDirectory(prefix=f"swebench-{instance_id}-") as tmp:
            repo_dir = Path(tmp) / "repo"
            url = f"https://github.com/{repo}.git"
            subprocess.run(["git", "clone", "--quiet", url, str(repo_dir)], check=True, timeout=900)
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", "--quiet", base_commit],
                check=True,
                timeout=120,
            )
            from belief.agents.brownfield_agent import fix_issue

            asyncio.run(fix_issue(repo_dir, problem))
            diff = subprocess.run(
                ["git", "-C", str(repo_dir), "diff"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            patch = diff.stdout or ""
    except Exception as e:  # pragma: no cover - real-path failure, logged not raised
        logger.warning("prediction failed for %s: %s", instance_id, e)
        patch = ""
    finally:
        _restore_env("BELIEF_SOIL_PATH", prev_soil)
        _restore_env("BELIEF_MODEL_MODE", prev_mode)
        _restore_env("BELIEF_LOCAL_MODEL", prev_local)

    return {
        "instance_id": instance_id,
        "model_name_or_path": PREDICTION_MODEL_LABEL,
        "model_patch": patch,
    }


def _restore_env(key: str, prev: Optional[str]) -> None:
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev


def evaluate_predictions(
    predictions: list[dict],
    instance_ids: tuple[str, ...],
    *,
    run_id: str,
    dataset_name: str = SWEBENCH_DATASET,
    work_dir: Optional[Path] = None,
) -> set[str]:
    """Run the official SWE-bench harness (Docker) and return resolved instance ids.

    Writes predictions to JSONL, invokes ``python -m swebench.harness.run_evaluation``,
    and parses ``resolved_ids`` from the emitted report JSON. Empty patches are
    skipped (counted as unresolved) so the harness isn't asked to evaluate no-ops.
    """
    work_dir = Path(work_dir or tempfile.mkdtemp(prefix=f"swebench-eval-{run_id}-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    preds_path = work_dir / "predictions.jsonl"
    nonempty = [p for p in predictions if p.get("model_patch", "").strip()]
    preds_path.write_text("\n".join(json.dumps(p) for p in nonempty))
    if not nonempty:
        return set()  # nothing applied -> nothing resolved; skip Docker entirely

    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(preds_path),
        "--run_id",
        run_id,
        "--instance_ids",
        *instance_ids,
        "--max_workers",
        "1",
        "--cache_level",
        "env",
    ]
    subprocess.run(cmd, cwd=str(work_dir), check=False, timeout=7200)

    # The harness writes <model>.<run_id>.json with a "resolved_ids" list.
    report = work_dir / f"{PREDICTION_MODEL_LABEL}.{run_id}.json"
    if not report.exists():
        # Fall back to a recursive search; harness output location varies by version.
        candidates = list(work_dir.rglob(f"*{run_id}*.json"))
        report = candidates[0] if candidates else report
    if not report.exists():
        logger.warning("no SWE-bench report found under %s", work_dir)
        return set()
    data = json.loads(report.read_text())
    return set(data.get("resolved_ids", []))


def run_instances(
    instance_ids: tuple[str, ...],
    soil_dir: Path,
    *,
    model: str = "qwen2.5-coder:14b",
    run_id: Optional[str] = None,
    loader: Callable[..., list[dict]] = load_verified_instances,
    predictor: Callable[..., dict] = generate_prediction,
    evaluator: Callable[..., set[str]] = evaluate_predictions,
) -> int:
    """Attempt SWE-bench Verified ``instance_ids`` against ``soil_dir``; return #resolved.

    Composes the three seams: load full instances → generate a brownfield-fix
    prediction per instance (soil-informed) → evaluate via the official harness.
    The three stages are injectable so this orchestration is unit-tested without
    Docker; the defaults are the real Mac path.
    """
    rid = run_id or f"probe-{int(time.time())}"
    instances = loader(instance_ids)
    predictions = [predictor(inst, soil_dir, model=model) for inst in instances]
    resolved = evaluator(predictions, instance_ids, run_id=rid)
    return len(resolved)


def smoke_one(instance_id: str, soil_dir: Path, *, model: str = "qwen2.5-coder:14b") -> dict:
    """One-instance feasibility smoke: resolve a single instance + time it.

    The decision gate before committing the full run to the probe — returns
    ``{instance_id, resolved, wall_clock_s}`` so a 14B's actual resolve outcome
    and per-instance cost are visible before wiring the checkpoint harness.
    """
    t0 = time.time()
    n_resolved = run_instances((instance_id,), Path(soil_dir), model=model)
    return {
        "instance_id": instance_id,
        "resolved": bool(n_resolved),
        "wall_clock_s": round(time.time() - t0, 1),
    }


def make_probe_fn(
    config: "StarvedConfig",
    instance_ids: Optional[tuple[str, ...]] = None,
    runner: Optional[Callable[[tuple[str, ...], Path], int]] = None,
) -> Callable[[int, str, Path], "ProbeResult"]:
    """Build the ``(gen, arm, soil_dir) -> ProbeResult`` probe callable.

    ``runner`` is injectable so tests can supply a deterministic resolver; when
    omitted the real :func:`run_instances` path is used, bound to the config's
    local model and a stable per-(gen, arm) run_id. Refuses to build a probe with
    an empty instance set, so ``--probe-at`` cannot silently run a zero-instance
    probe.
    """
    ids = instance_ids if instance_ids is not None else SWEBENCH_VERIFIED_PROBE_INSTANCES
    if not ids:
        raise ValueError(
            "No SWE-bench Verified probe instances pinned; set "
            "SWEBENCH_VERIFIED_PROBE_INSTANCES or pass instance_ids before using --probe-at."
        )

    def _probe(gen: int, arm: str, soil_dir: Path) -> "ProbeResult":
        from belief.experiments.starved_runner import ProbeResult

        if runner is not None:
            n_resolved = runner(ids, soil_dir)
        else:
            n_resolved = run_instances(
                ids,
                soil_dir,
                model=config.local_model,
                run_id=f"{config.experiment_id}-g{gen}-{arm}",
            )
        return ProbeResult(gen=gen, arm=arm, n_instances=len(ids), n_resolved=n_resolved)

    return _probe
