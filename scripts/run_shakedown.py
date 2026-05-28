"""Shakedown driver for the substrate-transfer experiment.

Runs run_substrate_transfer_experiment on a tiny subset to smoke out
integration bugs (snapshot restore between cells, BELIEF_EXPERIMENT_CONDITION
env-var propagation, runner dispatch) before committing to the full 140-build
production run.

Scope:
  - 2 in-distribution challenges (t1-wordcount, t2-health-api) — fastest in
    their respective domains; chosen so a single bad build doesn't waste hours
  - 3 conditions (raw_local, soil_only, full)
  - 1 build-seq point (build_seq=1, the empty baseline)
  - Total: 2 × 3 × 1 = 6 builds. ~1.5 hours including snapshot-restore overhead.

This deliberately does NOT exercise the novel-artifact validators — those
need task #10 (validator integration into the runner) to land first. The
shakedown's purpose is to verify the runner mechanics, which are
condition-independent of which validator runs.

Usage:
    python3 scripts/run_shakedown.py

Reads ~/.belief-engine/substrate_baselines.json for snapshot paths.
Results stored in ~/.belief-engine/experiments.db via the runner.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


SHAKEDOWN_CHALLENGES: list[dict] = [
    {
        "id": "t1-wordcount",
        "goal": (
            "Build a Python script that reads a text file from stdin and "
            "prints word count, line count, and character count."
        ),
    },
    {
        "id": "t2-health-api",
        "goal": (
            "Build a FastAPI server with GET /health returning "
            "{status: ok}, GET /time returning current UTC time, and "
            "GET /echo?msg=hello returning the message."
        ),
    },
]


def main() -> int:
    config_path = Path.home() / ".belief-engine" / "substrate_baselines.json"
    if not config_path.exists():
        print(
            f"ERROR: substrate_baselines.json not found at {config_path}",
            file=sys.stderr,
        )
        print(
            "Run task #9 (baseline prep) first to generate this file.",
            file=sys.stderr,
        )
        return 1

    with config_path.open() as fh:
        config = json.load(fh)

    # The shakedown only uses build_seq=1, which is the empty baseline.
    # Both soil_only and full share the empty snapshot as their b1.
    baseline_snapshots: dict[tuple[str, int], Path] = {
        ("soil_only", 1): Path(config["soil_only_b1"]),
        ("full", 1): Path(config["full_b1"]),
    }

    # Verify the snapshot paths actually exist before we start
    for key, path in baseline_snapshots.items():
        if not path.exists():
            print(
                f"ERROR: baseline snapshot for {key} does not exist: {path}",
                file=sys.stderr,
            )
            return 1

    # Defer the heavy import until after we've validated config —
    # importing ab_runner pulls in chromadb and the whole belief stack.
    from belief.experiments.ab_runner import run_substrate_transfer_experiment

    print("=" * 60)
    print("  Substrate-transfer SHAKEDOWN")
    print(f"  Challenges: {[c['id'] for c in SHAKEDOWN_CHALLENGES]}")
    print("  Conditions: raw_local, soil_only, full")
    print("  Build seq:  (1,)  — empty baseline only")
    print(f"  Total cells: {len(SHAKEDOWN_CHALLENGES) * 3 * 1}")
    print("=" * 60)

    exp_id = asyncio.run(
        run_substrate_transfer_experiment(
            challenges=SHAKEDOWN_CHALLENGES,
            baseline_snapshots=baseline_snapshots,
            conditions=["raw_local", "soil_only", "full"],
            build_seq_points=(1,),
        )
    )

    print(f"\nShakedown experiment id: {exp_id}")
    print("\nQuery results with:")
    print(
        f"  sqlite3 ~/.belief-engine/experiments.db "
        f'"SELECT condition, challenge_id, passed, weighted_score, '
        f"time_seconds, error FROM results WHERE experiment_id = '{exp_id}';\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
