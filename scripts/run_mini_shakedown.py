"""Mini-shakedown: validate the novel-artifact validator integration end-to-end.

Companion to scripts/run_shakedown.py. Runs ONE novel-artifact challenge
(novel-regex — cheapest, no z3/tla2tools dependencies) through all three
conditions to confirm task #10's wiring works on a real engine build.

If this passes (i.e. no runtime errors, results land in DB with
weighted_score = 1.0 or 0.0 per the validator override), task #6 (the
full 140-build run) is unblocked.

Scope:
  - 1 novel-artifact challenge (novel-regex)
  - 3 conditions (raw_local, soil_only, full)
  - 1 build-seq point (build_seq=1, empty baseline)
  - Total: 3 cells. ~30-45 minutes.

Why novel-regex specifically: zero external-tool dependencies (no z3, no
tla2tools.jar, no wordlist fixture). If the integration works on this
challenge, it will work on the others modulo their external-tool checks
which are already tested in tests/test_novel_artifact_validators.py.

Usage:
    python3 scripts/run_mini_shakedown.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from belief.experiments.novel_artifact_challenges import NOVEL_ARTIFACT_SPECS


def main() -> int:
    config_path = Path.home() / ".belief-engine" / "substrate_baselines.json"
    if not config_path.exists():
        print(
            f"ERROR: substrate_baselines.json not found at {config_path}",
            file=sys.stderr,
        )
        return 1

    with config_path.open() as fh:
        config = json.load(fh)

    baseline_snapshots: dict[tuple[str, int], Path] = {
        ("soil_only", 1): Path(config["soil_only_b1"]),
        ("full", 1): Path(config["full_b1"]),
    }
    for key, path in baseline_snapshots.items():
        if not path.exists():
            print(
                f"ERROR: baseline snapshot for {key} does not exist: {path}",
                file=sys.stderr,
            )
            return 1

    # Pull the regex challenge spec from the novel-artifact registry so
    # the goal text stays in sync with task #10's definitions.
    spec = NOVEL_ARTIFACT_SPECS["novel-regex"]
    challenges = [{"id": spec.challenge_id, "goal": spec.goal}]

    from belief.experiments.ab_runner import run_substrate_transfer_experiment

    print("=" * 60)
    print("  Substrate-transfer MINI-SHAKEDOWN (novel-artifact integration)")
    print(f"  Challenge:  {spec.challenge_id}")
    print("  Conditions: raw_local, soil_only, full")
    print("  Build seq:  (1,)  — empty baseline only")
    print("  Total cells: 3")
    print("=" * 60)

    exp_id = asyncio.run(
        run_substrate_transfer_experiment(
            challenges=challenges,
            baseline_snapshots=baseline_snapshots,
            conditions=["raw_local", "soil_only", "full"],
            build_seq_points=(1,),
        )
    )

    print(f"\nMini-shakedown experiment id: {exp_id}")
    print("\nQuery results with:")
    print(
        f"  sqlite3 ~/.belief-engine/experiments.db "
        f'"SELECT condition, challenge_id, passed, weighted_score, '
        f"time_seconds, error FROM results WHERE experiment_id = '{exp_id}';\""
    )
    print("\nExpected: 3 rows total. weighted_score should be 1.0 or 0.0")
    print("(novel-artifact validators report binary pass/fail, not partial credit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
