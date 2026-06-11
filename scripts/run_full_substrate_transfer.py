"""Full 140-build substrate-transfer experiment driver.

This is task #6: the production run. Exercises the full experimental
design from docs/experiments/substrate_transfer_challenges.md (the
locked thesis) and docs/experiments/in_distribution_inventory.md (the
15 in-distribution challenges).

Matrix:
  - 20 challenges = 15 in-distribution + 5 novel-artifact
  - 3 conditions  = raw_local, soil_only, full
  - 3 build_seq   = 1, 5, 15  (raw_local skips this axis)
  - Total cells   = 20 + (15 * 2 * 3) = 20 + 90 = 140... wait let me recount.
                    = 20 raw_local
                    + 20 * 3 = 60 soil_only
                    + 20 * 3 = 60 full
                    = 140 cells

Expected wall-clock: 30-35 hours.
Expected cost: ~$0.60 (safety overseer Haiku, ~$0.005/cell on engine builds).

Pre-requisites (must all be satisfied or the script aborts):
  1. ~/.belief-engine/substrate_baselines.json exists with all 5 paths
  2. All 5 baseline snapshots exist on disk
  3. Ollama daemon running with qwen2.5-coder:14b pulled
  4. (Optional) z3-solver installed (for novel-smtlib validator)
  5. (Optional) tla2tools.jar at ~/lib/ (for novel-tlaplus validator)

Safety contract:
  - Takes a fresh safety snapshot of the live working soil before any
    builds run.
  - Auto-restores the safety snapshot in a finally block, even if the
    experiment is interrupted or crashes.

Usage (FOREGROUND, with caffeinate -di in a second terminal — see ops notes
in project_substrate_transfer_infra_landed.md):

    python3 scripts/run_full_substrate_transfer.py

Results stored in ~/.belief-engine/experiments.db; query via:

    sqlite3 ~/.belief-engine/experiments.db "SELECT ... FROM results WHERE experiment_id = '<exp_id>';"
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


# The 15 in-distribution challenge ids selected in
# docs/experiments/in_distribution_inventory.md. Order: 5 microservices,
# 5 CLI, 5 data-pipeline. Their goal text is pulled from
# belief.benchmark.CHALLENGES at runtime so the canonical definition lives
# in one place.
IN_DISTRIBUTION_IDS: list[str] = [
    # Microservices
    "t2-health-api",
    "t3-url-shortener",
    "t3-bookmark-api",
    "t3-notes-api",
    "t3-contact-api",
    # CLI / scripts
    "t1-wordcount",
    "t2-todo-cli",
    "t2-calculator-cli",
    "t2-csv-stats",
    "t3-expense-tracker",
    # Data pipelines (weakest bucket — see inventory doc)
    "t3-data-pipeline",
    "t6-data-pipeline",
    "t5-workflow-engine",
    "t3-task-queue",
    "t3-schema-validator",
]


def main() -> int:
    # ---- 1. Validate prerequisites ----
    config_path = Path.home() / ".belief-engine" / "substrate_baselines.json"
    if not config_path.exists():
        print(
            f"ERROR: substrate_baselines.json not found at {config_path}",
            file=sys.stderr,
        )
        print(
            "Run task #9 (baseline prep, scripts/run_baseline_prep.sh) first.",
            file=sys.stderr,
        )
        return 1

    with config_path.open() as fh:
        config = json.load(fh)

    baseline_snapshots: dict[tuple[str, int], Path] = {
        ("soil_only", 1): Path(config["soil_only_b1"]),
        ("soil_only", 5): Path(config["soil_only_b5"]),
        ("soil_only", 15): Path(config["soil_only_b15"]),
        ("full", 1): Path(config["full_b1"]),
        ("full", 5): Path(config["full_b5"]),
        ("full", 15): Path(config["full_b15"]),
    }
    for key, path in baseline_snapshots.items():
        if not path.exists():
            print(
                f"ERROR: baseline snapshot for {key} not found on disk: {path}",
                file=sys.stderr,
            )
            return 1

    # ---- 2. Assemble the 20-challenge list ----
    from belief.benchmark import CHALLENGES
    from belief.experiments.novel_artifact_challenges import NOVEL_ARTIFACT_SPECS

    by_id = {c.id: c for c in CHALLENGES}
    missing = [cid for cid in IN_DISTRIBUTION_IDS if cid not in by_id]
    if missing:
        print(
            f"ERROR: in-distribution challenge ids not found in benchmark.CHALLENGES: {missing}",
            file=sys.stderr,
        )
        return 1

    challenges: list[dict] = []
    for cid in IN_DISTRIBUTION_IDS:
        challenges.append({"id": cid, "goal": by_id[cid].goal})
    for cid, spec in NOVEL_ARTIFACT_SPECS.items():
        challenges.append({"id": spec.challenge_id, "goal": spec.goal})

    assert len(challenges) == 20, f"expected 20 challenges, got {len(challenges)}"

    # ---- 3. Take a safety snapshot of live working soil ----
    from belief.memory.snapshot import SoilSnapshot

    snap = SoilSnapshot()
    print("Taking pre-experiment safety snapshot of live working soil...")
    safety_path = snap.take_snapshot(label="pre-full-substrate-transfer-experiment")
    print(f"Safety snapshot: {safety_path}")

    # ---- 4. Run the experiment with auto-restore in finally ----
    from belief.experiments.ab_runner import run_substrate_transfer_experiment

    print("=" * 60)
    print("  Substrate-transfer FULL EXPERIMENT (task #6)")
    print("  Challenges: 20 (15 in-distribution + 5 novel-artifact)")
    print("  Conditions: raw_local, soil_only, full")
    print("  Build seq:  1, 5, 15")
    print("  Total cells: 140")
    print("  Estimated wall-clock: 30-35 hours")
    print(f"  Safety snapshot for restore: {safety_path}")
    print("=" * 60)

    exp_id: str = "unknown"
    try:
        exp_id = asyncio.run(
            run_substrate_transfer_experiment(
                challenges=challenges,
                baseline_snapshots=baseline_snapshots,
                conditions=["raw_local", "soil_only", "full"],
                build_seq_points=(1, 5, 15),
            )
        )
    except KeyboardInterrupt:
        print("\n\nINTERRUPTED — restoring safety snapshot and exiting.", file=sys.stderr)
    except Exception as exc:
        print(f"\n\nCRASHED: {exc}\nRestoring safety snapshot.", file=sys.stderr)
        raise
    finally:
        print("\nRestoring live working soil from safety snapshot...")
        try:
            snap.restore_snapshot(safety_path)
            print("Live soil restored.")
        except Exception as exc:
            print(
                f"WARNING: safety restore failed: {exc}\n"
                f"Manual recovery: belief snapshot restore {safety_path}",
                file=sys.stderr,
            )

    print(f"\nFull experiment id: {exp_id}")
    print("\nQuery summary stats:")
    print(
        f"  sqlite3 ~/.belief-engine/experiments.db "
        f'"SELECT condition, build_seq, COUNT(*) as cells, '
        f"SUM(passed) as passes, AVG(weighted_score) as avg_score, "
        f"SUM(cost_usd) as total_cost FROM results WHERE experiment_id = "
        f"'{exp_id}' GROUP BY condition, build_seq ORDER BY condition, build_seq;\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
