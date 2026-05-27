# Substrate-Transfer Baseline Snapshot Prep

**Status:** Procedure for task #9 of the substrate-transfer experiment. Run on your Mac with Ollama running. Expected wall-clock: 5-6 hours.

**What this does:** Generates the 5 baseline soil snapshots the runner needs:

| Snapshot label | Used at | Soil state |
|---|---|---|
| `substrate-baseline-empty` | (soil_only, b1) AND (full, b1) | empty |
| `substrate-baseline-soil_only-b5` | (soil_only, b5) | 4 builds, no covenants stored |
| `substrate-baseline-soil_only-b15` | (soil_only, b15) | 14 builds, no covenants stored |
| `substrate-baseline-full-b5` | (full, b5) | 4 builds, covenants enforced |
| `substrate-baseline-full-b15` | (full, b15) | 14 builds, covenants enforced |

**Safety contract:** Step 1 takes a snapshot of your live working soil. The procedure ends with restoring it. If anything goes wrong mid-procedure, `belief snapshot restore /Users/joefiorillo/.belief-engine/snapshots/<live-pre-baseline>` brings you back to safety. **Do not skip step 1.**

---

## Pre-flight checks

```bash
which belief && belief --help | head -5
ollama list | grep qwen2.5-coder:14b
df -h ~ | head -2
```

Confirm: `belief` command works, model is pulled, you have at least 5 GB free in `~`. If any of these are missing, fix before continuing.

---

## Step 1 — Safety snapshot (DO NOT SKIP)

```bash
belief snapshot take --label "live-pre-substrate-baseline-prep"
```

Verify it landed:

```bash
belief snapshot list | head -3
```

You should see a snapshot path containing `live-pre-substrate-baseline-prep`. **Copy that path** — you'll need it for step 7.

```bash
# Save the safety snapshot path for later restore. Example:
SAFETY_SNAPSHOT="$HOME/.belief-engine/snapshots/2026-05-27T_live-pre-substrate-baseline-prep"
# Replace the path above with the actual one from `belief snapshot list`.
echo "Safety snapshot: $SAFETY_SNAPSHOT"
ls -la "$SAFETY_SNAPSHOT"
```

The `ls` should show the snapshot directory exists with a `manifest.json` inside.

---

## Step 2 — Wipe live state to create the empty baseline

This deletes your active soil. The safety snapshot from step 1 is the only restore point. **Re-verify the safety snapshot exists before running this:**

```bash
ls -la "$SAFETY_SNAPSHOT/manifest.json" || echo "STOP — safety snapshot missing"
```

If that says "STOP", do not continue. Re-run step 1.

Wipe:

```bash
rm -rf ~/.belief-engine/soil
rm -f ~/.belief-engine/builds.db ~/.belief-engine/builds.db-journal
rm -f ~/.belief-engine/niches.db ~/.belief-engine/niches.db-journal
rm -f ~/.belief-engine/reciprocity.db ~/.belief-engine/reciprocity.db-journal
rm -f ~/.belief-engine/routing.db ~/.belief-engine/routing.db-journal
```

Take the empty-state snapshot:

```bash
belief snapshot take --label "substrate-baseline-empty"
```

Record the path:

```bash
EMPTY_SNAPSHOT=$(belief snapshot list | grep substrate-baseline-empty | head -1 | awk '{print $1}')
echo "Empty: $EMPTY_SNAPSHOT"
```

---

## Step 3 — Build soil_only sequence: empty → build_seq=5

Run 4 builds under `soil_only` condition. Each build takes ~10-15 minutes. Total: ~45-60 minutes.

```bash
BELIEF_EXPERIMENT_CONDITION=soil_only \
  python3 scripts/baseline_build_sequence.py --count 4 --offset 0
```

Tail the log in another terminal if you want to watch progress:

```bash
tail -f ~/.belief-engine/baseline_prep.log
```

When step 3 completes, snapshot:

```bash
belief snapshot take --label "substrate-baseline-soil_only-b5"
SOIL_ONLY_B5=$(belief snapshot list | grep substrate-baseline-soil_only-b5 | head -1 | awk '{print $1}')
echo "soil_only b5: $SOIL_ONLY_B5"
```

---

## Step 4 — Build soil_only sequence: build_seq=5 → build_seq=15

10 more builds, ~2 hours.

```bash
BELIEF_EXPERIMENT_CONDITION=soil_only \
  python3 scripts/baseline_build_sequence.py --count 10 --offset 4
```

Snapshot when done:

```bash
belief snapshot take --label "substrate-baseline-soil_only-b15"
SOIL_ONLY_B15=$(belief snapshot list | grep substrate-baseline-soil_only-b15 | head -1 | awk '{print $1}')
echo "soil_only b15: $SOIL_ONLY_B15"
```

---

## Step 5 — Restore empty state, then build full sequence: empty → build_seq=5

Restore to empty so the `full`-condition baseline doesn't have any `soil_only` contamination:

```bash
belief snapshot restore "$EMPTY_SNAPSHOT"
```

Verify:

```bash
ls ~/.belief-engine/soil 2>&1 | head -3
# If soil dir doesn't exist or is empty, you're at empty state. Good.
```

Now run 4 builds under `full` condition (~45-60 minutes):

```bash
BELIEF_EXPERIMENT_CONDITION=full \
  python3 scripts/baseline_build_sequence.py --count 4 --offset 0
```

Snapshot:

```bash
belief snapshot take --label "substrate-baseline-full-b5"
FULL_B5=$(belief snapshot list | grep substrate-baseline-full-b5 | head -1 | awk '{print $1}')
echo "full b5: $FULL_B5"
```

---

## Step 6 — Build full sequence: build_seq=5 → build_seq=15

10 more builds, ~2 hours.

```bash
BELIEF_EXPERIMENT_CONDITION=full \
  python3 scripts/baseline_build_sequence.py --count 10 --offset 4
```

Snapshot:

```bash
belief snapshot take --label "substrate-baseline-full-b15"
FULL_B15=$(belief snapshot list | grep substrate-baseline-full-b15 | head -1 | awk '{print $1}')
echo "full b15: $FULL_B15"
```

---

## Step 7 — Verify all 5 baselines + restore live soil

Verify each snapshot is structurally sound (TLC-style verification by the snapshot module):

```bash
for snap in "$EMPTY_SNAPSHOT" "$SOIL_ONLY_B5" "$SOIL_ONLY_B15" "$FULL_B5" "$FULL_B15"; do
  echo "=== $snap ==="
  belief snapshot verify "$snap" || echo "VERIFY FAILED for $snap"
done
```

All 5 should print "verified" or equivalent success. If any fail, do not restore live soil yet — keep the failed snapshot for inspection and re-run only the affected sequence.

If all verified, restore your live working soil:

```bash
belief snapshot restore "$SAFETY_SNAPSHOT"
```

Confirm the live soil is back:

```bash
ls ~/.belief-engine/soil 2>&1 | head -3
# Should show soil files
```

---

## Step 8 — Save baseline paths for the experiment runner

Drop the 5 paths into a config file the runner can read. From the same terminal session (variables still set):

```bash
mkdir -p ~/.belief-engine && cat > ~/.belief-engine/substrate_baselines.json <<EOF
{
  "soil_only_b1": "$EMPTY_SNAPSHOT",
  "soil_only_b5": "$SOIL_ONLY_B5",
  "soil_only_b15": "$SOIL_ONLY_B15",
  "full_b1": "$EMPTY_SNAPSHOT",
  "full_b5": "$FULL_B5",
  "full_b15": "$FULL_B15"
}
EOF

cat ~/.belief-engine/substrate_baselines.json
```

This file is what task #5 (shakedown) and task #6 (full run) will read. It lives outside the repo (gitignored by default per the `.belief-engine/` rule in `.gitignore`).

---

## Step 9 — Commit the script + procedure doc

The procedure doc and the helper script should go into git so future sessions can re-run baseline prep:

```bash
git add scripts/baseline_build_sequence.py docs/experiments/baseline_prep_procedure.md
git commit -m "experiments: baseline-snapshot prep script + procedure doc"
```

---

## Recovery from mid-procedure failures

If a single `belief build` fails inside one of the sequences, the script logs the failure and moves on. You can re-run just that one challenge by adjusting `--offset` and `--count`, or accept that one missing build in the baseline. A baseline with 13 builds instead of 14 is still meaningful; you just describe build_seq as "approximate" in the writeup.

If a snapshot take fails (rare — usually means disk full), check `df -h`, free space, re-run the `belief snapshot take` for that specific label.

If you need to abort mid-procedure and come back later, just restore the safety snapshot now:

```bash
belief snapshot restore "$SAFETY_SNAPSHOT"
```

Any partial baseline snapshots you already took stay on disk in `~/.belief-engine/snapshots/`. You can resume by re-restoring the appropriate state and re-running just the missing sequences. Track what you've done in the log file `~/.belief-engine/baseline_prep.log`.

---

## What's next after this

Once `~/.belief-engine/substrate_baselines.json` exists with valid paths, task #5 (shakedown) is unblocked. The shakedown will read those paths and run the substrate-transfer experiment on a 2-challenge subset to smoke out integration bugs before the full 140-build run.
