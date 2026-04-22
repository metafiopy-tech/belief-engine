#!/usr/bin/env bash
# Speed benchmark for the local-mode pipeline.
#
# Runs three FizzBuzz builds back-to-back and reports each wall-clock
# time plus the average.  Session 1 target: <180s average.
# Session 2 target: ideally <120s.
#
# Usage:
#   scripts/speed_benchmark.sh                    # 3 builds, FizzBuzz goal
#   scripts/speed_benchmark.sh 5                  # 5 builds
#   scripts/speed_benchmark.sh 3 "build a CLI that reverses a string"
#
# Notes:
#   - Assumes Ollama is already running with the local model loaded.
#   - Suppresses the engine's stdout; only the timing summary is shown.
#   - Uses BELIEF_MODEL_MODE=local for all builds.
#   - Ollama keep_alive (30m) means the first build may be slower
#     than the rest if the model needs to be warmed up.

set -euo pipefail

N="${1:-3}"
GOAL="${2:-Build a Python FizzBuzz script}"

if ! command -v belief >/dev/null 2>&1; then
  echo "ERROR: 'belief' CLI not on PATH. Run 'pip install -e .' from the repo." >&2
  exit 1
fi

echo "Speed benchmark: $N builds on local model"
echo "Goal: $GOAL"
echo "----------------------------------------"

total=0
times=()

for i in $(seq 1 "$N"); do
  START=$(date +%s)
  BELIEF_MODEL_MODE=local belief --goal "$GOAL" >/dev/null 2>&1 || {
    echo "Build $i: FAILED"
    continue
  }
  END=$(date +%s)
  elapsed=$((END - START))
  times+=("$elapsed")
  total=$((total + elapsed))
  echo "Build $i: ${elapsed}s"
done

if [ "${#times[@]}" -eq 0 ]; then
  echo "All builds failed — nothing to average."
  exit 1
fi

avg=$((total / ${#times[@]}))
echo "----------------------------------------"
echo "Average: ${avg}s  (over ${#times[@]} successful builds)"

# Flag whether we hit the targets
if [ "$avg" -lt 120 ]; then
  echo "✓ Under 120s — Session 2 stretch goal hit"
elif [ "$avg" -lt 180 ]; then
  echo "✓ Under 180s — Session 1 target hit"
else
  echo "✗ Over 180s — further optimization needed"
fi
