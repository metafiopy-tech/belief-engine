#!/usr/bin/env bash
# scripts/run_bigcode_subset.sh
#
# Run a 50-problem HumanEval subset under both conditions:
#   A — raw qwen2.5-coder:14b directly via Ollama
#   B — Belief Engine via the OpenAI-compat shim (scripts/bigcode_shim.py)
#
# Outputs go to results/, named with a date stamp. Idempotent: existing
# result files are skipped, so re-running picks up where it left off.
#
# Prerequisites (run once on Joe's Mac):
#   1. ollama serve (and `ollama pull qwen2.5-coder:14b`)
#   2. pip3 install -e ".[bench]"
#   3. python3 -m uvicorn scripts.bigcode_shim:app --port 8000 \
#        (in a separate terminal, leave running)
#   4. cd ~/bigcode-eval && pip3 install -e .
#        (clone https://github.com/bigcode-project/bigcode-evaluation-harness
#        first if you haven't)
#
# Usage:
#   bash scripts/run_bigcode_subset.sh
#
# Tunables (override via env):
#   N_PROBLEMS       — how many HumanEval problems to run (default 50)
#   N_SAMPLES        — generations per problem (default 1, see notes)
#   TEMPERATURE      — sampling temperature (default 0.0)
#   SEED             — RNG seed (default 42)
#   HARNESS_DIR      — path to the bigcode-evaluation-harness clone
#                      (default $HOME/bigcode-eval)
#   SHIM_URL         — shim base URL (default http://127.0.0.1:8000/v1)
#   OLLAMA_URL       — ollama base URL (default http://127.0.0.1:11434)
#   MODEL_BASELINE   — baseline model name (default qwen2.5-coder:14b)
#
# Notes:
#   * The engine is deterministic at temperature=0 + seed=42, so
#     N_SAMPLES=1 is sufficient for pass@1. Bumping it produces
#     identical outputs (wasted wall-clock).
#   * BigCode harness flag names vary by version. If `--limit` doesn't
#     work in your install, try `--max_n_samples` or hand-edit the task
#     file. Check `python3 -m bigcode_eval.main --help` after install.
#   * Wall-clock estimate at defaults: raw ~30 min, engine ~2.5 hours.

set -euo pipefail

N_PROBLEMS="${N_PROBLEMS:-50}"
N_SAMPLES="${N_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
HARNESS_DIR="${HARNESS_DIR:-$HOME/bigcode-eval}"
SHIM_URL="${SHIM_URL:-http://127.0.0.1:8000/v1}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL_BASELINE="${MODEL_BASELINE:-qwen2.5-coder:14b}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
DATESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$RESULTS_DIR"

cd "$HARNESS_DIR"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "==> pre-flight"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not on PATH" >&2
    exit 1
fi

if ! curl -fs "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "ERROR: ollama not reachable at $OLLAMA_URL — start it with 'ollama serve'" >&2
    exit 1
fi

if ! curl -fs "$SHIM_URL/models" >/dev/null 2>&1; then
    echo "ERROR: shim not reachable at $SHIM_URL — start with:" >&2
    echo "       python3 -m uvicorn scripts.bigcode_shim:app --port 8000" >&2
    exit 1
fi

if [ ! -d "$HARNESS_DIR" ]; then
    echo "ERROR: BigCode harness not found at $HARNESS_DIR" >&2
    echo "       git clone https://github.com/bigcode-project/bigcode-evaluation-harness $HARNESS_DIR" >&2
    exit 1
fi

echo "    ollama up at $OLLAMA_URL"
echo "    shim up at $SHIM_URL"
echo "    harness at $HARNESS_DIR"
echo "    n_problems=$N_PROBLEMS  n_samples=$N_SAMPLES  temp=$TEMPERATURE  seed=$SEED"
echo

# ---------------------------------------------------------------------------
# Condition A — raw qwen baseline
# ---------------------------------------------------------------------------

RAW_OUT="$RESULTS_DIR/raw_humaneval_subset_${N_PROBLEMS}.json"
RAW_METRICS="$RESULTS_DIR/raw_humaneval_subset_${N_PROBLEMS}_metrics.json"

if [ -f "$RAW_METRICS" ]; then
    echo "==> condition A (raw qwen): SKIP — $RAW_METRICS exists"
else
    echo "==> condition A (raw qwen, $MODEL_BASELINE) starting at $(date)"
    python3 -m bigcode_eval.main \
        --model openai_chat \
        --model_args "model=$MODEL_BASELINE,base_url=$OLLAMA_URL/v1" \
        --tasks humaneval \
        --limit "$N_PROBLEMS" \
        --n_samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --seed "$SEED" \
        --save_generations \
        --save_generations_path "$RAW_OUT" \
        --metric_output_path "$RAW_METRICS" \
        --allow_code_execution \
        2>&1 | tee "$RESULTS_DIR/raw_run_${DATESTAMP}.log"
    echo "    raw done at $(date)"
fi

# ---------------------------------------------------------------------------
# Condition B — engine via shim
# ---------------------------------------------------------------------------

ENG_OUT="$RESULTS_DIR/engine_humaneval_subset_${N_PROBLEMS}.json"
ENG_METRICS="$RESULTS_DIR/engine_humaneval_subset_${N_PROBLEMS}_metrics.json"

if [ -f "$ENG_METRICS" ]; then
    echo "==> condition B (engine via shim): SKIP — $ENG_METRICS exists"
else
    echo "==> condition B (engine via shim) starting at $(date)"
    echo "    each problem ~3 min — expect ~$((N_PROBLEMS * 3)) min wall-clock"
    python3 -m bigcode_eval.main \
        --model openai_chat \
        --model_args "model=belief-engine-local,base_url=$SHIM_URL" \
        --tasks humaneval \
        --limit "$N_PROBLEMS" \
        --n_samples "$N_SAMPLES" \
        --temperature 0.0 \
        --seed "$SEED" \
        --save_generations \
        --save_generations_path "$ENG_OUT" \
        --metric_output_path "$ENG_METRICS" \
        --allow_code_execution \
        2>&1 | tee "$RESULTS_DIR/engine_run_${DATESTAMP}.log"
    echo "    engine done at $(date)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo
echo "==> done."
echo
echo "    raw metrics:    $RAW_METRICS"
echo "    engine metrics: $ENG_METRICS"
echo
echo "    Compare pass@1:"
echo "      python3 -c \"import json; print('raw:   ', json.load(open('$RAW_METRICS'))['humaneval']['pass@1'])\""
echo "      python3 -c \"import json; print('engine:', json.load(open('$ENG_METRICS'))['humaneval']['pass@1'])\""
