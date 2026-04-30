#!/usr/bin/env bash
# scripts/run_bigcode_subset.sh
#
# Run a 50-problem HumanEval subset under both conditions:
#   A — raw qwen2.5-coder:14b directly via Ollama
#   B — Belief Engine via subprocess
#
# Architecture: we own GENERATION (scripts/generate_humaneval_completions.py)
# and the BigCode harness only does SCORING via --load_generations_path.
# Earlier versions of this script used --model openai_chat against the
# harness, but that flag doesn't exist in this harness — it loads HF
# transformers models directly with no HTTP/API driver. The shim
# (scripts/bigcode_shim.py) survives as future infra but is not on
# this critical path.
#
# Outputs go to results/. Idempotent: existing result files are
# skipped; the generator is also resumable per-problem via .partial
# JSONL files.
#
# Prerequisites (run once on Joe's Mac):
#   1. ollama serve (and `ollama pull qwen2.5-coder:14b`)
#   2. pip3 install -e . --break-system-packages   (belief-engine)
#   3. ~/bigcode-eval cloned and `pip3 install -e . --break-system-packages`
#
# Usage:
#   bash scripts/run_bigcode_subset.sh
#
# Tunables (override via env):
#   N_PROBLEMS       — how many HumanEval problems to run (default 50)
#   N_SAMPLES        — generations per problem (default 1, see notes)
#   SEED             — RNG seed (default 42)
#   HARNESS_DIR      — path to the bigcode-evaluation-harness clone
#                      (default $HOME/bigcode-eval)
#   OLLAMA_URL       — ollama base URL (default http://127.0.0.1:11434)
#   MODEL_BASELINE   — baseline model name (default qwen2.5-coder:14b)
#   ENGINE_TIMEOUT_S — per-problem engine timeout (default 1800)
#
# Notes:
#   * Engine is deterministic at temp=0+seed=42, so N_SAMPLES=1 is
#     sufficient for pass@1. Bumping it would produce identical outputs.
#   * Wall-clock estimate at defaults: raw ~30 min, engine ~2.5 hours.

set -euo pipefail

N_PROBLEMS="${N_PROBLEMS:-50}"
N_SAMPLES="${N_SAMPLES:-1}"
SEED="${SEED:-42}"
HARNESS_DIR="${HARNESS_DIR:-$HOME/bigcode-eval}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL_BASELINE="${MODEL_BASELINE:-qwen2.5-coder:14b}"
ENGINE_TIMEOUT_S="${ENGINE_TIMEOUT_S:-1800}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
GENERATOR="$REPO_ROOT/scripts/generate_humaneval_completions.py"
DATESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$RESULTS_DIR"

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

if ! command -v belief >/dev/null 2>&1; then
    echo "ERROR: belief CLI not on PATH" >&2
    echo "       run: pip3 install -e $REPO_ROOT --break-system-packages" >&2
    exit 1
fi

if [ ! -d "$HARNESS_DIR" ]; then
    echo "ERROR: BigCode harness not found at $HARNESS_DIR" >&2
    echo "       run: git clone https://github.com/bigcode-project/bigcode-evaluation-harness $HARNESS_DIR" >&2
    echo "            && cd $HARNESS_DIR && pip3 install -e . --break-system-packages" >&2
    exit 1
fi

if [ ! -f "$HARNESS_DIR/main.py" ]; then
    echo "ERROR: $HARNESS_DIR/main.py missing — harness install incomplete" >&2
    exit 1
fi

if [ ! -f "$GENERATOR" ]; then
    echo "ERROR: generator script missing at $GENERATOR" >&2
    exit 1
fi

echo "    ollama up at $OLLAMA_URL"
echo "    belief CLI on PATH"
echo "    harness at $HARNESS_DIR"
echo "    n_problems=$N_PROBLEMS  n_samples=$N_SAMPLES  seed=$SEED"
echo

# ---------------------------------------------------------------------------
# Helper: generate then score one condition
# ---------------------------------------------------------------------------

run_condition() {
    local label="$1"
    local backend="$2"
    local generations="$3"
    local metrics="$4"
    local extra_args="$5"
    local log="$RESULTS_DIR/${label}_${DATESTAMP}.log"

    if [ -f "$metrics" ]; then
        echo "==> condition $label: SKIP — $metrics exists"
        return 0
    fi

    if [ ! -f "$generations" ]; then
        echo "==> condition $label: GENERATE starting at $(date)"
        # shellcheck disable=SC2086
        python3 "$GENERATOR" \
            --backend "$backend" \
            --output "$generations" \
            --limit "$N_PROBLEMS" \
            --seed "$SEED" \
            --ollama-url "$OLLAMA_URL" \
            --ollama-model "$MODEL_BASELINE" \
            --engine-timeout-s "$ENGINE_TIMEOUT_S" \
            $extra_args \
            2>&1 | tee -a "$log"
        echo "    generation done at $(date)"
    else
        echo "==> condition $label: GENERATE skipped — $generations exists"
    fi

    echo "==> condition $label: SCORE via harness"
    (
        cd "$HARNESS_DIR"
        python3 main.py \
            --tasks humaneval \
            --load_generations_path "$generations" \
            --metric_output_path "$metrics" \
            --allow_code_execution \
            --n_samples "$N_SAMPLES" \
            2>&1 | tee -a "$log"
    )
    echo "    scoring done at $(date)"
}

# ---------------------------------------------------------------------------
# Condition A — raw qwen baseline
# ---------------------------------------------------------------------------

run_condition \
    "raw" \
    "raw" \
    "$RESULTS_DIR/raw_humaneval_subset_${N_PROBLEMS}.json" \
    "$RESULTS_DIR/raw_humaneval_subset_${N_PROBLEMS}_metrics.json" \
    ""

# ---------------------------------------------------------------------------
# Condition B — engine via subprocess
# ---------------------------------------------------------------------------

run_condition \
    "engine" \
    "engine" \
    "$RESULTS_DIR/engine_humaneval_subset_${N_PROBLEMS}.json" \
    "$RESULTS_DIR/engine_humaneval_subset_${N_PROBLEMS}_metrics.json" \
    ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

RAW_METRICS="$RESULTS_DIR/raw_humaneval_subset_${N_PROBLEMS}_metrics.json"
ENG_METRICS="$RESULTS_DIR/engine_humaneval_subset_${N_PROBLEMS}_metrics.json"

echo
echo "==> done."
echo
echo "    raw metrics:    $RAW_METRICS"
echo "    engine metrics: $ENG_METRICS"
echo
echo "    pass@1 comparison:"
python3 - <<PYEOF
import json
import pathlib

def load(p):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception as e:
        return {"_error": str(e)}

raw = load("$RAW_METRICS")
eng = load("$ENG_METRICS")

def pass_at_1(d):
    if "_error" in d:
        return None
    # BigCode metric files are usually {"humaneval": {"pass@1": 0.xx}}
    he = d.get("humaneval") or {}
    return he.get("pass@1")

r = pass_at_1(raw)
e = pass_at_1(eng)
if r is None:
    print(f"    raw    : (could not parse {raw.get('_error', 'no humaneval block')})")
else:
    print(f"    raw    : pass@1 = {r:.3f}")
if e is None:
    print(f"    engine : (could not parse {eng.get('_error', 'no humaneval block')})")
else:
    print(f"    engine : pass@1 = {e:.3f}")
if r is not None and e is not None:
    print(f"    delta  : {(e - r):+.3f}")
PYEOF
