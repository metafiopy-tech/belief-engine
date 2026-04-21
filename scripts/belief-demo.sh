#!/usr/bin/env bash
# belief-demo.sh - the public-demo script (Session 18, Task 4).
#
# Flow (designed to read well in a screen recording):
#   1. `belief progression`           - every domain at Stage 0
#   2. 5 builds across different verticals (fastapi, cli, mcp, data, script)
#   3. `belief progression`           - domains advancing
#   4. `belief dashboard`             - metrics improving
#   5. `belief library`               - any tools the engine promoted
#
# The goals below are deliberately small so the demo runs in a
# reasonable wallclock.  Override GOALS with your own list if you
# want to showcase something else, or set BELIEF_DEMO_BUILDS=0 to
# skip the build phase and only show the cold-start state.
#
# Environment:
#   BELIEF_MODEL_MODE      'local' (default), 'hybrid', or 'cloud'
#   BELIEF_DEMO_BUILDS     Count to run (default: 5; set to 0 to skip)
#   BELIEF_DEMO_MAX_COST   Per-build cap in USD (default: 1.0)

set -euo pipefail

MODE="${BELIEF_MODEL_MODE:-local}"
BUILDS="${BELIEF_DEMO_BUILDS:-5}"
MAX_COST="${BELIEF_DEMO_MAX_COST:-1.0}"

# Default 5 goals, one per primary vertical.
if [ -z "${GOALS:-}" ]; then
    read -r -d '' GOALS <<'EOF' || true
Build a FastAPI endpoint that returns the current time as JSON
Build a Click CLI that converts CSV files to JSON
Build an MCP tool server exposing one 'echo' tool
Build a Python script that computes the first 100 primes
Build a data pipeline that reads a CSV and writes sorted rows
EOF
fi

_banner() {
    local msg="$*"
    printf '\n\033[1;36m'
    printf '═%.0s' $(seq 1 62)
    printf '\n  %s\n' "$msg"
    printf '═%.0s' $(seq 1 62)
    printf '\033[0m\n'
}

command -v belief >/dev/null 2>&1 || {
    printf 'belief-demo.sh: the `belief` CLI is not on PATH. Run scripts/belief-setup.sh first.\n' >&2
    exit 1
}

export BELIEF_MODEL_MODE="$MODE"
_banner "Mode: $BELIEF_MODEL_MODE"

# ── 1. Progression at cold start ──────────────────────────────────────────
_banner "1. belief progression  (cold start — every domain at Stage 0)"
belief progression || true

# ── 2. Five builds across different verticals ─────────────────────────────
if [ "$BUILDS" -gt 0 ]; then
    _banner "2. Running $BUILDS builds across different verticals"
    idx=0
    while IFS= read -r goal && [ "$idx" -lt "$BUILDS" ]; do
        [ -z "$goal" ] && continue
        idx=$((idx + 1))
        _banner "Build $idx/$BUILDS: $goal"
        if ! belief --goal "$goal" --max-cost "$MAX_COST"; then
            printf '\n  (build %d failed, moving on — the soil still records the remainder)\n' "$idx"
        fi
    done <<< "$GOALS"
else
    _banner "2. BELIEF_DEMO_BUILDS=0 — skipping the build phase"
fi

# ── 3. Progression after the builds ───────────────────────────────────────
_banner "3. belief progression  (after $BUILDS builds — domains advancing)"
belief progression || true

# ── 4. Dashboard ──────────────────────────────────────────────────────────
_banner "4. belief dashboard  (metrics: pass rate, cost, nutrients, covenants)"
belief dashboard || true

# ── 5. Library ────────────────────────────────────────────────────────────
_banner "5. belief library  (any tools the engine promoted)"
belief library || true

_banner "Demo complete."
echo "The same command on build 50 will look noticeably different —"
echo "that's the soil compounding.  Run 'belief manifold' to see the topology"
echo "or 'belief grinder start' to let it keep building on its own."
