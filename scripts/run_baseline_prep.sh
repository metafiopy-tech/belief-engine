#!/usr/bin/env bash
# Run the full substrate-transfer baseline-prep procedure unattended.
#
# Designed to be kicked off overnight. Takes the safety snapshot path
# as the only argument; everything else is automated. On any failure,
# auto-restores the safety snapshot so the live engine state is preserved.
#
# Usage:
#   ./scripts/run_baseline_prep.sh /path/to/safety-snapshot
#
# Output:
#   ~/.belief-engine/baseline_prep_run.log  — every step + timestamps
#   ~/.belief-engine/substrate_baselines.json — final paths config (on success)
#
# Time budget: ~5 hours (28 sequential builds + snapshot overhead).
# Cost budget: $0 (all builds forced through local Ollama).

set -uo pipefail

SAFETY_SNAPSHOT="${1:-}"
if [[ -z "${SAFETY_SNAPSHOT}" ]]; then
  echo "ERROR: pass the safety snapshot path as the first argument" >&2
  echo "  Example:" >&2
  echo "    $0 \"\$(ls -dt ~/.belief-engine/snapshots/*live-pre-substrate-baseline-prep | head -1)\"" >&2
  exit 2
fi

if [[ ! -f "${SAFETY_SNAPSHOT}/manifest.json" ]]; then
  echo "ERROR: safety snapshot manifest not found at ${SAFETY_SNAPSHOT}/manifest.json" >&2
  exit 2
fi

LOG="${HOME}/.belief-engine/baseline_prep_run.log"
mkdir -p "${HOME}/.belief-engine"
exec > >(tee -a "${LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

abort() {
  log "ABORTING: $*"
  log "Restoring safety snapshot ${SAFETY_SNAPSHOT}"
  belief snapshot restore "${SAFETY_SNAPSHOT}" || log "WARNING: safety restore failed; manual recovery needed"
  exit 1
}

trap 'abort "received signal"' INT TERM

log "============================================================"
log "Substrate-transfer baseline prep — start"
log "Safety snapshot: ${SAFETY_SNAPSHOT}"
log "============================================================"

# Pre-flight: confirm tools available. Note: we don't probe Ollama here
# because `ollama` may not be on the detached shell's PATH, even when the
# daemon is running fine. The first `belief build` will fail loudly if
# the model isn't actually reachable, and the per-build error handler
# logs the failure without crashing the whole sequence.
which belief >/dev/null || abort "belief CLI not found on PATH"
which python3 >/dev/null || abort "python3 not found on PATH"

run_step() {
  local label="$1"; shift
  log "STEP: ${label}"
  if ! "$@"; then
    abort "step failed: ${label}"
  fi
  log "STEP DONE: ${label}"
}

wipe_live_state() {
  rm -rf "${HOME}/.belief-engine/soil"
  rm -f "${HOME}/.belief-engine/builds.db" "${HOME}/.belief-engine/builds.db-journal"
  rm -f "${HOME}/.belief-engine/niches.db" "${HOME}/.belief-engine/niches.db-journal"
  rm -f "${HOME}/.belief-engine/reciprocity.db" "${HOME}/.belief-engine/reciprocity.db-journal"
  rm -f "${HOME}/.belief-engine/routing.db" "${HOME}/.belief-engine/routing.db-journal"
}

take_snapshot() {
  local label="$1"
  belief snapshot take --label "${label}" || abort "snapshot take failed for ${label}"
}

find_snapshot() {
  local pattern="$1"
  ls -dt "${HOME}/.belief-engine/snapshots/"*"${pattern}" 2>/dev/null | head -1
}

# -- Phase 1: empty state ----------------------------------------------------
run_step "wipe live state for empty baseline" wipe_live_state
run_step "take empty snapshot" take_snapshot "substrate-baseline-empty"
EMPTY_SNAPSHOT="$(find_snapshot 'substrate-baseline-empty')"
[[ -d "${EMPTY_SNAPSHOT}" ]] || abort "empty snapshot path not found after take"
log "EMPTY_SNAPSHOT=${EMPTY_SNAPSHOT}"

# -- Phase 2: soil_only b1->b5 -----------------------------------------------
log "Phase 2: soil_only build_seq=1->5 (4 builds, ~45-60 min)"
BELIEF_EXPERIMENT_CONDITION=soil_only \
  python3 scripts/baseline_build_sequence.py --count 4 --offset 0 \
  || abort "soil_only b1->b5 script exited non-zero"
run_step "take soil_only b5 snapshot" take_snapshot "substrate-baseline-soil_only-b5"
SOIL_ONLY_B5="$(find_snapshot 'substrate-baseline-soil_only-b5')"
[[ -d "${SOIL_ONLY_B5}" ]] || abort "soil_only-b5 snapshot path not found after take"
log "SOIL_ONLY_B5=${SOIL_ONLY_B5}"

# -- Phase 3: soil_only b5->b15 ----------------------------------------------
log "Phase 3: soil_only build_seq=5->15 (10 builds, ~2h)"
BELIEF_EXPERIMENT_CONDITION=soil_only \
  python3 scripts/baseline_build_sequence.py --count 10 --offset 4 \
  || abort "soil_only b5->b15 script exited non-zero"
run_step "take soil_only b15 snapshot" take_snapshot "substrate-baseline-soil_only-b15"
SOIL_ONLY_B15="$(find_snapshot 'substrate-baseline-soil_only-b15')"
[[ -d "${SOIL_ONLY_B15}" ]] || abort "soil_only-b15 snapshot path not found after take"
log "SOIL_ONLY_B15=${SOIL_ONLY_B15}"

# -- Phase 4: restore empty, then full b1->b5 -------------------------------
log "Phase 4: restore empty -> full build_seq=1->5 (4 builds, ~45-60 min)"
belief snapshot restore "${EMPTY_SNAPSHOT}" || abort "could not restore empty snapshot before full sequence"
BELIEF_EXPERIMENT_CONDITION=full \
  python3 scripts/baseline_build_sequence.py --count 4 --offset 0 \
  || abort "full b1->b5 script exited non-zero"
run_step "take full b5 snapshot" take_snapshot "substrate-baseline-full-b5"
FULL_B5="$(find_snapshot 'substrate-baseline-full-b5')"
[[ -d "${FULL_B5}" ]] || abort "full-b5 snapshot path not found after take"
log "FULL_B5=${FULL_B5}"

# -- Phase 5: full b5->b15 ---------------------------------------------------
log "Phase 5: full build_seq=5->15 (10 builds, ~2h)"
BELIEF_EXPERIMENT_CONDITION=full \
  python3 scripts/baseline_build_sequence.py --count 10 --offset 4 \
  || abort "full b5->b15 script exited non-zero"
run_step "take full b15 snapshot" take_snapshot "substrate-baseline-full-b15"
FULL_B15="$(find_snapshot 'substrate-baseline-full-b15')"
[[ -d "${FULL_B15}" ]] || abort "full-b15 snapshot path not found after take"
log "FULL_B15=${FULL_B15}"

# -- Phase 6: verify all 5 -------------------------------------------------
log "Phase 6: verify all 5 baselines"
for snap in "${EMPTY_SNAPSHOT}" "${SOIL_ONLY_B5}" "${SOIL_ONLY_B15}" "${FULL_B5}" "${FULL_B15}"; do
  if ! belief snapshot verify "${snap}"; then
    abort "snapshot verify failed for ${snap}"
  fi
  log "  verified: ${snap}"
done

# -- Phase 7: restore live -------------------------------------------------
log "Phase 7: restore live working soil"
belief snapshot restore "${SAFETY_SNAPSHOT}" || abort "could not restore live working soil at end"

# -- Phase 8: write the JSON config ----------------------------------------
log "Phase 8: write substrate_baselines.json"
cat > "${HOME}/.belief-engine/substrate_baselines.json" <<EOF
{
  "soil_only_b1": "${EMPTY_SNAPSHOT}",
  "soil_only_b5": "${SOIL_ONLY_B5}",
  "soil_only_b15": "${SOIL_ONLY_B15}",
  "full_b1": "${EMPTY_SNAPSHOT}",
  "full_b5": "${FULL_B5}",
  "full_b15": "${FULL_B15}"
}
EOF
log "Wrote: ${HOME}/.belief-engine/substrate_baselines.json"

log "============================================================"
log "DONE — all 5 baselines verified and config written"
log "Per-build log: ${HOME}/.belief-engine/baseline_prep.log"
log "Run log:       ${LOG}"
log "============================================================"
