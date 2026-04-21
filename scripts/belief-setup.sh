#!/usr/bin/env bash
# belief-setup.sh - one-command setup for the Belief Engine (Session 18).
#
# What this does, in order:
#   1. Detects macOS / Linux and refuses to run elsewhere.
#   2. Installs Ollama (only if not already installed).
#   3. Pulls the recommended local model (qwen2.5-coder:14b by default).
#   4. Creates ~/.belief-engine/soil if missing.
#   5. Installs the belief-engine Python package with optional 'full' extras.
#   6. Runs a smoke build in local mode to verify everything wires up.
#   7. Optionally starts the Grinder daemon for autonomous operation.
#
# Environment knobs:
#   BELIEF_LOCAL_MODEL      Override the Ollama model (default qwen2.5-coder:14b)
#   BELIEF_INSTALL_EXTRAS   pip extras spec (default 'full'; set to '' to skip)
#   BELIEF_SKIP_SMOKE       '1' to skip the smoke build
#   BELIEF_START_GRINDER    '1' to run the grinder in the background at the end
#
# Idempotent: re-running is safe.  Steps that detect an existing install
# log "already present" and move on.

set -euo pipefail

# ---------------------------------------------------------------- logging
_info()  { printf '\033[1;34m[belief-setup]\033[0m %s\n' "$*"; }
_warn()  { printf '\033[1;33m[belief-setup]\033[0m %s\n' "$*" >&2; }
_error() { printf '\033[1;31m[belief-setup]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- platform
OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM=macos ;;
    Linux)  PLATFORM=linux ;;
    *)
        _error "Unsupported platform: $OS. belief-setup.sh supports macOS and Linux."
        ;;
esac
_info "Platform: $PLATFORM"

# ---------------------------------------------------------------- ollama
if command -v ollama >/dev/null 2>&1; then
    _info "Ollama already installed ($(ollama --version 2>/dev/null | head -n1))."
else
    _info "Installing Ollama ..."
    case "$PLATFORM" in
        macos|linux)
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
    esac
    command -v ollama >/dev/null 2>&1 || _error "Ollama install appeared to complete but 'ollama' is not on PATH. Open a new shell and retry."
fi

# Ensure the Ollama daemon is running so 'ollama pull' can talk to it.
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    _info "Starting Ollama daemon in the background ..."
    ollama serve >/dev/null 2>&1 &
    OLLAMA_PID=$!
    # Poll the health endpoint rather than a blind sleep.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
    done
    if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
        _warn "Ollama daemon did not come up within 10s. Continuing anyway; 'ollama pull' will retry."
    fi
fi

LOCAL_MODEL="${BELIEF_LOCAL_MODEL:-qwen2.5-coder:14b}"
_info "Pulling local model: $LOCAL_MODEL (this may take a while on first run)"
ollama pull "$LOCAL_MODEL"

# ---------------------------------------------------------------- soil dir
SOIL_DIR="${HOME}/.belief-engine/soil"
if [ -d "$SOIL_DIR" ]; then
    _info "Soil directory exists: $SOIL_DIR"
else
    _info "Creating soil directory: $SOIL_DIR"
    mkdir -p "$SOIL_DIR"
fi

# ---------------------------------------------------------------- python package
EXTRAS="${BELIEF_INSTALL_EXTRAS-full}"
if command -v belief >/dev/null 2>&1; then
    _info "'belief' CLI already on PATH; skipping pip install. (unset BELIEF_INSTALL_EXTRAS to force a re-install)"
else
    if [ -n "$EXTRAS" ]; then
        _info "Installing belief-engine[$EXTRAS] via pip ..."
        pip install "belief-engine[$EXTRAS]"
    else
        _info "Installing belief-engine via pip ..."
        pip install "belief-engine"
    fi
fi

command -v belief >/dev/null 2>&1 || _error "'belief' CLI still not on PATH after install. Check your PYTHONPATH / venv."

# ---------------------------------------------------------------- smoke build
if [ "${BELIEF_SKIP_SMOKE:-0}" = "1" ]; then
    _info "BELIEF_SKIP_SMOKE=1 -- skipping smoke build."
else
    _info "Running a smoke build in local mode ..."
    export BELIEF_MODEL_MODE=local
    export BELIEF_LOCAL_MODEL="$LOCAL_MODEL"
    if belief --goal "Build a Python script that prints hello world" --max-cost 0.01; then
        _info "Smoke build succeeded."
    else
        _warn "Smoke build exited non-zero. Setup is done but the first run didn't produce a passing artefact. Check ~/.belief-engine/ logs."
    fi
fi

# ---------------------------------------------------------------- grinder
if [ "${BELIEF_START_GRINDER:-0}" = "1" ]; then
    _info "Starting the Grinder daemon in the background ..."
    nohup belief grinder start >/dev/null 2>&1 &
    _info "Grinder PID: $!"
else
    _info "Setup complete. Run 'belief grinder start' to begin autonomous operation."
fi
