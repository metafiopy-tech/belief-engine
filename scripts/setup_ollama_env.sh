#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_ollama_env.sh — Session 1 (v3.2) Belief Engine
# ---------------------------------------------------------------------------
# Configures the Ollama.app environment on macOS for the session-1
# streaming + prefix-cache improvements.  Uses `launchctl setenv` rather
# than ~/.zshrc because Ollama.app is launched by launchd (Finder / Dock)
# and does NOT read shell rc files — anything you put in .zshrc is
# invisible to the GUI-launched daemon.  See:
#   https://github.com/ollama/ollama/blob/main/docs/faq.md#setting-environment-variables-on-mac
#
# You MUST restart Ollama.app after running this script for the new
# environment variables to take effect.
#
# This script is deliberately kept out of the Python code path: Claude
# Code / the Belief Engine itself cannot run launchctl — only the
# logged-in user has permission.  Run manually, once, after install.
# ---------------------------------------------------------------------------

set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "[setup_ollama_env] This script only applies to macOS." >&2
    echo "  On Linux, add the equivalent exports to /etc/systemd/system/ollama.service.d/override.conf" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# The Session 1 Ollama environment
# ---------------------------------------------------------------------------
#
# OLLAMA_KEEP_ALIVE=-1
#   Never unload models from VRAM.  Without this, Ollama unloads a model
#   after 5 minutes of idle, and every agent that calls it pays a cold
#   reload (~3-5s on M2 Air for the 14B weights).  Across an overnight
#   run of 40 builds, that's ~15 minutes of pure reload latency.
#
# OLLAMA_FLASH_ATTENTION=1
#   Enables Flash Attention 2 on Metal.  Cuts prompt-eval latency by
#   ~20% on qwen2.5-coder:14b on M2 Air with no measurable quality
#   impact (llama.cpp has had FA2 stable since #7474, mid-2024).
#
# OLLAMA_KV_CACHE_TYPE=q8_0
#   Quantize the KV cache to 8-bit.  Halves KV memory vs f16, letting
#   num_ctx=8192 fit alongside qwen2.5-coder:7b as a resident
#   fallback.  DO NOT USE q4_0 for KV cache — Qwen2 family is
#   empirically sensitive (see llama.cpp PR#7527 and the subsequent
#   quality regression reports on the Qwen2.5-Coder benchmark set).
#
# OLLAMA_NUM_PARALLEL=1
#   Single concurrent request per model.  The Belief Engine is
#   inherently sequential (agent A must finish before agent B sees its
#   output), so parallelism here just wastes VRAM.
#
# OLLAMA_MAX_LOADED_MODELS=2
#   Keep up to 2 models resident.  Primary 14B + fallback 7B fit
#   together in 16GB on M2 Air.  The graceful-degradation cascade in
#   belief/llm.py depends on both being pre-loaded.
#
# OLLAMA_LOAD_TIMEOUT=10m
#   Cold-load ceiling.  Default 5 minutes occasionally fired on a
#   cold-boot M2 Air; 10 minutes is still below the per-role budget.
# ---------------------------------------------------------------------------

declare -A ENV_VARS=(
    [OLLAMA_KEEP_ALIVE]="-1"
    [OLLAMA_FLASH_ATTENTION]="1"
    [OLLAMA_KV_CACHE_TYPE]="q8_0"
    [OLLAMA_NUM_PARALLEL]="1"
    [OLLAMA_MAX_LOADED_MODELS]="2"
    [OLLAMA_LOAD_TIMEOUT]="10m"
)

echo "[setup_ollama_env] Setting Ollama environment via launchctl setenv..."

# Preserve bash key iteration order.
for KEY in OLLAMA_KEEP_ALIVE OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE \
          OLLAMA_NUM_PARALLEL OLLAMA_MAX_LOADED_MODELS OLLAMA_LOAD_TIMEOUT; do
    VAL="${ENV_VARS[$KEY]}"
    launchctl setenv "$KEY" "$VAL"
    echo "  ${KEY}=${VAL}"
done

# DO NOT USE — kept here as a commented warning so nobody re-introduces it.
# launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0
# ^ q4_0 on Qwen2 family causes measurable quality loss on code tasks.
#   Reference: llama.cpp PR#7527, Qwen2.5-Coder regression discussion.

echo ""
echo "[setup_ollama_env] Done.  IMPORTANT:"
echo "  1. Quit Ollama.app completely (menu bar icon → Quit)."
echo "  2. Re-launch Ollama.app from Applications."
echo "  3. Verify: ollama serve --help | head -1   (should still work)"
echo "  4. Verify env applied:"
echo "       launchctl getenv OLLAMA_KEEP_ALIVE   # should print -1"
echo ""
echo "[setup_ollama_env] These settings persist across reboots until you"
echo "  run: launchctl unsetenv OLLAMA_KEEP_ALIVE  (etc.)"
