#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# refresh_pypi_corpus.sh — Session 3 (v3.2) Belief Engine
# ---------------------------------------------------------------------------
# Refreshes two on-disk package corpora that the 6-layer package
# validator (belief/validators/package_validator.py) reads:
#
#   1. Top-15k PyPI packages — small (~0.5MB), authoritative positive
#      list.  Refreshed weekly by the validator automatically via
#      refresh_top_packages() when mtime > 7 days.  This script is for
#      manual / scheduled refreshes (cron, launchd).
#
#   2. Full PyPI name list (~13MB, ~600k names).  Used for offline
#      validation when the network is down.  Not required for normal
#      operation; optional.
#
# Both files live under ~/.belief-engine/.  Run this script once per
# week via cron or launchd, or on-demand when you notice validator
# misses.
#
# Dependencies: curl, jq (install via Homebrew: `brew install jq`).
# ---------------------------------------------------------------------------

set -euo pipefail

CACHE_DIR="${BELIEF_CACHE_DIR:-$HOME/.belief-engine}"
TOP15K_URL="https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
PYPI_SIMPLE_URL="https://pypi.org/simple/"

mkdir -p "$CACHE_DIR"

echo "[refresh_pypi_corpus] Cache dir: $CACHE_DIR"

# ---- Top-15k -------------------------------------------------------------

TOP15K_PATH="$CACHE_DIR/top-pypi-packages-15k.json"
TMP_TOP15K="$(mktemp)"
TMP_FULL="$(mktemp)"
trap 'rm -f "$TMP_TOP15K" "$TMP_FULL" 2>/dev/null || true' EXIT

echo "[refresh_pypi_corpus] Fetching top-15k ..."
if curl -sSL --max-time 60 -o "$TMP_TOP15K" "$TOP15K_URL"; then
    # Basic sanity: file is non-empty and starts with a JSON token.
    if [[ -s "$TMP_TOP15K" ]] && \
       { head -c 1 "$TMP_TOP15K" | grep -q '\['; } || \
       { head -c 1 "$TMP_TOP15K" | grep -q '{'; }; then
        mv "$TMP_TOP15K" "$TOP15K_PATH"
        echo "[refresh_pypi_corpus] ✓ Wrote $TOP15K_PATH ($(wc -c < "$TOP15K_PATH") bytes)"
    else
        echo "[refresh_pypi_corpus] ✗ top-15k download returned empty/invalid data; keeping old copy" >&2
        exit 1
    fi
else
    echo "[refresh_pypi_corpus] ✗ top-15k fetch failed; keeping old copy" >&2
    exit 1
fi

# ---- Full PyPI name list (~13MB, optional) -------------------------------

FULL_PATH="$CACHE_DIR/pypi-all-names.txt"

if ! command -v jq >/dev/null 2>&1; then
    echo "[refresh_pypi_corpus] jq not installed; skipping full-list refresh."
    echo "[refresh_pypi_corpus] Install jq with 'brew install jq' to enable offline fallback."
    exit 0
fi

echo "[refresh_pypi_corpus] Fetching full PyPI name list (~13MB, slow) ..."
if curl -sSL --max-time 120 \
     -H "Accept: application/vnd.pypi.simple.v1+json" \
     -H "User-Agent: belief-engine/3.2 (metafiopy@example.com)" \
     -o "$TMP_FULL" "$PYPI_SIMPLE_URL"; then
    # Extract just the names — smaller than the whole JSON blob.
    if jq -r '.projects[].name' "$TMP_FULL" > "$FULL_PATH.tmp"; then
        mv "$FULL_PATH.tmp" "$FULL_PATH"
        echo "[refresh_pypi_corpus] ✓ Wrote $FULL_PATH ($(wc -l < "$FULL_PATH") names)"
    else
        echo "[refresh_pypi_corpus] ✗ jq parse failed on full list; keeping old copy" >&2
    fi
else
    echo "[refresh_pypi_corpus] ✗ Full PyPI list fetch failed; keeping old copy" >&2
fi

# ---- Scheduling hints ----------------------------------------------------

cat <<EOF

[refresh_pypi_corpus] Done.

Scheduling suggestion (weekly):

  # Via cron (add to 'crontab -e'):
  0 3 * * 1  $HOME/Desktop/belief-engine/scripts/refresh_pypi_corpus.sh

  # Via launchd (macOS, preferred):
  Create ~/Library/LaunchAgents/com.belief-engine.pypi-refresh.plist
  with a StartCalendarInterval entry for weekly Monday 3am refresh.
  See 'man launchd.plist' for the schema.
EOF
