#!/usr/bin/env python3
"""CLI shim for belief.thermal — Session 1 (v3.2).

The real implementation lives at belief/thermal.py (which is importable
from the rest of the engine).  This file is provided at the path the
session-1 prompt document specifies (``scripts/thermal_gate.py``) and
adds a command-line entry point for manual inspection or use in a
shell loop (``while :; do python scripts/thermal_gate.py; done``).

Examples
--------

    $ python scripts/thermal_gate.py
    nominal 0 (slept 2.0s)

    $ python scripts/thermal_gate.py --dry-run
    nominal 0 (would sleep 2.0s — --dry-run passed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Let the script be runnable from anywhere without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from belief.thermal import (  # noqa: E402  (path set up above)
    ThermalPressure,
    _duration_for,
    read_thermal_pressure,
    thermal_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Belief Engine thermal gate CLI.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the observed pressure + the would-sleep duration, then exit. No sleep.",
    )
    args = parser.parse_args()

    if args.dry_run:
        pressure = read_thermal_pressure()
        dur = _duration_for(pressure)
        print(
            f"{pressure.value} ({_numeric(pressure)}) "
            f"(would sleep {dur:.1f}s — --dry-run passed)"
        )
        return 0

    pressure = thermal_gate()
    dur = _duration_for(pressure)
    print(f"{pressure.value} ({_numeric(pressure)}) (slept {dur:.1f}s)")
    return 0


def _numeric(pressure: ThermalPressure) -> int:
    return {
        ThermalPressure.NOMINAL: 0,
        ThermalPressure.MODERATE: 1,
        ThermalPressure.HEAVY: 2,
        ThermalPressure.TRAPPING: 3,
        ThermalPressure.UNKNOWN: -1,
    }[pressure]


if __name__ == "__main__":
    raise SystemExit(main())
