"""Thermal gate — M2 Air throttling mitigation (Session 1 v3.2).

Apple Silicon M2/M3 laptops (especially the fanless M2 Air) throttle
silently after 3–6 minutes of sustained inference, dropping tok/s by
25–35% with no log event.  The Ollama runner doesn't notice; it just
runs slower.  That's the opposite of what we want for per-chunk
watchdogs, because the degraded-but-not-stalled state looks like
progress but wall-clock is destroyed.

Solution (per the research report): poll macOS's thermal pressure
indicator before each agent session and back off proportionally:

    Nominal   → 2s   (barely a pause — just hand the scheduler a slice)
    Moderate  → 10s  (light cooldown, typical during long builds)
    Heavy     → 30s  (material cooldown, throttling already happening)
    Trapping  → 120s (emergency — the kernel is aggressively reducing frequency)

On non-macOS platforms or when `notifyutil` is missing, :func:`thermal_gate`
is a no-op.  This keeps Linux CI (where Claude Code's hermetic tests
actually run) completely unaffected.

Both a sync and an async variant are exposed:

  * :func:`thermal_gate`       — sync ``time.sleep`` (CLI / non-async callers)
  * :func:`async_thermal_gate` — uses ``asyncio.sleep`` (BaseAgent)

The async variant is what :class:`~belief.agents.base.BaseAgent` calls
at the start of ``__call__``.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import subprocess
import time
from enum import Enum
from typing import Final

logger = logging.getLogger("belief.thermal")


class ThermalPressure(str, Enum):
    NOMINAL = "nominal"
    MODERATE = "moderate"
    HEAVY = "heavy"
    TRAPPING = "trapping"
    UNKNOWN = "unknown"


# Durations in seconds — tuned for MacBook Air M2 16GB during a
# 40-challenge overnight benchmark.  The heavy/trapping values look
# expensive; they are, but 30s of proactive cooldown routinely prevents
# the 90s-180s penalty of a full thermal throttle event later.
_DURATIONS: Final[dict[ThermalPressure, float]] = {
    ThermalPressure.NOMINAL: 2.0,
    ThermalPressure.MODERATE: 10.0,
    ThermalPressure.HEAVY: 30.0,
    ThermalPressure.TRAPPING: 120.0,
    ThermalPressure.UNKNOWN: 0.0,
}


def _read_pressure_macos() -> ThermalPressure:
    """Query the macOS system-wide thermal pressure level.

    Prefers ``notifyutil -g com.apple.system.thermalpressurelevel`` as
    the session doc specifies.  Falls back silently on errors or when
    notifyutil is unavailable.
    """
    if shutil.which("notifyutil") is None:
        return ThermalPressure.UNKNOWN
    try:
        # ``notifyutil -g KEY`` prints ``KEY <current_value>`` where
        # <current_value> is a numeric level (0=Nominal, 1=Moderate,
        # 2=Heavy, 3=Trapping) as of macOS 13+.  Older releases may
        # print the string form directly; we handle both.
        result = subprocess.run(
            ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("notifyutil query failed: %s", e)
        return ThermalPressure.UNKNOWN

    if result.returncode != 0:
        logger.debug("notifyutil exit=%s stderr=%r", result.returncode, result.stderr[:200])
        return ThermalPressure.UNKNOWN

    raw = (result.stdout or "").strip().lower()
    # Formats we've seen:
    #   "com.apple.system.thermalpressurelevel 0"
    #   "com.apple.system.thermalpressurelevel nominal"
    #   "0"
    tokens = raw.split()
    if not tokens:
        return ThermalPressure.UNKNOWN
    candidate = tokens[-1]

    # Numeric form
    if candidate.isdigit():
        num_to_state = {
            "0": ThermalPressure.NOMINAL,
            "1": ThermalPressure.MODERATE,
            "2": ThermalPressure.HEAVY,
            "3": ThermalPressure.TRAPPING,
        }
        return num_to_state.get(candidate, ThermalPressure.UNKNOWN)

    # String form
    string_to_state = {
        "nominal": ThermalPressure.NOMINAL,
        "moderate": ThermalPressure.MODERATE,
        "heavy": ThermalPressure.HEAVY,
        "trapping": ThermalPressure.TRAPPING,
        "critical": ThermalPressure.TRAPPING,  # older wording
    }
    return string_to_state.get(candidate, ThermalPressure.UNKNOWN)


def read_thermal_pressure() -> ThermalPressure:
    """Return the current thermal pressure, or UNKNOWN off-macOS."""
    if platform.system() != "Darwin":
        return ThermalPressure.UNKNOWN
    return _read_pressure_macos()


def _duration_for(pressure: ThermalPressure) -> float:
    return _DURATIONS.get(pressure, 0.0)


def thermal_gate() -> ThermalPressure:
    """Synchronous thermal gate — sleeps proportionally to pressure.

    Returns the observed :class:`ThermalPressure` so callers can log.
    Safe to call on any platform: on non-macOS or when notifyutil is
    missing, returns :data:`ThermalPressure.UNKNOWN` and sleeps 0s.
    """
    pressure = read_thermal_pressure()
    dur = _duration_for(pressure)
    if dur > 0:
        logger.info(
            "thermal_gate: pressure=%s sleeping=%.0fs",
            pressure.value,
            dur,
        )
        time.sleep(dur)
    return pressure


async def async_thermal_gate() -> ThermalPressure:
    """Async variant — the same contract, backed by ``asyncio.sleep``.

    Used by :class:`~belief.agents.base.BaseAgent` at the start of each
    agent invocation.  ``read_thermal_pressure`` is a subprocess call,
    so we offload it to a thread to avoid blocking the event loop.
    """
    pressure = await asyncio.to_thread(read_thermal_pressure)
    dur = _duration_for(pressure)
    if dur > 0:
        logger.info(
            "thermal_gate: pressure=%s sleeping=%.0fs",
            pressure.value,
            dur,
        )
        await asyncio.sleep(dur)
    return pressure


__all__ = [
    "ThermalPressure",
    "async_thermal_gate",
    "read_thermal_pressure",
    "thermal_gate",
]
