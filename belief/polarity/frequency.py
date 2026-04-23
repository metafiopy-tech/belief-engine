"""FrequencyLayer — moment-to-moment coherence tracking.

EMA (alpha=0.3) for both Latios and Latias.
World state = f(latios_coherence, latias_coherence).
Decay per tick: -0.02 (prevents stale high values).

Source: frequency_layer.py
"""

from __future__ import annotations

import logging

from belief.models.state import PolarityState, WorldState

logger = logging.getLogger("belief.polarity.frequency")

ALPHA = 0.3
DECAY = 0.02


class FrequencyLayer:
    """Tracks moment-to-moment coherence between incompleteness and belief.

    This is the real-time pulse of the system. Updated after every
    agent action. The world state determines system behavior:
      - DORMANT: business as usual
      - RESONANCE: both coherent, productive flow
      - TENSION: significant disagreement, generative potential
      - EMERGENCE: both high coherence — genuine synthesis happening
    """

    def __init__(self) -> None:
        self._tick_count = 0

    def update(
        self, polarity: PolarityState, latios_signal: float, latias_signal: float
    ) -> PolarityState:
        """Update coherence from latest signals and return new state."""
        self._tick_count += 1

        # EMA update
        polarity.update_latios(latios_signal)
        polarity.update_latias(latias_signal)

        # Decay (prevents stale high values)
        polarity.latios_coherence = max(0.0, round(polarity.latios_coherence - DECAY, 4))
        polarity.latias_coherence = max(0.0, round(polarity.latias_coherence - DECAY, 4))

        # World state is recalculated inside update_latios/update_latias
        # but we re-check after decay
        lo, la = polarity.latios_coherence, polarity.latias_coherence
        diff = abs(lo - la)
        if lo > 0.75 and la > 0.75:
            polarity.world_state = WorldState.EMERGENCE
        elif diff > 0.4:
            polarity.world_state = WorldState.TENSION
        elif lo > 0.6 and la > 0.6:
            polarity.world_state = WorldState.RESONANCE
        else:
            polarity.world_state = WorldState.DORMANT

        logger.debug(
            f"Frequency tick {self._tick_count}: "
            f"lo={polarity.latios_coherence:.3f} la={polarity.latias_coherence:.3f} "
            f"state={polarity.world_state.value}"
        )
        return polarity

    def format_status(self, polarity: PolarityState) -> str:
        """Human-readable frequency status."""
        lo = polarity.latios_coherence
        la = polarity.latias_coherence

        def level(v: float) -> str:
            if v > 0.75:
                return "high"
            if v > 0.5:
                return "mid"
            if v > 0.25:
                return "low"
            return "flat"

        return (
            f"Latios: {lo:.2f} ({level(lo)}) | "
            f"Latias: {la:.2f} ({level(la)}) | "
            f"State: {polarity.world_state.value}"
        )
