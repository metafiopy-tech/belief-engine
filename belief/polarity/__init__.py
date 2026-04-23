"""Polarity engine — the core innovation.

IncompletenessLoop (Latios) finds what's missing.
BeliefLoop (Latias) protects what matters.
CrossTalkManager feeds each into the other's world.
FrequencyLayer tracks moment-to-moment coherence.
Convergence modules prevent spirals and oscillation.
"""

from belief.polarity.incompleteness import IncompletenessLoop, Remainder
from belief.polarity.belief import BeliefLoop, Covenant
from belief.polarity.crosstalk import CrossTalkManager
from belief.polarity.frequency import FrequencyLayer
from belief.polarity.convergence import LoopBlocker, OscillationDetector, ActionDeduplicator

__all__ = [
    "IncompletenessLoop",
    "Remainder",
    "BeliefLoop",
    "Covenant",
    "CrossTalkManager",
    "FrequencyLayer",
    "LoopBlocker",
    "OscillationDetector",
    "ActionDeduplicator",
]
