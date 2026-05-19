"""Signal alphabet and temporal integration (mycorrhizal Stage 4).

Areas 1 + 10 of the mycorrhizal brief. A small, fixed token alphabet with
ratio/blend semantics over a moving time window, plus a channel-capacity
measurement harness so the protocol's information density is empirical
rather than speculative.

This package ships the *protocol* and *measurement infrastructure*. No
consumers (autonomous agents emitting signals) exist yet — they arrive
in Session 5 (router) and later. The shape here is deliberately
agent-agnostic so a future router can drop in without protocol changes.
"""

from belief.signal.alphabet import (
    SIGNAL_TOKENS,
    Signal,
    SignalToken,
)
from belief.signal.store import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_HALF_LIFE,
    DEFAULT_WINDOW,
    SignalStore,
    get_default_store,
)
from belief.signal.triggers import (
    Trigger,
    TriggerRegistry,
    sustained_offer_predicate,
    stress_request_conjunction_predicate,
)
from belief.signal.capacity import (
    CapacityMeasurement,
    CapacityReport,
)

__all__ = [
    "SIGNAL_TOKENS",
    "Signal",
    "SignalToken",
    "SignalStore",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_HALF_LIFE",
    "DEFAULT_WINDOW",
    "get_default_store",
    "Trigger",
    "TriggerRegistry",
    "sustained_offer_predicate",
    "stress_request_conjunction_predicate",
    "CapacityMeasurement",
    "CapacityReport",
]
