"""Safety subsystem: cost budgets, kill switches, anomaly, audit, HITL, rate limits.

Session 5 fills in the real implementations. Stubs from Session 4 are
gone; the public surface (import paths, function/class names) is the
same, so Session-4 callers keep working.
"""

from belief.photosynthesis.safety.cost_tracker import (
    BreakerAnthropic,
    BreakerConfig,
    BudgetExceeded,
    CostTracker,
    PRICING,
    Usage,
    price_usd,
)
from belief.photosynthesis.safety.kill_switch import (
    ControlStatus,
    KillSwitchTripped,
    KillSwitchState,
    kill_switch,
)

__all__ = [
    "BreakerAnthropic",
    "BreakerConfig",
    "BudgetExceeded",
    "ControlStatus",
    "CostTracker",
    "KillSwitchState",
    "KillSwitchTripped",
    "PRICING",
    "Usage",
    "kill_switch",
    "price_usd",
]
