"""Safety subsystem: cost budgets, kill switches, anomaly detection.

Session 4 lands MINIMAL STUBS so synthesis modules can import safety
primitives without Session 5 being complete. Every stub is permissive:

  - CostTracker.under_budget() always True
  - @kill_switch(tag=...) is a pass-through decorator

Session 5 replaces these with real implementations (SQLite-backed
cost log, per-tag kill-switch control table, Discord webhook, etc.).
No public API changes expected — Session 4 modules already call the
stubs by the name they'll have post-Session-5.
"""

from belief.photosynthesis.safety.cost_tracker import (
    BudgetExceeded,
    CostTracker,
)
from belief.photosynthesis.safety.kill_switch import kill_switch, KillSwitchTripped

__all__ = [
    "BudgetExceeded",
    "CostTracker",
    "KillSwitchTripped",
    "kill_switch",
]
