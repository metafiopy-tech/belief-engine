"""Grinder: continuous autonomous build loop.

The Grinder is the consumer side of the Photosynthesis pipeline. It
watches `pending_sessions/` for goal specs, picks the highest-priority
one, runs the existing build graph, records results, and periodically
triggers the self-improvement cycles (SICA, jitterbug, crystallization).

See belief/grinder/daemon.py for the main class. Other modules:
  goal_queue.py  — pick_next_goal + fallback templates + file shuffling
  status.py      — atomic status-file writer for `belief grinder status`
"""

from belief.grinder.daemon import GrinderDaemon
from belief.grinder.goal_queue import (
    FALLBACK_GOAL_TEMPLATES,
    GoalEnvelope,
    GoalQueue,
    TEMPLATE_FILLS,
)
from belief.grinder.status import GrinderStatus

__all__ = [
    "FALLBACK_GOAL_TEMPLATES",
    "GoalEnvelope",
    "GoalQueue",
    "GrinderDaemon",
    "GrinderStatus",
    "TEMPLATE_FILLS",
]
