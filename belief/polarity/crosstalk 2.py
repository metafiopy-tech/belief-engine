"""CrossTalkManager — the tension between Latios and Latias.

Source: engine_loop.py lines 249-282
"""

from __future__ import annotations

import logging
from typing import Optional

from belief.polarity.incompleteness import IncompletenessLoop
from belief.polarity.belief import BeliefLoop

logger = logging.getLogger("belief.polarity.crosstalk")


class CrossTalkManager:
    """Latios's gap → Latias's world (new material to hold).
    Latias's covenant → Latios's world (thing that has a gap).
    Neither knows. They just feel it as context.

    Source: engine_loop.py CrossTalkManager
    """

    def get_latios_world(self, latios: IncompletenessLoop,
                         latias: BeliefLoop, goal: str) -> str:
        """Build world context for Latios, seeded with Latias's covenant."""
        covenant = latias.get_deepest_covenant()
        world = f"Goal: {goal}\n"
        if covenant:
            world += f"\nYour twin is standing somewhere: {covenant}\n"
            world += "She believes in it. Find what's still missing from that view."
        recent_gaps = latios.get_recent(3)
        if recent_gaps:
            world += "\nRecent gaps you've found:\n" + "\n".join(f"- {g}" for g in recent_gaps)
        return world

    def get_latias_world(self, latios: IncompletenessLoop,
                         latias: BeliefLoop, goal: str) -> str:
        """Build world context for Latias, seeded with Latios's remainder."""
        gap = latios.current
        world = f"Goal: {goal}\n"
        if gap:
            world += f"\nYour twin is chasing this gap: {gap}\n"
            world += "You don't have to chase it. What here is worth protecting?"
        recent_covenants = latias.get_recent(3)
        if recent_covenants:
            world += "\nWhat you've been holding:\n" + "\n".join(f"- {c}" for c in recent_covenants)
        return world
