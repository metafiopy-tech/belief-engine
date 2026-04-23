"""Convergence engineering — circuit breakers and loop detection.

Source: engine_loop.py lines 291-341 (LoopBlocker)
        taskforce_base.py (oscillation detection)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger("belief.polarity.convergence")


class LoopBlocker:
    """Detects think spirals and breaks them with forced action.

    Source: engine_loop.py LoopBlocker

    If an agent produces N consecutive "thinking" outputs without
    taking concrete action, the blocker injects a forced action prompt.
    """

    THRESHOLD = 5

    def __init__(self) -> None:
        self.consecutive_thinks = 0
        self.total_blocked = 0

    def check(self, agent_name: str, output: str) -> Optional[str]:
        """Check if the agent is in a think spiral.

        Returns a forced-action prompt if threshold is hit, else None.
        """
        output_lower = output.lower()
        is_think_only = (
            "i think" in output_lower
            or "let me consider" in output_lower
            or "perhaps" in output_lower
        ) and len(output) < 200  # Short vague output = probably spiraling

        if is_think_only:
            self.consecutive_thinks += 1
        else:
            self.consecutive_thinks = 0

        if self.consecutive_thinks >= self.THRESHOLD:
            self.consecutive_thinks = 0
            self.total_blocked += 1
            logger.warning(
                f"LoopBlocker: broke think spiral in {agent_name} (total: {self.total_blocked})"
            )
            return (
                "\nYou've been thinking in circles. Stop analyzing. "
                "Take ONE concrete action right now — write code, run a command, "
                "or produce a specific output. No more deliberation."
            )
        return None


class OscillationDetector:
    """Detects when gap reports are converging (same gaps repeating).

    If two consecutive gap summaries have >85% word overlap,
    the system is oscillating and should break the loop.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self._hashes: list[str] = []

    def check(self, gap_summary: str) -> bool:
        """Returns True if oscillation is detected."""
        h = hashlib.md5(gap_summary.encode()).hexdigest()

        # Exact duplicate
        if h in self._hashes[-3:]:
            logger.warning("OscillationDetector: exact duplicate gap summary")
            return True

        self._hashes.append(h)
        return False

    def check_word_overlap(self, summaries: list[str]) -> bool:
        """Check word overlap between last two summaries."""
        if len(summaries) < 2:
            return False
        last = set(summaries[-1].lower().split())
        prev = set(summaries[-2].lower().split())
        union = last | prev
        if not union:
            return False
        overlap = len(last & prev) / len(union)
        if overlap > self.threshold:
            logger.warning(f"OscillationDetector: {overlap:.2%} word overlap — oscillating")
            return True
        return False


class ActionDeduplicator:
    """Prevents the same action from being taken twice.

    Tracks action hashes. If the same tool + args combination
    appears twice, it's a wasted call.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, action_type: str, action_args: str) -> bool:
        key = hashlib.md5(f"{action_type}:{action_args}".encode()).hexdigest()
        if key in self._seen:
            logger.debug(f"ActionDedup: blocked duplicate {action_type}")
            return True
        self._seen.add(key)
        return False

    def reset(self) -> None:
        self._seen.clear()
