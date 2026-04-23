"""IncompletenessLoop — the Latios engine.

After every action, asks: what did this fail to account for?
The remainder becomes the seed for the next action.

Source: latios_latias_v1.py lines 385-461
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("belief.polarity.incompleteness")


class Remainder(BaseModel):
    """A single incompleteness observation."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    action: str = ""
    result_summary: str = ""
    remainder: str = ""
    confidence: float = 0.5


class IncompletenessLoop:
    """Latios — the gap-finder.

    After each action:
    1. What did I just do?
    2. What did it produce?
    3. What did it FAIL to account for? (the remainder)
    4. That remainder becomes the seed for the next action.

    Uses LLM for extraction but falls back to heuristics if unavailable.
    """

    def __init__(self, max_remainders: int = 100) -> None:
        self.remainders: list[Remainder] = []
        self.max_remainders = max_remainders

    async def extract_remainder(
        self,
        action: str,
        result: str,
        goal: str,
        llm=None,
        role: str = "latios",
    ) -> str:
        """Extract what the action failed to account for.

        The 0.000...001 — the infinitesimal gap that drives the next operation.
        """
        if llm:
            try:
                prompt = (
                    f"ACTION: {action[:200]}\n"
                    f"RESULT: {result[:400]}\n"
                    f"GOAL: {goal[:100]}\n\n"
                    "What ONE gap did this action reveal? What did it fail to account for?\n"
                    "ONE sentence, under 30 words. Be specific, not vague."
                )
                remainder_text = await llm.generate_text(
                    role=role,
                    system="You are the incompleteness engine. Find the gap. One sentence.",
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=100,
                )
            except Exception as e:
                logger.debug(f"LLM remainder extraction failed: {e}")
                remainder_text = self._heuristic_remainder(action, result, goal)
        else:
            remainder_text = self._heuristic_remainder(action, result, goal)

        remainder = Remainder(
            action=action[:200],
            result_summary=result[:200],
            remainder=remainder_text.strip(),
        )
        self.remainders.append(remainder)

        # Prune to max
        if len(self.remainders) > self.max_remainders:
            self.remainders = self.remainders[-self.max_remainders :]

        logger.debug(f"Remainder: {remainder_text[:80]}")
        return remainder_text

    def _heuristic_remainder(self, action: str, result: str, goal: str) -> str:
        """Zero-cost fallback — keyword-based gap detection."""
        result_lower = result.lower()
        if "error" in result_lower or "failed" in result_lower:
            return f"Action '{action[:50]}' produced errors that need resolution."
        if "todo" in result_lower or "placeholder" in result_lower:
            return f"Action '{action[:50]}' left placeholders that need implementation."
        if len(result) < 50:
            return f"Action '{action[:50]}' produced minimal output — may be incomplete."
        return f"Action '{action[:50]}' completed but alignment with goal needs verification."

    def get_recent(self, n: int = 5) -> list[str]:
        """Get the last N remainders as strings."""
        return [r.remainder for r in self.remainders[-n:]]

    def get_accumulated_context(self, n: int = 5) -> str:
        """Format recent remainders for injection into agent prompts."""
        recent = self.get_recent(n)
        if not recent:
            return ""
        return "## ACCUMULATED GAPS (from incompleteness loop)\n" + "\n".join(
            f"- {r}" for r in recent
        )

    @property
    def current(self) -> Optional[str]:
        """The most recent remainder."""
        return self.remainders[-1].remainder if self.remainders else None
