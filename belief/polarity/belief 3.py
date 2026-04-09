"""BeliefLoop — the Latias engine.

After every action, asks: what did this cause me to want to PROTECT?
The covenant becomes the anchor that prevents goal drift.

Source: belief_loop.py
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("belief.polarity.belief")


class Covenant(BaseModel):
    """A single belief observation — what's worth protecting."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    action: str = ""
    result_summary: str = ""
    covenant: str = ""
    intensity: float = 0.5


class BeliefLoop:
    """Latias — the covenant-holder.

    After each action:
    1. What did I just do?
    2. What did it produce?
    3. What did it cause me to want to PROTECT or return to?
    4. That covenant becomes the anchor for the next action.
    """

    def __init__(self, max_covenants: int = 100) -> None:
        self.covenants: list[Covenant] = []
        self.max_covenants = max_covenants

    async def extract_covenant(
        self, action: str, result: str, goal: str,
        llm=None, role: str = "latios",
    ) -> str:
        """Extract what this action revealed as worth protecting."""
        if llm:
            try:
                prompt = (
                    f"ACTION: {action[:200]}\n"
                    f"RESULT: {result[:400]}\n"
                    f"GOAL: {goal[:100]}\n\n"
                    "What did this action cause you to want to PROTECT or STAND FOR?\n"
                    "What quality or aspect of the result is worth preserving?\n"
                    "ONE sentence, under 30 words."
                )
                covenant_text = await llm.generate_text(
                    role=role,
                    system="You are the belief engine. Find what's worth protecting. One sentence.",
                    prompt=prompt,
                    temperature=0.4,
                    max_tokens=100,
                )
            except Exception as e:
                logger.debug(f"LLM covenant extraction failed: {e}")
                covenant_text = self._heuristic_covenant(action, result)
        else:
            covenant_text = self._heuristic_covenant(action, result)

        covenant = Covenant(
            action=action[:200],
            result_summary=result[:200],
            covenant=covenant_text.strip(),
        )
        self.covenants.append(covenant)

        if len(self.covenants) > self.max_covenants:
            self.covenants = self.covenants[-self.max_covenants:]

        logger.debug(f"Covenant: {covenant_text[:80]}")
        return covenant_text

    def _heuristic_covenant(self, action: str, result: str) -> str:
        """Zero-cost fallback."""
        if len(result) > 200:
            return f"The substantive output from '{action[:50]}' should be preserved."
        return f"The intent behind '{action[:50]}' is worth protecting."

    def get_deepest_covenant(self) -> Optional[str]:
        """What has she returned to most? Word frequency over last 10 covenants.

        Zero LLM cost — pure word frequency analysis.
        """
        if not self.covenants:
            return None

        recent = [c.covenant for c in self.covenants[-10:]]
        words = []
        stop_words = {"the", "a", "an", "is", "in", "it", "to", "of", "and", "or",
                      "for", "with", "on", "at", "by", "this", "that", "from", "was"}
        for text in recent:
            for word in text.lower().split():
                cleaned = "".join(c for c in word if c.isalpha())
                if cleaned and len(cleaned) > 2 and cleaned not in stop_words:
                    words.append(cleaned)

        if not words:
            return recent[-1] if recent else None

        most_common = Counter(words).most_common(3)
        theme = " ".join(w for w, _ in most_common)

        # Find the covenant that best matches the theme
        best_match = recent[-1]
        best_score = 0
        for cov in recent:
            score = sum(1 for w, _ in most_common if w in cov.lower())
            if score > best_score:
                best_score = score
                best_match = cov

        return best_match

    def get_recent(self, n: int = 5) -> list[str]:
        return [c.covenant for c in self.covenants[-n:]]

    def get_accumulated_context(self, n: int = 5) -> str:
        recent = self.get_recent(n)
        if not recent:
            return ""
        deepest = self.get_deepest_covenant()
        lines = ["## COVENANTS (what's worth protecting)"]
        if deepest:
            lines.append(f"Deepest: {deepest}")
        lines.extend(f"- {c}" for c in recent)
        return "\n".join(lines)

    @property
    def current(self) -> Optional[str]:
        return self.covenants[-1].covenant if self.covenants else None
