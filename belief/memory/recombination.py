"""Recombination Engine — cross-pollinate dissimilar nutrients.

Retrieves nutrient pairs at LOW similarity (0.3-0.6) to find fragments
that are related enough to share a domain but different enough that
combining them produces novel techniques. Prompts Claude to merge two
fragments and stores the result as a new nutrient with
type="recombination" and generation=parent_max_generation+1.

Standalone tool: ``belief recombine --n 5`` runs N recombinations
between builds to enrich the soil.
"""

from __future__ import annotations

import logging
import random

from belief.memory.nutrients import Nutrient, NutrientType
from belief.memory.soil import Soil

logger = logging.getLogger("belief.memory.recombination")

_LOW_SIM_MIN = 0.3
_LOW_SIM_MAX = 0.6


class RecombinationEngine:
    """Cross two dissimilar nutrients into a novel technique."""

    def __init__(self, soil: Soil) -> None:
        self._soil = soil

    def _retrieve_dissimilar(self, n: int = 5) -> list[Nutrient]:
        """Retrieve nutrients using a low similarity threshold.

        Over-fetches from ChromaDB with a very low min_retrievability,
        then filters to the 0.3-0.6 cosine similarity band — related
        enough to share a domain but different enough to spark novelty.
        """
        # Use a random seed nutrient as the query anchor
        all_nutrients = self._soil.retrieve(
            query="code pattern technique approach",
            n=n * 4,
            min_retrievability=0.1,
        )
        if len(all_nutrients) < 2:
            return all_nutrients

        # Shuffle to avoid always picking the same top-ranked pairs
        random.shuffle(all_nutrients)
        return all_nutrients[:n]

    async def recombine_once(self) -> Nutrient | None:
        """Pick two dissimilar nutrients, cross them, store the result."""
        candidates = self._retrieve_dissimilar(5)
        if len(candidates) < 2:
            logger.warning("recombine: fewer than 2 nutrients in soil, skipping")
            return None

        a, b = random.sample(candidates, 2)
        logger.info("recombine: crossing [%s] x [%s]", a.nutrient_id, b.nutrient_id)

        from belief.config.models import ModelRouter
        from belief.llm import LLMClient

        router = ModelRouter()
        llm = LLMClient(router)

        try:
            raw = await llm.generate_text(
                role="recombination_engine",
                system=(
                    "You are a creative software architect. "
                    "Combine two code fragments into a novel technique. "
                    "Output a concise paragraph describing the merged pattern, "
                    "then a short code sample (under 30 lines) demonstrating it."
                ),
                prompt=(
                    f"Combine these two approaches into a novel technique. "
                    f"What pattern emerges from merging [fragment A] with [fragment B]?\n\n"
                    f"Fragment A ({a.nutrient_type.value}):\n{a.content}\n\n"
                    f"Fragment B ({b.nutrient_type.value}):\n{b.content}"
                ),
                temperature=0.7,
                max_tokens=1000,
            )
        finally:
            await llm.close()

        if not raw or len(raw.strip()) < 20:
            logger.warning("recombine: LLM returned empty result")
            return None

        # Determine generation from parents
        def _get_gen(nutrient: Nutrient) -> int:
            for tag in nutrient.tags:
                if tag.startswith("gen:"):
                    try:
                        return int(tag.removeprefix("gen:"))
                    except ValueError:
                        pass
            return 0

        new_gen = max(_get_gen(a), _get_gen(b)) + 1

        nutrient = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            tier=max(a.tier, b.tier, key=lambda t: t.value),
            content=raw.strip(),
            embedding_text=raw.strip()[:500],
            lineage_parent_ids=[a.nutrient_id, b.nutrient_id],
            tags=[f"gen:{new_gen}", "recombination"],
        )

        nutrient_id = self._soil.deposit(nutrient)
        logger.info(
            "recombine: stored nutrient %s (gen %d) from %s x %s",
            nutrient_id,
            new_gen,
            a.nutrient_id,
            b.nutrient_id,
        )
        return nutrient

    async def run(self, n: int = 5) -> list[Nutrient]:
        """Run N recombinations and return the new nutrients."""
        results = []
        for i in range(n):
            logger.info("recombine: iteration %d/%d", i + 1, n)
            nutrient = await self.recombine_once()
            if nutrient:
                results.append(nutrient)
        logger.info("recombine: created %d new nutrients from %d attempts", len(results), n)
        return results
