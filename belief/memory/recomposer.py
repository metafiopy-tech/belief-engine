"""
Recomposer — pre-build nutrient retrieval for the Metabolization Architecture.

Runs before intake as the first node in the pipeline. Queries ChromaDB soil
for relevant nutrients based on the user's goal, formats them into a context
block, and injects it into state.nutrient_context for the architect.

No LLM call needed — this is pure retrieval + formatting. The heavy lifting
is done by Soil.retrieve_profile() and NutrientProfile.format_context_block().

The recomposer never fails the build — if soil is empty or retrieval fails,
it produces an empty context block and the build proceeds normally.

Source: METABOLIZATION_BUILD_PLAN.md Phase 4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.memory.recomposer")

# ── Soil accessor (shared with decomposer) ───────────────────────────────────

_soil_instance = None


def _get_soil():
    """Lazy-load the global Soil instance."""
    global _soil_instance
    if _soil_instance is None:
        from belief.memory.soil import Soil
        soil_dir = Path("~/.belief-engine/soil").expanduser()
        _soil_instance = Soil(soil_dir)
    return _soil_instance


# ── The node ─────────────────────────────────────────────────────────────────

async def recomposer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pre-build LangGraph node: retrieve nutrients and enrich architect context.

    Runs as the entry point of the pipeline, before intake. Queries ChromaDB
    soil with the user's goal and formats relevant nutrients into a context
    block that gets injected into the architect's prompt.

    If soil is empty or retrieval fails, the build proceeds with empty
    nutrient context — the recomposer never blocks the pipeline.

    Wiring (in graph.py):
        graph.add_node("recomposer", recomposer_node)
        graph.set_entry_point("recomposer")
        graph.add_edge("recomposer", "intake")
    """
    result = dict(state)

    try:
        goal = state.get("user_goal", "")
        if not goal:
            logger.debug("Recomposer: no user_goal in state, skipping")
            result["nutrient_context"] = ""
            result["nutrient_profile"] = None
            return result

        soil = _get_soil()

        # Run decay maintenance on startup (cleans stale nutrients)
        decay_result = soil.decay_all()
        if decay_result["archived"] > 0:
            logger.info(
                f"Soil maintenance: archived {decay_result['archived']} stale nutrients, "
                f"{decay_result['active']} active"
            )

        # Detect complexity for token budget
        complexity = state.get("complexity_score", 3)

        # Retrieve nutrient profile
        profile = soil.retrieve_profile(goal, complexity=complexity)

        if profile.is_empty:
            logger.info("Recomposer: soil is empty — fresh build, no institutional memory")
            result["nutrient_context"] = ""
            result["nutrient_profile"] = None
            return result

        # Format context block within token budget
        context_block = profile.format_context_block(complexity=complexity)

        # Serialize profile summary for state (lightweight dict, not full Nutrient objects)
        profile_summary = {
            "covenants": len(profile.covenants),
            "antipatterns": len(profile.antipatterns),
            "patterns": len(profile.patterns),
            "skeletons": len(profile.skeletons),
            "total": profile.total_nutrients,
            "context_chars": len(context_block),
        }

        result["nutrient_context"] = context_block
        result["nutrient_profile"] = profile_summary

        logger.info(
            f"Recomposer: injected {profile.total_nutrients} nutrients "
            f"(cov={len(profile.covenants)}, anti={len(profile.antipatterns)}, "
            f"pat={len(profile.patterns)}, skel={len(profile.skeletons)}, "
            f"{len(context_block)} chars)"
        )

    except Exception as e:
        logger.warning(f"Recomposer: failed (non-fatal): {e}")
        result["nutrient_context"] = ""
        result["nutrient_profile"] = None

    return result
