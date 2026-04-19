"""
Recomposer — pre-build nutrient retrieval for the Metabolization Architecture.

Runs before intake as the first node in the pipeline. Queries ChromaDB soil
for relevant nutrients based on the user's goal, formats them into a context
block, and injects it into state.nutrient_context for the architect.

Queries principles + tools + covenants collections and applies FSRS-based
filtering: only retrieve records where ``next_review <= now`` OR
``decay_state != "lapsed"``.

No LLM call needed — this is pure retrieval + formatting. The heavy lifting
is done by Soil.retrieve_profile() and NutrientProfile.format_context_block().

The recomposer never fails the build — if soil is empty or retrieval fails,
it produces an empty context block and the build proceeds normally.

Source: METABOLIZATION_BUILD_PLAN.md Phase 4
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from belief.memory.nutrients import NutrientType

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


def _fsrs_filter(nutrients: list, now_ts: float) -> list:
    """Filter out lapsed nutrients unless they are due for review.

    Keeps nutrients where:
      - decay_state != "lapsed", OR
      - fsrs_next_review <= now (due for review, so worth trying again)
    """
    filtered = []
    for n in nutrients:
        meta = n.to_chromadb_metadata() if hasattr(n, "to_chromadb_metadata") else {}
        decay_state = meta.get("fsrs_decay_state", "new")
        next_review = meta.get("fsrs_next_review", 0.0)

        if decay_state != "lapsed" or next_review <= now_ts:
            filtered.append(n)
    return filtered


# ── The node ─────────────────────────────────────────────────────────────

async def recomposer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pre-build LangGraph node: retrieve nutrients and enrich architect context.

    Runs as the entry point of the pipeline, before intake. Queries
    principles, tools, and covenants collections with the user's goal and
    formats relevant nutrients into a context block for the architect.

    FSRS filtering: excludes lapsed nutrients unless they are due for re-review.

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

        # Retrieve nutrient profile (queries principles + tools + covenants + failures)
        profile = soil.retrieve_profile(goal, complexity=complexity)

        if profile.is_empty:
            logger.info("Recomposer: soil is empty — fresh build, no institutional memory")
            result["nutrient_context"] = ""
            result["nutrient_profile"] = None
            return result

        # Apply FSRS-based filtering: exclude lapsed nutrients unless due for review
        now_ts = datetime.now(timezone.utc).timestamp()
        profile.covenants = _fsrs_filter(profile.covenants, now_ts)
        profile.antipatterns = _fsrs_filter(profile.antipatterns, now_ts)
        profile.patterns = _fsrs_filter(profile.patterns, now_ts)
        profile.skeletons = _fsrs_filter(profile.skeletons, now_ts)

        if profile.is_empty:
            logger.info("Recomposer: all nutrients filtered by FSRS — proceeding without context")
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

        # Retrieve self-authored tools relevant to this goal
        try:
            from belief.memory.tool_registry import ToolRegistry
            tool_registry = ToolRegistry(soil)
            relevant_tools = tool_registry.find_tools_for_goal(goal, k=3)
            if relevant_tools:
                tool_block = "\n\n## AVAILABLE TOOLS (self-authored validators)\n"
                for t in relevant_tools:
                    tool_block += (
                        f"- **{t.name}**: {t.description} "
                        f"(quality={t.quality_score:.1f}, uses={t.use_count})\n"
                    )
                    if t.input_description:
                        tool_block += f"  Input: {t.input_description}\n"
                result["nutrient_context"] = context_block + tool_block
                result.setdefault("available_tools", [])
                result["available_tools"] = [
                    {"id": t.id, "name": t.name} for t in relevant_tools
                ]
                logger.info(f"Recomposer: added {len(relevant_tools)} self-authored tools to context")
        except Exception as e:
            logger.debug(f"Tool retrieval skipped: {e}")

        # Retrieve past reflexions for similar goals
        try:
            from belief.memory.reflexion import retrieve_reflexions
            reflexions = retrieve_reflexions(goal, n=3)
            if reflexions:
                reflexion_block = "\n\n## LESSONS FROM PAST FAILURES\n" + "\n".join(
                    f"- {r.split('Reflection: ')[-1][:200]}" if 'Reflection: ' in r else f"- {r[:200]}"
                    for r in reflexions
                )
                result["nutrient_context"] = result.get("nutrient_context", context_block) + reflexion_block
                logger.info(f"Recomposer: added {len(reflexions)} reflexions to context")
        except Exception as e:
            logger.debug(f"Reflexion retrieval skipped: {e}")

    except Exception as e:
        logger.warning(f"Recomposer: failed (non-fatal): {e}")
        result["nutrient_context"] = ""
        result["nutrient_profile"] = None

    return result
