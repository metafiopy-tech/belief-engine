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

import copy
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("belief.memory.recomposer")


# ---------------------------------------------------------------------------
# Validation-phase Session 1: local-mode prompt compression
# ---------------------------------------------------------------------------

# Per-nutrient character cap under local mode. 500 chars ≈ 125 tokens —
# enough to carry the rule/pattern body, short enough that 3 patterns +
# 3 antipatterns fit inside the 3 KB total budget.
LOCAL_NUTRIENT_CHAR_CAP = 500

# Total nutrient-context-injection cap when running against a local model.
# Target is ~750 tokens — leaves headroom inside an 8K context window
# for the actual goal, code, and generation.
LOCAL_TOTAL_CONTEXT_CHAR_CAP = 3000

# Tag / content substrings that mark a nutrient as "human-documentation"
# rather than "model-actionable guidance". Dropped in local mode — a
# 14B model doesn't benefit from README prose and deployment YAML in
# its context window.
_LOCAL_DROP_TAG_SUBSTRINGS = (
    "readme",
    "changelog",
    "docs",
    "documentation",
    "deploy",
    "deployment",
    "dockerfile",
    "compose",
)


def _is_local_mode(state: dict[str, Any]) -> bool:
    """True iff the current build is routing through a local model.

    Checks both ``state['model_mode']`` (set by higher-level callers)
    and the ``BELIEF_MODEL_MODE`` env var (matches how
    :class:`~belief.config.models.ModelRouter` reads its own mode).
    Either saying 'local' triggers compression; anything else keeps
    the cloud-path behaviour intact.
    """
    mode = str(state.get("model_mode") or "").strip().lower()
    if mode == "local":
        return True
    env_mode = os.environ.get("BELIEF_MODEL_MODE", "").strip().lower()
    return env_mode == "local"


def _looks_like_human_docs(nutrient) -> bool:
    """Flag nutrients that are docs-for-humans rather than guidance."""
    tags = [str(t).lower() for t in (getattr(nutrient, "tags", []) or [])]
    framework = (getattr(nutrient, "framework", "") or "").lower()
    blob = " ".join(tags + [framework])
    return any(marker in blob for marker in _LOCAL_DROP_TAG_SUBSTRINGS)


def _truncate_nutrient(nutrient, cap: int = LOCAL_NUTRIENT_CHAR_CAP):
    """Return a shallow copy of ``nutrient`` with its content clipped.

    Pydantic models (the real Nutrient class) expose ``model_copy`` /
    ``copy`` — fall back to ``copy.copy`` + attribute assignment for
    the simple stand-ins used in tests.
    """
    content = getattr(nutrient, "content", None)
    if not isinstance(content, str) or len(content) <= cap:
        return nutrient
    clipped = content[: cap - 1].rstrip() + "…"
    try:
        # Pydantic BaseModel path — preserves validation
        return nutrient.model_copy(update={"content": clipped})
    except AttributeError:
        pass
    try:
        dup = copy.copy(nutrient)
        dup.content = clipped
        return dup
    except Exception:
        return nutrient


def _compress_for_local(profile, drop_human_docs: bool = True):
    """Apply local-mode trimming to a NutrientProfile in place-safe way.

    Drops human-documentation nutrients, truncates each survivor's
    content, and leans on the profile's existing ``compact()`` to cap
    per-category counts (top-3 patterns / antipatterns, 1 skeleton).
    Returns a new profile; the input is not mutated.
    """

    def _filter(nutrients):
        if not nutrients:
            return []
        if drop_human_docs:
            kept = [n for n in nutrients if not _looks_like_human_docs(n)]
        else:
            kept = list(nutrients)
        return [_truncate_nutrient(n) for n in kept]

    # compact() caps per-category counts. We call it AFTER filtering
    # so human-docs drop-out doesn't reduce the effective budget for
    # high-value guidance nutrients.
    from belief.memory.nutrients import NutrientProfile

    filtered = NutrientProfile(
        covenants=_filter(profile.covenants),
        antipatterns=_filter(profile.antipatterns),
        patterns=_filter(profile.patterns),
        skeletons=_filter(profile.skeletons),
    )
    return filtered.compact()


def _hard_cap_context(block: str, cap: int = LOCAL_TOTAL_CONTEXT_CHAR_CAP) -> str:
    """Final safety net: clip the rendered block if something outgrew it."""
    if len(block) <= cap:
        return block
    truncated = block[: cap - 80].rstrip()
    return truncated + "\n# ... (local-mode context truncated)\n"


# ---------------------------------------------------------------------------
# Session 7: domain-aware re-ranking
# ---------------------------------------------------------------------------


def _nutrient_domain_score(nutrient, target_domain: str) -> int:
    """Rank a nutrient against the current build's domain.

    Score ladder:
      3  same domain as the build
      2  "general" domain  (neutral prior — safe to promote)
      1  any other explicit domain
      0  no domain information at all (least-specific)
    """
    from belief.evolution.progression import (
        GENERAL_DOMAIN,
        detect_domain,
    )

    tags = [str(t).lower() for t in (getattr(nutrient, "tags", []) or [])]
    # A nutrient may encode its domain directly as a tag or via framework
    framework = (getattr(nutrient, "framework", "") or "").lower()
    content = (getattr(nutrient, "content", "") or "").lower()
    blob = " ".join([framework] + tags)
    if not blob.strip() and not content:
        return 0
    nutrient_domain = detect_domain(blob or content, tags=tags)
    if nutrient_domain == target_domain:
        return 3
    if nutrient_domain == GENERAL_DOMAIN:
        return 2
    return 1


def reorder_by_domain(nutrients: list, target_domain: str) -> list:
    """Stable re-sort: same-domain first, general next, others after.

    Returns a new list; input is not mutated. When target_domain is
    "general" the input list is returned unchanged (no preference).
    """
    from belief.evolution.progression import GENERAL_DOMAIN

    if target_domain == GENERAL_DOMAIN or not nutrients:
        return list(nutrients)
    scored = [(_nutrient_domain_score(n, target_domain), i, n) for i, n in enumerate(nutrients)]
    # Higher score first; stable by original index on ties
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [n for _, _, n in scored]


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

        # Session 7: domain-aware re-ranking. Detect the current build's
        # domain from the goal, then reorder each nutrient list so
        # same-domain > general > other. This keeps e.g. FastAPI patterns
        # from being crowded out by CLI nutrients on a CRUD build.
        from belief.evolution.progression import detect_domain

        build_domain = detect_domain(goal, tags=state.get("tags") or [])
        profile.covenants = reorder_by_domain(profile.covenants, build_domain)
        profile.antipatterns = reorder_by_domain(profile.antipatterns, build_domain)
        profile.patterns = reorder_by_domain(profile.patterns, build_domain)
        profile.skeletons = reorder_by_domain(profile.skeletons, build_domain)

        if profile.is_empty:
            logger.info("Recomposer: all nutrients filtered by FSRS — proceeding without context")
            result["nutrient_context"] = ""
            result["nutrient_profile"] = None
            return result

        # Validation-phase Session 1: compress aggressively in local mode.
        # - drop human-documentation nutrients
        # - truncate each surviving nutrient to 500 chars
        # - per-category compact (top-3 patterns/antipatterns, 1 skeleton)
        # - render with the compact formatter
        # - hard-cap the final block at LOCAL_TOTAL_CONTEXT_CHAR_CAP
        if _is_local_mode(state):
            profile = _compress_for_local(profile)
            if profile.is_empty:
                logger.info("Recomposer: compressed profile is empty — proceeding without context")
                result["nutrient_context"] = ""
                result["nutrient_profile"] = None
                return result
            context_block = _hard_cap_context(
                profile.format_context_block_compact(complexity=complexity)
            )
        else:
            # Format context block within token budget (cloud-mode unchanged)
            context_block = profile.format_context_block(complexity=complexity)

        # Serialize profile summary for state (lightweight dict, not full Nutrient objects)
        profile_summary = {
            "covenants": len(profile.covenants),
            "antipatterns": len(profile.antipatterns),
            "patterns": len(profile.patterns),
            "skeletons": len(profile.skeletons),
            "total": profile.total_nutrients,
            "context_chars": len(context_block),
            "domain": build_domain,
        }

        result["nutrient_context"] = context_block
        result["nutrient_profile"] = profile_summary

        logger.info(
            f"Recomposer: injected {profile.total_nutrients} nutrients "
            f"(cov={len(profile.covenants)}, anti={len(profile.antipatterns)}, "
            f"pat={len(profile.patterns)}, skel={len(profile.skeletons)}, "
            f"{len(context_block)} chars)"
        )

        local_mode = _is_local_mode(state)

        # Retrieve self-authored tools relevant to this goal
        try:
            from belief.memory.tool_registry import ToolRegistry

            tool_registry = ToolRegistry(soil)
            # Local mode: cap at 1 tool — 14B models rarely benefit from a menu.
            k_tools = 1 if local_mode else 3
            relevant_tools = tool_registry.find_tools_for_goal(goal, k=k_tools)
            if relevant_tools:
                tool_block = "\n\n## AVAILABLE TOOLS (self-authored validators)\n"
                for t in relevant_tools:
                    tool_block += (
                        f"- **{t.name}**: {t.description} "
                        f"(quality={t.quality_score:.1f}, uses={t.use_count})\n"
                    )
                    if t.input_description:
                        tool_block += f"  Input: {t.input_description}\n"
                candidate = context_block + tool_block
                # In local mode, only accept the addition if it still fits.
                if local_mode and len(candidate) > LOCAL_TOTAL_CONTEXT_CHAR_CAP:
                    logger.info(
                        "Recomposer: dropping self-authored tool block in local mode "
                        "(would overflow 3KB cap)"
                    )
                else:
                    result["nutrient_context"] = candidate
                    result.setdefault("available_tools", [])
                    result["available_tools"] = [
                        {"id": t.id, "name": t.name} for t in relevant_tools
                    ]
                    logger.info(
                        f"Recomposer: added {len(relevant_tools)} self-authored tools to context"
                    )

                # Mycorrhizal Stage 2: each surfaced tool counts as a
                # reference to the niche that registered it. The semantic
                # is "this niche was visible to this build" — a slightly
                # weaker signal than "definitely consumed" but the cleanest
                # chokepoint that has build_id in hand. Idempotency-keyed
                # on (niche_id, run_id) so a recomposer replay for the
                # same build never double-credits. Best-effort.
                try:
                    from belief.memory.niche_ledger import get_default_ledger

                    nl = get_default_ledger()
                    run_id = state.get("run_id") or "unknown"
                    for t in relevant_tools:
                        rec = nl.lookup_by_soil_reference(kind="tool", soil_reference=t.id)
                        if rec is not None:
                            nl.record_reference(
                                niche_id=rec.niche_id,
                                referring_build_id=str(run_id),
                            )
                except Exception as nl_err:  # pragma: no cover — best-effort
                    logger.debug(f"Niche-reference record skipped: {nl_err}")
        except Exception as e:
            logger.debug(f"Tool retrieval skipped: {e}")

        # Retrieve past reflexions for similar goals
        try:
            from belief.memory.reflexion import retrieve_reflexions

            reflexions = retrieve_reflexions(goal, n=3)
            if reflexions:
                reflexion_block = "\n\n## LESSONS FROM PAST FAILURES\n" + "\n".join(
                    f"- {r.split('Reflection: ')[-1][:200]}"
                    if "Reflection: " in r
                    else f"- {r[:200]}"
                    for r in reflexions
                )
                current_block = result.get("nutrient_context", context_block)
                candidate = current_block + reflexion_block
                if local_mode and len(candidate) > LOCAL_TOTAL_CONTEXT_CHAR_CAP:
                    logger.info(
                        "Recomposer: dropping reflexion block in local mode "
                        "(would overflow 3KB cap)"
                    )
                else:
                    result["nutrient_context"] = candidate
                    logger.info(f"Recomposer: added {len(reflexions)} reflexions to context")
        except Exception as e:
            logger.debug(f"Reflexion retrieval skipped: {e}")

    except Exception as e:
        logger.warning(f"Recomposer: failed (non-fatal): {e}")
        result["nutrient_context"] = ""
        result["nutrient_profile"] = None

    return result
