"""Reflexion — Verbal Self-Critique After Failed Builds.

Based on Reflexion (Shinn et al., NeurIPS 2023): instead of just storing
pass/fail outcomes, generate a natural language self-reflection explaining
WHAT went wrong and WHY. This reflection is stored in ChromaDB and retrieved
as context for future similar builds.

Key insight from the paper: raw test output ("AssertionError: expected 5, got 6")
is insufficient. The model needs SEMANTIC interpretation of what went wrong.
Without reflection, Reflexion's ablation showed ZERO improvement.

Usage:
    from belief.memory.reflexion import store_reflexion, retrieve_reflexions

    # After a failed build:
    await store_reflexion(goal, validation_result, code_files, llm)

    # Before a new build:
    reflections = retrieve_reflexions(goal, n=3)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("belief.memory.reflexion")


async def generate_reflexion(
    goal: str,
    validation_result: dict,
    code_files: dict[str, str],
    errors: list[str],
    llm=None,
) -> str | None:
    """Generate a verbal self-reflection on a failed build.

    Returns a natural language critique: what went wrong, why, and what
    to do differently next time. This is stored in ChromaDB for retrieval
    on future similar builds.
    """
    if llm is None:
        return None

    verdict = validation_result.get("verdict", "unknown") if isinstance(validation_result, dict) else getattr(validation_result, "verdict", "unknown")
    tests_passed = validation_result.get("tests_passed", 0) if isinstance(validation_result, dict) else getattr(validation_result, "tests_passed", 0)
    tests_total = validation_result.get("tests_total", 0) if isinstance(validation_result, dict) else getattr(validation_result, "tests_total", 0)
    summary = validation_result.get("summary", "") if isinstance(validation_result, dict) else getattr(validation_result, "summary", "")

    # Build a compact view of what was generated
    file_list = ", ".join(sorted(code_files.keys())[:10])
    error_summary = "; ".join(errors[:3]) if errors else "none"

    prompt = f"""A code generation build just FAILED. Reflect on what went wrong.

GOAL: {goal}

RESULT: {verdict} — {tests_passed}/{tests_total} tests passed
ERRORS: {error_summary}
FILES GENERATED: {file_list}
VALIDATION SUMMARY: {summary[:500]}

Write a 2-3 sentence reflection covering:
1. The root cause of the failure (not just the symptom)
2. What should be done differently next time for a similar goal
3. Any pattern or anti-pattern to remember

Be specific and actionable. Do NOT repeat the error message — interpret it."""

    try:
        reflection = await llm.generate_text(
            role="latios",  # Use Haiku — this is a mechanical task
            system="You are a build failure analyst. Write concise, actionable reflections.",
            prompt=prompt,
            temperature=0.2,
            max_tokens=300,
        )
        logger.info(f"Reflexion: generated {len(reflection)} chars")
        return reflection.strip()
    except Exception as e:
        logger.debug(f"Reflexion generation failed: {e}")
        return None


async def store_reflexion(
    goal: str,
    reflection: str,
    verdict: str,
    tests_passed: int,
    tests_total: int,
) -> bool:
    """Store a reflexion in ChromaDB soil as an antipattern nutrient."""
    try:
        from belief.memory.soil import Soil
        from belief.memory.nutrients import NutrientType

        soil_path = Path("~/.belief-engine/soil").expanduser()
        soil = Soil(soil_path)

        content = f"[REFLEXION] Goal: {goal}\nOutcome: {verdict} ({tests_passed}/{tests_total})\nReflection: {reflection}"
        soil.deposit(
            content=content,
            nutrient_type=NutrientType.ANTIPATTERN,
            source=f"reflexion:{verdict}",
            tags=["reflexion", verdict, f"pass_rate:{tests_passed}/{tests_total}"],
        )
        logger.info(f"Reflexion stored in soil: {reflection[:80]}...")
        return True
    except Exception as e:
        logger.debug(f"Reflexion storage failed: {e}")
        return False


def retrieve_reflexions(goal: str, n: int = 3) -> list[str]:
    """Retrieve relevant past reflexions for a similar goal."""
    try:
        from belief.memory.soil import Soil
        from belief.memory.nutrients import NutrientType

        soil_path = Path("~/.belief-engine/soil").expanduser()
        soil = Soil(soil_path)

        results = soil.retrieve(
            query=goal,
            nutrient_type=NutrientType.ANTIPATTERN,
            n=n,
        )
        reflexions = [r.content for r in results if "[REFLEXION]" in r.content]
        if reflexions:
            logger.info(f"Retrieved {len(reflexions)} past reflexions for context")
        return reflexions
    except Exception:
        return []
