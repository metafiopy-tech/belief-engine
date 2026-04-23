"""ZPD-fit difficulty estimator.

Zone of Proximal Development: a build goal is "at the right level" when
it stretches the agent modestly beyond its current repertoire. Signals
that are far too easy (high skill_coverage, low pred_time) waste
compute; signals far too hard (low coverage, high pred_time) burn
budget on almost-certain failures.

Model (POET minimal criterion):

    skill_coverage   = hits_above_0.6_cos / estimated_skills_needed
    zpd_fit          = gaussian(pred_time, target_time, sigma)
                       * clamp(skill_coverage, 0.2, 0.9)
    target_time      = 20 + 2 * floor(successful_builds / 10)
    sigma            = 15

Reject if skill_coverage outside [0.2, 0.9].

The LLM time estimate is issued via a deterministic Haiku call with
`max_tokens=150`. Session 5 will persist every call to the audit log
and honor the CostTracker cap. For Session 4, callers may pass
`llm_time_estimator=None` to fall back on a heuristic — useful in tests.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


from belief.photosynthesis.synthesis.prompts import DIFFICULTY_PROMPT

logger = logging.getLogger("belief.photosynthesis.synthesis.difficulty")


HAIKU_MODEL = "claude-haiku-4-5-20251001"
SIGMA = 15.0

# POET minimal criterion bounds
MIN_COVERAGE = 0.2
MAX_COVERAGE = 0.9

# Heuristic default if no LLM estimator is provided
HEURISTIC_BASE_MIN = 25.0
HEURISTIC_PER_SKILL_MIN = 8.0


@dataclass
class DifficultyResult:
    skill_coverage: float
    pred_time_min: int
    target_time_min: int
    zpd_fit: float
    accepted: bool
    reason: str
    estimated_skills_needed: int = 0
    skills_found: list[str] = field(default_factory=list)


async def async_estimate_difficulty(
    seed: dict[str, Any],
    *,
    skill_library_query: Callable[[str], list[tuple[str, float]]],
    successful_builds: int,
    estimated_skills_needed: int = 5,
    llm_time_estimator: Optional[Callable[[str], Awaitable[str]]] = None,
) -> DifficultyResult:
    """Estimate difficulty for one seed.

    `skill_library_query(text) -> [(skill_name, cosine)]` performs the
    embedding lookup over the skill library. Only hits with cosine >= 0.6
    are counted as "present."

    `successful_builds` drives the target_time ramp per the spec:
    start at 20 min, add 2 min per 10 built.

    Returns a DifficultyResult with `accepted=False` when skill coverage
    lies outside the POET minimal band — the ranker will still see the
    result and drop the seed from further pipeline consideration.
    """
    text = f"{seed.get('title', '')} {seed.get('summary', '')}".strip()

    hits = skill_library_query(text) if text else []
    skills_found = [name for name, cos in hits if cos >= 0.6]
    skill_count = len(skills_found)

    # Never divide by zero — floor estimated_skills_needed at 1.
    est_needed = max(1, int(estimated_skills_needed))
    skill_coverage = min(1.0, skill_count / est_needed)

    target_time = 20 + 2 * (max(0, successful_builds) // 10)

    # Predicted time
    if llm_time_estimator is not None:
        pred_time = await _call_time_estimator(
            llm_time_estimator,
            seed=seed,
            skills_found=skills_found,
            estimated_skills_needed=est_needed,
        )
    else:
        pred_time = _heuristic_pred_time(skills_found, est_needed)

    # ZPD fit
    gauss = math.exp(-((pred_time - target_time) ** 2) / (2.0 * SIGMA * SIGMA))
    clamped = max(MIN_COVERAGE, min(MAX_COVERAGE, skill_coverage))
    zpd_fit = gauss * clamped

    accepted = MIN_COVERAGE <= skill_coverage <= MAX_COVERAGE
    reason = "ok" if accepted else ("too_easy" if skill_coverage > MAX_COVERAGE else "too_hard")

    return DifficultyResult(
        skill_coverage=skill_coverage,
        pred_time_min=pred_time,
        target_time_min=target_time,
        zpd_fit=zpd_fit,
        accepted=accepted,
        reason=reason,
        estimated_skills_needed=est_needed,
        skills_found=skills_found,
    )


def estimate_difficulty(
    seed: dict[str, Any],
    *,
    skill_library_query: Callable[[str], list[tuple[str, float]]],
    successful_builds: int,
    estimated_skills_needed: int = 5,
    llm_time_estimator: Optional[Callable[[str], str]] = None,
) -> DifficultyResult:
    """Sync convenience wrapper (tests, scripts)."""
    import asyncio

    async def _adapter(prompt: str) -> str:
        if llm_time_estimator is None:
            return ""
        return llm_time_estimator(prompt)

    return asyncio.run(
        async_estimate_difficulty(
            seed,
            skill_library_query=skill_library_query,
            successful_builds=successful_builds,
            estimated_skills_needed=estimated_skills_needed,
            llm_time_estimator=_adapter if llm_time_estimator else None,
        )
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _heuristic_pred_time(skills_found: list[str], estimated_skills_needed: int) -> int:
    """Spec-agnostic fallback when the LLM estimator isn't wired up.

    Rough idea: more required skills than found => more exploration =>
    higher predicted time. Caps at 240 min / floors at 5 min per spec.
    """
    deficit = max(0, estimated_skills_needed - len(skills_found))
    pred = HEURISTIC_BASE_MIN + deficit * HEURISTIC_PER_SKILL_MIN
    return int(max(5.0, min(240.0, pred)))


async def _call_time_estimator(
    estimator: Callable[[str], Awaitable[str]],
    *,
    seed: dict[str, Any],
    skills_found: list[str],
    estimated_skills_needed: int,
) -> int:
    prompt = DIFFICULTY_PROMPT.format(
        title=seed.get("title", ""),
        summary=seed.get("summary", ""),
        skills_found=", ".join(skills_found) or "(none)",
        estimated_skills_needed=estimated_skills_needed,
    )
    try:
        raw = await estimator(prompt)
    except Exception as exc:
        logger.warning("time estimator raised %s; using heuristic", exc)
        return _heuristic_pred_time(skills_found, estimated_skills_needed)

    parsed = _parse_time_verdict(raw)
    if parsed is None:
        return _heuristic_pred_time(skills_found, estimated_skills_needed)
    # Clamp to a sane band
    return max(5, min(240, int(parsed)))


def _parse_time_verdict(raw: str) -> Optional[int]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    val = data.get("pred_time_min")
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    return None


__all__ = [
    "DifficultyResult",
    "HAIKU_MODEL",
    "MAX_COVERAGE",
    "MIN_COVERAGE",
    "SIGMA",
    "async_estimate_difficulty",
    "estimate_difficulty",
]
