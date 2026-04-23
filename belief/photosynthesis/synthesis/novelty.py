"""Hybrid novelty scoring for synthesis — OMNI-EPIC interestingness bands.

Given a filtered signal, decide whether it's novel enough to promote to
the generator. Three cosine bands gate the expensive Haiku judge:

    cosine >= 0.92     hard duplicate — reject, no LLM call
    0.75 <= c < 0.92   mid-band — call Haiku judge, keep if "interesting"
    cosine <  0.75     distinct — keep, no LLM call

The judge uses the INTERESTINGNESS_PROMPT (module-level constant) against
up to 5 archived neighbors. Verdict is validated against a strict JSON
schema; invalid responses retry once, then default to "not interesting"
(conservative — better to silently reject than to promote a malformed
response).

Session 5 will tighten this: hard per-cycle LLM call cap, structured
audit log of every judge verdict. The public API (NoveltyResult and
score_novelty / async_score_novelty) won't change.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


from belief.photosynthesis.synthesis.archives import ArchiveManager, Neighbor
from belief.photosynthesis.synthesis.prompts import (
    INTERESTINGNESS_PROMPT,
    format_neighbors,
)

logger = logging.getLogger("belief.photosynthesis.synthesis.novelty")


HARD_DUP_THRESHOLD = 0.92
DISTINCT_THRESHOLD = 0.75

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Pydantic-ish verdict keys we accept. Kept loose so the judge can
# return slightly different shapes without breaking us — see _parse_verdict.
VALID_CATEGORIES = {"a", "b", "c", "d", "x", "y", "z"}


# ---------------------------------------------------------------------------
# Dataclass — the result any caller consumes
# ---------------------------------------------------------------------------


@dataclass
class NoveltyResult:
    accepted: bool
    reason: str
    novelty: float  # 0..1
    cosine_top1: float = 0.0
    neighbors: list[Neighbor] = field(default_factory=list)
    judge_verdict: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def async_score_novelty(
    seed: dict[str, Any],
    *,
    archive: ArchiveManager,
    embedder: Callable[[str], Any],
    llm_judge: Optional[Callable[[str], Awaitable[str]]] = None,
    top_k: int = 10,
    judge_neighbors: int = 5,
) -> NoveltyResult:
    """Score a single seed.

    `seed` is a dict with at least keys 'title', 'summary' (and
    optionally 'source', 'source_id', 'domain_tags').
    `embedder(text) -> vector` produces the query embedding.
    `llm_judge(prompt) -> str` returns a JSON-stringy verdict. When
    unset (Session 4 default in tests), mid-band seeds are conservatively
    rejected with reason='mid_no_judge'.
    """
    text = f"{seed.get('title', '')} {seed.get('summary', '')}".strip()
    if not text:
        return NoveltyResult(accepted=False, reason="empty_seed", novelty=0.0)

    vec = embedder(text)
    neighbors = archive.query_neighbors("goal_archive", vec, top_k=top_k)

    if not neighbors:
        # Empty archive: accept outright, maximum novelty.
        return NoveltyResult(
            accepted=True,
            reason="archive_empty",
            novelty=1.0,
            cosine_top1=0.0,
            neighbors=[],
        )

    top = neighbors[0].cosine

    # Hard duplicate gate.
    if top >= HARD_DUP_THRESHOLD:
        return NoveltyResult(
            accepted=False,
            reason="hard_duplicate",
            novelty=0.0,
            cosine_top1=top,
            neighbors=neighbors,
        )

    # Distinct gate — no LLM call needed.
    if top < DISTINCT_THRESHOLD:
        return NoveltyResult(
            accepted=True,
            reason="distinct",
            novelty=_novelty_from_cos(top),
            cosine_top1=top,
            neighbors=neighbors,
        )

    # Mid-band — call judge.
    if llm_judge is None:
        return NoveltyResult(
            accepted=False,
            reason="mid_no_judge",
            novelty=0.0,
            cosine_top1=top,
            neighbors=neighbors,
        )

    prompt = _build_judge_prompt(seed, neighbors[:judge_neighbors])
    verdict = await _call_judge_with_retry(llm_judge, prompt)
    if verdict is None:
        return NoveltyResult(
            accepted=False,
            reason="judge_invalid",
            novelty=0.0,
            cosine_top1=top,
            neighbors=neighbors,
        )

    interesting = bool(verdict.get("interesting"))
    return NoveltyResult(
        accepted=interesting,
        reason=f"judge_{verdict.get('category', '?')}",
        novelty=_novelty_from_cos(top) if interesting else 0.0,
        cosine_top1=top,
        neighbors=neighbors,
        judge_verdict=verdict,
    )


def score_novelty(
    seed: dict[str, Any],
    *,
    archive: ArchiveManager,
    embedder: Callable[[str], Any],
    llm_judge: Optional[Callable[[str], str]] = None,
    **kwargs: Any,
) -> NoveltyResult:
    """Sync convenience wrapper.

    Wraps `async_score_novelty` and adapts a sync llm_judge callable
    into the expected awaitable. Useful in tests and one-shot scripts.
    """
    import asyncio

    async def _async_judge_adapter(prompt: str) -> str:
        if llm_judge is None:
            return ""
        return llm_judge(prompt)

    return asyncio.run(
        async_score_novelty(
            seed,
            archive=archive,
            embedder=embedder,
            llm_judge=_async_judge_adapter if llm_judge else None,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _novelty_from_cos(cos: float) -> float:
    """Map a top-1 cosine to a novelty score in [0, 1].

    cosine 0 -> novelty 1.0
    cosine DISTINCT_THRESHOLD (0.75) -> novelty ~0.65
    cosine HARD_DUP_THRESHOLD (0.92) -> novelty ~0.2 (but this band
                                        is rejected before we land here)
    """
    return max(0.0, 1.0 - cos)


def _build_judge_prompt(seed: dict[str, Any], neighbors: list[Neighbor]) -> str:
    neighbors_formatted = format_neighbors([n.to_prompt_dict() for n in neighbors])
    return INTERESTINGNESS_PROMPT.format(
        title=seed.get("title", ""),
        summary=seed.get("summary", ""),
        source=seed.get("source", ""),
        source_id=seed.get("source_id", ""),
        domain_tags=", ".join(seed.get("domain_tags", []) or []) or "(none)",
        neighbors_formatted=neighbors_formatted,
    )


async def _call_judge_with_retry(
    llm_judge: Callable[[str], Awaitable[str]],
    prompt: str,
    *,
    attempts: int = 2,
) -> Optional[dict[str, Any]]:
    for i in range(attempts):
        try:
            raw = await llm_judge(prompt)
        except Exception as exc:
            logger.warning("judge call failed on attempt %d: %s", i + 1, exc)
            continue
        parsed = _parse_verdict(raw)
        if parsed is not None:
            return parsed
        logger.warning("judge verdict invalid on attempt %d: %r", i + 1, (raw or "")[:200])
    return None


def _parse_verdict(raw: str) -> Optional[dict[str, Any]]:
    """Return the parsed verdict dict or None on any validation failure."""
    if not raw:
        return None
    # Prefer strict JSON parse; fall back to the first {...} block.
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
    if "interesting" not in data:
        return None
    if not isinstance(data["interesting"], bool):
        return None
    if "category" in data and data["category"] not in VALID_CATEGORIES:
        data["category"] = "?"
    # justification length guard
    just = data.get("one_line_justification") or ""
    if isinstance(just, str) and len(just) > 400:
        data["one_line_justification"] = just[:400]
    return data


__all__ = [
    "DISTINCT_THRESHOLD",
    "HAIKU_MODEL",
    "HARD_DUP_THRESHOLD",
    "NoveltyResult",
    "async_score_novelty",
    "score_novelty",
]
