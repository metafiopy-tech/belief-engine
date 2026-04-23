"""Format top-k prior AgentConfigurations for planner-prompt injection.

Session 6 (v3.2).  The planner agent calls this helper just before
:meth:`belief.llm.LLMClient.generate_structured` to append a few-shot
"Prior successful configuration for similar task" block to its system
prompt.

Append-only (keeps session-1's num_keep=512 KV-cache hit).  Returns an
empty string when the archive is empty or the query fails — the
planner proceeds with its base prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from belief.archive.outcome import BuildOutcome
from belief.archive.sampler import parent_sample
from belief.archive.store import AgentArchive

logger = logging.getLogger("belief.archive.priors")


_HEADER = (
    "\n\n---\n## PRIOR SUCCESSFUL CONFIGURATIONS (injected by agent archive)\n\n"
    "The archive has retrieved similar past builds.  Use these as "
    "few-shot priors — reproduce what worked, avoid what didn't.\n"
)


def format_priors_block(
    goal: str,
    *,
    archive: AgentArchive | None = None,
    k: int = 3,
    max_chars_per_prior: int = 600,
) -> str:
    """Sample top-k priors via :func:`parent_sample` and format them
    as an append-only text block.

    Returns an empty string if the archive is empty or the query
    yields nothing.  Truncates per-prior text to keep the total
    injection well under the 3 KB budget recomposer already respects.
    """
    try:
        results = parent_sample(goal, archive=archive, k=k)
    except Exception as e:
        logger.debug("parent_sample failed: %s", e)
        return ""
    if not results:
        return ""

    blocks: list[str] = []
    for i, hit in enumerate(results, start=1):
        outcome = hit.get("outcome")
        meta = hit.get("metadata") or {}
        if isinstance(outcome, BuildOutcome):
            summary = _format_outcome(outcome, i, max_chars_per_prior)
        else:
            summary = _format_from_metadata(meta, i, max_chars_per_prior)
        blocks.append(summary)

    return _HEADER + "\n".join(blocks) + "\n"


def _format_outcome(outcome: BuildOutcome, n: int, budget: int) -> str:
    goal_short = outcome.goal[:200]
    verdict = outcome.verdict
    score = outcome.weighted_score
    planner = outcome.agent_configurations.get("planner")
    snippet = ""
    if planner is not None:
        prompt = getattr(planner, "system_prompt", "") or ""
        snippet = prompt[:budget]
    lines = [
        f"### Prior {n} — goal: {goal_short}",
        f"  verdict={verdict}, weighted_score={score:.2f}",
    ]
    if snippet:
        lines.append("  planner system prompt (truncated):")
        for ln in snippet.strip().splitlines()[:6]:
            lines.append(f"    {ln[:120]}")
    return "\n".join(lines)


def _format_from_metadata(meta: dict[str, Any], n: int, budget: int) -> str:
    goal_short = str(meta.get("goal", ""))[:200]
    verdict = meta.get("verdict", "?")
    score = meta.get("weighted_score", 0.0)
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    return (
        f"### Prior {n} — goal: {goal_short}\n"
        f"  verdict={verdict}, weighted_score={score_f:.2f}"
    )


__all__ = ["format_priors_block"]
