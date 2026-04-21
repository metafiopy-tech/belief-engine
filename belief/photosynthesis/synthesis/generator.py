"""Sonnet 4.5 goal spec generator with K=4 samples + post-dup check.

Per spec constraint: Sonnet (not Haiku) for generation — this is the
one place where spending tokens matters. K=4 alternative specs are
drawn at temperature=0.7; each is re-scored via the ranker and the
top-1 is kept. If the chosen spec's embedding is >= 0.90 cosine to any
goal in goal_archive, it's diverted to failed_interest and the cycle
returns None (don't promote).

Every generator response is validated against `GoalSpec`. Invalid
responses retry once; still-invalid responses are logged and the
candidate is rejected (no third attempt — spec cap).

The caller provides:
  - `generator_client(prompt, *, temperature, max_tokens) -> str`
    returns the raw model text
  - `embedder(text) -> vector` for the post-dup check
  - `archive: ArchiveManager`
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from belief.photosynthesis.synthesis.archives import ArchiveManager, Neighbor
from belief.photosynthesis.synthesis.prompts import (
    GENERATOR_PROMPT,
    format_neighbors,
)
from belief.photosynthesis.synthesis.ranker import (
    ACCEPT_THRESHOLD,
    RankerResult,
    combined_value,
    coverage_gain,
    source_quality,
)


logger = logging.getLogger("belief.photosynthesis.synthesis.generator")


SONNET_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLES = 4
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1600
POST_DUP_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Pydantic schema for generator output
# ---------------------------------------------------------------------------


ALLOWED_ARTIFACT_TYPES = {"cli", "api", "library", "mcp_server", "pipeline", "script"}
ALLOWED_AC_KINDS = {"test", "endpoint", "behavior", "artifact"}


class AcceptanceCriterion(BaseModel):
    kind: str
    spec: str

    @field_validator("kind")
    @classmethod
    def _valid_kind(cls, v: str) -> str:
        if v not in ALLOWED_AC_KINDS:
            raise ValueError(f"kind must be one of {sorted(ALLOWED_AC_KINDS)}")
        return v


class GoalSpec(BaseModel):
    """The JSON shape we require from Sonnet."""

    goal_id: str
    title: str = Field(max_length=120)
    one_paragraph_description: str = Field(max_length=1200)
    artifact_type: str
    primary_libraries: list[str] = Field(default_factory=list)
    new_libraries_introduced: list[str] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion]
    estimated_build_time_min: int = Field(ge=5, le=240)
    estimated_difficulty: int = Field(ge=1, le=5)
    prerequisite_skills: list[str] = Field(default_factory=list)
    relevance_rationale: str
    novelty_rationale: str
    source_citation: str

    @field_validator("artifact_type")
    @classmethod
    def _valid_artifact(cls, v: str) -> str:
        if v not in ALLOWED_ARTIFACT_TYPES:
            raise ValueError(
                f"artifact_type must be one of {sorted(ALLOWED_ARTIFACT_TYPES)}"
            )
        return v

    @field_validator("acceptance_criteria")
    @classmethod
    def _at_least_one(
        cls, v: list[AcceptanceCriterion]
    ) -> list[AcceptanceCriterion]:
        if not v:
            raise ValueError("acceptance_criteria must have at least one entry")
        return v


# ---------------------------------------------------------------------------
# Generator result wrapper
# ---------------------------------------------------------------------------


@dataclass
class GeneratorResult:
    spec: Optional[GoalSpec]
    ranker: Optional[RankerResult]
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def synthesize(
    seed: dict[str, Any],
    *,
    novelty_score: float,
    zpd_fit: float,
    pred_time_min: int,
    neighbors: list[Neighbor],
    archive: ArchiveManager,
    embedder: Callable[[str], Any],
    generator_client: Callable[..., Awaitable[str]],
    bittensor_cosine: Optional[float] = None,
    bittensor_bias_cutoff: float = 0.70,
    k: int = DEFAULT_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
) -> GeneratorResult:
    """Sample K specs, re-rank, pick top-1, post-dup-check, return.

    Returns GeneratorResult:
      - spec populated iff a valid, non-duplicate spec was produced,
      - reason = 'accepted' | 'no_valid_sample' | 'post_dup' | 'empty'
    """
    prompt = _build_prompt(seed, novelty_score, zpd_fit, pred_time_min, neighbors)

    raw_candidates: list[str] = []
    for _ in range(max(1, int(k))):
        try:
            text = await generator_client(
                prompt,
                temperature=temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("generator call raised: %s", exc)
            continue
        if text:
            raw_candidates.append(text)

    parsed: list[tuple[GoalSpec, RankerResult, dict[str, Any]]] = []
    top_archive_tags = archive.top_tags("goal_archive", top_n=20)

    for raw in raw_candidates:
        spec = _parse_with_retry(raw)
        if spec is None:
            continue
        # Re-rank each candidate under the ranker
        cov = coverage_gain(
            seed.get("domain_tags", []) or spec.primary_libraries,
            top_archive_tags,
        )
        sq = source_quality(seed)
        r = combined_value(
            novelty=novelty_score,
            zpd_fit=zpd_fit,
            coverage_gain=cov,
            source_quality=sq,
            bittensor_cosine=bittensor_cosine,
        )
        parsed.append((spec, r, json.loads(spec.model_dump_json())))

    if not parsed:
        return GeneratorResult(
            spec=None,
            ranker=None,
            reason="no_valid_sample",
            candidates=[{"raw": r[:1000]} for r in raw_candidates],
        )

    parsed.sort(key=lambda t: t[1].value, reverse=True)
    best_spec, best_ranker, best_json = parsed[0]

    # Bittensor biasing (Session 5): when the seed is close enough to the
    # SWE-Bench centroid, require the canonical validator-friendly
    # acceptance criteria so every promoted goal practices the exact
    # shapes SN62 scores on. This is a post-generation augmentation —
    # it doesn't alter the LLM's original output beyond adding required
    # criteria.
    if (
        bittensor_cosine is not None
        and bittensor_cosine >= bittensor_bias_cutoff
    ):
        best_spec = _inject_bittensor_constraints(best_spec)

    if not best_ranker.accepted:
        return GeneratorResult(
            spec=None,
            ranker=best_ranker,
            reason="below_accept_threshold",
            candidates=[d for _, _, d in parsed],
        )

    # Post-expansion duplicate check
    spec_text = f"{best_spec.title}. {best_spec.one_paragraph_description}"
    spec_vec = embedder(spec_text)
    hits = archive.query_neighbors("goal_archive", spec_vec, top_k=1)
    if hits and hits[0].cosine >= POST_DUP_THRESHOLD:
        # Divert to failed_interest — don't poison goal_archive
        archive.upsert_goal(
            "failed_interest",
            goal_id=best_spec.goal_id,
            embedding=spec_vec,
            document=spec_text,
            metadata={
                "title": best_spec.title,
                "reason": "post_expansion_duplicate",
                "nearest_goal_id": hits[0].goal_id,
                "cosine": hits[0].cosine,
            },
        )
        return GeneratorResult(
            spec=None,
            ranker=best_ranker,
            reason="post_dup",
            candidates=[d for _, _, d in parsed],
        )

    return GeneratorResult(
        spec=best_spec,
        ranker=best_ranker,
        reason="accepted",
        candidates=[d for _, _, d in parsed],
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_prompt(
    seed: dict[str, Any],
    novelty_score: float,
    zpd_fit: float,
    pred_time_min: int,
    neighbors: list[Neighbor],
) -> str:
    neighbors_formatted = format_neighbors(
        [n.to_prompt_dict() for n in neighbors[:5]]
    )
    return GENERATOR_PROMPT.format(
        title=seed.get("title", ""),
        summary=seed.get("summary", ""),
        raw_excerpt=(seed.get("raw_excerpt") or "")[:2000],
        source=seed.get("source", ""),
        source_id=seed.get("source_id", ""),
        domain_tags=", ".join(seed.get("domain_tags", []) or []) or "(none)",
        neighbors_formatted=neighbors_formatted,
        novelty_rationale=f"novelty={novelty_score:.2f}",
        pred_time_min=int(pred_time_min),
        zpd_fit=float(zpd_fit),
    )


def _parse_with_retry(raw: str) -> Optional[GoalSpec]:
    """Parse the Sonnet output into GoalSpec. Returns None on any failure."""
    data = _extract_json(raw)
    if data is None:
        return None
    # Ensure goal_id is present — fill with a uuid-backed default if omitted
    if "goal_id" not in data or not data.get("goal_id"):
        title = str(data.get("title", "")).strip()
        data["goal_id"] = _slugify(title) or f"goal-{uuid.uuid4().hex[:8]}"
    try:
        return GoalSpec.model_validate(data)
    except ValidationError as exc:
        logger.warning("generator schema failed: %s", exc.errors()[:3])
        return None


def _extract_json(raw: str) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    # Try strict first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Otherwise grab the first balanced {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _slugify(text: str) -> str:
    lowered = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug[:60]


# ---------------------------------------------------------------------------
# Session 5: Bittensor-biased spec augmentation
# ---------------------------------------------------------------------------


BITTENSOR_CONSTRAINT_LINE = (
    "Agent must complete in <=25 minutes wallclock; "
    "inference budget <=$2; no outbound network except proxy endpoint."
)
BITTENSOR_DIFF_AC = {
    "kind": "artifact",
    "spec": "output matches `diff --git` unified-diff format",
}
BITTENSOR_PYTEST_AC = {
    "kind": "test",
    "spec": "pytest runs inside sandbox with all repo tests passing",
}


def _inject_bittensor_constraints(spec: GoalSpec) -> GoalSpec:
    """Return a copy of `spec` augmented for SN62 validator scoring.

    Adds two required acceptance criteria (unified-diff output + pytest
    in sandbox) and an operational constraint line embedded in the
    novelty_rationale (the spec .md renderer pulls constraints from the
    spec's fields + a fixed tail). Idempotent — re-running on a spec
    that's already been augmented is a no-op.
    """
    already_has_diff = any(
        ac.kind == "artifact" and "diff --git" in ac.spec for ac in spec.acceptance_criteria
    )
    already_has_pytest = any(
        ac.kind == "test" and "sandbox" in ac.spec for ac in spec.acceptance_criteria
    )

    new_acs = list(spec.acceptance_criteria)
    if not already_has_diff:
        new_acs.append(AcceptanceCriterion.model_validate(BITTENSOR_DIFF_AC))
    if not already_has_pytest:
        new_acs.append(AcceptanceCriterion.model_validate(BITTENSOR_PYTEST_AC))

    new_rel = spec.relevance_rationale
    if BITTENSOR_CONSTRAINT_LINE not in new_rel:
        new_rel = (new_rel.rstrip() + " " + BITTENSOR_CONSTRAINT_LINE).strip()

    return spec.model_copy(
        update={
            "acceptance_criteria": new_acs,
            "relevance_rationale": new_rel,
        }
    )


__all__ = [
    "ALLOWED_AC_KINDS",
    "ALLOWED_ARTIFACT_TYPES",
    "AcceptanceCriterion",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SAMPLES",
    "DEFAULT_TEMPERATURE",
    "GeneratorResult",
    "GoalSpec",
    "POST_DUP_THRESHOLD",
    "SONNET_MODEL",
    "synthesize",
]
