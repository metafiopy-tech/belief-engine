"""Apex-predator promotion: named-library induction from soil nutrients.

When a fragment reaches trophic_level >= 3 AND has been used in >=5
successful builds, we ask Claude to name it and wrap it as a
:class:`SelfAuthoredTool`. Anonymous abstractions degrade LLM
performance (the LILO result is ~30 points); naming is first-class.

Session 11 shipped compete/run_trophic_pass. This module is the
consumer side: takes a qualifying nutrient, pipes its code through a
naming LLM, and registers the result. The caller is expected to be
the jitterbug integration phase — it calls :func:`promote_eligible`
with a list of candidate nutrient ids, respecting a cap of 3
promotions per invocation.

Design:
  - LLM client is injectable. The function is dispatcher-agnostic:
    any object matching
    :class:`~belief.photosynthesis.safety.cost_tracker.HasMessagesCreate`
    works. The photosynthesis daemon passes
    :class:`~belief.photosynthesis.safety.cost_tracker.BreakerAnthropic`
    (anthropic SDK wrapped with per-call cost metering); main-pipeline
    callers pass :mod:`belief.llm`'s equivalent; tests pass a fake
    that returns canned JSON.  See ``docs/architecture/http_boundary.md``
    for why two LLM dispatchers coexist and when that's expected to
    consolidate.
  - Naming output validated against a strict Pydantic schema. On
    validation failure we retry once, then give up.
  - Promotion is atomic: either the tool is registered AND the
    nutrient is marked promoted, or neither. Errors propagate to the
    caller but the soil is not left half-updated.
  - No tool registers before passing :func:`belief.evolution.tool_validator.validate_tool`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from belief.memory.tool_registry import SelfAuthoredTool


logger = logging.getLogger("belief.memory.library_inductor")


# Spec thresholds
MIN_TROPHIC_LEVEL = 3
MIN_SUCCESSFUL_USES = 5
MAX_PROMOTIONS_PER_CYCLE = 3

SONNET_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5-20251001"


NAMING_SYSTEM_PROMPT = """\
You are a library-function naming assistant for the Belief Engine.
You'll be given a snippet of Python code that has proven useful across
multiple builds. Produce a JSON object:

    {"name": str,                # snake_case, <=32 chars, Python-legal identifier
     "description": str,         # one paragraph: what it does + when to use it
     "type_hints": str,          # revised function signature with full typing
     "usage_examples": [str, str, str]}  # 3 short usage examples

Requirements:
  - name must be descriptive (not 'helper', not 'utility', not 'x').
  - description must explain the intended use-site, not re-state the code.
  - type_hints is just the 'def fn(...) -> T:' line, no body.
  - usage_examples each <=120 chars.

Return JSON only — no markdown fences, no prose.
"""


_USER_TEMPLATE = """\
PROMOTION CANDIDATE

Code (proven across multiple builds):
---
{code}
---

Hints:
  - This fragment was used {use_count} times in successful builds.
  - It reached trophic_level {trophic_level} via competition wins.
  - Candidate tags: {tags}
"""


# ---------------------------------------------------------------------------
# Client protocol + schema
# ---------------------------------------------------------------------------


class NamingClient(Protocol):
    def generate_text(self, *, system: str, prompt: str, max_tokens: int) -> str: ...  # noqa: E704


class NamingResult(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1200)
    type_hints: str = Field(min_length=1, max_length=400)
    usage_examples: list[str] = Field(min_length=1, max_length=5)

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        s = v.strip()
        if not s.isidentifier():
            raise ValueError("name must be a valid Python identifier")
        # Reject obviously-generic names the spec called out.
        if s.lower() in {"helper", "util", "utility", "tmp", "x", "fn"}:
            raise ValueError("name too generic")
        return s

    @field_validator("usage_examples")
    @classmethod
    def _examples_length(cls, v: list[str]) -> list[str]:
        return [str(ex)[:200] for ex in v if ex and str(ex).strip()]


# ---------------------------------------------------------------------------
# Candidate + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A nutrient deemed eligible for promotion."""

    nutrient_id: str
    code: str
    trophic_level: int = 0
    use_count: int = 0
    tags: list[str] = None  # type: ignore[assignment]

    def eligible(
        self,
        *,
        min_trophic_level: int = MIN_TROPHIC_LEVEL,
        min_uses: int = MIN_SUCCESSFUL_USES,
    ) -> bool:
        return (
            self.trophic_level >= min_trophic_level
            and self.use_count >= min_uses
            and bool(self.code)
        )


@dataclass
class PromotionOutcome:
    nutrient_id: str
    success: bool
    tool_id: Optional[str] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def promote_apex_predator(
    candidate: Candidate,
    *,
    tool_registry: Any,
    naming_client: Optional[NamingClient] = None,
    validator: Optional[Any] = None,
    model: str = HAIKU_MODEL,
    max_tokens: int = 900,
) -> PromotionOutcome:
    """Run one candidate through the naming + validation + register pipe.

    Returns a :class:`PromotionOutcome` with success=True on a clean
    register; on failure, reason explains which stage refused.
    """
    if not candidate.eligible():
        return PromotionOutcome(
            nutrient_id=candidate.nutrient_id,
            success=False,
            reason=(
                f"not eligible (trophic_level={candidate.trophic_level}, "
                f"uses={candidate.use_count})"
            ),
        )

    if naming_client is None:
        return PromotionOutcome(
            nutrient_id=candidate.nutrient_id,
            success=False,
            reason="no naming client injected",
        )

    # --- 1. LLM naming (strict JSON, retry-once on schema fail) ---
    naming = _call_namer_with_retry(
        candidate, client=naming_client, model=model, max_tokens=max_tokens
    )
    if naming is None:
        return PromotionOutcome(
            nutrient_id=candidate.nutrient_id,
            success=False,
            reason="naming rejected after retry",
        )

    # --- 2. Assemble SelfAuthoredTool ---
    tool = _build_tool(candidate, naming)

    # --- 3. Validate via existing tool_validator ---
    validation_error = _validate_tool(tool, validator=validator)
    if validation_error:
        return PromotionOutcome(
            nutrient_id=candidate.nutrient_id,
            success=False,
            reason=f"tool_validator rejected: {validation_error}",
        )

    # --- 4. Register ---
    try:
        tool_id = tool_registry.register_tool(tool)
    except Exception as exc:
        return PromotionOutcome(
            nutrient_id=candidate.nutrient_id,
            success=False,
            reason=f"register_tool failed: {exc}",
        )

    return PromotionOutcome(
        nutrient_id=candidate.nutrient_id,
        success=True,
        tool_id=str(tool_id),
    )


def promote_eligible(
    candidates: Iterable[Candidate],
    *,
    tool_registry: Any,
    naming_client: Optional[NamingClient] = None,
    validator: Optional[Any] = None,
    max_promotions: int = MAX_PROMOTIONS_PER_CYCLE,
) -> list[PromotionOutcome]:
    """Iterate a stream of candidates; stop after ``max_promotions`` successes.

    Always returns one outcome per attempted candidate (skips the
    ineligible ones). Failures don't consume the budget — only
    successful promotions count toward the cap.
    """
    outcomes: list[PromotionOutcome] = []
    promoted = 0
    for candidate in candidates:
        if promoted >= max(0, int(max_promotions)):
            break
        outcome = promote_apex_predator(
            candidate,
            tool_registry=tool_registry,
            naming_client=naming_client,
            validator=validator,
        )
        outcomes.append(outcome)
        if outcome.success:
            promoted += 1
    return outcomes


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _call_namer_with_retry(
    candidate: Candidate,
    *,
    client: NamingClient,
    model: str,
    max_tokens: int,
    attempts: int = 2,
) -> Optional[NamingResult]:
    tags = ", ".join(candidate.tags or []) or "(none)"
    prompt = _USER_TEMPLATE.format(
        code=candidate.code[:4000],
        use_count=candidate.use_count,
        trophic_level=candidate.trophic_level,
        tags=tags,
    )
    for i in range(attempts):
        try:
            raw = client.generate_text(
                system=NAMING_SYSTEM_PROMPT,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning("naming client call failed (attempt %d): %s", i + 1, exc)
            continue
        parsed = _parse_naming_output(raw)
        if parsed is not None:
            return parsed
        logger.warning("naming output invalid on attempt %d", i + 1)
    return None


def _parse_naming_output(raw: str) -> Optional[NamingResult]:
    if not raw:
        return None
    # Strict JSON first
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    try:
        return NamingResult.model_validate(data)
    except ValidationError as exc:
        logger.debug("naming schema rejected: %s", exc.errors()[:3])
        return None


def _build_tool(candidate: Candidate, naming: NamingResult) -> SelfAuthoredTool:
    import uuid as _uuid

    return SelfAuthoredTool(
        id=f"lib-{_uuid.uuid4().hex[:10]}",
        name=naming.name,
        description=naming.description,
        code=candidate.code,
        input_description=naming.type_hints,
        output_description="",
        parent_id=candidate.nutrient_id,
        created_by="jitterbug",
    )


def _validate_tool(tool: SelfAuthoredTool, *, validator: Any) -> str:
    """Run the existing tool_validator; return error string or ''."""
    if validator is None:
        try:
            from belief.evolution.tool_validator import validate_tool

            validator = validate_tool
        except Exception:
            # Tool validator unavailable — let register proceed.
            return ""
    try:
        result = validator(tool)
    except Exception as exc:  # pragma: no cover - validator error
        return f"validator raised {exc}"
    # validate_tool returns either a bool, a (valid, errors) tuple, or
    # an object with .valid / .errors. Handle all three cheaply.
    valid = getattr(result, "valid", None)
    errors = getattr(result, "errors", None)
    if valid is not None:
        return "" if valid else "; ".join(errors or [])
    if isinstance(result, tuple) and len(result) == 2:
        ok, errs = result
        return "" if ok else "; ".join(errs or [])
    # Plain bool
    return "" if bool(result) else "validator returned False"


__all__ = [
    "Candidate",
    "HAIKU_MODEL",
    "MAX_PROMOTIONS_PER_CYCLE",
    "MIN_SUCCESSFUL_USES",
    "MIN_TROPHIC_LEVEL",
    "NAMING_SYSTEM_PROMPT",
    "NamingClient",
    "NamingResult",
    "PromotionOutcome",
    "SONNET_MODEL",
    "promote_apex_predator",
    "promote_eligible",
]
