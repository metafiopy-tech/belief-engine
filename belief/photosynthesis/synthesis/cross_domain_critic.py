"""CoVe-style critic for cross-domain mechanisms (SE Session 3).

The critic runs in an INDEPENDENT context: it sees only the candidate
``StructuralMechanism`` JSON, never the synthesizer's chain-of-thought
brainstorm or its intermediate predicate-forcing pass. This keeps the
critique from being primed by the synthesizer's own framing.

Eight checks (see ``CRITIC_PROMPT`` for the rubric). ACCEPT iff all
eight pass; REJECT otherwise. The two syntactic checks (1 and 6) run
deterministically before the LLM call to short-circuit obvious
failures and save tokens. The remaining six are sent to the
``critic_client`` callable as a single prompt.

Out of scope for Session 3:
  - Domain-document grounding: Session 4's ``BiologicalPrimitiveStore``
    will surface evidence the critic can compare against. Session 3's
    critic relies on the candidate JSON alone.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from belief.photosynthesis.synthesis.prompts_cross_domain import CRITIC_PROMPT
from belief.photosynthesis.synthesis.structural_mechanism import StructuralMechanism


logger = logging.getLogger("belief.photosynthesis.synthesis.cross_domain_critic")


# Attribute-style predicate name prefixes -- check 1 catches these
# without paying for an LLM call.
_ATTRIBUTE_PREFIXES: tuple[str, ...] = (
    "has_",
    "is_",
    "are_",
    "lacks_",
    "contains_",
    "owns_",
    "holds_",
    "wants_",
    "knows_",
)


# A small lexicon of process-y role tokens. Check 2 looks for at least
# one role containing one of these substrings. The list is intentionally
# permissive -- we are only screening out predicates whose roles read
# entirely as static descriptors.
_PROCESS_ROLE_HINTS: tuple[str, ...] = (
    "transducer",
    "sensor",
    "compress",
    "encode",
    "decode",
    "filter",
    "router",
    "controller",
    "coordinator",
    "allocator",
    "scheduler",
    "selector",
    "gate",
    "signal",
    "input",
    "output",
    "source",
    "target",
    "edge",
    "node",
    "agent",
    "actuator",
    "buffer",
    "queue",
    "consumer",
    "producer",
    "channel",
    "process",
    "step",
)


# Descriptive role tokens that explicitly fail check 2 when ALL roles
# are descriptive. These are static-attribute-style.
_DESCRIPTIVE_ONLY_TOKENS: tuple[str, ...] = (
    "color",
    "size",
    "count",
    "shape",
    "weight",
    "height",
    "depth",
    "length",
)


@dataclass
class CheckResult:
    """One row in the critic's output."""

    id: int
    name: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "passed": self.passed, "reason": self.reason}


@dataclass
class CriticResult:
    """Full critic output."""

    verdict: str  # "ACCEPT" or "REJECT"
    checks: list[CheckResult] = field(default_factory=list)
    short_circuited: bool = False
    error: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.verdict == "ACCEPT"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "checks": [c.to_dict() for c in self.checks],
            "short_circuited": self.short_circuited,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def _check_predicate_not_attribute_style(mech: StructuralMechanism) -> CheckResult:
    """Check 1: predicate name is verb-ish, not an attribute prefix."""
    name = mech.predicate_in_source.name
    for prefix in _ATTRIBUTE_PREFIXES:
        if name.startswith(prefix):
            return CheckResult(
                id=1,
                name="predicate_not_attribute_style",
                passed=False,
                reason=(
                    f"predicate name '{name}' starts with attribute-style prefix "
                    f"'{prefix}' -- predicates must describe relations, not properties"
                ),
            )
    return CheckResult(
        id=1,
        name="predicate_not_attribute_style",
        passed=True,
        reason=f"predicate name '{name}' is verb-ish, not attribute-style",
    )


def _check_roles_have_process_token(mech: StructuralMechanism) -> CheckResult:
    """Check 2: at least one role contains a process-y substring.

    A role like ``transducer`` or ``compressor_buffer`` passes; a set
    of roles that are entirely descriptive (``red``, ``small``,
    ``many``) fails. Conservative -- if we can't confidently say the
    roles are descriptive, we pass.
    """
    roles = mech.predicate_in_source.roles
    lowered = [r.lower() for r in roles]
    has_process_token = any(any(tok in role for tok in _PROCESS_ROLE_HINTS) for role in lowered)
    if has_process_token:
        return CheckResult(
            id=2,
            name="roles_are_process_oriented",
            passed=True,
            reason=f"at least one role in {roles} contains a process token",
        )
    # No process tokens. Check if all roles match obvious descriptive
    # patterns -- if so, fail confidently. Otherwise (custom domain
    # vocabulary), pass and let the LLM critique it.
    all_descriptive = all(any(d in role for d in _DESCRIPTIVE_ONLY_TOKENS) for role in lowered)
    if all_descriptive:
        return CheckResult(
            id=2,
            name="roles_are_process_oriented",
            passed=False,
            reason=f"all roles in {roles} are static descriptors, none are process-y",
        )
    return CheckResult(
        id=2,
        name="roles_are_process_oriented",
        passed=True,
        reason=(f"roles {roles} use custom vocabulary; no descriptive-only failure detected"),
    )


# ---------------------------------------------------------------------------
# LLM-driven checks (3-8 in the rubric)
# ---------------------------------------------------------------------------


_CHECK_NAMES: dict[int, str] = {
    1: "predicate_not_attribute_style",
    2: "roles_are_process_oriented",
    3: "higher_order_relations_describe_processes",
    4: "near_miss_plausibly_fits_domains",
    5: "near_miss_breaks_at_substantive_slot",
    6: "rejected_attributes_are_surface_level",
    7: "predicate_transfers_to_target_domain",
    8: "analogy_is_non_trivial",
}


def _parse_critic_response(raw: str) -> tuple[Optional[str], list[CheckResult]]:
    """Parse the JSON returned by the critic LLM.

    Returns ``(verdict_or_None, list_of_CheckResults)``. On any parse
    failure returns ``(None, [])`` so the caller can default to a
    REJECT verdict.
    """
    try:
        data = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return None, []
    verdict = data.get("verdict")
    checks_raw = data.get("checks") or []
    out: list[CheckResult] = []
    for c in checks_raw:
        if not isinstance(c, dict):
            continue
        cid = int(c.get("id", 0)) if str(c.get("id", "")).isdigit() else 0
        if cid == 0:
            continue
        out.append(
            CheckResult(
                id=cid,
                name=str(c.get("name") or _CHECK_NAMES.get(cid, f"check_{cid}")),
                passed=bool(c.get("passed")),
                reason=str(c.get("reason") or ""),
            )
        )
    if verdict not in ("ACCEPT", "REJECT"):
        verdict = None
    return verdict, out


def _extract_json(raw: str) -> str:
    """Strip code fences and locate the first balanced JSON object."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in critic response")
    return match.group(0)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


async def critique(
    mechanism: StructuralMechanism,
    *,
    critic_client: Callable[..., Awaitable[str]],
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> CriticResult:
    """Run the 8-check CoVe critique against ``mechanism``.

    ``critic_client(prompt, *, temperature, max_tokens) -> str`` is an
    async callable that returns the LLM's raw text. Callers inject
    this so tests can replay canned responses.

    Returns a :class:`CriticResult`. ``verdict`` is ``"ACCEPT"`` only
    when all 8 checks pass. Deterministic checks (1, 2) run inline and
    short-circuit obvious failures before the LLM call -- in that case
    ``short_circuited=True`` and the LLM is never invoked.
    """
    # Run deterministic checks first.
    det_results: list[CheckResult] = [
        _check_predicate_not_attribute_style(mechanism),
        _check_roles_have_process_token(mechanism),
    ]

    # Short-circuit on deterministic failure -- save the LLM call.
    failed_det = [c for c in det_results if not c.passed]
    if failed_det:
        return CriticResult(
            verdict="REJECT",
            checks=det_results,
            short_circuited=True,
            error=None,
        )

    # Run the LLM critique for checks 3-8.
    prompt = CRITIC_PROMPT.format(mechanism_json=mechanism.model_dump_json(indent=2))
    try:
        raw = await critic_client(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("critic_client raised: %s", exc)
        return CriticResult(
            verdict="REJECT",
            checks=det_results,
            error=f"critic_client error: {exc}",
        )

    verdict, llm_checks = _parse_critic_response(raw)

    # Merge deterministic + LLM check rows; deterministic wins on
    # overlapping IDs (1, 2). The critic is supposed to skip those
    # but we defend against duplicates.
    seen_ids = {c.id for c in det_results}
    merged = list(det_results) + [c for c in llm_checks if c.id not in seen_ids]
    merged.sort(key=lambda c: c.id)

    # If the LLM didn't return all 8 expected checks, mark missing ones
    # as failed -- a malformed critic response is the same as REJECT.
    present_ids = {c.id for c in merged}
    for cid in range(1, 9):
        if cid not in present_ids:
            merged.append(
                CheckResult(
                    id=cid,
                    name=_CHECK_NAMES.get(cid, f"check_{cid}"),
                    passed=False,
                    reason="check missing from critic response",
                )
            )
    merged.sort(key=lambda c: c.id)

    final_verdict = (
        "ACCEPT" if (verdict == "ACCEPT" and all(c.passed for c in merged)) else "REJECT"
    )
    return CriticResult(verdict=final_verdict, checks=merged)


__all__ = [
    "CheckResult",
    "CriticResult",
    "critique",
]
