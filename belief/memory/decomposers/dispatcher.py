"""Decomposition dispatcher (mycorrhizal Stage 7, Area 4).

Inspects a build outcome and routes it to one or more of the three
decomposition tiers. A build can be processed by several paths — partial
recovery from the easy + structural tiers is fine and expected.

This is a *pure analysis* function: it returns a ``DecompositionResult``
describing what was extracted. The caller (the decomposer node hook) decides
what to persist and where. Keeping persistence out of here makes the tiers
trivially testable and means a dispatch never has soil side effects of its
own.

Routing rules (from the brief):
  * Any failed build with parseable code → easy tier (harvest clean frags).
  * Integration-level failure (parts parsed, composition failed) →
    structural tier (import + call edges).
  * Opaque/systemic failure (error trail present) → recalcitrant tier
    (failure signature).
A successful build is left to the existing LLM decomposer; the three-tier
path is specifically for extracting value from *failures*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from belief.memory.decomposers.easy import CleanFragment, extract_clean_fragments
from belief.memory.decomposers.recalcitrant import (
    FailureSignature,
    extract_failure_signature,
)
from belief.memory.decomposers.structural import (
    CompositionEdge,
    extract_composition_edges,
)

logger = logging.getLogger("belief.memory.decomposers.dispatcher")


@dataclass
class DecompositionResult:
    """What the three tiers recovered from one failed build."""

    build_id: str
    build_passed: bool
    tiers_run: list[str] = field(default_factory=list)
    clean_fragments: list[CleanFragment] = field(default_factory=list)
    composition_edges: list[CompositionEdge] = field(default_factory=list)
    failure_signature: FailureSignature | None = None

    @property
    def recovered_anything(self) -> bool:
        return bool(self.clean_fragments or self.composition_edges or self.failure_signature)

    def summary(self) -> dict:
        return {
            "build_id": self.build_id,
            "build_passed": self.build_passed,
            "tiers_run": list(self.tiers_run),
            "clean_fragments": len(self.clean_fragments),
            "composition_edges": len(self.composition_edges),
            "failure_signature": (
                self.failure_signature.signature_id if self.failure_signature else None
            ),
        }


def _state_get(state: dict[str, Any], key: str, default=None):
    return state.get(key, default)


def decompose_failed_build(state: dict[str, Any]) -> DecompositionResult:
    """Run the appropriate decomposition tiers for a build state.

    ``state`` is the build's UnifiedState dict (run_id, code_files, errors,
    execution_result, validation_result). Pure: returns a result, persists
    nothing. Safe to call on a passing build — it simply runs no tiers and
    returns an empty result (the LLM decomposer owns successes).
    """
    build_id = str(_state_get(state, "run_id", "unknown"))

    # Determine pass/fail (mirror the existing decomposer's verdict logic).
    validation = _state_get(state, "validation_result")
    verdict = None
    if validation:
        verdict = (
            validation.get("verdict")
            if isinstance(validation, dict)
            else getattr(validation, "verdict", None)
        )
        if verdict is not None and hasattr(verdict, "value"):
            verdict = verdict.value
    exec_result = _state_get(state, "execution_result")
    exec_success = False
    exec_error = ""
    if exec_result:
        exec_success = (
            exec_result.get("success")
            if isinstance(exec_result, dict)
            else getattr(exec_result, "success", False)
        )
        exec_error = (
            exec_result.get("error_summary", "")
            if isinstance(exec_result, dict)
            else getattr(exec_result, "error_summary", "")
        ) or ""
    build_passed = (verdict == "pass") or bool(exec_success)

    result = DecompositionResult(build_id=build_id, build_passed=build_passed)
    if build_passed:
        # Successful builds are the LLM decomposer's domain.
        return result

    code_files: dict[str, str] = _state_get(state, "code_files", {}) or {}
    errors: list[str] = _state_get(state, "errors", []) or []

    # ── Easy tier: clean fragments from parseable files ─────────────────
    try:
        frags = extract_clean_fragments(code_files, source_build_id=build_id)
        if frags:
            result.clean_fragments = frags
            result.tiers_run.append("easy")
    except Exception as e:  # pragma: no cover — best-effort
        logger.debug("easy tier failed: %s", e)

    # ── Structural tier: composition edges (integration-level failures) ──
    # Run when the build produced multiple files OR multiple clean frags —
    # i.e., there was something to compose. The failure annotation carries
    # the last error so the edge records why the composition didn't hold.
    try:
        if len(code_files) >= 1:
            annotation = errors[-1][:200] if errors else exec_error[:200]
            edges = extract_composition_edges(
                code_files, source_build_id=build_id, failure_annotation=annotation
            )
            if edges:
                result.composition_edges = edges
                result.tiers_run.append("structural")
    except Exception as e:  # pragma: no cover — best-effort
        logger.debug("structural tier failed: %s", e)

    # ── Recalcitrant tier: failure signature (opaque/systemic) ──────────
    try:
        sig = extract_failure_signature(
            errors=errors,
            exec_error=exec_error,
            last_agent_messages=_state_get(state, "agent_messages", []),
            source_build_id=build_id,
        )
        if sig is not None:
            result.failure_signature = sig
            result.tiers_run.append("recalcitrant")
    except Exception as e:  # pragma: no cover — best-effort
        logger.debug("recalcitrant tier failed: %s", e)

    return result
