"""
Decomposer — post-build nutrient extraction for the Metabolization Architecture.

Runs after every build (pass or fail) as the last node before END.
Extracts atomic, verified, reusable nutrients from the build result
and deposits them in ChromaDB soil.

Extraction rules:
  SUCCESS → PATTERNs for what worked, SKELETON if skeleton was clean
  FAILURE → ANTIPATTERNs for root cause, COVENANT if 3+ same antipattern
  ALWAYS  → cost/timing signals as metadata

Uses Sonnet (not Haiku) for genuine abstraction in nutrient extraction.
The decomposer never fails the build — errors are logged and swallowed.

Verification gate (Voyager pattern):
  PATTERN must come from a passing build
  ANTIPATTERN must come from a build with a concrete error
  SKELETON must come from a build with 0 syntax errors
  COVENANT requires 3+ antipatterns with the same root cause

Source: METABOLIZATION_BUILD_PLAN.md Phase 3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from belief.memory.nutrients import Nutrient, NutrientTier, NutrientType

logger = logging.getLogger("belief.memory.decomposer")

# ── Soil accessor ────────────────────────────────────────────────────────────

_soil_instance = None


def _get_soil():
    """Lazy-load the global Soil instance."""
    global _soil_instance
    if _soil_instance is None:
        from belief.memory.soil import Soil

        soil_dir = Path("~/.belief-engine/soil").expanduser()
        _soil_instance = Soil(soil_dir)
    return _soil_instance


# ── Pydantic schema for LLM extraction ───────────────────────────────────────


class ExtractedNutrient(BaseModel):
    """A single nutrient extracted by the LLM from a build result."""

    nutrient_type: str = Field(description="One of: pattern, antipattern, skeleton, covenant")
    content: str = Field(
        description="The knowledge itself — a concise, transferable insight. "
        "Must be generalizable beyond this specific build."
    )
    embedding_text: str = Field(
        description="Natural language description optimized for semantic search. "
        "Should describe WHAT the pattern is about, not the specific build."
    )
    difficulty: float = Field(
        default=5.0,
        description="How complex is this pattern? 1=trivial idiom, 10=architectural pattern",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags: framework names, concepts, error types",
    )
    framework: str = Field(
        default="", description="Primary framework if applicable: fastapi, fastmcp, etc."
    )
    code_sample: str = Field(
        default="", description="Representative code snippet if this is a code pattern (optional)"
    )


class DecomposerOutput(BaseModel):
    """Structured output from the decomposer LLM call."""

    nutrients: list[ExtractedNutrient] = Field(
        description="3-7 atomic nutrients extracted from this build result"
    )
    build_summary: str = Field(default="", description="One-line summary of the build outcome")


# ── Prompt ───────────────────────────────────────────────────────────────────

DECOMPOSER_SYSTEM = """\
You are the Decomposer for an autonomous code generation engine's memory system.

Your job: extract atomic, reusable NUTRIENTS from a build result. A nutrient is
a single piece of knowledge that could help a future build succeed.

NUTRIENT TYPES:
- pattern: What worked. A transferable insight about code structure, imports,
  architecture, or process. Must come from something that SUCCEEDED.
- antipattern: What failed and why. A transferable warning. Must reference a
  CONCRETE error, not speculation.
- skeleton: A SkeletonArtifact or file structure that produced clean code.
  Only extract if the skeleton led to 0 syntax/import errors.
- covenant: An immutable rule. Only propose if you see evidence of the SAME
  failure pattern appearing 3+ times.

EXTRACTION RULES:
1. Each nutrient must be SELF-CONTAINED — understandable without this build's context
2. Each nutrient must be SINGLE-CONCEPT — one idea per nutrient
3. Each nutrient must be GENERALIZABLE — strip project-specific names
4. Each nutrient must be SEARCHABLE — embedding_text should match future queries
5. Do NOT extract trivially obvious things ("code should have correct syntax")
6. Do NOT speculate about what MIGHT have gone wrong — only extract from evidence
7. Extract 3-7 nutrients. Quality over quantity.

For SUCCESSFUL builds: focus on patterns and architectural decisions that worked.
For FAILED builds: focus on root causes and transferable warnings."""

DECOMPOSER_PROMPT = """\
Extract nutrients from this build result:

BUILD ID: {build_id}
GOAL: {goal}
VERDICT: {verdict}
ITERATION COUNT: {iterations}

FILES GENERATED ({file_count} files):
{file_summary}

EXECUTION RESULT:
  Success: {exec_success}
  Error: {exec_error}

ERRORS/WARNINGS:
{errors}

{skeleton_context}

Extract 3-7 atomic nutrients. For each, provide:
- nutrient_type (pattern/antipattern/skeleton/covenant)
- content (the transferable knowledge)
- embedding_text (search-optimized description)
- difficulty (1-10)
- tags and framework if applicable"""


# ── Tier detection ───────────────────────────────────────────────────────────


def _detect_tier(file_count: int, has_packages: bool) -> NutrientTier:
    """Detect the complexity tier of a build from its output."""
    if file_count <= 1:
        return NutrientTier.TIER_1
    if file_count <= 4 and not has_packages:
        return NutrientTier.TIER_2
    if file_count <= 15:
        return NutrientTier.TIER_3
    if file_count <= 35:
        return NutrientTier.TIER_4
    return NutrientTier.TIER_5


# ── Verification gate ────────────────────────────────────────────────────────


def _verify_nutrient(
    extracted: ExtractedNutrient,
    build_passed: bool,
    has_concrete_error: bool,
    skeleton_clean: bool,
) -> bool:
    """Voyager-pattern verification: only store verified nutrients.

    Returns True if the nutrient passes its type-specific gate.
    """
    ntype = extracted.nutrient_type.lower()

    if ntype == "pattern":
        # Patterns must come from successful builds or specific fixes
        return build_passed

    if ntype == "antipattern":
        # Antipatterns must reference a concrete error
        return has_concrete_error

    if ntype == "skeleton":
        # Skeletons must come from builds with clean syntax
        return skeleton_clean and build_passed

    if ntype == "covenant":
        # Covenants are proposed by LLM but verified via soil history
        # (3+ matching antipatterns). We store them as antipatterns first,
        # and promote to covenants in the soil's lineage system.
        # For now, accept LLM-proposed covenants at reduced confidence.
        return has_concrete_error

    return False


# ── The node ─────────────────────────────────────────────────────────────────


async def decomposer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Post-build LangGraph node: extract nutrients and deposit in soil.

    Runs after polarity_check, before END. Always succeeds — errors are
    logged but never fail the build.

    Wiring (in graph.py):
        graph.add_node("decomposer", decomposer_node)
        # Replace the polarity_check → END edge with:
        # polarity_check → decomposer → END
    """
    result = dict(state)

    try:
        nutrients = await _extract_and_deposit(state)
        result["extracted_nutrients"] = [
            {"id": n.nutrient_id, "type": n.nutrient_type.value, "content": n.content}
            for n in nutrients
        ]
        if nutrients:
            logger.info(
                f"Decomposer: extracted {len(nutrients)} nutrients "
                f"({', '.join(n.nutrient_type.value for n in nutrients)})"
            )
        else:
            logger.info("Decomposer: no nutrients extracted from this build")
    except Exception as e:
        logger.warning(f"Decomposer: failed (non-fatal): {e}")
        result["extracted_nutrients"] = []

    # Record build episode for crystallizer analysis
    try:
        from belief.memory.episode_recorder import record_episode

        soil = _get_soil()
        episode_id = record_episode(soil, state)
        logger.debug(f"Decomposer: recorded episode {episode_id}")
    except Exception as e:
        logger.debug(f"Decomposer: episode recording skipped: {e}")

    # Every 10 builds, run one recombination to cross-pollinate the soil
    try:
        soil = _get_soil()
        build_count = soil.count()
        if build_count > 0 and build_count % 10 == 0:
            from belief.memory.recombination import RecombinationEngine

            engine = RecombinationEngine(soil)
            nutrient = await engine.recombine_once()
            if nutrient:
                logger.info(f"Decomposer: recombination produced {nutrient.nutrient_id}")
    except Exception as e:
        logger.debug(f"Decomposer: recombination skipped: {e}")

    # Mycorrhizal Stage 1: charge the constructing agent for compute spent
    # on this build. We hook here (not at every agent boundary) because the
    # decomposer is the canonical end-of-build node — exactly one charge
    # per run, idempotent on run_id, never raises. Token totals are pulled
    # from state.token_usage if the upstream agents populated it; otherwise
    # we fall back to a placeholder cost of 1.0 so the agent still appears
    # in the ledger with a non-zero denominator.
    try:
        from belief.memory.reciprocity import get_default_ledger

        constructor = state.get("agent_id") or "belief_engine"
        run_id = state.get("run_id") or "unknown"
        token_usage = state.get("token_usage") or {}
        if isinstance(token_usage, dict):
            cost = float(
                token_usage.get("total_prompt_tokens", 0)
                + token_usage.get("total_completion_tokens", 0)
            )
        else:
            cost = float(
                getattr(token_usage, "total_prompt_tokens", 0)
                + getattr(token_usage, "total_completion_tokens", 0)
            )
        if cost <= 0:
            cost = 1.0  # placeholder for builds without token tracking
        get_default_ledger().record_request(
            agent_id=str(constructor),
            cost=cost,
            idempotency_key=f"request:{run_id}",
        )
    except Exception as e:  # pragma: no cover — best effort, never blocks build
        logger.debug(f"Reciprocity request charge skipped: {e}")

    return result


async def _extract_and_deposit(state: dict[str, Any]) -> list[Nutrient]:
    """Core extraction logic: LLM call → verify → deposit."""
    from belief.config.models import ModelRouter
    from belief.llm import LLMClient

    # Gather build context from state
    build_id = state.get("run_id", "unknown")
    goal = state.get("user_goal", "")
    spec = state.get("requirement_spec")
    if spec:
        goal = (
            spec.get("goal_refined")
            if isinstance(spec, dict)
            else getattr(spec, "goal_refined", "")
        ) or goal

    # Determine verdict
    validation = state.get("validation_result")
    verdict = "unknown"
    if validation:
        v = (
            validation.get("verdict")
            if isinstance(validation, dict)
            else getattr(validation, "verdict", None)
        )
        if v:
            verdict = v.value if hasattr(v, "value") else str(v)

    exec_result = state.get("execution_result")
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
        )

    build_passed = verdict == "pass" or exec_success
    has_concrete_error = bool(exec_error)

    # File info
    code_files = state.get("code_files", {})
    file_count = len(code_files)
    has_packages = any("/" in f for f in code_files)
    tier = _detect_tier(file_count, has_packages)

    # File summary (truncated for token budget)
    file_lines = []
    for fname in sorted(code_files.keys()):
        content = code_files[fname]
        line_count = content.count("\n") + 1
        file_lines.append(f"  {fname} ({line_count} lines)")
    file_summary = "\n".join(file_lines[:20])
    if len(code_files) > 20:
        file_summary += f"\n  ... and {len(code_files) - 20} more files"

    # Skeleton context
    skeleton = state.get("skeleton_artifact")
    skeleton_clean = False
    skeleton_context = ""
    if skeleton:
        skeleton_clean = True  # If skeleton exists, it passed architect validation
        if isinstance(skeleton, dict):
            skeleton_context = f"SKELETON USED: {skeleton.get('name', 'unnamed')}"
        else:
            skeleton_context = f"SKELETON USED: {getattr(skeleton, 'name', 'unnamed')}"

    # Errors and warnings
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])
    error_text = "\n".join(f"  ERROR: {e}" for e in errors[-5:])
    if warnings:
        error_text += "\n" + "\n".join(f"  WARNING: {w}" for w in warnings[-3:])
    if exec_error:
        error_text = f"  EXEC: {exec_error}\n" + error_text
    if not error_text.strip():
        error_text = "  (none)"

    iterations = state.get("iteration", 0)

    # Build the prompt
    prompt = DECOMPOSER_PROMPT.format(
        build_id=build_id,
        goal=goal,
        verdict=verdict,
        iterations=iterations,
        file_count=file_count,
        file_summary=file_summary or "  (no files)",
        exec_success=exec_success,
        exec_error=exec_error or "(none)",
        errors=error_text,
        skeleton_context=skeleton_context,
    )

    # LLM call — uses Sonnet for genuine abstraction (review correction #2)
    router = ModelRouter()
    llm = LLMClient(router)
    deposited: list[Nutrient] = []

    try:
        extraction = await llm.generate_structured(
            role="decomposer",  # Falls through to default Sonnet
            system=DECOMPOSER_SYSTEM,
            prompt=prompt,
            response_schema=DecomposerOutput,
            temperature=0.3,
            max_tokens=4000,
        )

        soil = _get_soil()

        for ext in extraction.nutrients:
            # Verification gate
            if not _verify_nutrient(ext, build_passed, has_concrete_error, skeleton_clean):
                logger.debug(f"Decomposer: skipped {ext.nutrient_type} (failed verification gate)")
                continue

            # Map to NutrientType enum
            try:
                ntype = NutrientType(ext.nutrient_type.lower())
            except ValueError:
                logger.debug(f"Decomposer: unknown type '{ext.nutrient_type}', skipping")
                continue

            # Build Nutrient
            nutrient = Nutrient(
                nutrient_type=ntype,
                tier=tier,
                content=ext.content,
                embedding_text=ext.embedding_text,
                code_sample=ext.code_sample or None,
                difficulty=max(1.0, min(10.0, ext.difficulty)),
                source_build_id=build_id,
                tags=ext.tags,
                framework=ext.framework or None,
            )

            # Deposit (handles dedup + lineage internally). The returned
            # id is the nutrient's own when novel, or the dedup-target's
            # id when it merged into an existing nutrient — Stage 2 reads
            # that signal to gate the niche-primitive registration.
            deposited_id = soil.deposit(nutrient)
            deposited.append(nutrient)
            is_novel = deposited_id == nutrient.nutrient_id

            # Mycorrhizal Stage 1: credit the constructing agent in the
            # reciprocity ledger. The credit is per-nutrient (value 1.0),
            # keyed by nutrient_id so a re-run of the decomposer for the
            # same build never double-counts. Failures here MUST NOT
            # propagate — the decomposer never fails the build.
            try:
                from belief.memory.reciprocity import get_default_ledger

                constructor = state.get("agent_id") or "belief_engine"
                get_default_ledger().record_contribution(
                    agent_id=str(constructor),
                    nutrient_value=1.0,
                    nutrient_id=nutrient.nutrient_id,
                    idempotency_key=f"contrib:{nutrient.nutrient_id}",
                )
            except Exception as rec_err:  # pragma: no cover — best effort
                logger.debug(f"Reciprocity contribution credit skipped: {rec_err}")

            # Mycorrhizal Stage 2: novel nutrients are niche modifications
            # (new primitives the system can compose with). Dedup-merged
            # nutrients reinforce an existing niche and don't register a
            # new one. The pre/post descriptions stay short — the
            # nutrient's own content carries the detail. Best-effort.
            if is_novel:
                try:
                    from belief.memory.niche_ledger import get_default_ledger as _nl

                    constructor = state.get("agent_id") or "belief_engine"
                    _nl().record_modification(
                        constructing_agent_id=str(constructor),
                        kind="primitive",
                        soil_reference=nutrient.nutrient_id,
                        pre_state_description=(f"no soil primitive for: {ext.embedding_text[:80]}"),
                        post_state_description=(f"{ext.nutrient_type}: {ext.content[:120]}"),
                    )
                except Exception as nl_err:  # pragma: no cover — best effort
                    logger.debug(f"Niche primitive record skipped: {nl_err}")

            # Mycorrhizal Stage 6: a freshly-extracted antipattern is an
            # unconfirmed failure mode — emit a priming-class warning so
            # future operations on the same pattern raise a sentinel. This
            # is priming, NOT a covenant block: it never refuses anything
            # (the build pipeline doesn't call check_operation). Best-effort.
            if ntype == NutrientType.ANTIPATTERN:
                try:
                    from belief.safety.priming import get_default_propagator

                    constructor = state.get("agent_id") or "belief_engine"
                    get_default_propagator().emit_priming(
                        pattern=ext.embedding_text[:80],
                        evidence={
                            "build_id": build_id,
                            "nutrient_id": nutrient.nutrient_id,
                        },
                        originating_agent_id=str(constructor),
                    )
                except Exception as pr_err:  # pragma: no cover — best effort
                    logger.debug(f"Priming warning emit skipped: {pr_err}")

    except Exception as e:
        logger.warning(f"Decomposer LLM call failed: {e}")
    finally:
        await llm.close()

    return deposited
