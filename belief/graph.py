"""The Belief Engine Pipeline — a LangGraph conditional state graph.

Source: forge/pipeline.py (conditional routing with circuit breakers)
       + taskforce_base.py (Latios post-build gap check)
       + METABOLIZATION_BUILD_PLAN.md (recomposer + decomposer nodes)

    recomposer → intake → research → planner → architect → builder → tester → executor → gap_analyst
                                                                                              │
    ┌──── requires_research ──────────────────────────────────────────────────────────────────┘
    │          ┌── has_blockers (code crashed) ────────────────────────────────────────────────┘
    │          │     ┌── has_major (code ran, incomplete) ─────────────────────────────────────┘
    │          │     │          ┌── clean ─────────────────────────────────────────────────────┘
    ▼          ▼     ▼          ▼
 research   debugger builder   synthesizer → validator → polarity_check
                │                                             │
                └→ executor → gap_analyst             ┌───────┴───────┐
                                                  no gap         significant gap
                                                      │              │
                                                  decomposer   increment → planner
                                                      │
                                                     END
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph

from belief.agents.architect import ArchitectAgent
from belief.agents.builder import BuilderAgent
from belief.agents.debugger import DebuggerAgent
from belief.agents.executor import ExecutorAgent
from belief.agents.gap_analyst import GapAnalystAgent
from belief.agents.intake import IntakeAgent
from belief.agents.planner import PlannerAgent
from belief.agents.research import ResearchAgent
from belief.agents.skeleton_pass1 import skeleton_pass1_node
from belief.agents.synthesizer import SynthesizerAgent
from belief.agents.tester import TesterAgent
from belief.agents.validator import ValidatorAgent
from belief.config.models import ModelRouter
from belief.llm import ceiling_for_node
from belief.memory.decomposer import decomposer_node
from belief.memory.recomposer import recomposer_node
from belief.models.state import Phase

logger = logging.getLogger("belief.graph")


# ── Routing functions ─────────────────────────────────────────────────────────


def _normalize_error(error: str) -> str:
    """Normalize an error string for deduplication (strip line numbers, paths, timestamps)."""
    import re

    s = error.lower().strip()
    s = re.sub(r"line \d+", "line N", s)
    s = re.sub(r"/tmp/belief_\w+/", "/tmp/", s)
    s = re.sub(r"belief-\w{8}", "belief-XXXX", s)
    s = re.sub(r"\d{2}:\d{2}:\d{2}", "HH:MM:SS", s)
    return s


def _error_hash(error: str) -> str:
    """Hash a normalized error for dedup tracking."""
    import hashlib

    return hashlib.md5(_normalize_error(error).encode()).hexdigest()[:12]


def _maybe_apply_confidence_probe(state: dict[str, Any]) -> Optional[str]:
    """Session 10: consult the confidence probe before normal routing.

    Returns a next-node name ONLY when the probe wants to circuit-break;
    otherwise None (caller proceeds with the normal routing logic).
    Also sets state['escalate_to_cloud'] when running local and
    confidence is in the mid-band. No-op unless the probe file exists
    and BELIEF_ENABLE_PROBE env var is set, so default routing is
    unchanged.
    """
    try:
        from belief.safety.confidence_probe import (
            get_default_probe,
            is_probe_routing_enabled,
        )
    except Exception:
        return None
    if not is_probe_routing_enabled():
        return None

    probe = get_default_probe()
    # Feature-row from the most-recently-traced step. The probe wants a
    # dict that matches TraceCollector's row shape; we synthesize one
    # from current state.
    budget = state.get("budget")
    cost = float(getattr(budget, "spent_usd", 0.0) or 0.0) if budget else 0.0
    row = {
        "agent_name": state.get("_last_agent", "gap_analyst"),
        "iteration": int(state.get("iteration", 0) or 0),
        "cost_so_far": cost,
        "output_summary": str(state.get("last_agent_output") or "")[:500],
        "step_index": int(state.get("_step_index", 0) or 0),
        "build_passed": None,
    }
    try:
        confidence = probe.predict_confidence(row)
    except Exception:
        return None

    if confidence < 0.4:
        logger.warning(
            "confidence probe: low confidence (%.2f) — circuit-breaking to synthesizer",
            confidence,
        )
        return "synthesizer"
    # Mid-band with local mode: flag for escalation
    if confidence < 0.8 and str(state.get("model_mode") or "").lower() == "local":
        state["escalate_to_cloud"] = True
    return None


def _route_after_gap(
    state: dict[str, Any],
) -> Literal["research", "debugger", "builder", "synthesizer"]:
    # Session 10: probe-driven circuit-break. No-op unless enabled + trained.
    probe_decision = _maybe_apply_confidence_probe(state)
    if probe_decision == "synthesizer":
        return "synthesizer"

    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # ── Hard iteration cap (unconditional) ────────────────────────────
    # After 3 debugger iterations, ALWAYS circuit-break regardless of
    # error classification. This prevents infinite retry loops when a
    # transient classification (e.g. "429" in an import error string)
    # masks a non-transient failure.
    if iteration >= max_iter:
        logger.warning(
            f"Hard iteration cap reached ({iteration}/{max_iter}) — circuit-breaking to synthesizer"
        )
        return "synthesizer"

    # ── OTP-style error classification ──────────────────────────────
    # Instead of ad-hoc checks, classify the error and route based on
    # the recovery strategy. This is the Erlang supervision pattern
    # applied to LLM agent pipelines.
    exec_r = state.get("execution_result")

    # If execution succeeded, the code WORKS — go to synthesizer
    exec_success = False
    if exec_r:
        exec_success = (
            exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)
        )
    if exec_success:
        return "synthesizer"

    # Classify the error
    if exec_r:
        error = (
            exec_r.get("error_summary", "")
            if isinstance(exec_r, dict)
            else getattr(exec_r, "error_summary", "")
        )
        stderr = (
            exec_r.get("stderr", "") if isinstance(exec_r, dict) else getattr(exec_r, "stderr", "")
        )

        if error or stderr:
            from belief.agents.error_classifier import classify_error, RecoveryStrategy

            previous_errors = state.get("_previous_error_summaries", [])
            classified = classify_error(
                error_summary=error,
                stderr=stderr,
                iteration=iteration,
                max_iterations=max_iter,
                previous_errors=previous_errors,
            )

            logger.info(
                f"Error classified: {classified.category.value} → {classified.strategy.value} ({classified.reason})"
            )

            if classified.strategy == RecoveryStrategy.FAIL_FAST:
                return "synthesizer"
            elif classified.strategy == RecoveryStrategy.CIRCUIT_BREAK:
                return "synthesizer"
            elif classified.strategy == RecoveryStrategy.RETRY_BACKOFF:
                # Transient — retry via debugger (which will re-execute)
                return "debugger"
            elif classified.strategy == RecoveryStrategy.REPROMPT:
                if classified.should_rebuild:
                    return "builder"
                return "debugger"

    # No execution result or no error — check gap report
    gap = state.get("gap_report")
    if gap is None:
        return "synthesizer"

    if isinstance(gap, dict):
        requires_research = gap.get("requires_research", False)
        total_blockers = gap.get("total_blockers", 0)
    else:
        requires_research = getattr(gap, "requires_research", False)
        total_blockers = getattr(gap, "total_blockers", 0)

    if requires_research:
        return "research"
    if total_blockers > 0:
        return "debugger"

    return "builder"


_REFINEMENT_ELIGIBLE_ERROR_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "cannot import name",
    "Test collection failed",
    "Smoke test:",
    "missing_symbol",
    # Failing tests are the water cycle's core use case — the fixer is
    # literally designed to read pytest output and patch the offending
    # code. Previously excluded because the marker list was scoped to
    # import errors only, which meant exec-failed builds with genuine
    # test failures circuit-broke instead of getting a refinement pass.
    "NameError",
    "AttributeError",
    "TypeError",
)


def _exec_error_is_refinable(exec_r: Any) -> bool:
    """True if the executor's error summary is in a class refinement can patch.

    Refinement's fixer is good at surgical import/export edits but can't
    untangle genuine logic errors. We route to refinement only for the
    failure classes the fixer actually handles — missing imports, missing
    symbols, failed test collection, smoke-test import failures.
    """
    if not exec_r:
        return False
    summary = (
        exec_r.get("error_summary")
        if isinstance(exec_r, dict)
        else getattr(exec_r, "error_summary", "")
    ) or ""
    stderr = (
        exec_r.get("stderr") if isinstance(exec_r, dict) else getattr(exec_r, "stderr", "")
    ) or ""
    haystack = f"{summary}\n{stderr}"
    return any(marker in haystack for marker in _REFINEMENT_ELIGIBLE_ERROR_MARKERS)


def _route_after_validation(
    state: dict[str, Any],
) -> Literal["polarity_check", "builder", "research", "refinement"]:
    exec_r = state.get("execution_result")
    exec_ok = False
    if exec_r:
        exec_ok = (
            exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)
        )

    validation = state.get("validation_result")
    if validation is None:
        return "polarity_check"

    verdict = (
        validation.get("verdict")
        if isinstance(validation, dict)
        else getattr(validation, "verdict", "pass")
    )
    if isinstance(verdict, str):
        v = verdict
    else:
        v = verdict.value if hasattr(verdict, "value") else str(verdict)

    # Water Cycle: if the code RUNS but has quality issues,
    # ALWAYS send to refinement — regardless of iteration count.
    # Refinement is polishing, not rebuilding. It doesn't cost a rebuild iteration.
    if v == "fail_fixable" and exec_ok:
        return "refinement"

    # Executor-failed, but the failure is in a class refinement's fixer
    # can patch (missing exports, import paths, failed test collection,
    # failing tests). Refinement is a terminal sink (refinement →
    # decomposer → END), so it's safe to let it run regardless of
    # iteration budget — it can't loop back into the main pipeline.
    # Previously these bypassed refinement entirely and burned rebuild
    # iterations on the debugger's immutable-skeleton deadlock.
    if v == "fail_fixable" and not exec_ok and _exec_error_is_refinable(exec_r):
        logger.info("Router: executor failed with refinable error — routing to refinement")
        return "refinement"

    # For rebuilds (not refinement), check iteration limit
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if iteration >= max_iter:
        return "polarity_check"

    if v in ("pass", "fail_unfixable"):
        return "polarity_check"
    if v == "fail_rethink":
        return "research"
    if v == "fail_fixable":
        return "builder"
    return "polarity_check"


def _route_after_polarity(state: dict[str, Any]) -> Literal["planner", "__end__"]:
    """The outer loop — Latios convergence check."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if iteration >= max_iter:
        return END

    # If code passed execution, don't rebuild — it WORKS.
    # Latios "significant gap" on working code is a quality complaint, not a functional failure.
    # Rebuilding working code always produces worse results.
    exec_r = state.get("execution_result")
    if exec_r:
        exec_ok = (
            exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)
        )
        if exec_ok:
            return END

    polarity = state.get("polarity", {})
    remainder = (
        polarity.get("current_remainder")
        if isinstance(polarity, dict)
        else getattr(polarity, "current_remainder", None)
    )
    if remainder and "significant" in str(remainder).lower():
        return "planner"

    return END


# ── Polarity check node ──────────────────────────────────────────────────────


def _make_polarity_check_node(router: ModelRouter):
    """Factory: returns a _polarity_check_node that closes over the pipeline router.

    Using a closure keeps the router consistent with the rest of the pipeline so
    BELIEF_MODEL_MODE=local routes the Latios call to Ollama instead of cloud.
    """

    async def _polarity_check_node(state: dict[str, Any]) -> dict[str, Any]:
        return await _polarity_check_impl(state, router)

    return _polarity_check_node


async def _polarity_check_impl(state: dict[str, Any], router: ModelRouter) -> dict[str, Any]:
    """Post-validation Latios gap check.

    OVERRIDE: When tests pass AND validator says pass, skip the LLM call entirely.
    Latios hallucinates "no code provided" on passing builds, wasting tokens
    and triggering unnecessary rebuild iterations.
    """
    from belief.llm import LLMClient
    from belief.prompts import LATIOS_SYSTEM, LATIOS_PROMPT

    result = dict(state)
    result["phase"] = Phase.POLARITY_CHECK.value

    # ── Deterministic override: tests pass + validator pass = done ──
    validation = state.get("validation_result")
    exec_result = state.get("execution_result")

    exec_success = False
    if exec_result:
        exec_success = (
            exec_result.get("success")
            if isinstance(exec_result, dict)
            else getattr(exec_result, "success", False)
        )

    verdict = "unknown"
    if validation:
        verdict = (
            validation.get("verdict")
            if isinstance(validation, dict)
            else getattr(validation, "verdict", "unknown")
        )
        if hasattr(verdict, "value"):
            verdict = verdict.value

    if exec_success and verdict == "pass":
        logger.info("Latios: SKIPPED — tests pass + validator pass (deterministic override)")
        result["polarity"] = {"current_remainder": None, "accumulated_remainders": []}
        return result

    spec = state.get("requirement_spec")

    goal = ""
    criteria = ""
    if spec:
        if isinstance(spec, dict):
            goal = spec.get("goal", state.get("user_goal", ""))
            criteria = "\n".join(f"  - {c}" for c in spec.get("acceptance_criteria", []))
        else:
            goal = spec.goal
            criteria = "\n".join(f"  - {c}" for c in spec.acceptance_criteria)

    if not goal:
        goal = state.get("user_goal", "")

    validator_summary = ""
    if validation:
        if isinstance(validation, dict):
            validator_summary = validation.get("summary", "")
        else:
            validator_summary = getattr(validation, "summary", "")

    llm = LLMClient(router)
    # Session A: enforce the build cost ceiling for this tail node's LLM call.
    try:
        async with ceiling_for_node(state):
            prompt = LATIOS_PROMPT.format(
                goal=goal,
                acceptance_criteria=criteria or "  (none specified)",
                validator_summary=validator_summary or "  (no validator output)",
                exec_success=exec_success,
            )
            raw = await llm.generate_text(
                role="latios",
                system=LATIOS_SYSTEM,
                prompt=prompt,
                temperature=0.2,
                max_tokens=800,
            )

            # Parse JSON response
            match = re.search(r'\{[^{}]*"significant_gap"[^{}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                polarity = dict(result.get("polarity", {}))
                if data.get("significant_gap"):
                    polarity["current_remainder"] = f"significant: {data.get('gap_summary', '')}"
                    polarity["accumulated_remainders"] = polarity.get(
                        "accumulated_remainders", []
                    ) + [data.get("gap_summary", "")]
                    logger.info(f"Latios: SIGNIFICANT gap — {data.get('gap_summary', '')[:100]}")
                    # Increment iteration for the re-run
                    result["iteration"] = state.get("iteration", 0) + 1
                else:
                    polarity["current_remainder"] = None
                    logger.info(f"Latios: gap is {data.get('gap_level', 'NONE')} — complete")

                result["polarity"] = polarity
            else:
                logger.warning("Latios: could not parse output — treating as complete")

    except Exception as e:
        logger.warning(f"Latios check failed: {e}")
    finally:
        await llm.close()

    return result


# ── Increment iteration node ─────────────────────────────────────────────────


def _increment_iteration(state: dict[str, Any]) -> dict[str, Any]:
    result = dict(state)
    result["iteration"] = state.get("iteration", 0) + 1

    # Track errors for the OTP-style classifier
    exec_r = state.get("execution_result")
    if exec_r:
        error = (
            exec_r.get("error_summary", "")
            if isinstance(exec_r, dict)
            else getattr(exec_r, "error_summary", "")
        )
        if error:
            # Hash tracking (legacy)
            hashes = list(state.get("_error_hashes", []))
            hashes.append(_error_hash(error))
            result["_error_hashes"] = hashes

            # Summary tracking (for classifier)
            summaries = list(state.get("_previous_error_summaries", []))
            summaries.append(error)
            result["_previous_error_summaries"] = summaries

    return result


# ── Covenant Enforcement node (Move 2) ───────────────────────────────────────


async def _covenant_enforce_node(state: dict[str, Any]) -> dict[str, Any]:
    """Structural enforcement of self-learned covenants.

    Runs AST-based validators that auto-fix violations:
    Python: Remove __future__ from SQLAlchemy, add missing imports, etc.
    TypeScript: Fix .js extensions, replace jest→vi, fix ethers v5→v6, etc.

    Zero LLM calls. Deterministic. Fast.

    Substrate-transfer experiment: short-circuits to a no-op when the
    BELIEF_EXPERIMENT_CONDITION env var disables covenants (``soil_only`` or
    ``raw_local``). In production / hard-gate runs the env var is unset and
    this branch is never taken.
    """
    result = dict(state)
    code_files = state.get("code_files", {})

    if not code_files:
        return result

    # Substrate-transfer experiment short-circuit.
    from belief.experiments.conditions import covenants_enabled

    if not covenants_enabled():
        logger.info("Covenant enforcer skipped: experiment condition disables covenants")
        return result

    # ── Session 2 (v3.2): LibCST Pydantic v2 + forbidden-imports pipeline ──
    # Runs BEFORE the existing AST validators so the debugger never
    # observes a v1 import.  The 4-stage pipeline (regex prepass →
    # LibCST → ruff --fix → bump-pydantic) handles what pure-AST
    # enforcement can't: Config → ConfigDict conversion, validator
    # decorator rewrites, method renames, stdlib-in-requirements.
    try:
        from belief.covenants import enforce_python_covenants_on_files

        fixed_cov, applied = enforce_python_covenants_on_files(code_files)
        if applied:
            code_files = fixed_cov
            result["code_files"] = fixed_cov
            # Group by rule for a compact log line — one per build,
            # not one per rewrite, so the logs stay readable.
            from collections import Counter

            rule_counts = Counter(a.rule for a in applied)
            summary = ", ".join(f"{rule}×{n}" for rule, n in rule_counts.most_common())
            logger.info(
                "Covenant pipeline (LibCST+ruff): %d rewrites across %d files — %s",
                len(applied),
                len({a.file for a in applied if a.file}),
                summary,
            )
    except Exception as e:
        logger.debug("LibCST covenant pipeline skipped: %s", e)

    # ── Python covenants (existing AST validators — complement LibCST) ──
    try:
        from belief.validators import enforce_all

        fixed, enforcement = enforce_all(code_files, auto_fix=True)

        if enforcement.fixes_applied > 0:
            result["code_files"] = fixed
            code_files = fixed  # Use fixed version for TS pass
            logger.info(
                f"Covenant enforcer (Python): {enforcement.fixes_applied} fixes "
                f"across {len(enforcement.files_modified)} files"
            )

        warnings = [v for v in enforcement.violations if v.severity == "warning" and not v.auto_fix]
        for w in warnings:
            logger.warning(f"Covenant warning: {w.covenant} — {w.message}")

    except Exception as e:
        logger.debug(f"Python covenant enforcement skipped: {e}")

    # ── TypeScript covenants ──
    has_ts = any(f.endswith((".ts", ".tsx", ".jsx")) for f in code_files)
    if has_ts:
        try:
            from belief.validators.typescript_fixup import fixup_typescript_output

            goal = state.get("user_goal", "")
            ts_fixed = fixup_typescript_output(code_files, goal=goal)

            if ts_fixed != code_files:
                result["code_files"] = ts_fixed
                code_files = ts_fixed
                logger.info("Covenant enforcer (TypeScript): fixup pipeline applied")

        except Exception as e:
            logger.debug(f"TypeScript fixup skipped: {e}")

    # ── Multi-service health check injection ──────────────────────
    svc_arch = state.get("service_architecture")
    if svc_arch:
        try:
            from belief.tools.multi_service import inject_health_endpoints, verify_services

            code_files = result.get("code_files", code_files)
            svc_arch_dict = svc_arch if isinstance(svc_arch, dict) else svc_arch.model_dump()

            # Auto-inject /health endpoints into services missing them
            fixed = inject_health_endpoints(code_files, svc_arch_dict)
            if fixed != code_files:
                result["code_files"] = fixed
                logger.info("Covenant enforcer: injected missing /health endpoints")

            # Verify service structure (non-blocking — logged as warnings)
            verification = verify_services(fixed, svc_arch_dict)
            if not verification.passed:
                for issue in verification.issues[:5]:
                    logger.warning(f"Multi-service: {issue}")

        except Exception as e:
            logger.debug(f"Multi-service verification skipped: {e}")

    return result


# ── Static Import Fix node (Covenant #3) ─────────────────────────────────────


async def _import_fix_node(state: dict[str, Any]) -> dict[str, Any]:
    """Statically verify and auto-fix cross-module imports after building.

    Implements Covenant #3: "Before finalizing any multi-module Python project,
    statically verify every cross-module import by tracing each `from X import Y`
    to confirm Y is defined in X."

    Runs after the builder, before the tester. Fixes what it can deterministically.
    """
    result = dict(state)
    code_files = state.get("code_files", {})

    if not code_files:
        return result

    try:
        from belief.codebase.imports import verify_imports, auto_fix_imports

        issues = verify_imports(code_files)
        if not issues:
            return result

        # Auto-fix what we can
        fixed = auto_fix_imports(code_files, issues)
        remaining = verify_imports(fixed)
        fixed_count = len(issues) - len(remaining)

        if fixed_count > 0:
            result["code_files"] = fixed
            logger.info(f"Import fix: auto-fixed {fixed_count}/{len(issues)} import issues")

        if remaining:
            unfixed = [f"{i.source_file}: {i.symbol} from {i.target_module}" for i in remaining[:3]]
            logger.warning(f"Import fix: {len(remaining)} unresolved: {'; '.join(unfixed)}")

    except Exception as e:
        logger.debug(f"Import fix skipped: {e}")

    return result


# ── Water Cycle refinement node ──────────────────────────────────────────────


def _make_refinement_node(router: ModelRouter):
    """Factory: returns a _refinement_node that closes over the pipeline router."""

    async def _refinement_node(state: dict[str, Any]) -> dict[str, Any]:
        return await _refinement_impl(state, router)

    return _refinement_node


async def _refinement_impl(state: dict[str, Any], router: ModelRouter) -> dict[str, Any]:
    """Water Cycle: targeted polishing of working code via test-driven refinement.

    Runs up to 3 cycles of analyze→fix→revalidate.
    Deposits refinement lessons into ChromaDB soil.
    """
    result = dict(state)

    try:
        from belief.refinement.runner import run_refinement_loop, store_refinement_lessons

        code_files = state.get("code_files", {})
        test_files = state.get("test_files", {})

        # Get test output — try multiple sources
        test_output = ""

        # Source 1: ExecutionResult.pytest_result.raw_output (best source)
        exec_r = state.get("execution_result")
        if exec_r:
            pytest_r = (
                exec_r.get("pytest_result")
                if isinstance(exec_r, dict)
                else getattr(exec_r, "pytest_result", None)
            )
            if pytest_r:
                test_output = (
                    pytest_r.get("raw_output", "")
                    if isinstance(pytest_r, dict)
                    else getattr(pytest_r, "raw_output", "")
                )
            if not test_output:
                test_output = (
                    exec_r.get("stdout", "")
                    if isinstance(exec_r, dict)
                    else getattr(exec_r, "stdout", "")
                )

        # Source 2: validator test output
        if not test_output:
            validation = state.get("validation_result")
            if isinstance(validation, dict):
                test_output = validation.get("test_output", "") or validation.get("summary", "")
            elif validation:
                test_output = getattr(validation, "test_output", "") or getattr(
                    validation, "summary", ""
                )

        # Source 3: tester output
        if not test_output:
            test_output = state.get("test_output", "")

        if not code_files:
            logger.warning("Refinement: no code files available")
            return result

        # If we don't have test output, or it's unparseable, run tests ourselves
        if test_files:
            from belief.refinement.analyzer import parse_test_results as _parse

            _p, _t, _, _ = _parse(test_output) if test_output else (0, 0, [], [])
            if _t == 0:
                from belief.refinement.runner import _run_tests

                test_output, _, _, _ = _run_tests(code_files, test_files)
                logger.info("Refinement: ran baseline tests to get initial output")

        # Run the water cycle (pass router so refinement uses the same backend).
        # Session A: under the build cost ceiling — the loop is the largest tail
        # spender, so its per-cycle calls are gated against --max-cost too.
        async with ceiling_for_node(state):
            refinement = await run_refinement_loop(
                code_files=code_files,
                test_files=test_files,
                initial_test_output=test_output,
                max_cycles=3,
                router=router,
            )

        # Update state with refined code
        result["code_files"] = refinement["code_files"]

        # Compute actual weighted verdict from real test results
        final_passed = refinement.get("final_passed", 0)
        final_total = refinement.get("final_total", 0)

        if refinement["verdict"] == "pass":
            result["validation_result"] = {
                "verdict": "pass",
                "weighted_score": 1.0,
                "tests_passed": final_passed,
                "tests_total": final_total,
            }
            logger.info("Refinement: verdict upgraded to PASS")
        elif final_total > 0:
            # Use actual pass rate as weighted score approximation
            actual_score = final_passed / final_total
            exec_ok = state.get("execution_result", {})
            if isinstance(exec_ok, dict):
                exec_ok = exec_ok.get("success", False)
            else:
                exec_ok = getattr(exec_ok, "success", False)

            # If code runs AND pass rate >= 75%, upgrade to PASS
            # Rationale: executor success means the code WORKS. Test failures
            # above 25% are almost always phantom failures from over-aggressive
            # test generation, not real bugs. 85% was too strict — it left
            # working builds (e.g. 17/20 tests) marked as failures.
            if exec_ok and actual_score >= 0.75:
                result["validation_result"] = {
                    "verdict": "pass",
                    "weighted_score": actual_score,
                    "tests_passed": final_passed,
                    "tests_total": final_total,
                }
                logger.info(
                    f"Refinement: verdict upgraded to PASS ({final_passed}/{final_total} = {actual_score:.0%})"
                )

        # Store lessons in soil
        lessons = refinement.get("lessons", [])
        if lessons:
            deposited = await store_refinement_lessons(lessons)
            logger.info(f"Refinement: deposited {deposited} lessons into soil")

        logger.info(
            f"Refinement: {refinement['exit_reason']} — "
            f"{refinement.get('cycles_used', 0)} cycles, "
            f"+{refinement.get('improvement', 0)} tests"
        )

    except Exception as e:
        logger.warning(f"Refinement failed: {e}")

    return result


# ── Build the graph ───────────────────────────────────────────────────────────


def _traced(node: Any, agent_name: str) -> Any:
    """Session 9: wrap a graph node with per-step trace recording.

    Does nothing when BELIEF_ENABLE_TRACE is unset (the default) — so
    existing tests and builds see no behavior change. When the env var
    is truthy, the wrapper calls record_step_from_state after the node
    returns, capturing agent_name, cost_so_far, iteration, and a short
    output summary.

    Handles both sync functions AND LangGraph-compatible agent
    instances (whose __call__ is async). Exceptions from the trace
    recorder are always swallowed — tracing must never fail a build.
    """
    try:
        from belief.metrics.trace_collector import (
            is_tracing_enabled,
            record_step_from_state,
        )
    except Exception:
        return node
    if not is_tracing_enabled():
        return node

    import asyncio as _asyncio
    import functools as _functools

    call_target = node if callable(node) else getattr(node, "__call__", node)
    is_async = _asyncio.iscoroutinefunction(call_target) or (
        hasattr(node, "__call__") and _asyncio.iscoroutinefunction(node.__call__)
    )

    if is_async:

        @_functools.wraps(call_target)
        async def _async_wrapper(state: Any) -> Any:
            result = await node(state)
            try:
                record_step_from_state(
                    result if isinstance(result, dict) else state,
                    agent_name=agent_name,
                )
            except Exception:
                pass
            return result

        return _async_wrapper

    @_functools.wraps(call_target)
    def _sync_wrapper(state: Any) -> Any:
        result = node(state)
        try:
            record_step_from_state(
                result if isinstance(result, dict) else state,
                agent_name=agent_name,
            )
        except Exception:
            pass
        return result

    return _sync_wrapper


def _compile_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic terminal gate: no build reports ``pass`` while shipping a
    ``.py`` file that does not parse.

    Runs after refinement (cloud: before decomposer; local: before END), so it
    is the post-polish backstop the per-stage validator cannot be — the
    validator runs before the refinement pass that can re-truncate files. On a
    parse failure it forces the verdict to ``fail_fixable`` and floors the
    score, so truncated output can never archive as a 1.00 exemplar.
    """
    from belief.validators.compile_gate import gate_validation_result

    code_files = state.get("code_files") or {}
    validation_result, broken = gate_validation_result(code_files, state.get("validation_result"))
    if broken:
        state["validation_result"] = validation_result
        logger.warning(
            "compile_gate: %d generated file(s) failed to parse — "
            "verdict forced to fail_fixable: %s",
            len(broken),
            ", ".join(fname for fname, _ in broken),
        )
    else:
        n_py = sum(1 for f in code_files if f.endswith(".py"))
        logger.info("compile_gate: all %d Python file(s) parse cleanly", n_py)
    return state


def _coverage_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic terminal gate: a build can't report ``pass`` while
    producing only a fraction of the planned files, or shipping hollow stubs.

    Runs right after the compile gate (which catches files that don't parse).
    This catches files that should exist and don't, and files that exist but
    are empty stubs — handoff Q2. Records the produced-vs-planned coverage
    fraction in state so the build record / dashboard can show it, and on a
    shortfall forces ``fail_fixable`` and caps the score at what shipped.

    Detection of structural incompleteness only — no judgement about whether
    the produced files are *correct* (the deferred research question).
    """
    import os as _os

    from belief.validators.coverage_gate import gate_validation_result

    raw = _os.environ.get("BELIEF_COVERAGE_THRESHOLD", "").strip()
    threshold = 1.0
    if raw:
        try:
            threshold = max(0.0, min(1.0, float(raw)))
        except ValueError:
            logger.warning("Ignoring non-numeric BELIEF_COVERAGE_THRESHOLD=%r", raw)

    # Produced universe = source + test files: the architect's manifest can
    # list test files that land in state.test_files, so counting only
    # code_files would falsely report them missing. The hollow check inside
    # the gate already excludes test paths, so merging is safe.
    code_files = state.get("code_files") or {}
    produced = {**code_files, **(state.get("test_files") or {})}
    validation_result, coverage_fraction, missing, hollow = gate_validation_result(
        state.get("file_manifest"),
        produced,
        state.get("validation_result"),
        threshold=threshold,
    )
    state["coverage_fraction"] = coverage_fraction
    if missing or hollow:
        state["validation_result"] = validation_result
        logger.warning(
            "coverage_gate: coverage=%.0f%% — %d missing, %d hollow; "
            "verdict forced to fail_fixable%s%s",
            coverage_fraction * 100,
            len(missing),
            len(hollow),
            f" [missing: {', '.join(missing)}]" if missing else "",
            f" [hollow: {', '.join(hollow)}]" if hollow else "",
        )
    else:
        logger.info("coverage_gate: coverage=%.0f%%, no hollow files", coverage_fraction * 100)
    return state


def _structure_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic terminal gate: a build can't report ``pass`` while shipping
    two parallel implementations of the same thing (handoff Q1).

    Runs after the coverage gate. Flags the architect-fallback failure mode
    where a packaged implementation and a competing root-level monolith both
    exist (duplicate core modules / competing entry points), forcing
    ``fail_fixable`` so "which one is real?" can never archive as a pass.

    Detection only — no judgement about which implementation is correct.
    """
    from belief.validators.structure_gate import gate_validation_result

    code_files = state.get("code_files") or {}
    validation_result, findings = gate_validation_result(code_files, state.get("validation_result"))
    if findings:
        state["validation_result"] = validation_result
        logger.warning(
            "structure_gate: %d coherence defect(s) — verdict forced to fail_fixable: %s",
            len(findings),
            "; ".join(findings),
        )
    else:
        logger.info("structure_gate: single coherent structure")
    return state


def build_pipeline(router: ModelRouter | None = None) -> StateGraph:
    """Construct and compile the Belief Engine pipeline."""
    if router is None:
        router = ModelRouter()

    # Instantiate agents
    intake = IntakeAgent(router)
    research = ResearchAgent(router)
    planner = PlannerAgent(router)
    architect = ArchitectAgent(router)
    builder = BuilderAgent(router)
    tester = TesterAgent(router)
    executor = ExecutorAgent(router)
    debugger = DebuggerAgent(router)
    gap_analyst = GapAnalystAgent(router)
    synthesizer = SynthesizerAgent(router)
    validator = ValidatorAgent(router)

    graph = StateGraph(dict)

    # Nodes — each wrapped with _traced so BELIEF_ENABLE_TRACE=1 captures
    # a StepTrace after the node runs. Default (env unset): no-op.
    graph.add_node("intake", _traced(intake, "intake"))
    graph.add_node("research", _traced(research, "research"))
    graph.add_node("planner", _traced(planner, "planner"))
    graph.add_node("architect", _traced(architect, "architect"))
    graph.add_node("skeleton_pass1", _traced(skeleton_pass1_node, "skeleton_pass1"))
    graph.add_node("builder", _traced(builder, "builder"))
    graph.add_node("tester", _traced(tester, "tester"))
    graph.add_node("executor", _traced(executor, "executor"))
    graph.add_node("debugger", _traced(debugger, "debugger"))
    graph.add_node("gap_analyst", _traced(gap_analyst, "gap_analyst"))
    graph.add_node("increment_iteration", _traced(_increment_iteration, "increment_iteration"))
    graph.add_node("synthesizer", _traced(synthesizer, "synthesizer"))
    graph.add_node("validator", _traced(validator, "validator"))
    graph.add_node("polarity_check", _traced(_make_polarity_check_node(router), "polarity_check"))
    graph.add_node("decomposer", _traced(decomposer_node, "decomposer"))
    graph.add_node("recomposer", _traced(recomposer_node, "recomposer"))
    graph.add_node("refinement", _traced(_make_refinement_node(router), "refinement"))
    graph.add_node("compile_gate", _traced(_compile_gate_node, "compile_gate"))
    graph.add_node("coverage_gate", _traced(_coverage_gate_node, "coverage_gate"))
    graph.add_node("structure_gate", _traced(_structure_gate_node, "structure_gate"))
    graph.add_node("import_fix", _traced(_import_fix_node, "import_fix"))
    graph.add_node("covenant_enforce", _traced(_covenant_enforce_node, "covenant_enforce"))

    # Entry — recomposer runs first (retrieves nutrients from soil)
    graph.set_entry_point("recomposer")

    # Recomposer → intake (nutrients enriched, then normal pipeline)
    graph.add_edge("recomposer", "intake")

    # Happy path — covenant enforcement after builder, before import fix
    graph.add_edge("intake", "research")
    graph.add_edge("research", "planner")
    graph.add_edge("planner", "architect")
    graph.add_edge("architect", "skeleton_pass1")
    graph.add_edge("skeleton_pass1", "builder")
    graph.add_edge("builder", "covenant_enforce")
    graph.add_edge("covenant_enforce", "import_fix")
    graph.add_edge("import_fix", "tester")
    graph.add_edge("tester", "executor")
    graph.add_edge("executor", "gap_analyst")

    # Gap routing
    graph.add_conditional_edges(
        "gap_analyst",
        _route_after_gap,
        {
            "research": "increment_iteration",
            "debugger": "increment_iteration",
            "builder": "increment_iteration",
            "synthesizer": "synthesizer",
        },
    )

    # After increment, re-route to the right target
    def _re_route(state: dict[str, Any]) -> Literal["research", "debugger", "builder"]:
        gap = state.get("gap_report")
        if gap is None:
            return "builder"
        requires_research = (
            gap.get("requires_research")
            if isinstance(gap, dict)
            else getattr(gap, "requires_research", False)
        )
        if requires_research:
            return "research"
        blockers = (
            gap.get("total_blockers", 0)
            if isinstance(gap, dict)
            else getattr(gap, "total_blockers", 0)
        )
        if blockers > 0:
            exec_r = state.get("execution_result")
            if exec_r and (
                exec_r.get("success")
                if isinstance(exec_r, dict)
                else getattr(exec_r, "success", False)
            ):
                return "builder"
            return "debugger"
        return "builder"

    graph.add_conditional_edges(
        "increment_iteration",
        _re_route,
        {
            "research": "research",
            "debugger": "debugger",
            "builder": "builder",
        },
    )

    # Debugger goes back to executor
    graph.add_edge("debugger", "executor")

    # End path
    graph.add_edge("synthesizer", "validator")

    # Validator routing — includes refinement path
    graph.add_conditional_edges(
        "validator",
        _route_after_validation,
        {
            "polarity_check": "polarity_check",
            "builder": "increment_iteration",
            "research": "research",
            "refinement": "refinement",
        },
    )

    # Refinement (Water Cycle) → compile_gate → decomposer → END
    graph.add_edge("refinement", "compile_gate")

    # Polarity check — outer loop
    # Terminal path: polarity_check → compile_gate → decomposer → END
    # Loop-back path: polarity_check → planner (decomposer skipped on intermediate iterations)
    graph.add_conditional_edges(
        "polarity_check",
        _route_after_polarity,
        {
            "planner": "planner",
            END: "compile_gate",
        },
    )

    # Terminal gates → decomposer. Every finalising path passes through the
    # compile gate (non-parsing build can't report pass), the coverage gate
    # (planned-vs-produced shortfall / hollow stubs can't report pass), and the
    # structure gate (two parallel implementations can't report pass).
    graph.add_edge("compile_gate", "coverage_gate")
    graph.add_edge("coverage_gate", "structure_gate")
    graph.add_edge("structure_gate", "decomposer")

    # Decomposer always terminates the pipeline
    graph.add_edge("decomposer", END)

    return graph.compile()
