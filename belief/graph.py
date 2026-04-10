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
from typing import Any, Literal

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
from belief.memory.decomposer import decomposer_node
from belief.memory.recomposer import recomposer_node
from belief.models.state import Phase

logger = logging.getLogger("belief.graph")


# ── Routing functions ─────────────────────────────────────────────────────────

def _normalize_error(error: str) -> str:
    """Normalize an error string for deduplication (strip line numbers, paths, timestamps)."""
    import re
    s = error.lower().strip()
    s = re.sub(r'line \d+', 'line N', s)
    s = re.sub(r'/tmp/belief_\w+/', '/tmp/', s)
    s = re.sub(r'belief-\w{8}', 'belief-XXXX', s)
    s = re.sub(r'\d{2}:\d{2}:\d{2}', 'HH:MM:SS', s)
    return s


def _error_hash(error: str) -> str:
    """Hash a normalized error for dedup tracking."""
    import hashlib
    return hashlib.md5(_normalize_error(error).encode()).hexdigest()[:12]


def _route_after_gap(state: dict[str, Any]) -> Literal["research", "debugger", "builder", "synthesizer"]:
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # ── OTP-style error classification ──────────────────────────────
    # Instead of ad-hoc checks, classify the error and route based on
    # the recovery strategy. This is the Erlang supervision pattern
    # applied to LLM agent pipelines.
    exec_r = state.get("execution_result")

    # If execution succeeded, the code WORKS — go to synthesizer
    exec_success = False
    if exec_r:
        exec_success = exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)
    if exec_success:
        return "synthesizer"

    # Classify the error
    if exec_r:
        error = exec_r.get("error_summary", "") if isinstance(exec_r, dict) else getattr(exec_r, "error_summary", "")
        stderr = exec_r.get("stderr", "") if isinstance(exec_r, dict) else getattr(exec_r, "stderr", "")

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

            logger.info(f"Error classified: {classified.category.value} → {classified.strategy.value} ({classified.reason})")

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

    if iteration >= max_iter:
        return "synthesizer"
    if requires_research:
        return "research"
    if total_blockers > 0:
        return "debugger"

    return "builder"


def _route_after_validation(state: dict[str, Any]) -> Literal["polarity_check", "builder", "research", "refinement"]:
    exec_r = state.get("execution_result")
    exec_ok = False
    if exec_r:
        exec_ok = exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)

    validation = state.get("validation_result")
    if validation is None:
        return "polarity_check"

    verdict = validation.get("verdict") if isinstance(validation, dict) else getattr(validation, "verdict", "pass")
    if isinstance(verdict, str):
        v = verdict
    else:
        v = verdict.value if hasattr(verdict, "value") else str(verdict)

    # Water Cycle: if the code RUNS but has quality issues,
    # ALWAYS send to refinement — regardless of iteration count.
    # Refinement is polishing, not rebuilding. It doesn't cost a rebuild iteration.
    if v == "fail_fixable" and exec_ok:
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
        exec_ok = exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)
        if exec_ok:
            return END

    polarity = state.get("polarity", {})
    remainder = polarity.get("current_remainder") if isinstance(polarity, dict) else getattr(polarity, "current_remainder", None)
    if remainder and "significant" in str(remainder).lower():
        return "planner"

    return END


# ── Polarity check node ──────────────────────────────────────────────────────

async def _polarity_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """Post-validation Latios gap check.

    Source: taskforce_base.py latios_gap_check() lines 358-447
    """
    from belief.config.settings import settings
    from belief.llm import LLMClient
    from belief.prompts import LATIOS_SYSTEM, LATIOS_PROMPT

    result = dict(state)
    result["phase"] = Phase.POLARITY_CHECK.value

    spec = state.get("requirement_spec")
    validation = state.get("validation_result")
    exec_result = state.get("execution_result")

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

    exec_success = False
    if exec_result:
        exec_success = exec_result.get("success") if isinstance(exec_result, dict) else getattr(exec_result, "success", False)

    router = ModelRouter()
    llm = LLMClient(router)
    try:
        prompt = LATIOS_PROMPT.format(
            goal=goal,
            acceptance_criteria=criteria or "  (none specified)",
            validator_summary=validator_summary or "  (no validator output)",
            exec_success=exec_success,
        )
        raw = await llm.generate_text(
            role="latios", system=LATIOS_SYSTEM, prompt=prompt,
            temperature=0.2, max_tokens=800,
        )

        # Parse JSON response
        match = re.search(r'\{[^{}]*"significant_gap"[^{}]*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            polarity = dict(result.get("polarity", {}))
            if data.get("significant_gap"):
                polarity["current_remainder"] = f"significant: {data.get('gap_summary', '')}"
                polarity["accumulated_remainders"] = polarity.get("accumulated_remainders", []) + [data.get("gap_summary", "")]
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
        error = exec_r.get("error_summary", "") if isinstance(exec_r, dict) else getattr(exec_r, "error_summary", "")
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
    """
    result = dict(state)
    code_files = state.get("code_files", {})

    if not code_files:
        return result

    # ── Python covenants ──
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

async def _refinement_node(state: dict[str, Any]) -> dict[str, Any]:
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
            pytest_r = exec_r.get("pytest_result") if isinstance(exec_r, dict) else getattr(exec_r, "pytest_result", None)
            if pytest_r:
                test_output = pytest_r.get("raw_output", "") if isinstance(pytest_r, dict) else getattr(pytest_r, "raw_output", "")
            if not test_output:
                test_output = exec_r.get("stdout", "") if isinstance(exec_r, dict) else getattr(exec_r, "stdout", "")
        
        # Source 2: validator test output
        if not test_output:
            validation = state.get("validation_result")
            if isinstance(validation, dict):
                test_output = validation.get("test_output", "") or validation.get("summary", "")
            elif validation:
                test_output = getattr(validation, "test_output", "") or getattr(validation, "summary", "")
        
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
        
        # Run the water cycle
        refinement = await run_refinement_loop(
            code_files=code_files,
            test_files=test_files,
            initial_test_output=test_output,
            max_cycles=3,
        )
        
        # Update state with refined code
        result["code_files"] = refinement["code_files"]

        # Compute actual weighted verdict from real test results
        final_passed = refinement.get("final_passed", 0)
        final_total = refinement.get("final_total", 0)
        initial_passed = refinement.get("initial_passed", 0)
        improvement = final_passed - initial_passed

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
                logger.info(f"Refinement: verdict upgraded to PASS ({final_passed}/{final_total} = {actual_score:.0%})")
        
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

    # Nodes
    graph.add_node("intake", intake)
    graph.add_node("research", research)
    graph.add_node("planner", planner)
    graph.add_node("architect", architect)
    graph.add_node("skeleton_pass1", skeleton_pass1_node)
    graph.add_node("builder", builder)
    graph.add_node("tester", tester)
    graph.add_node("executor", executor)
    graph.add_node("debugger", debugger)
    graph.add_node("gap_analyst", gap_analyst)
    graph.add_node("increment_iteration", _increment_iteration)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("validator", validator)
    graph.add_node("polarity_check", _polarity_check_node)
    graph.add_node("decomposer", decomposer_node)
    graph.add_node("recomposer", recomposer_node)
    graph.add_node("refinement", _refinement_node)
    graph.add_node("import_fix", _import_fix_node)
    graph.add_node("covenant_enforce", _covenant_enforce_node)

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
    graph.add_conditional_edges("gap_analyst", _route_after_gap, {
        "research": "increment_iteration",
        "debugger": "increment_iteration",
        "builder": "increment_iteration",
        "synthesizer": "synthesizer",
    })

    # After increment, re-route to the right target
    def _re_route(state: dict[str, Any]) -> Literal["research", "debugger", "builder"]:
        gap = state.get("gap_report")
        if gap is None:
            return "builder"
        requires_research = gap.get("requires_research") if isinstance(gap, dict) else getattr(gap, "requires_research", False)
        if requires_research:
            return "research"
        blockers = gap.get("total_blockers", 0) if isinstance(gap, dict) else getattr(gap, "total_blockers", 0)
        if blockers > 0:
            exec_r = state.get("execution_result")
            if exec_r and (exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)):
                return "builder"
            return "debugger"
        return "builder"

    graph.add_conditional_edges("increment_iteration", _re_route, {
        "research": "research", "debugger": "debugger", "builder": "builder",
    })

    # Debugger goes back to executor
    graph.add_edge("debugger", "executor")

    # End path
    graph.add_edge("synthesizer", "validator")

    # Validator routing — includes refinement path
    graph.add_conditional_edges("validator", _route_after_validation, {
        "polarity_check": "polarity_check",
        "builder": "increment_iteration",
        "research": "research",
        "refinement": "refinement",
    })

    # Refinement (Water Cycle) → decomposer → END
    graph.add_edge("refinement", "decomposer")

    # Polarity check — outer loop
    # Terminal path: polarity_check → decomposer → END
    # Loop-back path: polarity_check → planner (decomposer skipped on intermediate iterations)
    graph.add_conditional_edges("polarity_check", _route_after_polarity, {
        "planner": "planner",
        END: "decomposer",
    })

    # Decomposer always terminates the pipeline
    graph.add_edge("decomposer", END)

    return graph.compile()
