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

    # Circuit breakers
    if iteration >= max_iter:
        return "synthesizer"

    # Oscillation detection
    prev = state.get("previous_gap_summaries", [])
    if len(prev) >= 2:
        last = set(prev[-1].lower().split())
        prev_set = set(prev[-2].lower().split())
        union = last | prev_set
        if union and len(last & prev_set) / len(union) > 0.85:
            return "synthesizer"

    # Error hash deduplication — if same error appears 3+ times, stop debugging
    exec_r = state.get("execution_result")
    if exec_r:
        error = exec_r.get("error_summary", "") if isinstance(exec_r, dict) else getattr(exec_r, "error_summary", "")
        if error:
            error_hashes = state.get("_error_hashes", [])
            h = _error_hash(error)
            if error_hashes.count(h) >= 2:  # Already seen twice + this = 3
                logger.info(f"Error dedup: same error seen 3 times (hash={h}), skipping to synthesizer")
                return "synthesizer"

    # Check execution status
    exec_success = False
    if exec_r:
        exec_success = exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)

    # KEY INSIGHT: If execution succeeded, the code WORKS.
    if exec_success:
        return "synthesizer"

    gap = state.get("gap_report")
    if gap is None:
        return "synthesizer"

    # Extract gap fields (handle both dict and model)
    if isinstance(gap, dict):
        requires_research = gap.get("requires_research", False)
        total_blockers = gap.get("total_blockers", 0)
    else:
        requires_research = getattr(gap, "requires_research", False)
        total_blockers = getattr(gap, "total_blockers", 0)

    # Code didn't run — route based on gap severity
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

    # Track error hashes for deduplication
    exec_r = state.get("execution_result")
    if exec_r:
        error = exec_r.get("error_summary", "") if isinstance(exec_r, dict) else getattr(exec_r, "error_summary", "")
        if error:
            hashes = list(state.get("_error_hashes", []))
            hashes.append(_error_hash(error))
            result["_error_hashes"] = hashes

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
        
        # Update verdict if improved
        if refinement["verdict"] == "pass":
            result["validation_result"] = {"verdict": "pass"}
            logger.info("Refinement: verdict upgraded to PASS")
        
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

    # Entry — recomposer runs first (retrieves nutrients from soil)
    graph.set_entry_point("recomposer")

    # Recomposer → intake (nutrients enriched, then normal pipeline)
    graph.add_edge("recomposer", "intake")

    # Happy path — skeleton_pass1 inserted between architect and builder
    graph.add_edge("intake", "research")
    graph.add_edge("research", "planner")
    graph.add_edge("planner", "architect")
    graph.add_edge("architect", "skeleton_pass1")
    graph.add_edge("skeleton_pass1", "builder")
    graph.add_edge("builder", "import_fix")
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
