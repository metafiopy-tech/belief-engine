"""Multi-Service Pipeline — Tier 6 LangGraph with Fan-Out/Fan-In.

Extends the single-service pipeline to handle ServiceArchitecture specs
where multiple services are generated in parallel, then validated together
via docker-compose integration testing.

Pipeline flow:
  recomposer → intake → research → planner → architect
    → contract_agent (generates OpenAPI specs + docker-compose)
    → Send() fan-out: [service_pipeline(A), service_pipeline(B), ...]
    → fan-in (Annotated[dict, merge_dicts] merges code_files)
    → integration_tester (generates cross-service contract tests)
    → decomposer → END

Each service branch runs: skeleton → builder → covenant → import_fix → tester → executor
Then results merge and integration tests validate the whole system.

Usage:
    from belief.graph_multi import build_multi_pipeline
    pipeline = build_multi_pipeline()
    result = await pipeline.ainvoke({"user_goal": "Build a microservice system..."})
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from belief.agents.contract_agent import contract_agent_node
from belief.agents.integration_tester import integration_test_node
from belief.config.models import ModelRouter
from belief.memory.decomposer import decomposer_node
from belief.memory.recomposer import recomposer_node

logger = logging.getLogger("belief.graph_multi")


# ── Custom reducer for merging dicts from parallel branches ──────────────────

def _merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Merge two dicts — used by Annotated reducer for fan-in.

    When parallel branches each produce code_files with different keys
    (e.g., branch A: {"api_service/main.py": ...}, branch B: {"worker/main.py": ...}),
    this merger combines them into a single dict.
    """
    merged = (left or {}).copy()
    merged.update(right or {})
    return merged


def _merge_lists(left: list | None, right: list | None) -> list:
    """Merge two lists for fan-in (errors, warnings, service_results)."""
    return (left or []) + (right or [])


def _last_value(left: Any, right: Any) -> Any:
    """Keep the last value written — default reducer for non-merged keys."""
    return right if right is not None else left


# ── Typed state schema with reducers for parallel fan-in ─────────────────────
# This fixes the INVALID_CONCURRENT_GRAPH_UPDATE error.
# LangGraph requires Annotated reducers on any key that parallel
# Send() branches write to simultaneously.

class MultiServiceState(TypedDict, total=False):
    """State schema for the multi-service pipeline.

    Keys with Annotated reducers can receive writes from parallel branches.
    Keys without reducers use last-write-wins semantics.
    """
    # ── Merged keys (parallel branches write to these) ──
    code_files: Annotated[dict, _merge_dicts]
    test_files: Annotated[dict, _merge_dicts]
    errors: Annotated[list, _merge_lists]
    warnings: Annotated[list, _merge_lists]
    service_results: Annotated[list, _merge_lists]
    agent_timings: Annotated[dict, _merge_dicts]

    # ── Single-writer keys (last value wins) ──
    run_id: str
    user_goal: str
    phase: str
    iteration: int
    max_iterations: int
    max_cost_usd: float
    # These are the critical keys the shared agents (intake, research,
    # planner, architect) read and write. Missing any of these causes
    # the pipeline to loop on "no requirement_spec" errors.
    requirement_spec: Any
    spec: Any
    research_report: Any
    architecture: Any
    service_architecture: Any
    skeleton_artifact: Any
    openapi_specs: dict
    file_specs: list
    file_manifest: Any
    dag: Any
    code_plan: Any
    implementation_plan: Any
    gap_report: Any
    token_usage: Any
    validation_result: Any
    execution_result: Any
    build_budget: Any
    complexity_score: int
    similar_builds_context: str
    previous_gap_summaries: list
    polarity: dict
    nutrient_context: str
    nutrient_profile: Any
    extracted_nutrients: Any
    service_spec: Any
    shared_models: Any
    openapi_spec: str


# ── Service build sub-pipeline ───────────────────────────────────────────────

async def _build_one_service(state: dict[str, Any]) -> dict[str, Any]:
    """Build a single service using the existing Tier 5 pipeline.

    This node runs the full skeleton → builder → tester → executor pipeline
    for one service. It's dispatched via Send() for parallel execution.

    Receives:
        state["service_spec"]: ServiceSpec for this service
        state["shared_models"]: SharedModelSpec list
        state["openapi_spec"]: OpenAPI YAML for this service
        state["user_goal"]: original goal (for context)

    Returns:
        state with code_files, test_files, execution_result for this service
    """
    from belief.agents.architect import ArchitectAgent
    from belief.agents.builder import BuilderAgent
    from belief.agents.executor import ExecutorAgent
    from belief.agents.skeleton_pass1 import skeleton_pass1_node
    from belief.agents.tester import TesterAgent
    from belief.config.models import ModelRouter
    from belief.models.state import UnifiedState

    service_spec = state.get("service_spec")
    if not service_spec:
        return {"code_files": {}, "test_files": {}, "service_errors": ["No service_spec"]}

    # Hydrate
    if isinstance(service_spec, dict):
        from belief.models.service_architecture import ServiceSpec
        service_spec = ServiceSpec.model_validate(service_spec)

    router = ModelRouter()
    package = service_spec.package

    # Build goal context for this specific service
    service_goal = (
        f"Build the {service_spec.name} service: {service_spec.description}. "
        f"Framework: {service_spec.framework}. Port: {service_spec.port}. "
        f"Package directory: {package}/. "
        f"Routes: {', '.join(f'{r.method} {r.path}' for r in service_spec.routes[:10])}."
    )

    logger.info(f"Building service: {service_spec.name} ({len(service_spec.routes)} routes)")

    try:
        # Create a mini-state for the single-service pipeline
        from belief.graph import build_pipeline
        pipeline = build_pipeline(router)

        # Run the full pipeline for this one service
        result = await pipeline.ainvoke({
            "user_goal": service_goal,
            "max_iterations": 2,  # Fewer iterations per service to save cost
            "complexity_score": min(6, len(service_spec.routes) + 2),
        })

        # Prefix all file paths with the package directory
        code_files = {}
        for fname, content in result.get("code_files", {}).items():
            # Prefix with package/ unless already prefixed
            if not fname.startswith(f"{package}/"):
                code_files[f"{package}/{fname}"] = content
            else:
                code_files[fname] = content

        test_files = {}
        for fname, content in result.get("test_files", {}).items():
            if not fname.startswith(f"{package}/"):
                test_files[f"{package}/{fname}"] = content
            else:
                test_files[fname] = content

        return {
            "code_files": code_files,
            "test_files": test_files,
            "service_results": [{
                "name": service_spec.name,
                "verdict": result.get("validation_result", {}).get("verdict", "unknown")
                    if isinstance(result.get("validation_result"), dict)
                    else getattr(result.get("validation_result"), "verdict", "unknown"),
                "files": len(code_files),
            }],
        }

    except Exception as e:
        logger.warning(f"Service build failed for {service_spec.name}: {e}")
        return {
            "code_files": {},
            "test_files": {},
            "service_results": [{"name": service_spec.name, "error": str(e)}],
        }


# ── Fan-out routing ──────────────────────────────────────────────────────────

def _fan_out_services(state: dict[str, Any]) -> list[Send]:
    """Route to parallel service builds via Send().

    Each service gets its own branch running the full build pipeline.
    Results merge via _merge_dicts reducer on code_files/test_files.
    """
    architecture = state.get("service_architecture")
    if not architecture:
        return []

    if isinstance(architecture, dict):
        from belief.models.service_architecture import ServiceArchitecture
        architecture = ServiceArchitecture.model_validate(architecture)

    openapi_specs = state.get("openapi_specs", {})

    sends = []
    for svc in architecture.services:
        sends.append(Send("build_service", {
            "service_spec": svc.model_dump() if hasattr(svc, "model_dump") else svc,
            "shared_models": [m.model_dump() for m in architecture.shared_models]
                if architecture.shared_models else [],
            "openapi_spec": openapi_specs.get(svc.name, ""),
            "user_goal": state.get("user_goal", ""),
        }))

    logger.info(f"Fan-out: dispatching {len(sends)} service builds in parallel")
    return sends


# ── Multi-service detection ──────────────────────────────────────────────────

def _route_after_architect(state: dict[str, Any]) -> Literal["contract_agent", "skeleton_pass1"]:
    """Route based on whether the architect produced a multi-service architecture."""
    if state.get("service_architecture"):
        logger.info("Architect produced ServiceArchitecture — routing to multi-service pipeline")
        return "contract_agent"
    return "skeleton_pass1"


# ── Collect results after fan-in ─────────────────────────────────────────────

async def _collect_service_results(state: dict[str, Any]) -> dict[str, Any]:
    """Aggregate results from parallel service builds."""
    result = dict(state)

    service_results = state.get("service_results", [])
    code_files = state.get("code_files", {})
    test_files = state.get("test_files", {})

    total_files = len(code_files)
    total_tests = len(test_files)
    passed = sum(1 for r in service_results if r.get("verdict") == "pass")
    total = len(service_results)

    logger.info(
        f"Fan-in: {passed}/{total} services passed, "
        f"{total_files} code files, {total_tests} test files"
    )

    result["validation_result"] = {
        "verdict": "pass" if passed == total else "fail_fixable",
        "weighted_score": passed / max(total, 1),
        "tests_passed": passed,
        "tests_total": total,
        "summary": f"{passed}/{total} services built successfully",
    }

    return result


# ── Build the multi-service graph ────────────────────────────────────────────

def build_multi_pipeline(router: ModelRouter | None = None) -> Any:
    """Construct and compile the Tier 6 multi-service pipeline.

    This extends the base pipeline with:
    - Contract agent (generates OpenAPI specs after architect)
    - Fan-out via Send() for parallel service generation
    - Fan-in with dict merging for code_files
    - Integration testing node
    - Decomposer for memory storage

    For single-service builds (no ServiceArchitecture), falls through
    to the standard skeleton → builder path.
    """
    if router is None:
        router = ModelRouter()

    # Import the base pipeline's agents
    from belief.agents.architect import ArchitectAgent
    from belief.agents.intake import IntakeAgent
    from belief.agents.planner import PlannerAgent
    from belief.agents.research import ResearchAgent

    # Import single-service nodes (for fallback path)
    from belief.agents.skeleton_pass1 import skeleton_pass1_node

    intake = IntakeAgent(router)
    research = ResearchAgent(router)
    planner = PlannerAgent(router)
    architect = ArchitectAgent(router)

    graph = StateGraph(MultiServiceState)

    # Shared front-end nodes
    graph.add_node("recomposer", recomposer_node)
    graph.add_node("intake", intake)
    graph.add_node("research", research)
    graph.add_node("planner", planner)
    graph.add_node("architect", architect)

    # Multi-service branch
    graph.add_node("contract_agent", contract_agent_node)
    graph.add_node("build_service", _build_one_service)
    graph.add_node("collect_results", _collect_service_results)
    graph.add_node("integration_tester", integration_test_node)

    # Single-service fallback — delegates to the full base pipeline
    graph.add_node("skeleton_pass1", skeleton_pass1_node)

    # Terminal
    graph.add_node("decomposer", decomposer_node)

    # Wiring — shared front-end
    graph.set_entry_point("recomposer")
    graph.add_edge("recomposer", "intake")
    graph.add_edge("intake", "research")
    graph.add_edge("research", "planner")
    graph.add_edge("planner", "architect")

    # Branch point: multi-service or single-service
    graph.add_conditional_edges("architect", _route_after_architect, {
        "contract_agent": "contract_agent",
        "skeleton_pass1": "skeleton_pass1",
    })

    # Multi-service path: contract → fan-out → fan-in → integration → decomposer
    graph.add_conditional_edges("contract_agent", _fan_out_services, ["build_service"])
    graph.add_edge("build_service", "collect_results")
    graph.add_edge("collect_results", "integration_tester")
    graph.add_edge("integration_tester", "decomposer")

    # Single-service fallback goes to the base pipeline
    # (For now, just end — the caller should use build_pipeline() for single-service)
    graph.add_edge("skeleton_pass1", "decomposer")

    # Terminal
    graph.add_edge("decomposer", END)

    return graph.compile()
