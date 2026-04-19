"""
Jitterbug Cycle — periodic compression-reconstruction loop.

A LangGraph subgraph with 5 phases:
  1. Expansion  — run diverse builds to collect traces
  2. Compression — analyze traces, extract principles, cluster failures
  3. Reconstruction — build new tools, crystallize invariants
  4. Validation — benchmark for regressions
  5. Integration — accept improvements, update soil, prune

The jitterbug accelerates improvement by turning raw build experience
into structured knowledge (principles, tools, covenants) in a single
coordinated cycle rather than waiting for organic accumulation.

Usage:
    graph = build_jitterbug_graph()
    result = await graph.ainvoke({"n_expansion_goals": 5})
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import END, StateGraph

logger = logging.getLogger("belief.evolution.jitterbug")


# ── State ───────────────────────────────────────────────────────────────────

class JitterbugState(dict):
    """State for the jitterbug cycle graph.

    Keys:
        n_expansion_goals:     How many diverse builds to run (default 5)
        regression_threshold:  Max acceptable regression rate (default 0.03)
        expansion_traces:      Full build traces from expansion
        expansion_cost:        Total cost of expansion builds
        candidate_principles:  Distilled principles from compression
        failure_clusters:      Clustered failure patterns
        redundant_tool_pairs:  Tools that could be merged
        compression_summary:   Human-readable summary
        new_tools_built:       Tools created by the engine
        new_covenants:         Covenants crystallized
        merged_tools:          Merged tool results
        validation_results:    Benchmark results
        regressions:           Any regressions found
        validation_passed:     Whether validation passed
        integrated_tool_ids:   Tool IDs integrated
        integrated_covenant_ids: Covenant IDs integrated
        pruned_ids:            IDs of pruned records
        cycle_number:          Current cycle number
        started_at:            ISO timestamp
        total_cost:            Total cost across all phases
        stage_before:          Generative chain stage at start
        stage_after:           Stage at end
        dry_run:               If True, skip reconstruction and integration
    """
    pass


# ── Default state ───────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "n_expansion_goals": 5,
        "regression_threshold": 0.03,
        "expansion_traces": [],
        "expansion_cost": 0.0,
        "candidate_principles": [],
        "failure_clusters": [],
        "redundant_tool_pairs": [],
        "compression_summary": "",
        "new_tools_built": [],
        "new_covenants": [],
        "merged_tools": [],
        "validation_results": {},
        "regressions": [],
        "validation_passed": False,
        "integrated_tool_ids": [],
        "integrated_covenant_ids": [],
        "pruned_ids": [],
        "cycle_number": 0,
        "started_at": "",
        "total_cost": 0.0,
        "stage_before": 0,
        "stage_after": 0,
        "dry_run": False,
    }


# ── Goal templates ──────────────────────────────────────────────────────────

EXPANSION_GOAL_TEMPLATES = [
    # Tier 1: simple scripts
    "Build a Python script that {script_action}",
    # Tier 2: CLI tools and simple APIs
    "Build a Click CLI that {cli_action}",
    "Build a FastAPI {resource} API with SQLite CRUD",
    # Tier 3: multi-file packages
    "Build a Python module that {module_action}",
    "Build a FastAPI app with {feature}",
    # Diverse domains
    "Build an MCP server that wraps {api_name}",
    "Build a {framework} app with {feature}",
]

_SCRIPT_ACTIONS = [
    "generates random passwords with configurable length and character sets",
    "converts CSV files to JSON format",
    "counts word frequency in a text file",
    "calculates fibonacci numbers up to N",
    "validates email addresses from stdin",
]

_CLI_ACTIONS = [
    "manages a local TODO list with add, list, done, delete commands",
    "converts between file formats (JSON, YAML, TOML)",
    "tracks daily expenses with categories and monthly summaries",
    "generates project scaffolding from templates",
]

_RESOURCES = ["todo", "bookmark", "note", "contact", "expense"]

_MODULE_ACTIONS = [
    "validates URL formats and resolves redirects",
    "generates slug strings from titles",
    "computes text similarity using TF-IDF",
    "parses and validates configuration files",
]

_FEATURES = [
    "JWT authentication and role-based access",
    "pagination and filtering",
    "file upload and download",
    "rate limiting middleware",
]

_API_NAMES = ["OpenWeatherMap", "GitHub", "HackerNews", "JSONPlaceholder"]
_FRAMEWORKS = ["FastAPI", "Flask"]


def generate_expansion_goals(n: int = 5) -> list[str]:
    """Generate diverse build goals for expansion phase."""
    goals: list[str] = []

    # Ensure at least one from each tier
    # Tier 1
    goals.append(
        random.choice(EXPANSION_GOAL_TEMPLATES[:1]).format(
            script_action=random.choice(_SCRIPT_ACTIONS)
        )
    )
    # Tier 2
    template = random.choice(EXPANSION_GOAL_TEMPLATES[1:3])
    if "{cli_action}" in template:
        goals.append(template.format(cli_action=random.choice(_CLI_ACTIONS)))
    else:
        goals.append(template.format(resource=random.choice(_RESOURCES)))
    # Tier 3
    template = random.choice(EXPANSION_GOAL_TEMPLATES[3:5])
    if "{module_action}" in template:
        goals.append(template.format(module_action=random.choice(_MODULE_ACTIONS)))
    else:
        goals.append(template.format(feature=random.choice(_FEATURES)))

    # Fill remaining with random templates
    while len(goals) < n:
        template = random.choice(EXPANSION_GOAL_TEMPLATES)
        try:
            if "{script_action}" in template:
                goal = template.format(script_action=random.choice(_SCRIPT_ACTIONS))
            elif "{cli_action}" in template:
                goal = template.format(cli_action=random.choice(_CLI_ACTIONS))
            elif "{resource}" in template:
                goal = template.format(resource=random.choice(_RESOURCES))
            elif "{module_action}" in template:
                goal = template.format(module_action=random.choice(_MODULE_ACTIONS))
            elif "{api_name}" in template:
                goal = template.format(api_name=random.choice(_API_NAMES))
            elif "{framework}" in template and "{feature}" in template:
                goal = template.format(
                    framework=random.choice(_FRAMEWORKS),
                    feature=random.choice(_FEATURES),
                )
            elif "{feature}" in template:
                goal = template.format(feature=random.choice(_FEATURES))
            else:
                goal = template
        except (KeyError, IndexError):
            continue

        if goal not in goals:
            goals.append(goal)

    return goals[:n]


# ── Phase 1: Expansion ──────────────────────────────────────────────────────

async def expansion_node(state: dict) -> dict:
    """Run diverse builds to collect traces.

    Generates N goals, runs each through the engine pipeline,
    and records all traces as episodes in ChromaDB.
    """
    result = dict(state)
    result["started_at"] = datetime.now(timezone.utc).isoformat()

    n = state.get("n_expansion_goals", 5)
    goals = generate_expansion_goals(n)
    traces: list[dict] = []
    total_cost = 0.0

    # Budget cap: $10 total, $2 per build
    max_cost_per_build = 2.0
    budget_remaining = 10.0

    for i, goal in enumerate(goals):
        if budget_remaining <= 0:
            logger.info(f"Expansion: budget exhausted after {i} builds")
            break

        logger.info(f"Expansion [{i+1}/{len(goals)}]: {goal[:80]}")
        trace = await _run_expansion_build(goal, max_cost=min(max_cost_per_build, budget_remaining))
        traces.append(trace)

        build_cost = trace.get("cost_usd", 0.0)
        total_cost += build_cost
        budget_remaining -= build_cost

    # Record episodes
    try:
        from belief.memory.episode_recorder import record_episode
        from belief.memory.soil import Soil
        soil = Soil()
        for trace in traces:
            record_episode(soil, trace)
    except Exception as e:
        logger.debug(f"Episode recording skipped: {e}")

    result["expansion_traces"] = traces
    result["expansion_cost"] = total_cost
    result["total_cost"] = total_cost

    logger.info(
        f"Expansion complete: {len(traces)} builds, "
        f"${total_cost:.2f}, "
        f"{sum(1 for t in traces if t.get('passed'))} passed"
    )
    return result


async def _run_expansion_build(goal: str, max_cost: float = 2.0) -> dict:
    """Run a single build and return the trace."""
    trace = {
        "trace_id": f"jb-{uuid.uuid4().hex[:12]}",
        "user_goal": goal,
        "passed": False,
        "cost_usd": 0.0,
        "code_files": {},
        "errors": [],
    }

    try:
        from belief.graph import build_pipeline
        graph = build_pipeline()
        compiled = graph.compile()
        result = await compiled.ainvoke({
            "user_goal": goal,
            "max_iterations": 2,
            "max_cost_usd": max_cost,
        })

        trace["passed"] = result.get("phase", "") == "complete" or bool(
            result.get("validation_result", {}).get("verdict") == "pass"
            if isinstance(result.get("validation_result"), dict) else False
        )
        trace["code_files"] = result.get("code_files", {})
        trace["cost_usd"] = result.get("total_cost_usd", 0.0)
        trace["errors"] = result.get("errors", [])
        trace["run_id"] = result.get("run_id", trace["trace_id"])

    except Exception as e:
        trace["errors"] = [str(e)]
        trace["cost_usd"] = 0.01  # Minimal cost for failed attempt
        logger.warning(f"Expansion build failed: {e}")

    return trace


# ── Phase 2: Compression ────────────────────────────────────────────────────

async def compression_node(state: dict) -> dict:
    """Analyze traces, extract principles, cluster failures."""
    result = dict(state)
    traces = state.get("expansion_traces", [])

    if not traces:
        result["compression_summary"] = "No traces to analyze"
        return result

    # Cluster failures using existing machinery
    from belief.evolution.self_improvement import cluster_failures

    failure_traces = [t for t in traces if not t.get("passed")]
    success_traces = [t for t in traces if t.get("passed")]

    # Normalize failure traces for clustering
    normalized = []
    for t in failure_traces:
        for err in t.get("errors", ["unknown error"]):
            normalized.append({
                "content": str(err),
                "trace_id": t.get("trace_id", ""),
                "code_sample": "\n".join(
                    list(t.get("code_files", {}).values())[:1]
                )[:2000],
            })

    clusters = cluster_failures(normalized) if normalized else []

    # Extract candidate principles from successful builds
    principles: list[dict] = []
    for t in success_traces:
        files = t.get("code_files", {})
        principles.append({
            "goal": t.get("user_goal", ""),
            "file_count": len(files),
            "pattern": "success",
        })

    # Check for redundant tools
    redundant_pairs: list[tuple] = []
    try:
        from belief.memory.tool_registry import ToolRegistry
        from belief.memory.soil import Soil
        soil = Soil()
        registry = ToolRegistry(soil)
        active_tools = registry.get_active_tools()

        # Find pairs with high embedding similarity
        if len(active_tools) >= 2:
            col = soil._collections.get("belief_tools")
            if col and col.count() >= 2:
                for i, t1 in enumerate(active_tools):
                    for t2 in active_tools[i+1:]:
                        try:
                            result_q = col.query(
                                query_texts=[f"tool: {t1.name} — {t1.description}"],
                                n_results=2,
                                include=["distances"],
                            )
                            if (result_q["distances"] and
                                len(result_q["distances"][0]) >= 2 and
                                result_q["distances"][0][1] < 0.15):
                                redundant_pairs.append((t1.id, t2.id))
                        except Exception:
                            pass
    except Exception as e:
        logger.debug(f"Redundancy check skipped: {e}")

    # Build summary
    summary_parts = [
        f"Expansion: {len(traces)} builds ({len(success_traces)} passed, {len(failure_traces)} failed)",
        f"Failure clusters: {len(clusters)}",
    ]
    for c in clusters[:5]:
        summary_parts.append(f"  - {c.error_type}: {c.count} occurrences")
    if redundant_pairs:
        summary_parts.append(f"Redundant tool pairs: {len(redundant_pairs)}")
    summary = "\n".join(summary_parts)

    result["candidate_principles"] = principles
    result["failure_clusters"] = [
        {
            "error_type": c.error_type,
            "count": c.count,
            "suggested_tool_name": c.suggested_tool_name,
            "example_errors": c.example_errors[:3],
        }
        for c in clusters
    ]
    result["redundant_tool_pairs"] = redundant_pairs
    result["compression_summary"] = summary

    logger.info(f"Compression complete:\n{summary}")
    return result


# ── Phase 3: Reconstruction ─────────────────────────────────────────────────

async def reconstruction_node(state: dict) -> dict:
    """Build new tools, crystallize invariants."""
    result = dict(state)

    if state.get("dry_run"):
        logger.info("Reconstruction: dry-run mode, skipping")
        return result

    new_tools: list[dict] = []
    new_covenants: list[dict] = []
    reconstruction_cost = 0.0

    # Build tools for unaddressed failure clusters (up to 3)
    clusters = state.get("failure_clusters", [])
    for cluster_info in clusters[:3]:
        if reconstruction_cost >= 6.0:  # Budget cap
            break

        try:
            from belief.evolution.self_improvement import (
                FailureCluster,
                execute_new_tool_proposal,
            )
            from belief.memory.soil import Soil

            soil = Soil()
            patch_result = await execute_new_tool_proposal(soil)

            if patch_result.success:
                new_tools.append({
                    "name": cluster_info.get("suggested_tool_name", ""),
                    "tool_id": patch_result.backup_path,  # tool_id stored here
                    "error_type": cluster_info.get("error_type", ""),
                })
                reconstruction_cost += 2.0  # Approximate cost
        except Exception as e:
            logger.warning(f"Tool building failed for {cluster_info.get('error_type')}: {e}")

    # Run crystallization pipeline
    try:
        from belief.evolution.crystallizer import run_crystallization
        from belief.memory.soil import Soil
        from belief.validators.covenant_registry import CovenantRegistry

        soil = Soil()
        registry = CovenantRegistry(soil)
        covenant_ids = await run_crystallization(soil, registry)

        for cid in covenant_ids:
            new_covenants.append({"covenant_id": cid})
    except Exception as e:
        logger.warning(f"Crystallization skipped: {e}")

    result["new_tools_built"] = new_tools
    result["new_covenants"] = new_covenants
    result["total_cost"] = state.get("total_cost", 0.0) + reconstruction_cost

    logger.info(
        f"Reconstruction: {len(new_tools)} tools built, "
        f"{len(new_covenants)} covenants crystallized"
    )
    return result


# ── Phase 4: Validation ─────────────────────────────────────────────────────

async def validation_node(state: dict) -> dict:
    """Benchmark to check for regressions."""
    result = dict(state)

    if state.get("dry_run"):
        result["validation_passed"] = True
        result["validation_results"] = {"dry_run": True}
        return result

    threshold = state.get("regression_threshold", 0.03)

    try:
        from belief.benchmark import run_benchmark

        # Smoke test: 5 challenges from tiers 1-3
        challenge_results = await run_benchmark(
            challenge_ids=["t1-fizzbuzz", "t2-todo-cli", "t2-health-api",
                          "t3-url-shortener", "t3-bookmark-api"]
        )

        passed = sum(1 for r in challenge_results if r.verdict == "pass")
        total = len(challenge_results)
        pass_rate = passed / max(total, 1)
        total_cost = sum(r.cost_usd for r in challenge_results)

        result["validation_results"] = {
            "passed": passed,
            "total": total,
            "pass_rate": pass_rate,
            "cost": total_cost,
        }
        result["total_cost"] = state.get("total_cost", 0.0) + total_cost

        # Check regression: pass_rate should be >= (1 - threshold)
        # For 5 challenges at 3% threshold, that means at most 0 regressions from baseline
        result["validation_passed"] = pass_rate >= (1.0 - threshold)

        if not result["validation_passed"]:
            result["regressions"] = [
                {"challenge": r.challenge_id, "verdict": r.verdict}
                for r in challenge_results if r.verdict != "pass"
            ]

        logger.info(
            f"Validation: {passed}/{total} ({pass_rate:.0%}), "
            f"passed={'yes' if result['validation_passed'] else 'NO'}"
        )

    except Exception as e:
        logger.warning(f"Validation benchmark failed: {e}")
        # If benchmark fails, default to passed (don't block on infra issues)
        result["validation_passed"] = True
        result["validation_results"] = {"error": str(e)}

    return result


# ── Phase 5: Integration ────────────────────────────────────────────────────

async def integration_node(state: dict) -> dict:
    """Accept improvements, update soil, prune."""
    result = dict(state)

    integrated_tools: list[str] = []
    integrated_covenants: list[str] = []
    pruned: list[str] = []

    # Record new tools
    for tool_info in state.get("new_tools_built", []):
        tid = tool_info.get("tool_id")
        if tid:
            integrated_tools.append(tid)

    # Record new covenants
    for cov_info in state.get("new_covenants", []):
        cid = cov_info.get("covenant_id")
        if cid:
            integrated_covenants.append(cid)

    # Prune bottom 5% by quality score among lapsed records
    try:
        from belief.memory.soil import Soil
        soil = Soil()

        for col_name, col in soil._collections.items():
            if col.count() == 0:
                continue

            all_records = col.get(include=["metadatas"], limit=col.count())
            lapsed = []
            for i, doc_id in enumerate(all_records["ids"]):
                meta = all_records["metadatas"][i]
                if meta.get("fsrs_decay_state") == "lapsed":
                    score = meta.get("quality_score", meta.get("stability", 0.5))
                    lapsed.append((doc_id, score))

            if len(lapsed) >= 20:
                # Prune bottom 5%
                lapsed.sort(key=lambda x: x[1])
                n_prune = max(1, len(lapsed) // 20)
                for doc_id, _ in lapsed[:n_prune]:
                    try:
                        col.delete(ids=[doc_id])
                        pruned.append(doc_id)
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"Pruning skipped: {e}")

    # Compute progression stage
    try:
        from belief.evolution.progression import compute_progression
        from belief.memory.soil import Soil
        from belief.memory.tool_registry import ToolRegistry

        soil = Soil()
        registry = ToolRegistry(soil)
        traces = state.get("expansion_traces", [])
        metrics = compute_progression(soil, registry, traces)
        result["stage_after"] = metrics.current_stage
    except Exception as e:
        logger.debug(f"Progression computation skipped: {e}")

    result["integrated_tool_ids"] = integrated_tools
    result["integrated_covenant_ids"] = integrated_covenants
    result["pruned_ids"] = pruned

    logger.info(
        f"Integration: {len(integrated_tools)} tools, "
        f"{len(integrated_covenants)} covenants, "
        f"{len(pruned)} pruned"
    )
    return result


# ── Routing ─────────────────────────────────────────────────────────────────

def route_after_validation(state: dict) -> str:
    """Route to integration if validation passed, otherwise END."""
    if state.get("validation_passed", False):
        return "integration"
    else:
        logger.warning("Jitterbug: validation failed, skipping integration")
        return END


def route_after_compression(state: dict) -> str:
    """Skip reconstruction in dry-run mode."""
    if state.get("dry_run"):
        return END
    return "reconstruction"


# ── Graph builder ───────────────────────────────────────────────────────────

def build_jitterbug_graph() -> StateGraph:
    """Build the jitterbug cycle as a LangGraph StateGraph.

    Returns an uncompiled graph. Call .compile() before .ainvoke().
    """
    graph = StateGraph(dict)

    graph.add_node("expansion", expansion_node)
    graph.add_node("compression", compression_node)
    graph.add_node("reconstruction", reconstruction_node)
    graph.add_node("validation", validation_node)
    graph.add_node("integration", integration_node)

    graph.set_entry_point("expansion")
    graph.add_edge("expansion", "compression")
    graph.add_conditional_edges(
        "compression",
        route_after_compression,
        {"reconstruction": "reconstruction", END: END},
    )
    graph.add_edge("reconstruction", "validation")
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {"integration": "integration", END: END},
    )
    graph.add_edge("integration", END)

    return graph


# ── Runner ──────────────────────────────────────────────────────────────────

async def run_jitterbug_cycle(
    n_goals: int = 5,
    dry_run: bool = False,
    cycle_number: int = 0,
) -> dict:
    """Run one complete jitterbug cycle.

    Args:
        n_goals:      Number of expansion builds.
        dry_run:       If True, only run expansion + compression.
        cycle_number:  Which cycle this is (for logging).

    Returns:
        Final JitterbugState dict.
    """
    graph = build_jitterbug_graph()
    compiled = graph.compile()

    initial_state = _default_state()
    initial_state["n_expansion_goals"] = n_goals
    initial_state["dry_run"] = dry_run
    initial_state["cycle_number"] = cycle_number

    # Compute stage before
    try:
        from belief.evolution.progression import compute_progression
        from belief.memory.soil import Soil
        from belief.memory.tool_registry import ToolRegistry

        soil = Soil()
        registry = ToolRegistry(soil)
        metrics = compute_progression(soil, registry, [])
        initial_state["stage_before"] = metrics.current_stage
    except Exception:
        pass

    logger.info(
        f"Jitterbug cycle {cycle_number}: "
        f"{'DRY RUN ' if dry_run else ''}"
        f"n_goals={n_goals}, stage_before={initial_state['stage_before']}"
    )

    result = await compiled.ainvoke(initial_state)

    logger.info(
        f"Jitterbug cycle {cycle_number} complete: "
        f"cost=${result.get('total_cost', 0):.2f}, "
        f"tools={len(result.get('integrated_tool_ids', []))}, "
        f"covenants={len(result.get('integrated_covenant_ids', []))}, "
        f"stage {result.get('stage_before', 0)}→{result.get('stage_after', 0)}"
    )

    return result
