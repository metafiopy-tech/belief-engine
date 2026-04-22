"""Tests for the collapsed local-mode pipeline (validation Session 1).

These cover the graph_local module's structural contract — node set,
pipeline selection precedence, route functions — without requiring a
running Ollama or API key.  End-to-end behavior is tested on the
user's Mac during the full build verification.
"""

from __future__ import annotations

import os

import pytest


def _router(mode: str):
    from belief.config.models import ModelRouter, RouteMode
    r = ModelRouter()
    r.set_mode(RouteMode(mode))
    return r


def test_local_pipeline_compiles():
    """Smoke: build_local_pipeline returns a compiled StateGraph."""
    from belief.graph_local import build_local_pipeline
    pipeline = build_local_pipeline(_router("local"))
    assert pipeline is not None
    # Compiled graphs expose a .nodes mapping
    assert hasattr(pipeline, "nodes")


def test_local_pipeline_has_collapsed_node_set():
    """The collapsed pipeline keeps the 4 logical stages and drops the
    agents we deliberately cut (research, tester, gap_analyst, polarity)."""
    from belief.graph_local import build_local_pipeline
    pipeline = build_local_pipeline(_router("local"))
    nodes = {n for n in pipeline.nodes.keys() if not n.startswith("__")}

    expected_present = {
        "recomposer",
        "intake",
        "planner",
        "architect",
        "skeleton_pass1",
        "builder",
        "covenant_enforce",
        "import_fix",
        "executor",
        "debugger",
        "increment_iteration",
        "synthesizer",
        "validator",
        "refinement",
    }
    # decomposer is intentionally run post-print by the CLI, not
    # inside the graph — see graph_local docstring.
    expected_absent = {
        "research",
        "tester",
        "gap_analyst",
        "polarity_check",
        "decomposer",
    }

    assert expected_present <= nodes, f"missing: {expected_present - nodes}"
    assert not (expected_absent & nodes), (
        f"forbidden nodes leaked: {expected_absent & nodes}"
    )


def test_cloud_pipeline_unaffected():
    """Building the local pipeline must not alter the cloud pipeline's
    node set.  The two should coexist."""
    from belief.graph import build_pipeline
    from belief.graph_local import build_local_pipeline

    cloud = build_pipeline(_router("cloud"))
    local = build_local_pipeline(_router("local"))

    cloud_nodes = {n for n in cloud.nodes.keys() if not n.startswith("__")}
    local_nodes = {n for n in local.nodes.keys() if not n.startswith("__")}

    # Cloud keeps everything, including decomposer
    assert {"research", "tester", "gap_analyst", "polarity_check", "decomposer"} <= cloud_nodes
    # Local stays lean — those same agents are absent
    assert not ({"research", "tester", "gap_analyst", "polarity_check", "decomposer"} & local_nodes)


def test_route_after_executor_success_goes_to_synthesizer():
    from belief.graph_local import _route_after_executor

    state = {"execution_result": {"success": True}, "iteration": 0}
    assert _route_after_executor(state) == "synthesizer"


def test_route_after_executor_failure_debugs():
    from belief.graph_local import _route_after_executor

    state = {"execution_result": {"success": False}, "iteration": 0}
    assert _route_after_executor(state) == "debugger"


def test_route_after_executor_respects_local_budget():
    """LOCAL_MAX_DEBUG_ITERATIONS caps the debug loop at 2 (cloud is 3)."""
    from belief.graph_local import (
        LOCAL_MAX_DEBUG_ITERATIONS,
        _route_after_executor,
    )
    # Use assertion-relative value so the test doesn't drift if the cap
    # is later bumped.
    at_cap = {
        "execution_result": {"success": False},
        "iteration": LOCAL_MAX_DEBUG_ITERATIONS,
    }
    assert _route_after_executor(at_cap) == "synthesizer"


def test_route_after_validator_sends_fixable_runnable_to_refinement():
    from belief.graph_local import _route_after_validator

    state = {
        "execution_result": {"success": True},
        "validation_result": {"verdict": "fail_fixable"},
    }
    assert _route_after_validator(state) == "refinement"


def test_route_after_validator_pass_ends_graph():
    """Pass verdict now ends the graph directly — the CLI fires the
    decomposer post-print so the user sees BUILD COMPLETE sooner."""
    from belief.graph_local import _route_after_validator

    state = {
        "execution_result": {"success": True},
        "validation_result": {"verdict": "pass"},
    }
    assert _route_after_validator(state) == "__end__"


def test_route_after_validator_missing_validation_ends_graph():
    from belief.graph_local import _route_after_validator

    assert _route_after_validator({}) == "__end__"
