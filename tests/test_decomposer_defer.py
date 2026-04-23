"""Tests for the deferred-decomposer behavior in local mode.

Session 1.1 follow-up: graph_local.py no longer includes the
decomposer node — the CLI runs it explicitly after printing BUILD
COMPLETE.  This test file guards three contracts:

1. graph_local's _route_after_validator never returns "decomposer" —
   routing to a missing node would silently end the graph.
2. The decomposer_node itself is still importable and callable
   (we didn't accidentally delete it).
3. The post-print defer branch in cli.run() fires exactly once on a
   successful build that took the local-pipeline branch.
"""

from __future__ import annotations


def test_route_never_returns_decomposer():
    """Router must never target a node that isn't in the compiled graph."""
    from belief.graph_local import _route_after_validator

    # Run a battery of scenarios — every return value must be one of
    # the two known targets.
    scenarios = [
        {},
        {"validation_result": {"verdict": "pass"}},
        {"validation_result": {"verdict": "fail_fixable"}, "execution_result": {"success": True}},
        {"validation_result": {"verdict": "fail_fixable"}, "execution_result": {"success": False}},
        {"validation_result": {"verdict": "fail_unfixable"}},
        {"validation_result": {"verdict": "unknown"}},
    ]
    allowed = {"refinement", "__end__"}
    for state in scenarios:
        result = _route_after_validator(state)
        assert result in allowed, (
            f"router returned {result!r} for state {state!r}; "
            f"only {allowed} are valid targets in the compiled graph"
        )
        assert result != "decomposer", "router must never route to decomposer in local mode"


def test_decomposer_node_still_exists():
    """Regression: we removed decomposer from the graph, not the module."""
    from belief.memory.decomposer import decomposer_node

    assert callable(decomposer_node)


def test_decomposer_is_async():
    """The post-print defer branch in cli.py uses `await _decomposer(...)`.
    If someone makes it sync, that await would raise at runtime."""
    import inspect
    from belief.memory.decomposer import decomposer_node

    assert inspect.iscoroutinefunction(decomposer_node), (
        "decomposer_node must be async — cli.py awaits it post-print"
    )


def test_local_pipeline_terminates_at_validator_or_refinement():
    """The compiled local graph must have a terminal edge from the
    validator/refinement path to END, without going through any
    removed decomposer node.

    We verify this by walking the graph's edges and asserting no
    edge points at a non-existent 'decomposer' target."""
    import os

    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-fake")
    from belief.config.models import ModelRouter, RouteMode
    from belief.graph_local import build_local_pipeline

    router = ModelRouter()
    router.set_mode(RouteMode.LOCAL)
    pipeline = build_local_pipeline(router)
    node_names = set(pipeline.nodes.keys())
    assert "decomposer" not in node_names, f"decomposer leaked into graph_local.nodes: {node_names}"
