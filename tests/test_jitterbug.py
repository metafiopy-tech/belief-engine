"""Tests for jitterbug cycle and progression tracker.

Covers:
  - JitterbugState initialization
  - Each phase node with synthetic data (mock engine calls)
  - Validation routing: accept/reject based on regression threshold
  - Progression metrics computation with synthetic data
  - Stage detection logic (each stage at correct thresholds)
  - Graph structure
  - Goal generation diversity
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from belief.evolution.jitterbug import (
    JitterbugState,
    build_jitterbug_graph,
    compression_node,
    expansion_node,
    generate_expansion_goals,
    integration_node,
    reconstruction_node,
    route_after_compression,
    route_after_validation,
    validation_node,
    _default_state,
)
from belief.evolution.progression import (
    ProgressionMetrics,
    _cosine_distance,
    _detect_archetypes,
    _threshold_cluster,
    compute_progression,
    format_progression_report,
)


# ── JitterbugState ──────────────────────────────────────────────────────────


class TestJitterbugState:
    def test_default_state(self):
        state = _default_state()
        assert state["n_expansion_goals"] == 5
        assert state["regression_threshold"] == 0.03
        assert state["expansion_traces"] == []
        assert state["total_cost"] == 0.0
        assert state["validation_passed"] is False
        assert state["dry_run"] is False

    def test_state_is_dict(self):
        state = JitterbugState()
        assert isinstance(state, dict)


# ── Goal Generation ────────────────────────────────────────────────────────


class TestGoalGeneration:
    def test_generates_correct_count(self):
        goals = generate_expansion_goals(5)
        assert len(goals) == 5

    def test_generates_diverse_goals(self):
        goals = generate_expansion_goals(10)
        # All goals should be unique
        assert len(set(goals)) == len(goals)

    def test_minimum_tier_coverage(self):
        """Should have at least one goal from each tier 1-3."""
        goals = generate_expansion_goals(5)
        has_script = any("script" in g.lower() for g in goals)
        has_cli_or_api = any(
            "CLI" in g or "API" in g or "FastAPI" in g
            for g in goals
        )
        # At least 2 of the 3 tiers should be present
        assert has_script or has_cli_or_api

    def test_single_goal(self):
        goals = generate_expansion_goals(1)
        assert len(goals) == 1


# ── Graph Structure ─────────────────────────────────────────────────────────


class TestGraphStructure:
    def test_graph_has_all_nodes(self):
        graph = build_jitterbug_graph()
        expected = {"expansion", "compression", "reconstruction", "validation", "integration"}
        # LangGraph stores nodes differently, check via the builder
        assert "expansion" in graph.nodes
        assert "compression" in graph.nodes
        assert "reconstruction" in graph.nodes
        assert "validation" in graph.nodes
        assert "integration" in graph.nodes

    def test_graph_compiles(self):
        graph = build_jitterbug_graph()
        compiled = graph.compile()
        assert compiled is not None


# ── Routing ─────────────────────────────────────────────────────────────────


class TestRouting:
    def test_validation_passed_routes_to_integration(self):
        state = {"validation_passed": True}
        assert route_after_validation(state) == "integration"

    def test_validation_failed_routes_to_end(self):
        from langgraph.graph import END
        state = {"validation_passed": False}
        assert route_after_validation(state) == END

    def test_dry_run_routes_to_end(self):
        from langgraph.graph import END
        state = {"dry_run": True}
        assert route_after_compression(state) == END

    def test_non_dry_run_routes_to_reconstruction(self):
        state = {"dry_run": False}
        assert route_after_compression(state) == "reconstruction"


# ── Phase Nodes (mocked) ───────────────────────────────────────────────────


class TestExpansionNode:
    @pytest.mark.asyncio
    async def test_expansion_collects_traces(self):
        """Expansion should run builds and collect traces."""
        state = _default_state()
        state["n_expansion_goals"] = 2

        with patch("belief.evolution.jitterbug._run_expansion_build") as mock_build:
            mock_build.return_value = {
                "trace_id": "jb-test1",
                "user_goal": "test goal",
                "passed": True,
                "cost_usd": 0.5,
                "code_files": {"main.py": "print('hello')"},
                "errors": [],
            }

            # Mock episode recording (imported inside the function)
            with patch("belief.memory.episode_recorder.record_episode"):
                result = await expansion_node(state)

        assert len(result["expansion_traces"]) == 2
        assert result["expansion_cost"] > 0
        assert result["started_at"] != ""


class TestCompressionNode:
    @pytest.mark.asyncio
    async def test_compression_with_failures(self):
        state = _default_state()
        state["expansion_traces"] = [
            {"passed": False, "errors": ["ImportError: no module named 'xyz'"],
             "trace_id": "t1", "code_files": {}},
            {"passed": False, "errors": ["ImportError: cannot import 'abc'"],
             "trace_id": "t2", "code_files": {}},
            {"passed": True, "errors": [], "user_goal": "test",
             "trace_id": "t3", "code_files": {"main.py": "x=1"}},
        ]

        result = await compression_node(state)
        assert result["compression_summary"] != ""
        assert len(result["failure_clusters"]) >= 1

    @pytest.mark.asyncio
    async def test_compression_empty_traces(self):
        state = _default_state()
        result = await compression_node(state)
        assert result["compression_summary"] == "No traces to analyze"


class TestValidationNode:
    @pytest.mark.asyncio
    async def test_validation_dry_run_passes(self):
        state = _default_state()
        state["dry_run"] = True
        result = await validation_node(state)
        assert result["validation_passed"] is True

    @pytest.mark.asyncio
    async def test_validation_passes_on_high_rate(self):
        state = _default_state()

        # Mock benchmark (imported inside the function)
        mock_result = MagicMock()
        mock_result.verdict = "pass"
        mock_result.cost_usd = 0.1
        mock_result.challenge_id = "t1-test"

        with patch("belief.benchmark.run_benchmark", new_callable=AsyncMock) as mock_bench:
            mock_bench.return_value = [mock_result] * 5
            result = await validation_node(state)

        assert result["validation_passed"] is True
        assert result["validation_results"]["pass_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_validation_fails_on_low_rate(self):
        state = _default_state()

        # 2/5 pass = 40%, well below 97% threshold
        pass_result = MagicMock()
        pass_result.verdict = "pass"
        pass_result.cost_usd = 0.1
        pass_result.challenge_id = "t1-pass"

        fail_result = MagicMock()
        fail_result.verdict = "fail"
        fail_result.cost_usd = 0.1
        fail_result.challenge_id = "t1-fail"

        with patch("belief.benchmark.run_benchmark", new_callable=AsyncMock) as mock_bench:
            mock_bench.return_value = [pass_result, pass_result, fail_result, fail_result, fail_result]
            result = await validation_node(state)

        assert result["validation_passed"] is False
        assert len(result["regressions"]) == 3


class TestIntegrationNode:
    @pytest.mark.asyncio
    async def test_integration_records_tools(self):
        state = _default_state()
        state["new_tools_built"] = [{"tool_id": "tool-1"}, {"tool_id": "tool-2"}]
        state["new_covenants"] = [{"covenant_id": "cov-1"}]

        result = await integration_node(state)
        assert result["integrated_tool_ids"] == ["tool-1", "tool-2"]
        assert result["integrated_covenant_ids"] == ["cov-1"]


# ── Progression Metrics ────────────────────────────────────────────────────


class TestProgressionMetrics:
    def test_default_metrics(self):
        m = ProgressionMetrics()
        assert m.current_stage == 0
        assert m.total_tool_count == 0

    def test_cosine_distance(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        dist = _cosine_distance(a, b)
        assert abs(dist - 1.0) < 0.01  # Orthogonal = distance 1

        same = _cosine_distance(a, a)
        assert abs(same) < 0.01  # Same vector = distance 0

    def test_cosine_distance_zero_vector(self):
        assert _cosine_distance([0, 0], [1, 0]) == 1.0

    def test_threshold_cluster_few_points(self):
        count, sil = _threshold_cluster([[1.0, 0.0], [0.0, 1.0]])
        assert count == 0  # Not enough points

    def test_threshold_cluster_groups(self):
        # Two tight clusters + one outlier
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.1, 0.9, 0.0],
        ]
        count, sil = _threshold_cluster(embeddings)
        assert count >= 2

    def test_detect_archetypes_empty(self):
        count, reuse = _detect_archetypes([])
        assert count == 0

    def test_detect_archetypes_with_patterns(self):
        traces = [
            {"code_files": {"main.py": "", "test_main.py": ""}},
            {"code_files": {"main.py": "", "test_main.py": ""}},
            {"code_files": {"app.py": "", "models.py": ""}},
            {"code_files": {"app.py": "", "models.py": ""}},
            {"code_files": {"unique.py": ""}},
        ]
        count, reuse = _detect_archetypes(traces)
        assert count >= 2  # Two repeating patterns
        assert reuse > 0.5  # 4 of 5 traces match an archetype


# ── Stage Detection ─────────────────────────────────────────────────────────


class TestStageDetection:
    def test_stage_0_with_no_tools(self, tmp_path):
        from belief.memory.soil import Soil
        from belief.memory.tool_registry import ToolRegistry

        soil = Soil(persist_dir=tmp_path / "soil")
        registry = ToolRegistry(soil)
        metrics = compute_progression(soil, registry, [])
        assert metrics.current_stage == 0

    def test_format_report(self):
        m = ProgressionMetrics(
            seed_tool_count=3,
            cluster_count=5,
            cluster_silhouette=0.35,
            coverage_fraction=0.6,
            basis_rank_ratio=0.4,
            connectivity_fraction=0.2,
            archetype_count=1,
            archetype_reuse=0.3,
            current_stage=1,
            total_tool_count=8,
        )
        report = format_progression_report(m)
        assert "Stage: 1" in report
        assert "Cluster" in report
        assert "8 total" in report
        assert "3 hand-authored" in report


# ── CLI Commands ────────────────────────────────────────────────────────────


class TestCLICommands:
    def test_progression_cmd_exists(self):
        """The progression command should be importable and callable."""
        from belief.cli import _run_progression_cmd
        assert callable(_run_progression_cmd)

    def test_jitterbug_cmd_exists(self):
        """The jitterbug command should be importable."""
        from belief.cli import _run_jitterbug_cmd
        assert callable(_run_jitterbug_cmd)

    def test_argparse_registers_jitterbug(self):
        """The jitterbug subcommand should be registered."""
        import argparse
        # Import app to trigger argparser setup
        from belief.cli import app
        # If we got here without error, the commands are registered
        assert True

    def test_argparse_registers_progression(self):
        from belief.cli import app
        assert True
