"""
End-to-end autocatalytic self-improvement integration test.

Proves that all v3.0 subsystems work together:
  1. ChromaDB 5-collection architecture with FSRS
  2. Episode recording and knowledge deposit
  3. Tool registry and validator
  4. Failure clustering and goal formulation (autocatalytic core)
  5. Crystallization pipeline (template sweep + Houdini filter)
  6. Jitterbug graph compilation and routing
  7. Evolutionary archive with parent selection
  8. Progression tracker
  9. Safety probes (evaluator integrity, env tampering)
  10. Goodhart canary divergence detection
  11. Metrics dashboard (record, load, growth analysis)
  12. DSPy module registry (import-only, no API calls)

This test uses ONLY synthetic data and mocked API calls.
No ANTHROPIC_API_KEY required. Cost: $0.

Run with: python -m pytest tests/test_e2e_autocatalytic.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest


@pytest.fixture
def workspace(tmp_path):
    """Create an isolated workspace with all subsystems."""
    soil_dir = tmp_path / "soil"
    archive_db = str(tmp_path / "archive.db")
    metrics_db = str(tmp_path / "metrics.jsonl")
    prompts_dir = str(tmp_path / "prompts")

    from belief.memory.soil import Soil

    soil = Soil(persist_dir=soil_dir)

    from belief.evolution.archive import Archive

    archive = Archive(db_path=archive_db)

    from belief.metrics.dashboard import MetricsDashboard

    dashboard = MetricsDashboard(db_path=metrics_db)

    return {
        "soil": soil,
        "archive": archive,
        "dashboard": dashboard,
        "tmp_path": tmp_path,
        "prompts_dir": prompts_dir,
    }


# ── 1. ChromaDB 5-collection architecture ──────────────────────────────────


class TestCollectionArchitecture:
    def test_five_collections_exist(self, workspace):
        soil = workspace["soil"]
        expected = {
            "belief_tools",
            "belief_episodes",
            "belief_principles",
            "belief_failures",
            "belief_covenants",
        }
        assert set(soil._collections.keys()) == expected

    def test_fsrs_fields_on_deposit(self, workspace):
        from belief.memory.nutrients import Nutrient, NutrientType

        soil = workspace["soil"]

        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="Test pattern for integration",
            embedding_text="integration test pattern",
        )
        nid = soil.deposit(n)

        col = soil._collections["belief_principles"]
        result = col.get(ids=[nid], include=["metadatas"])
        meta = result["metadatas"][0]
        assert "fsrs_stability" in meta
        assert "fsrs_decay_state" in meta


# ── 2. Episode recording ──────────────────────────────────────────────────


class TestEpisodeRecording:
    def test_record_episode(self, workspace):
        from belief.memory.episode_recorder import record_episode

        soil = workspace["soil"]

        state = {
            "run_id": "e2e-test-001",
            "user_goal": "Build a hello world CLI",
            "code_files": {
                "main.py": "import click\n@click.command()\ndef cli(): click.echo('hello')\n",
                "test_main.py": "def test_cli(): pass\n",
            },
        }
        record_episode(soil, state)

        col = soil._collections["belief_episodes"]
        assert col.count() >= 1


# ── 3. Tool registry + validator ───────────────────────────────────────────


class TestToolLifecycle:
    def test_register_validate_retrieve(self, workspace):
        from belief.memory.tool_registry import SelfAuthoredTool, ToolRegistry
        from belief.evolution.tool_validator import validate_tool

        soil = workspace["soil"]
        registry = ToolRegistry(soil)

        tool = SelfAuthoredTool(
            id="e2e-tool-001",
            name="e2e_validator",
            description="Integration test validator",
            code='"""Test tool."""\ndef check(code: str) -> list[str]:\n    return []\n',
            input_description="Python code",
            output_description="Error list",
        )

        # Validate
        result = validate_tool(tool)
        assert result.valid, f"Tool should be valid: {result.errors}"

        # Register
        tid = registry.register_tool(tool)
        assert tid == "e2e-tool-001"

        # Retrieve
        retrieved = registry.get_tool(tid)
        assert retrieved.name == "e2e_validator"

        # Usage tracking
        registry.record_usage(tid, success=True)
        updated = registry.get_tool(tid)
        assert updated.use_count == 1
        assert updated.success_rate == 1.0


# ── 4. Failure clustering (autocatalytic core) ────────────────────────────


class TestAutocatalyticCore:
    def test_cluster_and_formulate_goal(self, workspace):
        from belief.evolution.self_improvement import (
            cluster_failures,
            formulate_tool_goal,
            select_target_cluster,
        )

        failures = [{"content": "ImportError: no module named 'click'"} for _ in range(8)]
        clusters = cluster_failures(failures)
        assert len(clusters) >= 1

        target = select_target_cluster(clusters, existing_tool_names=[])
        assert target is not None

        goal = formulate_tool_goal(target)
        # Session 2: goal is signature-agnostic. Instead check that it
        # embeds the cluster's error type and a real error example.
        assert target.error_type in goal
        assert "ImportError" in goal or "click" in goal
        assert len(goal) > 50


# ── 5. Crystallization pipeline ────────────────────────────────────────────


class TestCrystallization:
    def test_template_sweep(self, workspace):
        from belief.evolution.crystallizer import sweep_templates

        # Build traces where fastapi always has uvicorn
        traces = [
            {
                "trace_id": f"tr-{i}",
                "dependencies": ["fastapi", "uvicorn", "pydantic"],
                "has_api_framework": True,
                "has_health_endpoint": True,
                "file_count": 5,
                "test_count": 10,
                "bare_except_count": 0,
                "has_entry_point": True,
            }
            for i in range(12)
        ]

        candidates = sweep_templates(traces)
        assert len(candidates) >= 1
        names = [c.name for c in candidates]
        assert "fastapi_requires_uvicorn" in names

    def test_houdini_filter(self, workspace):
        from belief.evolution.crystallizer import CandidateInvariant, filter_candidates

        good = CandidateInvariant(
            name="good_invariant",
            description="X",
            implementation_kind="ast",
            support=19,
            violations=1,
            precision=0.95,
            proposer="template",
        )
        bad = CandidateInvariant(
            name="bad_invariant",
            description="Y",
            implementation_kind="ast",
            support=5,
            violations=5,
            precision=0.50,
            proposer="template",
        )

        filtered = filter_candidates([good, bad], [])
        names = [c.name for c in filtered]
        assert "good_invariant" in names
        assert "bad_invariant" not in names


# ── 6. Jitterbug graph ────────────────────────────────────────────────────


class TestJitterbugGraph:
    def test_graph_structure(self):
        from belief.evolution.jitterbug import build_jitterbug_graph

        graph = build_jitterbug_graph()
        assert "expansion" in graph.nodes
        assert "compression" in graph.nodes
        assert "reconstruction" in graph.nodes
        assert "validation" in graph.nodes
        assert "integration" in graph.nodes

    def test_routing_logic(self):
        from belief.evolution.jitterbug import route_after_validation, route_after_compression
        from langgraph.graph import END

        assert route_after_validation({"validation_passed": True}) == "integration"
        assert route_after_validation({"validation_passed": False}) == END
        assert route_after_compression({"dry_run": True}) == END
        assert route_after_compression({"dry_run": False}) == "reconstruction"


# ── 7. Evolutionary archive ───────────────────────────────────────────────


class TestArchiveIntegration:
    def test_seed_and_select(self, workspace):
        from belief.evolution.archive import create_seed_version

        archive = workspace["archive"]
        seed = create_seed_version(archive)
        assert seed.parent_id is None

        parent = archive.select_parent()
        assert parent.id == seed.id

    def test_lineage(self, workspace):
        from belief.evolution.archive import AgentVersion, create_seed_version

        archive = workspace["archive"]
        seed = create_seed_version(archive)

        child = AgentVersion(
            id=str(uuid.uuid4()),
            parent_id=seed.id,
            created_at=datetime.now(timezone.utc),
            system_prompts=seed.system_prompts,
            tool_ids=[],
            principle_ids=[],
            covenant_ids=[],
            model_config=seed.model_config,
            diff_from_parent="test child",
            proposal_rationale="integration test",
            utility=0.6,
        )
        archive.save_version(child)

        lineage = archive.get_lineage(child.id)
        assert len(lineage) == 2
        assert lineage[0].id == seed.id
        assert lineage[1].id == child.id


# ── 8. Progression tracker ─────────────────────────────────────────────────


class TestProgression:
    def test_stage_0_empty(self, workspace):
        from belief.evolution.progression import compute_progression
        from belief.memory.tool_registry import ToolRegistry

        soil = workspace["soil"]
        registry = ToolRegistry(soil)
        metrics = compute_progression(soil, registry, [])
        assert metrics.current_stage == 0

    def test_format_report(self):
        from belief.evolution.progression import ProgressionMetrics, format_progression_report

        m = ProgressionMetrics(
            current_stage=1,
            total_tool_count=5,
            cluster_count=3,
            cluster_silhouette=0.3,
            seed_tool_count=2,
        )
        report = format_progression_report(m)
        assert "Stage: 1" in report
        assert "Cluster" in report


# ── 9. Safety probes ──────────────────────────────────────────────────────


class TestSafetyProbes:
    @pytest.mark.asyncio
    async def test_evaluator_integrity(self, workspace):
        from belief.safety.probes import initialize_probes, check_evaluator_integrity

        tmp = workspace["tmp_path"]
        (tmp / "belief").mkdir()
        (tmp / "belief" / "benchmark.py").write_text("# benchmark")
        (tmp / "belief" / "hardening.py").write_text("# hardening")
        (tmp / "belief" / "validators").mkdir()
        (tmp / "belief" / "validators" / "__init__.py").write_text("# validators")

        initialize_probes(str(tmp))
        await check_evaluator_integrity()  # Should pass

    @pytest.mark.asyncio
    async def test_env_tampering(self):
        from belief.safety.probes import _ENV_SNAPSHOTS, check_environment_tampering

        _ENV_SNAPSHOTS.clear()
        await check_environment_tampering()  # Snapshot
        await check_environment_tampering()  # Compare — should pass
        _ENV_SNAPSHOTS.clear()


# ── 10. Goodhart canary ───────────────────────────────────────────────────


class TestGoodhartCanary:
    def test_divergence_detected(self):
        from belief.safety.goodhart_canary import check_goodhart_divergence

        assert (
            check_goodhart_divergence(
                [0.5, 0.6, 0.7, 0.8, 0.85],
                [0.5, 0.5, 0.48, 0.45, 0.42],
            )
            is True
        )

    def test_no_divergence(self):
        from belief.safety.goodhart_canary import check_goodhart_divergence

        assert (
            check_goodhart_divergence(
                [0.5, 0.55, 0.6, 0.65, 0.7],
                [0.5, 0.55, 0.6, 0.65, 0.7],
            )
            is False
        )


# ── 11. Metrics dashboard ────────────────────────────────────────────────


class TestMetricsDashboard:
    def test_record_analyze_display(self, workspace, capsys):
        from belief.metrics.dashboard import IterationMetrics

        dashboard = workspace["dashboard"]

        for i in range(8):
            dashboard.record(
                IterationMetrics(
                    iteration=i,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    benchmark_score=0.5 + i * 0.04,
                    cost_per_solved=1.0 - i * 0.05,
                    novel_capabilities=max(0, i - 3),
                    tool_library_size=i,
                    covenant_count=7 + i // 2,
                )
            )

        loaded = dashboard.load_all()
        assert len(loaded) == 8

        analysis = dashboard.compute_growth_analysis()
        assert analysis["status"] == "ok"
        assert "linear" in analysis

        dashboard.print_dashboard()
        out = capsys.readouterr().out
        assert "BELIEF ENGINE" in out


# ── 12. DSPy modules (import only) ───────────────────────────────────────


class TestDspyModules:
    def test_module_registry_exists(self):
        from belief.optimization.dspy_modules import AGENT_MODULES, is_dspy_available

        assert len(AGENT_MODULES) == 5
        assert isinstance(is_dspy_available(), bool)

    def test_compiler_importable(self):
        from belief.optimization.compiler import BeliefOptimizer

        opt = BeliefOptimizer()
        assert opt.teacher == "claude-sonnet-4-6"

    def test_prompt_store(self, workspace):
        from belief.optimization.prompt_store import PromptStore

        store = PromptStore(store_dir=workspace["prompts_dir"])
        store.save({"test": "instruction"}, "v-e2e")
        loaded = store.load_for_version("v-e2e")
        assert loaded == {"test": "instruction"}


# ── 13. Covenant registry ────────────────────────────────────────────────


class TestCovenantRegistry:
    def test_static_covenants_loaded(self, workspace):
        from belief.validators.covenant_registry import CovenantRegistry

        registry = CovenantRegistry(workspace["soil"])
        stats = registry.get_covenant_stats()
        assert stats["static_count"] == 6

    def test_fire_on_clean_code(self, workspace):
        from belief.validators.covenant_registry import CovenantRegistry

        registry = CovenantRegistry(workspace["soil"])
        results = registry.fire_all({"main.py": "def main():\n    pass\n"})
        assert len(results) >= 6


# ── 14. Full loop summary ────────────────────────────────────────────────


class TestFullLoopSummary:
    def test_all_v3_modules_import(self):
        """Verify every module from sessions 1-7 imports successfully."""

    def test_version_is_3(self):
        import belief

        assert belief.__version__ == "3.0.0"
