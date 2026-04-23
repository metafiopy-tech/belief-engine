"""Tests for autocatalytic NEW_TOOL support.

Covers:
  - FailureCluster creation from synthetic failure data
  - Goal formulation produces clear, buildable goals
  - Tool validation catches bad tools (syntax errors, belief imports, dangerous calls)
  - Tool validation passes good tools
  - Tool registry stores and retrieves tools
  - Usage tracking updates quality scores
  - End-to-end: synthetic failures -> cluster -> formulate goal -> validate -> register
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from belief.evolution.self_improvement import (
    FailureCluster,
    ImprovementType,
    Mentor,
    ImprovementProposal,
    cluster_failures,
    evaluate_tool_against_failures,
    formulate_tool_goal,
    select_target_cluster,
)
from belief.evolution.tool_validator import validate_tool
from belief.memory.tool_registry import SelfAuthoredTool, ToolRegistry


# ── Test helpers ────────────────────────────────────────────────────────────

GOOD_TOOL_CODE = '''\
"""Validates test file structure."""


def validate_tests(code: str) -> list[str]:
    """Check that test files have proper structure.

    Args:
        code: Python test file source code.

    Returns:
        List of validation error strings.
    """
    errors = []
    if "def test_" not in code:
        errors.append("No test functions found")
    return errors
'''

BAD_TOOL_SYNTAX = "def broken(:\n    pass"

BAD_TOOL_BELIEF_IMPORT = '''\
"""Bad tool that imports from belief."""
from belief.memory.soil import Soil

def check(code):
    return []
'''

BAD_TOOL_DANGEROUS = '''\
"""Bad tool with dangerous calls."""

def check(code: str) -> list[str]:
    exec(code)
    return []
'''

BAD_TOOL_OS_REMOVE = '''\
"""Bad tool that deletes files."""
import os

def cleanup(path: str) -> list[str]:
    os.remove(path)
    return []
'''

GOOD_TOOL_AST = '''\
"""Checks for bare except clauses."""
import ast


def check_bare_except(code: str) -> list[str]:
    """Find bare except clauses in Python code."""
    errors = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Could not parse code"]
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            errors.append(f"Bare except at line {node.lineno}")
    return errors
'''


def _make_tool(**kwargs) -> SelfAuthoredTool:
    """Create a test SelfAuthoredTool with sensible defaults."""
    defaults = {
        "id": str(uuid.uuid4()),
        "name": "test_tool",
        "description": "A test tool",
        "code": GOOD_TOOL_CODE,
        "input_description": "Python source code",
        "output_description": "List of error strings",
    }
    defaults.update(kwargs)
    return SelfAuthoredTool(**defaults)


# ── FailureCluster ──────────────────────────────────────────────────────────


class TestFailureCluster:
    def test_cluster_creation(self):
        c = FailureCluster(
            error_type="missing_import",
            count=5,
            example_errors=["ImportError: no module named 'foo'"],
            suggested_tool_name="import_checker",
            suggested_tool_description="Checks for missing imports",
            input_description="Python code",
            output_description="List of errors",
        )
        assert c.count == 5
        assert c.error_type == "missing_import"
        assert not c.addressed_by_existing_tool

    def test_cluster_failures_groups_by_type(self):
        failures = [
            {"content": "ImportError: no module named 'xyz'", "trace_id": "f1"},
            {"content": "ImportError: cannot import 'abc' from 'foo'", "trace_id": "f2"},
            {"content": "SyntaxError: invalid syntax line 5", "trace_id": "f3"},
            {"content": "ImportError: module not found 'bar'", "trace_id": "f4"},
        ]
        clusters = cluster_failures(failures)
        assert len(clusters) >= 2

        # The import cluster should have count >= 3
        import_cluster = next((c for c in clusters if c.error_type == "missing_import"), None)
        assert import_cluster is not None
        assert import_cluster.count >= 3

    def test_cluster_sorted_by_count(self):
        failures = [{"content": "ImportError: no module named 'x'"}] * 5 + [
            {"content": "SyntaxError: invalid syntax"}
        ] * 2
        clusters = cluster_failures(failures)
        if len(clusters) >= 2:
            assert clusters[0].count >= clusters[1].count

    def test_empty_failures(self):
        assert cluster_failures([]) == []


class TestSelectTargetCluster:
    def test_selects_largest_unaddressed(self):
        clusters = [
            FailureCluster(
                error_type="missing_import",
                count=10,
                suggested_tool_name="import_checker",
            ),
            FailureCluster(
                error_type="syntax_error",
                count=5,
                suggested_tool_name="syntax_validator",
            ),
        ]
        target = select_target_cluster(clusters, existing_tool_names=[])
        assert target is not None
        assert target.error_type == "missing_import"

    def test_skips_addressed_clusters(self):
        clusters = [
            FailureCluster(
                error_type="missing_import",
                count=10,
                suggested_tool_name="import_checker",
            ),
            FailureCluster(
                error_type="syntax_error",
                count=5,
                suggested_tool_name="syntax_validator",
            ),
        ]
        target = select_target_cluster(clusters, existing_tool_names=["import_checker"])
        assert target is not None
        assert target.error_type == "syntax_error"

    def test_returns_none_when_all_addressed(self):
        clusters = [
            FailureCluster(
                error_type="missing_import",
                count=10,
                suggested_tool_name="import_checker",
            ),
        ]
        target = select_target_cluster(clusters, existing_tool_names=["import_checker"])
        assert target is None

    def test_skips_small_clusters(self):
        clusters = [
            FailureCluster(error_type="rare_error", count=2, suggested_tool_name="rare_checker"),
        ]
        target = select_target_cluster(clusters, existing_tool_names=[])
        assert target is None  # count < 3


# ── Goal Formulation ───────────────────────────────────────────────────────


class TestGoalFormulation:
    def test_produces_buildable_goal(self):
        cluster = FailureCluster(
            error_type="missing_import",
            count=10,
            example_errors=[
                "ImportError: no module named 'uvicorn'",
                "ImportError: cannot import 'FastAPI'",
            ],
            suggested_tool_name="import_checker",
            suggested_tool_description="missing or incorrect import statements",
            input_description="Python source code as a string",
            output_description="List of validation error strings",
        )
        goal = formulate_tool_goal(cluster)

        # Session 2: goals are signature-agnostic but must embed
        # enough cluster context to be buildable.
        # 1. Error type anchors the "what are we fixing" context.
        assert "missing_import" in goal
        # 2. At least one concrete example error should make it in
        #    (pulled from failure_traces or example_errors).
        assert "ImportError" in goal
        # 3. Self-contained stdlib-only constraint still holds.
        assert "self-contained" in goal.lower() or "Self-contained" in goal
        # 4. Spec requires "Include pytest tests that verify..."
        assert "test" in goal.lower()

    def test_goal_includes_examples(self):
        cluster = FailureCluster(
            error_type="test",
            count=5,
            example_errors=["Error A", "Error B"],
            suggested_tool_name="test_tool",
        )
        goal = formulate_tool_goal(cluster)
        assert "Error A" in goal
        assert "Error B" in goal


# ── Tool Validation ─────────────────────────────────────────────────────────


class TestToolValidation:
    def test_valid_tool_passes(self):
        tool = _make_tool(code=GOOD_TOOL_CODE)
        result = validate_tool(tool)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_ast_tool_passes(self):
        tool = _make_tool(code=GOOD_TOOL_AST)
        result = validate_tool(tool)
        assert result.valid is True

    def test_syntax_error_fails(self):
        tool = _make_tool(code=BAD_TOOL_SYNTAX)
        result = validate_tool(tool)
        assert result.valid is False
        assert any("Syntax error" in e for e in result.errors)

    def test_belief_import_fails(self):
        tool = _make_tool(code=BAD_TOOL_BELIEF_IMPORT)
        result = validate_tool(tool)
        assert result.valid is False
        assert any("belief internals" in e for e in result.errors)

    def test_dangerous_call_fails(self):
        tool = _make_tool(code=BAD_TOOL_DANGEROUS)
        result = validate_tool(tool)
        assert result.valid is False
        assert any("Dangerous call" in e for e in result.errors)

    def test_os_remove_fails(self):
        tool = _make_tool(code=BAD_TOOL_OS_REMOVE)
        result = validate_tool(tool)
        assert result.valid is False
        assert any("os.remove" in e for e in result.errors)

    def test_empty_code_fails(self):
        tool = _make_tool(code="")
        result = validate_tool(tool)
        assert result.valid is False

    def test_no_docstring_warns(self):
        tool = _make_tool(code="def foo():\n    pass\n")
        result = validate_tool(tool)
        assert result.valid is True
        assert any("docstring" in w.lower() for w in result.warnings)

    def test_long_code_warns(self):
        long_code = '"""Long tool."""\n' + "x = 1\n" * 201
        tool = _make_tool(code=long_code)
        result = validate_tool(tool)
        assert result.valid is True
        assert any("200" in w for w in result.warnings)


# ── Tool Registry ──────────────────────────────────────────────────────────


class TestToolRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        from belief.memory.soil import Soil

        soil = Soil(persist_dir=tmp_path / "soil")
        return ToolRegistry(soil)

    def test_register_and_get(self, registry):
        tool = _make_tool()
        tool_id = registry.register_tool(tool)
        assert tool_id == tool.id

        retrieved = registry.get_tool(tool_id)
        assert retrieved.name == tool.name
        assert retrieved.code == tool.code

    def test_get_active_tools(self, registry):
        registry.register_tool(_make_tool(id="t1", name="tool_one"))
        registry.register_tool(_make_tool(id="t2", name="tool_two"))

        active = registry.get_active_tools()
        assert len(active) == 2

    def test_get_active_excludes_lapsed(self, registry):
        tool = _make_tool(id="t1", name="lapsed_tool")
        tool.fsrs_decay_state = "lapsed"
        registry.register_tool(tool)

        active = registry.get_active_tools()
        # Lapsed tools should be excluded
        assert all(t.name != "lapsed_tool" or t.fsrs_decay_state != "lapsed" for t in active)

    def test_find_tools_for_goal(self, registry):
        registry.register_tool(
            _make_tool(
                id="t1",
                name="fastapi_validator",
                description="Validates FastAPI routes",
            )
        )
        registry.register_tool(
            _make_tool(
                id="t2",
                name="import_checker",
                description="Checks Python imports",
            )
        )

        results = registry.find_tools_for_goal("Build a FastAPI API")
        assert len(results) >= 1

    def test_tool_not_found(self, registry):
        with pytest.raises(KeyError):
            registry.get_tool("nonexistent")

    def test_get_tool_health_empty(self, registry):
        health = registry.get_tool_health()
        assert health["count"] == 0


# ── Usage Tracking ─────────────────────────────────────────────────────────


class TestUsageTracking:
    @pytest.fixture
    def registry(self, tmp_path):
        from belief.memory.soil import Soil

        soil = Soil(persist_dir=tmp_path / "soil")
        return ToolRegistry(soil)

    def test_record_success_updates_stats(self, registry):
        tool = _make_tool(id="t1")
        registry.register_tool(tool)

        registry.record_usage("t1", success=True)
        updated = registry.get_tool("t1")
        assert updated.use_count == 1
        assert updated.success_rate == 1.0
        assert updated.last_used is not None

    def test_record_failure_updates_stats(self, registry):
        tool = _make_tool(id="t1")
        registry.register_tool(tool)

        registry.record_usage("t1", success=False)
        updated = registry.get_tool("t1")
        assert updated.use_count == 1
        assert updated.success_rate == 0.0

    def test_mixed_usage_running_average(self, registry):
        tool = _make_tool(id="t1")
        registry.register_tool(tool)

        registry.record_usage("t1", success=True)
        registry.record_usage("t1", success=True)
        registry.record_usage("t1", success=False)

        updated = registry.get_tool("t1")
        assert updated.use_count == 3
        assert abs(updated.success_rate - 2.0 / 3.0) < 0.01

    def test_fsrs_stability_grows_on_success(self, registry):
        tool = _make_tool(id="t1")
        tool.fsrs_stability = 1.0
        registry.register_tool(tool)

        registry.record_usage("t1", success=True)
        updated = registry.get_tool("t1")
        assert updated.fsrs_stability > 1.0

    def test_fsrs_stability_drops_on_failure(self, registry):
        tool = _make_tool(id="t1")
        tool.fsrs_stability = 10.0
        tool.fsrs_decay_state = "stable"
        registry.register_tool(tool)

        registry.record_usage("t1", success=False)
        updated = registry.get_tool("t1")
        assert updated.fsrs_stability < 10.0
        assert updated.fsrs_decay_state == "lapsed"

    def test_tool_health_with_data(self, registry):
        registry.register_tool(_make_tool(id="t1"))
        registry.register_tool(_make_tool(id="t2"))
        registry.record_usage("t1", success=True)

        health = registry.get_tool_health()
        assert health["count"] == 2
        assert health["total_uses"] == 1


# ── Test Tool Against Failures ──────────────────────────────────────────────


class TestToolAgainstFailures:
    def test_tool_catches_failures(self):
        tool_code = GOOD_TOOL_AST
        failures = [
            {"code_sample": "try:\n    pass\nexcept:\n    pass\n"},
            {"code_sample": "try:\n    x = 1\nexcept:\n    pass\n"},
            {"code_sample": "x = 1\ny = 2\n"},  # No bare except
        ]
        catch_rate = evaluate_tool_against_failures(tool_code, failures)
        # Should catch 2/3 failures
        assert catch_rate >= 0.5

    def test_empty_failures_returns_zero(self):
        assert evaluate_tool_against_failures("def f(): pass", []) == 0.0

    def test_invalid_tool_returns_zero(self):
        assert evaluate_tool_against_failures("invalid python {{", [{"code_sample": "x=1"}]) == 0.0


# ── Mentor NEW_TOOL Approval ──────────────────────────────────────────────


class TestMentorNewTool:
    def test_approves_new_tool(self):
        mentor = Mentor()
        proposal = ImprovementProposal(
            title="Add import checker tool",
            description="Build a tool that checks imports",
            improvement_type=ImprovementType.NEW_TOOL,
            target_file="tools/import_checker.py",
            current_code="",
            proposed_code="",
            expected_benefit="Catch import errors before runtime",
            risk_level="low",
        )
        verdict = mentor.evaluate(proposal)
        assert verdict.approved is True
        assert "tool" in verdict.reasoning.lower()


# ── End-to-End (mocked pipeline) ───────────────────────────────────────────


class TestEndToEnd:
    def test_cluster_to_goal_to_validate_to_register(self, tmp_path):
        """Full flow with synthetic data, no API calls."""
        from belief.memory.soil import Soil

        soil = Soil(persist_dir=tmp_path / "soil")
        registry = ToolRegistry(soil)

        # 1. Create synthetic failures
        failures = [
            {"content": "ImportError: no module named 'uvicorn'", "trace_id": f"f{i}"}
            for i in range(8)
        ]

        # 2. Cluster
        clusters = cluster_failures(failures)
        assert len(clusters) >= 1

        # 3. Select target
        target = select_target_cluster(clusters, existing_tool_names=[])
        assert target is not None

        # 4. Formulate goal
        goal = formulate_tool_goal(target)
        assert len(goal) > 50

        # 5. Create a mock tool (simulating engine output)
        tool = SelfAuthoredTool(
            id=str(uuid.uuid4()),
            name=target.suggested_tool_name,
            description=target.suggested_tool_description,
            code=GOOD_TOOL_CODE,
            input_description=target.input_description,
            output_description=target.output_description,
        )

        # 6. Validate
        validation = validate_tool(tool)
        assert validation.valid is True

        # 7. Register
        tool_id = registry.register_tool(tool)
        assert tool_id == tool.id

        # 8. Retrieve
        retrieved = registry.get_tool(tool_id)
        assert retrieved.name == target.suggested_tool_name

        # 9. Tool health
        health = registry.get_tool_health()
        assert health["count"] == 1

    def test_metadata_roundtrip(self, tmp_path):
        """Tool metadata survives ChromaDB save/load cycle."""
        from belief.memory.soil import Soil

        soil = Soil(persist_dir=tmp_path / "soil")
        registry = ToolRegistry(soil)

        tool = _make_tool(
            name="roundtrip_test",
            description="Tests metadata persistence",
            input_description="Python code",
            output_description="Error list",
            dependencies=["pytest"],
            version=2,
        )
        registry.register_tool(tool)

        retrieved = registry.get_tool(tool.id)
        assert retrieved.name == "roundtrip_test"
        assert retrieved.description == "Tests metadata persistence"
        assert retrieved.input_description == "Python code"
        assert retrieved.version == 2
        assert retrieved.dependencies == ["pytest"]


# ── SICA dispatch tests ──────────────────────────────────────────────────────


class TestSICANewToolDispatch:
    """Verify SICA has the NEW_TOOL dispatch path wired in."""

    def test_should_propose_new_tool_exists(self, tmp_path):
        """_should_propose_new_tool method exists and is callable."""
        from belief.evolution.sica import SelfImprovementCycle

        sica = SelfImprovementCycle(
            project_root=tmp_path,
            archive_path=tmp_path / "archive.json",
        )
        assert hasattr(sica, "_should_propose_new_tool"), (
            "SelfImprovementCycle must have _should_propose_new_tool"
        )

    def test_should_propose_new_tool_returns_false_without_failures(self, tmp_path):
        """Returns False when there are no failure traces (empty soil)."""
        import asyncio
        from belief.evolution.sica import SelfImprovementCycle

        mock_soil = MagicMock()
        sica = SelfImprovementCycle(
            project_root=tmp_path,
            archive_path=tmp_path / "archive.json",
            soil=mock_soil,
        )
        # Patch get_recent_failures to return empty list (no failures recorded)
        with patch(
            "belief.evolution.self_improvement.get_recent_failures",
            return_value=[],
        ):
            result = asyncio.run(sica._should_propose_new_tool())
        assert result is False, "Should not propose NEW_TOOL with no failure traces"

    def test_run_one_iteration_handles_new_tool_type(self, tmp_path):
        """run_one_iteration source contains new_tool handling."""
        import inspect
        from belief.evolution.sica import SelfImprovementCycle

        sica = SelfImprovementCycle(
            project_root=tmp_path,
            archive_path=tmp_path / "archive.json",
        )
        source = inspect.getsource(sica.run_one_iteration)
        assert "new_tool" in source.lower(), (
            "run_one_iteration must contain new_tool dispatch logic"
        )

    def test_soil_initialized_on_instance(self, tmp_path):
        """SelfImprovementCycle initializes a soil attribute."""
        from belief.evolution.sica import SelfImprovementCycle

        sica = SelfImprovementCycle(
            project_root=tmp_path,
            archive_path=tmp_path / "archive.json",
        )
        assert hasattr(sica, "soil"), "SelfImprovementCycle must have a soil attribute"

    def test_custom_soil_injected(self, tmp_path):
        """Soil can be injected via constructor."""
        from belief.evolution.sica import SelfImprovementCycle

        mock_soil = MagicMock()
        sica = SelfImprovementCycle(
            project_root=tmp_path,
            archive_path=tmp_path / "archive.json",
            soil=mock_soil,
        )
        assert sica.soil is mock_soil
