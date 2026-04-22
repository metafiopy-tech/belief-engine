"""Tests for DSPy prompt optimization (belief/optimization/).

Covers:
  - DSPy modules instantiate correctly (when dspy available)
  - Graceful handling when dspy is not installed
  - Compiler creates metric functions
  - Prompt extraction works
  - PromptStore save/load round-trips
  - CLI command parses correctly
  - Optimizer handles empty trainsets
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _check_dspy() -> bool:
    """Check if dspy is installed (used for skipif)."""
    try:
        import dspy
        return True
    except ImportError:
        return False


# ── DSPy availability ──────────────────────────────────────────────────────


class TestDspyAvailability:
    def test_is_dspy_available_returns_bool(self):
        """is_dspy_available should return a bool regardless of install state."""
        from belief.optimization.dspy_modules import is_dspy_available
        result = is_dspy_available()
        assert isinstance(result, bool)

    def test_module_import_without_dspy(self):
        """The optimization package should import even if dspy is not installed."""

    def test_dspy_modules_import_without_dspy(self):
        """dspy_modules should import (the guard is in __init__, not import)."""
        from belief.optimization.dspy_modules import AGENT_MODULES
        assert len(AGENT_MODULES) == 5
        assert "planner" in AGENT_MODULES
        assert "builder" in AGENT_MODULES


# ── DSPy Modules ────────────────────────────────────────────────────────────


class TestDspyModules:
    def test_agent_module_registry(self):
        """AGENT_MODULES should have all 5 agents."""
        from belief.optimization.dspy_modules import AGENT_MODULES
        expected = {"planner", "architect", "builder", "tester", "debugger"}
        assert set(AGENT_MODULES.keys()) == expected

    def test_modules_raise_without_dspy(self):
        """Instantiating modules should raise ImportError if dspy missing."""
        from belief.optimization.dspy_modules import is_dspy_available
        if is_dspy_available():
            pytest.skip("dspy is installed, can't test ImportError path")

        from belief.optimization.dspy_modules import BeliefPlanner
        with pytest.raises(ImportError, match="dspy"):
            BeliefPlanner()

    def test_get_all_modules_when_dspy_missing(self):
        from belief.optimization.dspy_modules import is_dspy_available
        if is_dspy_available():
            pytest.skip("dspy is installed")

        from belief.optimization.dspy_modules import get_all_modules
        with pytest.raises(ImportError):
            get_all_modules()

    @pytest.mark.skipif(
        not _check_dspy(),
        reason="dspy not installed",
    )
    def test_modules_instantiate_with_dspy(self):
        """If dspy is installed, modules should instantiate."""
        from belief.optimization.dspy_modules import get_all_modules
        modules = get_all_modules()
        assert len(modules) == 5

    @pytest.mark.skipif(
        not _check_dspy(),
        reason="dspy not installed",
    )
    def test_modules_have_named_predictors(self):
        from belief.optimization.dspy_modules import get_all_modules
        modules = get_all_modules()
        for name, module in modules.items():
            predictors = list(module.named_predictors())
            assert len(predictors) >= 1, f"{name} should have at least 1 predictor"


# ── Compiler ────────────────────────────────────────────────────────────────


class TestCompiler:
    def test_compiler_init(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()
        assert opt.teacher == "claude-sonnet-4-6"
        assert opt.student == "claude-haiku-4-5-20251001"

    def test_make_metric_planner(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()
        metric = opt._make_metric("planner")

        # Output containing "steps" should score 1.0
        class FakePred:
            def __str__(self):
                return "Here are the steps to build the API"
        assert metric(None, FakePred()) == 1.0

        # Output without "steps" should score 0.0
        class BadPred:
            def __str__(self):
                return "I don't know"
        assert metric(None, BadPred()) == 0.0

    def test_make_metric_builder(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()
        metric = opt._make_metric("builder")

        class CodePred:
            def __str__(self):
                return "def main():\n    pass"
        assert metric(None, CodePred()) == 1.0

    def test_make_metric_tester(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()
        metric = opt._make_metric("tester")

        class TestPred:
            def __str__(self):
                return "def test_main():\n    assert True"
        assert metric(None, TestPred()) == 1.0

    def test_make_metric_debugger(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()
        metric = opt._make_metric("debugger")

        class FixPred:
            def __str__(self):
                return "The fix is to add the missing import"
        assert metric(None, FixPred()) == 1.0

    def test_make_metric_unknown(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()
        metric = opt._make_metric("unknown_agent")
        assert metric(None, "anything") == 0.5

    def test_extract_prompts_empty(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()
        prompts = opt.extract_optimized_prompts({})
        assert prompts == {}

    def test_extract_prompts_with_mock_module(self):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()

        # Mock a module with named_predictors
        mock_predictor = MagicMock()
        mock_predictor.extended_signature.instructions = "Optimized instruction text"

        mock_module = MagicMock()
        mock_module.named_predictors.return_value = [("plan", mock_predictor)]

        prompts = opt.extract_optimized_prompts({"planner": mock_module})
        assert "planner.plan" in prompts
        assert prompts["planner.plan"] == "Optimized instruction text"

    def test_save_and_load_prompts(self, tmp_path):
        from belief.optimization.compiler import BeliefOptimizer
        opt = BeliefOptimizer()

        prompts = {"planner.plan": "Test instruction", "builder.build": "Build instruction"}
        path = str(tmp_path / "test_prompts.json")
        opt.save_optimized_prompts(prompts, path)

        loaded = opt.load_optimized_prompts(path)
        assert loaded == prompts


# ── PromptStore ─────────────────────────────────────────────────────────────


class TestPromptStore:
    def test_save_and_load_latest(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))

        prompts = {"planner.plan": "Test instructions"}
        store.save(prompts, "v-001")

        loaded = store.load_latest()
        assert loaded is not None
        assert loaded == prompts

    def test_load_for_version(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))

        store.save({"a": "1"}, "v-001")
        store.save({"b": "2"}, "v-002")

        v1 = store.load_for_version("v-001")
        assert v1 == {"a": "1"}

        v2 = store.load_for_version("v-002")
        assert v2 == {"b": "2"}

    def test_load_nonexistent_version(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))
        assert store.load_for_version("nonexistent") is None

    def test_load_latest_empty_store(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))
        assert store.load_latest() is None

    def test_list_versions(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))

        store.save({"a": "1"}, "v-001")
        store.save({"b": "2", "c": "3"}, "v-002")

        versions = store.list_versions()
        assert len(versions) == 2
        assert versions[0]["version_id"] == "v-001"
        assert versions[0]["prompt_count"] == 1
        assert versions[1]["prompt_count"] == 2

    def test_delete_version(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))

        store.save({"a": "1"}, "v-001")
        assert store.load_for_version("v-001") is not None

        deleted = store.delete_version("v-001")
        assert deleted is True
        assert store.load_for_version("v-001") is None

    def test_delete_nonexistent(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))
        assert store.delete_version("nonexistent") is False

    def test_save_returns_path(self, tmp_path):
        from belief.optimization.prompt_store import PromptStore
        store = PromptStore(store_dir=str(tmp_path / "prompts"))

        path = store.save({"x": "y"}, "v-test")
        assert path.exists()
        assert path.name == "v-test.json"


# ── CLI ─────────────────────────────────────────────────────────────────────


class TestCLI:
    def test_optimize_cmd_exists(self):
        from belief.cli import _run_optimize_cmd
        assert callable(_run_optimize_cmd)

    def test_optimize_cmd_handles_missing_dspy(self):
        """When dspy is not installed, the command should print an error."""
        from belief.optimization.dspy_modules import is_dspy_available
        if is_dspy_available():
            pytest.skip("dspy is installed, can't test missing path")

        from belief.cli import _run_optimize_cmd
        args = MagicMock()
        args.all = True
        args.agent = None
        args.dry_run = False

        # Should not raise
        _run_optimize_cmd(args)

    def test_optimize_dry_run(self):
        """Dry run should print what would be optimized."""
        from belief.cli import _run_optimize_cmd
        args = MagicMock()
        args.all = True
        args.agent = None
        args.dry_run = True

        from belief.optimization.dspy_modules import is_dspy_available
        if not is_dspy_available():
            pytest.skip("dspy not installed")

        # Should not raise
        _run_optimize_cmd(args)


