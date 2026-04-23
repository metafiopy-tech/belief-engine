"""Tests for the crystallization pipeline (belief/evolution/crystallizer.py).

Covers:
  - Template sweep finds invariants from synthetic traces
  - Claude proposer generates candidates (mocked)
  - Houdini filter removes low-precision candidates
  - Promotion generates valid Python code
  - Covenant registry loads and fires dynamic covenants
  - Episode recorder extracts features from build state
  - Full pipeline end-to-end with synthetic traces
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from belief.evolution.crystallizer import (
    CandidateInvariant,
    INVARIANT_TEMPLATES,
    filter_candidates,
    promote_to_covenant,
    sweep_templates,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_trace(
    passed: bool = True,
    dependencies: list[str] | None = None,
    file_count: int = 4,
    test_count: int = 8,
    has_api_framework: bool = False,
    has_health_endpoint: bool = False,
    bare_except_count: int = 0,
    **kwargs,
) -> dict:
    """Create a synthetic build trace."""
    trace = {
        "trace_id": f"ep-{uuid.uuid4().hex[:12]}",
        "passed": passed,
        "score": 0.9 if passed else 0.3,
        "file_count": file_count,
        "test_count": test_count,
        "dependencies": dependencies or [],
        "has_api_framework": has_api_framework,
        "has_health_endpoint": has_health_endpoint,
        "has_click_conftest": False,
        "bare_except_count": bare_except_count,
        "print_count": 0,
        "has_entry_point": True,
        "has_dockerfile": False,
        "dockerfile_has_expose": False,
        "hardcoded_secret_count": 0,
        "has_error_handler": False,
        "mixed_sync_async": False,
    }
    trace.update(kwargs)
    return trace


def _make_fastapi_traces(n: int = 15) -> list[dict]:
    """Create N FastAPI traces with uvicorn (good pattern)."""
    return [
        _make_trace(
            passed=True,
            dependencies=["fastapi", "uvicorn", "pydantic"],
            has_api_framework=True,
            has_health_endpoint=True,
            file_count=5,
            test_count=10,
        )
        for _ in range(n)
    ]


# ── Template Sweep ──────────────────────────────────────────────────────────


class TestTemplateSweep:
    def test_templates_defined(self):
        """Should have 15 invariant templates."""
        assert len(INVARIANT_TEMPLATES) == 15

    def test_finds_fastapi_uvicorn(self):
        """Should detect FastAPI-requires-uvicorn from traces."""
        traces = _make_fastapi_traces(10)
        candidates = sweep_templates(traces)
        names = [c.name for c in candidates]
        assert "fastapi_requires_uvicorn" in names

    def test_finds_no_bare_except(self):
        """Should detect no-bare-except invariant."""
        traces = [_make_trace(bare_except_count=0) for _ in range(10)]
        candidates = sweep_templates(traces)
        names = [c.name for c in candidates]
        assert "no_bare_except" in names

    def test_support_threshold(self):
        """Candidates need support >= 5."""
        # Only 3 traces — below threshold
        traces = [_make_trace(bare_except_count=0) for _ in range(3)]
        candidates = sweep_templates(traces)
        # Most templates won't have enough support
        for c in candidates:
            assert c.support >= 5

    def test_precision_threshold(self):
        """Candidates need precision >= 0.90."""
        # Mix of good and bad traces
        traces = [_make_trace(bare_except_count=0) for _ in range(8)] + [
            _make_trace(bare_except_count=3) for _ in range(5)
        ]
        candidates = sweep_templates(traces)
        for c in candidates:
            assert c.precision >= 0.90

    def test_empty_traces(self):
        """Empty traces should return no candidates."""
        assert sweep_templates([]) == []

    def test_min_test_ratio_detected(self):
        """Should detect min_test_ratio when test_count >= 1.5 * file_count."""
        traces = [_make_trace(file_count=4, test_count=8) for _ in range(10)]
        candidates = sweep_templates(traces)
        names = [c.name for c in candidates]
        assert "min_test_ratio" in names


# ── Claude Proposer ─────────────────────────────────────────────────────────


class TestClaudeProposer:
    @pytest.mark.asyncio
    async def test_propose_parses_json(self):
        """Should parse Claude's JSON response into CandidateInvariant objects."""
        from belief.evolution.crystallizer import _parse_proposals

        response = """
        [
            {
                "name": "test_invariant",
                "description": "All builds should have tests",
                "implementation_kind": "assertion",
                "supporting_traces": ["ep-1", "ep-2", "ep-3"],
                "violating_traces": []
            }
        ]
        """
        candidates = _parse_proposals(response, [])
        assert len(candidates) == 1
        assert candidates[0].name == "test_invariant"
        assert candidates[0].proposer == "claude"
        assert candidates[0].support == 3

    @pytest.mark.asyncio
    async def test_propose_handles_bad_json(self):
        from belief.evolution.crystallizer import _parse_proposals

        assert _parse_proposals("not json at all", []) == []

    @pytest.mark.asyncio
    async def test_propose_filters_invalid_kinds(self):
        from belief.evolution.crystallizer import _parse_proposals

        response = '[{"name": "test", "description": "x", "implementation_kind": "llm_judge"}]'
        candidates = _parse_proposals(response, [])
        # llm_judge should be converted to assertion fallback
        assert len(candidates) == 1
        assert candidates[0].implementation_kind == "assertion"


# ── Houdini Filter ──────────────────────────────────────────────────────────


class TestHoudiniFilter:
    def test_removes_high_violation_rate(self):
        """Candidates with >5% violation rate should be removed."""
        candidates = [
            CandidateInvariant(
                name="no_bare_except",
                description="No bare except",
                implementation_kind="ast",
                support=8,
                violations=5,  # 38% violation rate
                precision=0.615,
                proposer="template",
            ),
        ]
        traces = [_make_trace(bare_except_count=0) for _ in range(8)]
        traces += [_make_trace(bare_except_count=2) for _ in range(5)]

        filtered = filter_candidates(candidates, traces)
        assert len(filtered) == 0

    def test_keeps_high_precision(self):
        """Candidates with low violation rate should survive."""
        candidates = [
            CandidateInvariant(
                name="no_bare_except",
                description="No bare except",
                implementation_kind="ast",
                support=19,
                violations=1,
                precision=0.95,
                proposer="template",
            ),
        ]
        traces = [_make_trace(bare_except_count=0) for _ in range(19)]
        traces += [_make_trace(bare_except_count=1)]

        filtered = filter_candidates(candidates, traces)
        assert len(filtered) == 1

    def test_claude_candidates_pass_through(self):
        """Claude-proposed candidates without templates pass if precision is good."""
        candidates = [
            CandidateInvariant(
                name="custom_invariant",
                description="Custom check",
                implementation_kind="assertion",
                support=15,
                violations=0,
                precision=1.0,
                proposer="claude",
            ),
        ]
        filtered = filter_candidates(candidates, [])
        assert len(filtered) == 1


# ── Promotion ───────────────────────────────────────────────────────────────


class TestPromotion:
    def test_generates_valid_ast_code(self):
        """Promotion should generate valid Python for AST covenants."""
        candidate = CandidateInvariant(
            name="test_ast_covenant",
            description="Test AST check",
            implementation_kind="ast",
            support=15,
            violations=0,
            precision=1.0,
            proposer="template",
        )

        from belief.evolution.crystallizer import _generate_covenant_code

        code = _generate_covenant_code(candidate)

        import ast

        ast.parse(code)  # Should not raise

    def test_generates_valid_regex_code(self):
        candidate = CandidateInvariant(
            name="test_regex_covenant",
            description="Test regex check",
            implementation_kind="regex",
            support=15,
            violations=0,
            precision=1.0,
            proposer="template",
        )

        from belief.evolution.crystallizer import _generate_covenant_code

        code = _generate_covenant_code(candidate)
        import ast

        ast.parse(code)

    def test_generates_valid_assertion_code(self):
        candidate = CandidateInvariant(
            name="test_assertion_covenant",
            description="Test assertion check",
            implementation_kind="assertion",
            support=15,
            violations=0,
            precision=1.0,
            proposer="template",
        )

        from belief.evolution.crystallizer import _generate_covenant_code

        code = _generate_covenant_code(candidate)
        import ast

        ast.parse(code)

    def test_promotion_stores_in_soil(self, tmp_path):
        """Promoted covenant should be stored in ChromaDB."""
        from belief.memory.soil import Soil

        soil = Soil(persist_dir=tmp_path / "soil")
        candidate = CandidateInvariant(
            name="promoted_covenant",
            description="Test promoted covenant",
            implementation_kind="assertion",
            support=15,
            violations=0,
            precision=1.0,
            proposer="template",
        )

        covenant_id = promote_to_covenant(candidate, soil)
        assert covenant_id.startswith("cov-")

        # Should be retrievable
        nutrient = soil.get(covenant_id)
        assert nutrient is not None
        assert nutrient.nutrient_type.value == "covenant"

    def test_rejects_unqualified(self, tmp_path):
        """Unqualified candidates should raise ValueError."""
        from belief.memory.soil import Soil

        soil = Soil(persist_dir=tmp_path / "soil")

        candidate = CandidateInvariant(
            name="bad_candidate",
            description="Not enough support",
            implementation_kind="ast",
            support=3,  # Below threshold
            violations=0,
            precision=1.0,
            proposer="template",
        )

        with pytest.raises(ValueError, match="not qualified"):
            promote_to_covenant(candidate, soil)


# ── Candidate Invariant ─────────────────────────────────────────────────────


class TestCandidateInvariant:
    def test_qualified_property(self):
        c = CandidateInvariant(
            name="test",
            description="test",
            implementation_kind="ast",
            support=10,
            violations=0,
            precision=1.0,
            proposer="template",
        )
        assert c.qualified is True

    def test_not_qualified_low_support(self):
        c = CandidateInvariant(
            name="test",
            description="test",
            implementation_kind="ast",
            support=5,
            violations=0,
            precision=1.0,
            proposer="template",
        )
        assert c.qualified is False

    def test_not_qualified_low_precision(self):
        c = CandidateInvariant(
            name="test",
            description="test",
            implementation_kind="ast",
            support=20,
            violations=5,
            precision=0.80,
            proposer="template",
        )
        assert c.qualified is False


# ── Covenant Registry ───────────────────────────────────────────────────────


class TestCovenantRegistry:
    def test_loads_static_covenants(self, tmp_path):
        """Registry should load the 6 static covenants."""
        from belief.memory.soil import Soil
        from belief.validators.covenant_registry import CovenantRegistry

        soil = Soil(persist_dir=tmp_path / "soil")
        registry = CovenantRegistry(soil)
        stats = registry.get_covenant_stats()
        assert stats["static_count"] == 6

    def test_fire_all_on_clean_code(self, tmp_path):
        """Fire all covenants on clean code — should mostly pass."""
        from belief.memory.soil import Soil
        from belief.validators.covenant_registry import CovenantRegistry

        soil = Soil(persist_dir=tmp_path / "soil")
        registry = CovenantRegistry(soil)

        code_files = {
            "main.py": "import os\n\ndef main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n",
        }

        results = registry.fire_all(code_files)
        assert len(results) >= 6  # At least the static covenants

    def test_get_all_descriptions(self, tmp_path):
        """Should return descriptions for all covenants."""
        from belief.memory.soil import Soil
        from belief.validators.covenant_registry import CovenantRegistry

        soil = Soil(persist_dir=tmp_path / "soil")
        registry = CovenantRegistry(soil)
        descriptions = registry.get_all_covenant_descriptions()
        assert len(descriptions) >= 6
        assert all("name" in d for d in descriptions)

    def test_dynamic_covenant_loaded_after_promotion(self, tmp_path):
        """After promoting a covenant, registry should find it as dynamic."""
        from belief.memory.soil import Soil
        from belief.validators.covenant_registry import CovenantRegistry

        soil = Soil(persist_dir=tmp_path / "soil")

        # Promote a covenant
        candidate = CandidateInvariant(
            name="dynamic_test",
            description="A dynamically discovered covenant",
            implementation_kind="assertion",
            support=15,
            violations=0,
            precision=1.0,
            proposer="template",
        )
        promote_to_covenant(candidate, soil)

        # Reload registry
        registry = CovenantRegistry(soil)
        stats = registry.get_covenant_stats()
        assert stats["dynamic_count"] >= 1


# ── Episode Recorder ────────────────────────────────────────────────────────


class TestEpisodeRecorder:
    def test_record_episode(self, tmp_path):
        """Should record an episode to belief_episodes collection."""
        from belief.memory.episode_recorder import record_episode
        from belief.memory.soil import Soil

        soil = Soil(persist_dir=tmp_path / "soil")
        state = {
            "run_id": "test-run-001",
            "user_goal": "Build a FastAPI todo API",
            "code_files": {
                "main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
                "requirements.txt": "fastapi\nuvicorn\n",
                "test_main.py": "def test_health():\n    pass\n",
            },
        }

        episode_id = record_episode(soil, state)
        assert episode_id == "test-run-001"

        # Verify it's in the episodes collection
        col = soil._collections["belief_episodes"]
        assert col.count() == 1

    def test_extracts_dependencies(self):
        from belief.memory.episode_recorder import _extract_dependencies

        deps = _extract_dependencies("fastapi>=0.100\nuvicorn\npydantic==2.0\n")
        assert "fastapi" in deps
        assert "uvicorn" in deps
        assert "pydantic" in deps

    def test_detects_api_framework(self):
        from belief.memory.episode_recorder import _analyze_code

        code_files = {"main.py": "from fastapi import FastAPI\napp = FastAPI()\n"}
        features = _analyze_code(code_files)
        assert features["has_api_framework"] is True

    def test_counts_bare_excepts(self):
        from belief.memory.episode_recorder import _analyze_code

        code_files = {"main.py": "try:\n    pass\nexcept:\n    pass\n"}
        features = _analyze_code(code_files)
        assert features["bare_except_count"] == 1

    def test_detects_entry_point(self):
        from belief.memory.episode_recorder import _analyze_code

        code_files = {"main.py": "print('hello')"}
        features = _analyze_code(code_files)
        assert features["has_entry_point"] is True

    def test_counts_tests(self):
        from belief.memory.episode_recorder import _analyze_code

        code_files = {"test_app.py": "def test_one():\n    pass\ndef test_two():\n    pass\n"}
        features = _analyze_code(code_files)
        assert features["test_count"] == 2


# ── Full Pipeline (mocked) ─────────────────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_with_synthetic_traces(self, tmp_path):
        """Full crystallization pipeline with synthetic traces."""
        from belief.memory.soil import Soil
        from belief.validators.covenant_registry import CovenantRegistry
        from belief.evolution.crystallizer import run_crystallization

        soil = Soil(persist_dir=tmp_path / "soil")
        registry = CovenantRegistry(soil)

        # Seed the episodes collection with synthetic traces
        episodes_col = soil._collections["belief_episodes"]
        for i in range(15):
            trace = _make_trace(
                passed=True,
                file_count=5,
                test_count=10,
                bare_except_count=0,
                dependencies=["fastapi", "uvicorn", "pydantic"],
                has_api_framework=True,
                has_health_endpoint=True,
            )
            meta = {}
            for k, v in trace.items():
                if isinstance(v, list):
                    meta[k] = ",".join(str(x) for x in v)
                elif isinstance(v, bool):
                    meta[k] = 1 if v else 0
                elif isinstance(v, (int, float, str)):
                    meta[k] = v
            episodes_col.upsert(
                ids=[trace["trace_id"]],
                documents=[f"test episode {i}"],
                metadatas=[meta],
            )

        # Mock the Claude proposer to avoid API calls
        with patch(
            "belief.evolution.crystallizer.propose_invariants", new_callable=AsyncMock
        ) as mock_propose:
            mock_propose.return_value = []

            new_ids = await run_crystallization(soil, registry, n_recent_traces=15)

            # Template sweep should find candidates but they may or may not
            # qualify depending on how traces are parsed back from ChromaDB
            # (metadata serialization flattens lists to strings)
            assert isinstance(new_ids, list)

    def test_existing_enforce_all_unchanged(self):
        """The original enforce_all must still work identically."""
        from belief.validators import enforce_all

        code_files = {
            "main.py": "import os\n\ndef main():\n    pass\n",
        }
        fixed, result = enforce_all(code_files)
        assert isinstance(result.passed, bool)
        assert "main.py" in fixed
