"""
Tests for Milestone 1: Skeleton Architecture

Covers:
1. SkeletonArtifact model validation
2. SymbolRegistry parsing and context generation
3. SkeletonBuilder deterministic generation
4. Pipeline integration (Pass 1 only, no LLM)
"""

import ast
import pytest

pytest.skip(
    "Legacy milestone-1 scaffold: references belief.agents.skeleton_pipeline "
    "and sibling modules removed in the v2 architecture rewrite. Kept for "
    "historical reference until rewritten against the current skeleton_pass1 "
    "+ builder pipeline.",
    allow_module_level=True,
)

from belief.models.skeleton import (  # noqa: E402
    SkeletonArtifact,
    FileTreeEntry,
    FileRole,
    DependencyEdge,
    DependencyKind,
    ModelChain,
    ModelSpec,
    ModelFieldSpec,
    ABCDefinition,
    MethodSignature,
    ConfigSchema,
    ExceptionSpec,
)
from belief.models.symbol_registry import (  # noqa: E402
    SymbolRegistry,
    parse_file,
)
from belief.agents.skeleton_builder import (  # noqa: E402
    generate_skeleton_file,
    generate_all_skeletons,
)
from belief.agents.skeleton_pipeline import (  # noqa: E402
    parse_skeleton_from_llm,
    run_skeleton_pipeline,
    skeleton_artifact_to_state,
)


# ---------------------------------------------------------------------------
# Fixtures — a minimal lead-gen-style skeleton for testing
# ---------------------------------------------------------------------------


def _make_lead_gen_skeleton() -> SkeletonArtifact:
    """Build a small SkeletonArtifact resembling a lead gen pipeline."""
    return SkeletonArtifact(
        project_name="lead_gen_pipeline",
        description="4-stage lead generation pipeline with progressive Pydantic models",
        file_tree=[
            FileTreeEntry(
                path="models/lead.py",
                role=FileRole.MODEL,
                description="Progressive lead models: RawLead → EnrichedLead → ScoredLead",
                skeleton=True,
            ),
            FileTreeEntry(
                path="models/config.py",
                role=FileRole.CONFIG,
                description="Pipeline configuration and settings",
                skeleton=True,
            ),
            FileTreeEntry(
                path="pipeline/base.py",
                role=FileRole.ABC,
                description="Abstract base class for pipeline stages",
                skeleton=True,
            ),
            FileTreeEntry(
                path="pipeline/exceptions.py",
                role=FileRole.EXCEPTION,
                description="Custom exception hierarchy",
                skeleton=True,
            ),
            FileTreeEntry(
                path="pipeline/discovery.py",
                role=FileRole.IMPLEMENTATION,
                description="Stage 1: Google Places discovery",
                skeleton=False,
            ),
            FileTreeEntry(
                path="pipeline/enrichment.py",
                role=FileRole.IMPLEMENTATION,
                description="Stage 2: Waterfall enrichment",
                skeleton=False,
            ),
            FileTreeEntry(
                path="main.py",
                role=FileRole.ENTRY_POINT,
                description="CLI entry point",
                skeleton=False,
            ),
        ],
        dependency_edges=[
            DependencyEdge(
                source="pipeline/base.py",
                target="models/lead.py",
                kind=DependencyKind.IMPORTS,
                symbols=["RawLead"],
            ),
            DependencyEdge(
                source="pipeline/discovery.py",
                target="pipeline/base.py",
                kind=DependencyKind.INHERITS,
                symbols=["BasePipelineStage"],
            ),
            DependencyEdge(
                source="pipeline/discovery.py",
                target="models/lead.py",
                kind=DependencyKind.IMPORTS,
                symbols=["RawLead"],
            ),
            DependencyEdge(
                source="pipeline/enrichment.py",
                target="pipeline/base.py",
                kind=DependencyKind.INHERITS,
                symbols=["BasePipelineStage"],
            ),
            DependencyEdge(
                source="pipeline/enrichment.py",
                target="models/lead.py",
                kind=DependencyKind.IMPORTS,
                symbols=["RawLead", "EnrichedLead"],
            ),
        ],
        model_chains=[
            ModelChain(
                name="LeadProgression",
                models=[
                    ModelSpec(
                        name="RawLead",
                        file_path="models/lead.py",
                        base_class="BaseModel",
                        fields=[
                            ModelFieldSpec(
                                name="name", type_annotation="str", description="Business name"
                            ),
                            ModelFieldSpec(
                                name="address", type_annotation="Optional[str]", default="None"
                            ),
                            ModelFieldSpec(
                                name="phone", type_annotation="Optional[str]", default="None"
                            ),
                            ModelFieldSpec(
                                name="place_id",
                                type_annotation="str",
                                description="Google Places ID",
                            ),
                        ],
                        docstring="Raw lead from Google Places API",
                    ),
                    ModelSpec(
                        name="EnrichedLead",
                        file_path="models/lead.py",
                        base_class="RawLead",
                        fields=[
                            ModelFieldSpec(
                                name="website", type_annotation="Optional[str]", default="None"
                            ),
                            ModelFieldSpec(
                                name="email", type_annotation="Optional[str]", default="None"
                            ),
                            ModelFieldSpec(
                                name="tech_stack", type_annotation="list[str]", default="[]"
                            ),
                        ],
                        docstring="Lead enriched with website, email, and tech detection",
                    ),
                    ModelSpec(
                        name="ScoredLead",
                        file_path="models/lead.py",
                        base_class="EnrichedLead",
                        fields=[
                            ModelFieldSpec(name="score", type_annotation="float", default="0.0"),
                            ModelFieldSpec(
                                name="score_reasons", type_annotation="list[str]", default="[]"
                            ),
                        ],
                        docstring="Lead with ICP scoring applied",
                        validators=["validate_score"],
                    ),
                ],
            ),
        ],
        abc_definitions=[
            ABCDefinition(
                name="BasePipelineStage",
                file_path="pipeline/base.py",
                base_classes=["ABC"],
                methods=[
                    MethodSignature(
                        name="process",
                        params="self, lead: RawLead",
                        return_type="RawLead",
                        is_async=True,
                        is_abstract=True,
                        docstring="Process a single lead through this stage",
                    ),
                    MethodSignature(
                        name="batch_process",
                        params="self, leads: list[RawLead], *, concurrency: int = 5",
                        return_type="list[RawLead]",
                        is_async=True,
                        is_abstract=True,
                        docstring="Process a batch of leads with concurrency control",
                    ),
                ],
                class_attributes=[
                    ModelFieldSpec(name="stage_name", type_annotation="str"),
                ],
                docstring="Base class for all pipeline stages",
            ),
        ],
        config_schemas=[
            ConfigSchema(
                name="PipelineSettings",
                file_path="models/config.py",
                fields=[
                    ModelFieldSpec(
                        name="google_api_key",
                        type_annotation="str",
                        description="Google Places API key",
                    ),
                    ModelFieldSpec(name="max_concurrency", type_annotation="int", default="5"),
                    ModelFieldSpec(name="cost_budget", type_annotation="float", default="2.0"),
                ],
                env_prefix="PIPELINE_",
                docstring="Pipeline configuration loaded from environment",
            ),
        ],
        exception_specs=[
            ExceptionSpec(
                name="PipelineError",
                file_path="pipeline/exceptions.py",
                base_class="Exception",
                docstring="Base exception for pipeline errors",
            ),
            ExceptionSpec(
                name="StageError",
                file_path="pipeline/exceptions.py",
                base_class="PipelineError",
                message_template="Stage '{kwargs.get('stage', 'unknown')}' failed: {kwargs.get('reason', '')}",
                docstring="Error in a specific pipeline stage",
            ),
            ExceptionSpec(
                name="BudgetExceededError",
                file_path="pipeline/exceptions.py",
                base_class="PipelineError",
                docstring="Cost budget exceeded",
            ),
        ],
        external_dependencies=["pydantic", "pydantic-settings", "httpx", "fastapi"],
        entry_point="main.py",
    )


# ===========================================================================
# Test SkeletonArtifact Model
# ===========================================================================


class TestSkeletonArtifact:
    def test_basic_creation(self):
        s = _make_lead_gen_skeleton()
        assert s.project_name == "lead_gen_pipeline"
        assert len(s.file_tree) == 7

    def test_skeleton_files_filter(self):
        s = _make_lead_gen_skeleton()
        skeletons = s.skeleton_files()
        assert len(skeletons) == 4
        assert all(f.skeleton for f in skeletons)

    def test_implementation_files_filter(self):
        s = _make_lead_gen_skeleton()
        impls = s.implementation_files()
        assert len(impls) == 3
        assert all(not f.skeleton for f in impls)

    def test_dependencies_for(self):
        s = _make_lead_gen_skeleton()
        deps = s.dependencies_for("pipeline/discovery.py")
        assert len(deps) == 2
        targets = {d.target for d in deps}
        assert "pipeline/base.py" in targets
        assert "models/lead.py" in targets

    def test_dependents_of(self):
        s = _make_lead_gen_skeleton()
        dependents = s.dependents_of("models/lead.py")
        sources = {d.source for d in dependents}
        assert "pipeline/base.py" in sources
        assert "pipeline/discovery.py" in sources

    def test_serialization_roundtrip(self):
        s = _make_lead_gen_skeleton()
        dumped = s.model_dump()
        restored = SkeletonArtifact.model_validate(dumped)
        assert restored.project_name == s.project_name
        assert len(restored.file_tree) == len(s.file_tree)
        assert len(restored.model_chains) == len(s.model_chains)

    def test_json_roundtrip(self):
        s = _make_lead_gen_skeleton()
        json_str = s.model_dump_json()
        restored = SkeletonArtifact.model_validate_json(json_str)
        assert restored.project_name == s.project_name


# ===========================================================================
# Test Symbol Registry
# ===========================================================================

SAMPLE_SOURCE = '''\
"""Sample module for testing."""

from typing import Optional
from pydantic import BaseModel, Field

MAX_RETRIES: int = 3

class RawLead(BaseModel):
    """Raw lead from Google Places."""
    name: str
    address: Optional[str] = None
    place_id: str = Field(description="Google Places ID")

    def full_address(self) -> str:
        """Get formatted address."""
        return f"{self.name}, {self.address or 'unknown'}"

class EnrichedLead(RawLead):
    """Lead with enrichment data."""
    website: Optional[str] = None
    tech_stack: list[str] = []

async def fetch_lead(place_id: str, *, timeout: float = 30.0) -> RawLead:
    """Fetch a lead from Google Places API."""
    pass
'''


class TestSymbolRegistry:
    def test_parse_file_classes(self):
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        assert len(symbols.classes) == 2
        assert symbols.classes[0].name == "RawLead"
        assert symbols.classes[1].name == "EnrichedLead"

    def test_parse_file_functions(self):
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        assert len(symbols.functions) == 1
        assert symbols.functions[0].name == "fetch_lead"
        assert symbols.functions[0].is_async

    def test_parse_file_constants(self):
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        assert any(c.name == "MAX_RETRIES" for c in symbols.constants)

    def test_module_path_computation(self):
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        assert symbols.module_path == "models.lead"

    def test_compressed_context(self):
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        ctx = symbols.as_compressed_context()
        assert "class RawLead" in ctx
        assert "class EnrichedLead" in ctx
        assert "async def fetch_lead" in ctx
        assert "MAX_RETRIES" in ctx

    def test_class_methods_extracted(self):
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        raw_lead = symbols.classes[0]
        method_names = [m.name for m in raw_lead.methods]
        assert "full_address" in method_names

    def test_class_attributes_extracted(self):
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        raw_lead = symbols.classes[0]
        attr_names = [a[0] for a in raw_lead.class_attributes]
        assert "name" in attr_names
        assert "place_id" in attr_names

    def test_registry_register_and_get(self):
        registry = SymbolRegistry()
        symbols = parse_file(SAMPLE_SOURCE, "models/lead.py")
        registry.register(symbols)
        assert registry.get("models/lead.py") is not None
        assert registry.get("nonexistent.py") is None

    def test_registry_resolve_symbol(self):
        registry = SymbolRegistry()
        registry.register_source(SAMPLE_SOURCE, "models/lead.py")
        result = registry.resolve_symbol("RawLead")
        assert result == ("models.lead", "RawLead")
        assert registry.resolve_symbol("NonExistent") is None

    def test_registry_context_for_file(self):
        registry = SymbolRegistry()
        registry.register_source(SAMPLE_SOURCE, "models/lead.py")
        ctx = registry.context_for_file("pipeline/discovery.py", ["models/lead.py"])
        assert "RawLead" in ctx
        assert "models.lead" in ctx

    def test_registry_context_for_file_no_deps(self):
        registry = SymbolRegistry()
        ctx = registry.context_for_file("models/lead.py", [])
        assert "No dependencies" in ctx


# ===========================================================================
# Test Skeleton Builder (Pass 1)
# ===========================================================================


class TestSkeletonBuilder:
    def setup_method(self):
        self.skeleton = _make_lead_gen_skeleton()
        self.registry = SymbolRegistry()

    def test_generate_model_chain(self):
        code = generate_skeleton_file(self.skeleton, "models/lead.py", self.registry)
        assert code is not None
        assert "class RawLead" in code
        assert "class EnrichedLead" in code
        assert "class ScoredLead" in code
        # Verify valid Python
        ast.parse(code)

    def test_generate_abc(self):
        code = generate_skeleton_file(self.skeleton, "pipeline/base.py", self.registry)
        assert code is not None
        assert "class BasePipelineStage" in code
        assert "abstractmethod" in code
        assert "async def process" in code
        ast.parse(code)

    def test_generate_config(self):
        code = generate_skeleton_file(self.skeleton, "models/config.py", self.registry)
        assert code is not None
        assert "class PipelineSettings" in code
        assert "BaseSettings" in code
        assert "google_api_key" in code
        ast.parse(code)

    def test_generate_exceptions(self):
        code = generate_skeleton_file(self.skeleton, "pipeline/exceptions.py", self.registry)
        assert code is not None
        assert "class PipelineError" in code
        assert "class StageError" in code
        assert "class BudgetExceededError" in code
        ast.parse(code)

    def test_implementation_file_returns_none(self):
        """Pass 1 should return None for implementation files."""
        code = generate_skeleton_file(self.skeleton, "pipeline/discovery.py", self.registry)
        assert code is None

    def test_generate_all_skeletons(self):
        results = generate_all_skeletons(self.skeleton, self.registry)
        assert len(results) == 4  # 4 skeleton files
        # All should be valid Python
        for path, code in results.items():
            ast.parse(code)

    def test_registry_populated_after_generation(self):
        generate_all_skeletons(self.skeleton, self.registry)
        # Registry should have entries for generated files
        assert self.registry.get("models/lead.py") is not None
        assert self.registry.get("pipeline/base.py") is not None
        # Should be able to resolve model names
        assert self.registry.resolve_symbol("RawLead") is not None
        assert self.registry.resolve_symbol("BasePipelineStage") is not None

    def test_model_chain_inheritance(self):
        code = generate_skeleton_file(self.skeleton, "models/lead.py", self.registry)
        assert "class EnrichedLead(RawLead):" in code
        assert "class ScoredLead(EnrichedLead):" in code

    def test_validator_stubs_generated(self):
        code = generate_skeleton_file(self.skeleton, "models/lead.py", self.registry)
        assert "def validate_score" in code


# ===========================================================================
# Test Pipeline Integration
# ===========================================================================


class TestSkeletonPipeline:
    def test_pass1_only(self):
        """Pipeline runs Pass 1 without LLM function."""
        skeleton = _make_lead_gen_skeleton()
        result = run_skeleton_pipeline(skeleton, llm_generate_fn=None)
        assert len(result["skeleton_files"]) == 4
        assert len(result["implementation_files"]) == 0
        assert len(result["errors"]) == 0

    def test_parse_skeleton_from_llm(self):
        """Parse SkeletonArtifact from LLM JSON output."""
        skeleton = _make_lead_gen_skeleton()
        json_str = skeleton.model_dump_json()
        parsed = parse_skeleton_from_llm(json_str)
        assert parsed.project_name == "lead_gen_pipeline"

    def test_parse_skeleton_with_markdown_fences(self):
        """Handle LLM output wrapped in code fences."""
        skeleton = _make_lead_gen_skeleton()
        json_str = f"```json\n{skeleton.model_dump_json()}\n```"
        parsed = parse_skeleton_from_llm(json_str)
        assert parsed.project_name == "lead_gen_pipeline"

    def test_skeleton_to_state(self):
        """Convert SkeletonArtifact to UnifiedState updates."""
        skeleton = _make_lead_gen_skeleton()
        state = skeleton_artifact_to_state(skeleton)
        assert "skeleton_artifact" in state
        assert "file_manifest" in state
        assert len(state["file_manifest"]) == 7
        assert "dependency_edges" in state
        assert "external_dependencies" in state

    def test_full_context_available_after_pass1(self):
        """After Pass 1, symbol context is available for all skeleton files."""
        skeleton = _make_lead_gen_skeleton()
        result = run_skeleton_pipeline(skeleton, llm_generate_fn=None)
        registry = result["registry"]

        # Build context for an implementation file
        ctx = registry.context_for_file(
            "pipeline/discovery.py", ["pipeline/base.py", "models/lead.py"]
        )
        assert "BasePipelineStage" in ctx
        assert "RawLead" in ctx
        assert "EnrichedLead" in ctx


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
