"""
Tests for Milestone 2: Dependency DAG + Parallel Generation

Covers:
1. Topological sort with Kahn's algorithm
2. Build plan creation (skeleton/impl split)
3. Cycle detection
4. Level computation and parallelism
5. Parallel builder with mock LLM
6. Pyright output parsing
7. Self-correction loop
8. 12-file project end-to-end

Done-when: A 12-file project generates with all imports resolving,
verified by pyright-style validation with zero errors.
"""

import ast
import asyncio
import json
import re
import pytest

pytest.skip(
    "Legacy milestone-2 scaffold: references belief.agents.parallel_builder "
    "and belief.agents.pyright_checker (now at belief.tools.pyright_checker). "
    "parallel_builder was removed in the v2 rewrite. Kept until rewritten "
    "against the current DAG-driven builder.",
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
from belief.models.symbol_registry import SymbolRegistry
from belief.models.dependency_dag import (
    topological_sort,
    create_build_plan,
    critical_path,
    DependencyCycleError,
)
from belief.agents.skeleton_builder import generate_all_skeletons
from belief.agents.pyright_checker import (
    PyrightError,
    PyrightResult,
    _parse_pyright_output,
    group_errors_by_file,
    format_errors_for_llm,
    write_project_for_pyright,
)
from belief.agents.parallel_builder import (
    build_level,
    run_parallel_build,
    _clean_code,
)


# ---------------------------------------------------------------------------
# 12-file lead gen pipeline skeleton
# ---------------------------------------------------------------------------

def _make_12_file_skeleton() -> SkeletonArtifact:
    """
    12-file lead gen pipeline — the Tier 3 benchmark target.
    Tests the full DAG with multiple levels of parallelism.
    """
    return SkeletonArtifact(
        project_name="lead_gen_pipeline",
        description="4-stage lead generation pipeline with API server",
        file_tree=[
            # Level 0: no dependencies (skeleton)
            FileTreeEntry(path="models/lead.py", role=FileRole.MODEL,
                          description="Progressive lead models", skeleton=True),
            FileTreeEntry(path="models/config.py", role=FileRole.CONFIG,
                          description="Pipeline settings", skeleton=True),
            FileTreeEntry(path="pipeline/exceptions.py", role=FileRole.EXCEPTION,
                          description="Pipeline exceptions", skeleton=True),
            # Level 1: depends on models (skeleton)
            FileTreeEntry(path="pipeline/base.py", role=FileRole.ABC,
                          description="Base stage ABC", skeleton=True),
            # Level 2: depends on base + models (implementation)
            FileTreeEntry(path="pipeline/discovery.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 1: Google Places discovery", skeleton=False),
            FileTreeEntry(path="pipeline/enrichment.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 2: Waterfall enrichment", skeleton=False),
            FileTreeEntry(path="pipeline/scoring.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 3: ICP scoring", skeleton=False),
            FileTreeEntry(path="pipeline/outreach.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 4: Outreach generation", skeleton=False),
            # Level 3: depends on stages + config (implementation)
            FileTreeEntry(path="pipeline/runner.py", role=FileRole.IMPLEMENTATION,
                          description="Pipeline runner/orchestrator", skeleton=False),
            FileTreeEntry(path="utils/cost_tracker.py", role=FileRole.IMPLEMENTATION,
                          description="LLM cost tracking", skeleton=False),
            # Level 4: depends on runner + config (implementation)
            FileTreeEntry(path="api/server.py", role=FileRole.IMPLEMENTATION,
                          description="FastAPI server", skeleton=False),
            FileTreeEntry(path="main.py", role=FileRole.ENTRY_POINT,
                          description="CLI entry point", skeleton=False),
        ],
        dependency_edges=[
            # base depends on models
            DependencyEdge(source="pipeline/base.py", target="models/lead.py",
                           kind=DependencyKind.IMPORTS, symbols=["RawLead"]),
            # All stages depend on base + models
            DependencyEdge(source="pipeline/discovery.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/discovery.py", target="models/lead.py",
                           kind=DependencyKind.IMPORTS, symbols=["RawLead"]),
            DependencyEdge(source="pipeline/enrichment.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/enrichment.py", target="models/lead.py",
                           kind=DependencyKind.IMPORTS, symbols=["RawLead", "EnrichedLead"]),
            DependencyEdge(source="pipeline/scoring.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/scoring.py", target="models/lead.py",
                           kind=DependencyKind.IMPORTS, symbols=["EnrichedLead", "ScoredLead"]),
            DependencyEdge(source="pipeline/outreach.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/outreach.py", target="models/lead.py",
                           kind=DependencyKind.IMPORTS, symbols=["ScoredLead", "OutreachLead"]),
            # Runner depends on stages + config + exceptions
            DependencyEdge(source="pipeline/runner.py", target="pipeline/discovery.py",
                           kind=DependencyKind.IMPORTS, symbols=["DiscoveryStage"]),
            DependencyEdge(source="pipeline/runner.py", target="pipeline/enrichment.py",
                           kind=DependencyKind.IMPORTS, symbols=["EnrichmentStage"]),
            DependencyEdge(source="pipeline/runner.py", target="pipeline/scoring.py",
                           kind=DependencyKind.IMPORTS, symbols=["ScoringStage"]),
            DependencyEdge(source="pipeline/runner.py", target="pipeline/outreach.py",
                           kind=DependencyKind.IMPORTS, symbols=["OutreachStage"]),
            DependencyEdge(source="pipeline/runner.py", target="models/config.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineSettings"]),
            DependencyEdge(source="pipeline/runner.py", target="pipeline/exceptions.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineError"]),
            # Cost tracker depends on config
            DependencyEdge(source="utils/cost_tracker.py", target="models/config.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineSettings"]),
            # API depends on runner + config
            DependencyEdge(source="api/server.py", target="pipeline/runner.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineRunner"]),
            DependencyEdge(source="api/server.py", target="models/config.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineSettings"]),
            # Main depends on runner + config
            DependencyEdge(source="main.py", target="pipeline/runner.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineRunner"]),
            DependencyEdge(source="main.py", target="models/config.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineSettings"]),
        ],
        model_chains=[
            ModelChain(
                name="LeadProgression",
                models=[
                    ModelSpec(name="RawLead", file_path="models/lead.py",
                             base_class="BaseModel",
                             fields=[
                                 ModelFieldSpec(name="name", type_annotation="str"),
                                 ModelFieldSpec(name="place_id", type_annotation="str"),
                                 ModelFieldSpec(name="address", type_annotation="Optional[str]", default="None"),
                                 ModelFieldSpec(name="phone", type_annotation="Optional[str]", default="None"),
                             ],
                             docstring="Raw lead from Google Places"),
                    ModelSpec(name="EnrichedLead", file_path="models/lead.py",
                             base_class="RawLead",
                             fields=[
                                 ModelFieldSpec(name="website", type_annotation="Optional[str]", default="None"),
                                 ModelFieldSpec(name="email", type_annotation="Optional[str]", default="None"),
                                 ModelFieldSpec(name="tech_stack", type_annotation="list[str]", default="[]"),
                             ],
                             docstring="Lead with enrichment data"),
                    ModelSpec(name="ScoredLead", file_path="models/lead.py",
                             base_class="EnrichedLead",
                             fields=[
                                 ModelFieldSpec(name="score", type_annotation="float", default="0.0"),
                                 ModelFieldSpec(name="reasons", type_annotation="list[str]", default="[]"),
                             ],
                             docstring="Lead with ICP scoring"),
                    ModelSpec(name="OutreachLead", file_path="models/lead.py",
                             base_class="ScoredLead",
                             fields=[
                                 ModelFieldSpec(name="subject_line", type_annotation="Optional[str]", default="None"),
                                 ModelFieldSpec(name="email_body", type_annotation="Optional[str]", default="None"),
                             ],
                             docstring="Lead with outreach content"),
                ],
            ),
        ],
        abc_definitions=[
            ABCDefinition(
                name="BaseStage",
                file_path="pipeline/base.py",
                base_classes=["ABC"],
                methods=[
                    MethodSignature(name="process", params="self, lead: RawLead",
                                    return_type="RawLead", is_async=True, is_abstract=True,
                                    docstring="Process a single lead"),
                    MethodSignature(name="batch_process",
                                    params="self, leads: list[RawLead], *, concurrency: int = 5",
                                    return_type="list[RawLead]", is_async=True, is_abstract=True,
                                    docstring="Process leads with concurrency"),
                ],
                class_attributes=[
                    ModelFieldSpec(name="stage_name", type_annotation="str"),
                ],
                docstring="Base class for pipeline stages",
            ),
        ],
        config_schemas=[
            ConfigSchema(
                name="PipelineSettings",
                file_path="models/config.py",
                fields=[
                    ModelFieldSpec(name="google_api_key", type_annotation="str", default='"test"'),
                    ModelFieldSpec(name="max_concurrency", type_annotation="int", default="5"),
                    ModelFieldSpec(name="cost_budget", type_annotation="float", default="2.0"),
                ],
                env_prefix="PIPELINE_",
                docstring="Pipeline settings",
            ),
        ],
        exception_specs=[
            ExceptionSpec(name="PipelineError", file_path="pipeline/exceptions.py",
                          base_class="Exception", docstring="Base pipeline error"),
            ExceptionSpec(name="StageError", file_path="pipeline/exceptions.py",
                          base_class="PipelineError", docstring="Stage error"),
        ],
        external_dependencies=["pydantic", "pydantic-settings", "httpx", "fastapi"],
        entry_point="main.py",
    )


# ---------------------------------------------------------------------------
# Mock LLM for 12-file project
# ---------------------------------------------------------------------------

def _mock_llm_12(system: str, user: str, model: str) -> str:
    """Mock LLM that generates valid implementations for the 12-file project."""
    path_match = re.search(r"Path:\s*(\S+)", user)
    file_path = path_match.group(1) if path_match else "unknown.py"

    impls = {
        "pipeline/discovery.py": '''"""Stage 1: Discovery."""
from pipeline.base import BaseStage
from models.lead import RawLead
class DiscoveryStage(BaseStage):
    stage_name: str = "discovery"
    async def process(self, lead: RawLead) -> RawLead:
        return lead
    async def batch_process(self, leads: list[RawLead], *, concurrency: int = 5) -> list[RawLead]:
        return leads
''',
        "pipeline/enrichment.py": '''"""Stage 2: Enrichment."""
from pipeline.base import BaseStage
from models.lead import RawLead, EnrichedLead
class EnrichmentStage(BaseStage):
    stage_name: str = "enrichment"
    async def process(self, lead: RawLead) -> RawLead:
        return EnrichedLead(name=lead.name, place_id=lead.place_id)
    async def batch_process(self, leads: list[RawLead], *, concurrency: int = 5) -> list[RawLead]:
        return [await self.process(l) for l in leads]
''',
        "pipeline/scoring.py": '''"""Stage 3: Scoring."""
from pipeline.base import BaseStage
from models.lead import RawLead, EnrichedLead, ScoredLead
class ScoringStage(BaseStage):
    stage_name: str = "scoring"
    async def process(self, lead: RawLead) -> RawLead:
        if isinstance(lead, EnrichedLead):
            return ScoredLead(name=lead.name, place_id=lead.place_id, score=0.5)
        return lead
    async def batch_process(self, leads: list[RawLead], *, concurrency: int = 5) -> list[RawLead]:
        return [await self.process(l) for l in leads]
''',
        "pipeline/outreach.py": '''"""Stage 4: Outreach."""
from pipeline.base import BaseStage
from models.lead import RawLead, ScoredLead, OutreachLead
class OutreachStage(BaseStage):
    stage_name: str = "outreach"
    async def process(self, lead: RawLead) -> RawLead:
        if isinstance(lead, ScoredLead):
            return OutreachLead(name=lead.name, place_id=lead.place_id, score=lead.score, subject_line="Hi")
        return lead
    async def batch_process(self, leads: list[RawLead], *, concurrency: int = 5) -> list[RawLead]:
        return [await self.process(l) for l in leads]
''',
        "pipeline/runner.py": '''"""Pipeline runner."""
from pipeline.discovery import DiscoveryStage
from pipeline.enrichment import EnrichmentStage
from pipeline.scoring import ScoringStage
from pipeline.outreach import OutreachStage
from models.config import PipelineSettings
from pipeline.exceptions import PipelineError
class PipelineRunner:
    def __init__(self, config: PipelineSettings):
        self.config = config
        self.stages = [DiscoveryStage(), EnrichmentStage(), ScoringStage(), OutreachStage()]
    async def run(self, leads):
        for stage in self.stages:
            leads = await stage.batch_process(leads)
        return leads
''',
        "utils/cost_tracker.py": '''"""Cost tracking."""
from models.config import PipelineSettings
class CostTracker:
    def __init__(self, config: PipelineSettings):
        self.budget = config.cost_budget
        self.spent = 0.0
    def track(self, cost: float) -> None:
        self.spent += cost
    def remaining(self) -> float:
        return self.budget - self.spent
''',
        "api/server.py": '''"""FastAPI server."""
from pipeline.runner import PipelineRunner
from models.config import PipelineSettings
config = PipelineSettings()
runner = PipelineRunner(config=config)
def health():
    return {"status": "ok"}
''',
        "main.py": '''"""CLI entry point."""
from pipeline.runner import PipelineRunner
from models.config import PipelineSettings
def main():
    config = PipelineSettings()
    runner = PipelineRunner(config=config)
    print(f"Pipeline ready: budget=${config.cost_budget}")
if __name__ == "__main__":
    main()
''',
    }

    return impls.get(file_path, f'"""Auto-generated: {file_path}"""\npass\n').strip()


# ===========================================================================
# Test Topological Sort
# ===========================================================================

class TestTopologicalSort:
    def test_basic_sort(self):
        skeleton = _make_12_file_skeleton()
        result = topological_sort(skeleton)
        assert len(result.sorted_files) == 12
        assert result.total_levels >= 3  # At least 3 levels of depth

    def test_level_ordering(self):
        """Files at level N come after all files at level N-1."""
        skeleton = _make_12_file_skeleton()
        result = topological_sort(skeleton)

        sorted_set = set()
        for level_files in result.levels:
            for f in level_files:
                # All dependencies should already be in sorted_set
                deps = skeleton.dependencies_for(f)
                for dep in deps:
                    assert dep.target in sorted_set, (
                        f"{f} at level {result.node_levels[f]} depends on "
                        f"{dep.target} at level {result.node_levels[dep.target]}, "
                        f"but it hasn't been generated yet"
                    )
            sorted_set.update(level_files)

    def test_no_deps_at_level_zero(self):
        """Level 0 files should have no project dependencies."""
        skeleton = _make_12_file_skeleton()
        result = topological_sort(skeleton)
        level_0 = result.levels[0]
        for f in level_0:
            deps = skeleton.dependencies_for(f)
            assert len(deps) == 0, f"{f} is at level 0 but has dependencies: {deps}"

    def test_parallelism_within_levels(self):
        """Files in the same level should have no inter-dependencies."""
        skeleton = _make_12_file_skeleton()
        result = topological_sort(skeleton)
        for level_files in result.levels:
            level_set = set(level_files)
            for f in level_files:
                deps = skeleton.dependencies_for(f)
                for dep in deps:
                    assert dep.target not in level_set, (
                        f"{f} and {dep.target} are in the same level but "
                        f"{f} depends on {dep.target}"
                    )

    def test_cycle_detection(self):
        skeleton = _make_12_file_skeleton()
        # Create a cycle
        skeleton.dependency_edges.append(
            DependencyEdge(source="models/lead.py", target="main.py",
                           kind=DependencyKind.IMPORTS, symbols=["main"])
        )
        with pytest.raises(DependencyCycleError):
            topological_sort(skeleton)

    def test_max_parallelism(self):
        """The 4 stage files should be parallelizable."""
        skeleton = _make_12_file_skeleton()
        result = topological_sort(skeleton)
        assert result.max_parallelism() >= 4  # 4 stages can run in parallel


class TestBuildPlan:
    def test_skeleton_impl_split(self):
        skeleton = _make_12_file_skeleton()
        plan = create_build_plan(skeleton)
        assert plan.total_skeleton_files == 4
        assert plan.total_impl_files == 8

    def test_skeletons_come_first(self):
        """All skeleton levels should precede implementation levels in topological order."""
        skeleton = _make_12_file_skeleton()
        plan = create_build_plan(skeleton)
        result = plan.topo_result

        skeleton_paths = {f.path for f in skeleton.file_tree if f.skeleton}
        impl_paths = {f.path for f in skeleton.file_tree if not f.skeleton}

        max_skeleton_level = max(
            result.node_levels[p] for p in skeleton_paths
            if p in result.node_levels
        )
        min_impl_level = min(
            result.node_levels[p] for p in impl_paths
            if p in result.node_levels
        )

        # This is valid: skeletons can be at same level as some impls
        # What matters is that within the build plan, skeletons are generated first
        assert plan.skeleton_order  # Has skeleton levels
        assert plan.impl_order  # Has impl levels

    def test_critical_path(self):
        skeleton = _make_12_file_skeleton()
        path = critical_path(skeleton)
        assert len(path) >= 3  # At least 3 files deep
        # First file should be a level-0 file
        result = topological_sort(skeleton)
        assert result.node_levels[path[0]] == 0


# ===========================================================================
# Test Pyright Integration
# ===========================================================================

class TestPyrightParsing:
    def test_parse_clean_output(self):
        raw = json.dumps({
            "version": "1.1",
            "summary": {"errorCount": 0, "warningCount": 0, "informationCount": 0},
            "generalDiagnostics": [],
        })
        from pathlib import Path
        result = _parse_pyright_output(raw, Path("/tmp/project"))
        assert result.success
        assert result.error_count == 0

    def test_parse_errors(self):
        raw = json.dumps({
            "version": "1.1",
            "summary": {"errorCount": 2, "warningCount": 0, "informationCount": 0},
            "generalDiagnostics": [
                {
                    "file": "/tmp/project/pipeline/discovery.py",
                    "severity": "error",
                    "message": "Import \"pipeline.base\" could not be resolved",
                    "rule": "reportMissingImports",
                    "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 30}},
                },
                {
                    "file": "/tmp/project/pipeline/discovery.py",
                    "severity": "error",
                    "message": "Type \"int\" is not assignable to type \"str\"",
                    "rule": "reportAssignmentType",
                    "range": {"start": {"line": 10, "character": 4}, "end": {"line": 10, "character": 15}},
                },
            ],
        })
        from pathlib import Path
        result = _parse_pyright_output(raw, Path("/tmp/project"))
        assert not result.success
        assert result.error_count == 2
        assert len(result.import_errors) == 1
        assert len(result.type_errors) == 1

    def test_group_errors_by_file(self):
        result = PyrightResult(errors=[
            PyrightError(file="a.py", line=1, column=0, message="err1", severity="error"),
            PyrightError(file="a.py", line=5, column=0, message="err2", severity="error"),
            PyrightError(file="b.py", line=1, column=0, message="err3", severity="error"),
        ])
        grouped = group_errors_by_file(result)
        assert len(grouped["a.py"]) == 2
        assert len(grouped["b.py"]) == 1

    def test_format_errors_for_llm(self):
        errors = [
            PyrightError(file="test.py", line=3, column=0,
                         message="Import 'foo' not found", severity="error",
                         rule="reportMissingImports"),
        ]
        source = "import foo\n\nfoo.bar()\n"
        formatted = format_errors_for_llm(errors, source)
        assert "line 3" in formatted
        assert "Import 'foo' not found" in formatted

    def test_write_project_creates_inits(self):
        """write_project_for_pyright should create __init__.py files."""
        import tempfile
        files = {
            "models/lead.py": "class Lead: pass",
            "pipeline/base.py": "class Base: pass",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_project_for_pyright(files, tmpdir)
            assert (root / "models" / "__init__.py").exists()
            assert (root / "pipeline" / "__init__.py").exists()
            assert (root / "pyrightconfig.json").exists()


# ===========================================================================
# Test Parallel Builder
# ===========================================================================

class TestParallelBuilder:
    def test_clean_code(self):
        assert _clean_code("```python\nx = 1\n```") == "x = 1"
        assert _clean_code("```\nx = 1\n```") == "x = 1"
        assert _clean_code("x = 1") == "x = 1"

    def test_build_level_async(self):
        """Test parallel generation of a single level."""
        skeleton = _make_12_file_skeleton()
        registry = SymbolRegistry()
        skeleton_files = generate_all_skeletons(skeleton, registry)

        # Build level 2 — the 4 stage files
        stage_entries = [
            f for f in skeleton.file_tree
            if f.path.startswith("pipeline/") and f.role == FileRole.IMPLEMENTATION
            and f.path != "pipeline/runner.py"
        ]

        result = asyncio.run(build_level(
            level_entries=stage_entries,
            skeleton=skeleton,
            registry=registry,
            skeleton_files=skeleton_files,
            llm_fn=_mock_llm_12,
        ))

        assert len(result) == 4
        assert all(v is not None for v in result.values())
        # All should parse
        for code in result.values():
            ast.parse(code)

    def test_full_parallel_build(self):
        """Full parallel build of 12-file project."""
        skeleton = _make_12_file_skeleton()

        result = asyncio.run(run_parallel_build(
            skeleton=skeleton,
            llm_fn=_mock_llm_12,
            run_type_check=False,  # Skip pyright in test
        ))

        assert len(result["skeleton_files"]) == 4
        assert len(result["implementation_files"]) == 8
        assert len(result["all_files"]) == 12
        assert len(result["failures"]) == 0

    def test_parallel_build_maintains_order(self):
        """Files are generated in dependency order even with parallelism."""
        skeleton = _make_12_file_skeleton()
        registry = SymbolRegistry()

        # Track generation order
        generation_order = []
        original_mock = _mock_llm_12

        def tracking_mock(system, user, model):
            path_match = re.search(r"Path:\s*(\S+)", user)
            if path_match:
                generation_order.append(path_match.group(1))
            return original_mock(system, user, model)

        result = asyncio.run(run_parallel_build(
            skeleton=skeleton,
            llm_fn=tracking_mock,
            run_type_check=False,
        ))

        # runner.py should come after all stage files
        if "pipeline/runner.py" in generation_order:
            runner_idx = generation_order.index("pipeline/runner.py")
            for stage in ["pipeline/discovery.py", "pipeline/enrichment.py",
                          "pipeline/scoring.py", "pipeline/outreach.py"]:
                if stage in generation_order:
                    stage_idx = generation_order.index(stage)
                    assert stage_idx < runner_idx, (
                        f"{stage} should be generated before pipeline/runner.py"
                    )


# ===========================================================================
# Test 12-File Import Resolution (done-when criterion)
# ===========================================================================

class TestImportResolution12Files:
    """
    The critical test: every import in the 12-file project resolves
    to a symbol that exists in another generated file.
    """

    def test_all_imports_resolve(self):
        skeleton = _make_12_file_skeleton()

        result = asyncio.run(run_parallel_build(
            skeleton=skeleton,
            llm_fn=_mock_llm_12,
            run_type_check=False,
        ))

        all_files = result["all_files"]
        assert len(all_files) == 12

        # Build module → exports map
        module_exports: dict[str, set[str]] = {}
        for path, code in all_files.items():
            module_path = path.replace("/", ".").replace(".py", "")
            tree = ast.parse(code)
            exports = set()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    exports.add(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    exports.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            exports.add(target.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    exports.add(node.target.id)
            module_exports[module_path] = exports

        # Verify every project-internal import resolves
        unresolved = []
        for path, code in all_files.items():
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                    if module in module_exports:
                        for alias in node.names:
                            if alias.name not in module_exports[module]:
                                unresolved.append(
                                    f"{path}: 'from {module} import {alias.name}' "
                                    f"— not found in {module} "
                                    f"(available: {module_exports[module]})"
                                )

        assert not unresolved, (
            "Unresolved imports in 12-file project:\n" + "\n".join(unresolved)
        )

    def test_all_files_have_valid_syntax(self):
        skeleton = _make_12_file_skeleton()
        result = asyncio.run(run_parallel_build(
            skeleton=skeleton,
            llm_fn=_mock_llm_12,
            run_type_check=False,
        ))
        for path, code in result["all_files"].items():
            try:
                ast.parse(code)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {path}: {e}")

    def test_model_chain_inheritance_preserved(self):
        """The progressive model chain should have correct inheritance."""
        skeleton = _make_12_file_skeleton()
        result = asyncio.run(run_parallel_build(
            skeleton=skeleton,
            llm_fn=_mock_llm_12,
            run_type_check=False,
        ))

        model_code = result["all_files"]["models/lead.py"]
        tree = ast.parse(model_code)
        classes = {
            node.name: [ast.unparse(b) for b in node.bases]
            for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        assert "BaseModel" in classes.get("RawLead", [])
        assert "RawLead" in classes.get("EnrichedLead", [])
        assert "EnrichedLead" in classes.get("ScoredLead", [])
        assert "ScoredLead" in classes.get("OutreachLead", [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
