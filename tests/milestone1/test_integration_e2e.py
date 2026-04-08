"""
End-to-End Integration Test — Milestone 1

Simulates the complete skeleton build pipeline:
1. Architect produces SkeletonArtifact (mocked — returns pre-built artifact)
2. Pass 1 generates skeleton files deterministically
3. Pass 2 generates implementation files (mocked — returns valid Python)
4. Validates all imports resolve across files

This test proves the done-when criterion:
"produces files where imports resolve correctly across all files"
"""

import ast
import re
import pytest

from belief.models.skeleton import (
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
from belief.agents.skeleton_builder import generate_all_skeletons
from belief.agents.builder_skeleton import build_implementation_file, build_all_implementations
from belief.agents.architect_skeleton import validate_skeleton
from belief.agents.graph_integration import (
    skeleton_architect_node,
    skeleton_pass1_node,
    skeleton_builder_node,
    run_full_skeleton_build,
)


# ---------------------------------------------------------------------------
# Test fixture: 4-stage data pipeline skeleton
# ---------------------------------------------------------------------------

def _make_data_pipeline_skeleton() -> SkeletonArtifact:
    """
    A 4-stage data pipeline with progressive Pydantic models.
    This matches the done-when target from the milestone spec.
    """
    return SkeletonArtifact(
        project_name="data_pipeline",
        description="4-stage data pipeline with progressive Pydantic models",
        file_tree=[
            # Skeleton files (Pass 1)
            FileTreeEntry(path="models/data.py", role=FileRole.MODEL,
                          description="Progressive data models", skeleton=True),
            FileTreeEntry(path="models/config.py", role=FileRole.CONFIG,
                          description="Pipeline settings", skeleton=True),
            FileTreeEntry(path="pipeline/base.py", role=FileRole.ABC,
                          description="Base stage ABC", skeleton=True),
            FileTreeEntry(path="pipeline/exceptions.py", role=FileRole.EXCEPTION,
                          description="Pipeline exceptions", skeleton=True),
            # Implementation files (Pass 2)
            FileTreeEntry(path="pipeline/stage_ingest.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 1: Data ingestion", skeleton=False),
            FileTreeEntry(path="pipeline/stage_transform.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 2: Data transformation", skeleton=False),
            FileTreeEntry(path="pipeline/stage_validate.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 3: Data validation", skeleton=False),
            FileTreeEntry(path="pipeline/stage_output.py", role=FileRole.IMPLEMENTATION,
                          description="Stage 4: Output writing", skeleton=False),
            FileTreeEntry(path="pipeline/runner.py", role=FileRole.IMPLEMENTATION,
                          description="Pipeline runner/orchestrator", skeleton=False),
            FileTreeEntry(path="main.py", role=FileRole.ENTRY_POINT,
                          description="CLI entry point", skeleton=False),
        ],
        dependency_edges=[
            # ABC depends on models
            DependencyEdge(source="pipeline/base.py", target="models/data.py",
                           kind=DependencyKind.IMPORTS, symbols=["RawRecord"]),
            # All stages depend on ABC and models
            DependencyEdge(source="pipeline/stage_ingest.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/stage_ingest.py", target="models/data.py",
                           kind=DependencyKind.IMPORTS, symbols=["RawRecord"]),
            DependencyEdge(source="pipeline/stage_transform.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/stage_transform.py", target="models/data.py",
                           kind=DependencyKind.IMPORTS, symbols=["RawRecord", "TransformedRecord"]),
            DependencyEdge(source="pipeline/stage_validate.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/stage_validate.py", target="models/data.py",
                           kind=DependencyKind.IMPORTS, symbols=["TransformedRecord", "ValidatedRecord"]),
            DependencyEdge(source="pipeline/stage_output.py", target="pipeline/base.py",
                           kind=DependencyKind.INHERITS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/stage_output.py", target="models/data.py",
                           kind=DependencyKind.IMPORTS, symbols=["ValidatedRecord", "OutputRecord"]),
            # Runner depends on stages and config
            DependencyEdge(source="pipeline/runner.py", target="models/config.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineConfig"]),
            DependencyEdge(source="pipeline/runner.py", target="pipeline/base.py",
                           kind=DependencyKind.IMPORTS, symbols=["BaseStage"]),
            DependencyEdge(source="pipeline/runner.py", target="pipeline/exceptions.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineError"]),
            # Main depends on runner and config
            DependencyEdge(source="main.py", target="pipeline/runner.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineRunner"]),
            DependencyEdge(source="main.py", target="models/config.py",
                           kind=DependencyKind.IMPORTS, symbols=["PipelineConfig"]),
        ],
        model_chains=[
            ModelChain(
                name="DataProgression",
                models=[
                    ModelSpec(name="RawRecord", file_path="models/data.py",
                             base_class="BaseModel",
                             fields=[
                                 ModelFieldSpec(name="id", type_annotation="str"),
                                 ModelFieldSpec(name="source", type_annotation="str"),
                                 ModelFieldSpec(name="raw_data", type_annotation="dict[str, Any]"),
                                 ModelFieldSpec(name="ingested_at", type_annotation="Optional[str]", default="None"),
                             ],
                             docstring="Raw data record from ingestion"),
                    ModelSpec(name="TransformedRecord", file_path="models/data.py",
                             base_class="RawRecord",
                             fields=[
                                 ModelFieldSpec(name="transformed_data", type_annotation="dict[str, Any]"),
                                 ModelFieldSpec(name="transform_log", type_annotation="list[str]", default="[]"),
                             ],
                             docstring="Record after transformation stage"),
                    ModelSpec(name="ValidatedRecord", file_path="models/data.py",
                             base_class="TransformedRecord",
                             fields=[
                                 ModelFieldSpec(name="is_valid", type_annotation="bool", default="True"),
                                 ModelFieldSpec(name="validation_errors", type_annotation="list[str]", default="[]"),
                             ],
                             docstring="Record after validation stage"),
                    ModelSpec(name="OutputRecord", file_path="models/data.py",
                             base_class="ValidatedRecord",
                             fields=[
                                 ModelFieldSpec(name="output_path", type_annotation="Optional[str]", default="None"),
                                 ModelFieldSpec(name="written_at", type_annotation="Optional[str]", default="None"),
                             ],
                             docstring="Final output record"),
                ],
            ),
        ],
        abc_definitions=[
            ABCDefinition(
                name="BaseStage",
                file_path="pipeline/base.py",
                base_classes=["ABC"],
                methods=[
                    MethodSignature(name="process", params="self, record: RawRecord",
                                    return_type="RawRecord", is_async=True, is_abstract=True,
                                    docstring="Process a single record"),
                    MethodSignature(name="validate_input", params="self, record: RawRecord",
                                    return_type="bool", is_abstract=True,
                                    docstring="Validate input before processing"),
                ],
                class_attributes=[
                    ModelFieldSpec(name="stage_name", type_annotation="str"),
                ],
                docstring="Abstract base for pipeline stages",
            ),
        ],
        config_schemas=[
            ConfigSchema(
                name="PipelineConfig",
                file_path="models/config.py",
                fields=[
                    ModelFieldSpec(name="input_path", type_annotation="str", default='"./data/input"'),
                    ModelFieldSpec(name="output_path", type_annotation="str", default='"./data/output"'),
                    ModelFieldSpec(name="batch_size", type_annotation="int", default="100"),
                    ModelFieldSpec(name="max_retries", type_annotation="int", default="3"),
                ],
                env_prefix="PIPELINE_",
                docstring="Pipeline configuration",
            ),
        ],
        exception_specs=[
            ExceptionSpec(name="PipelineError", file_path="pipeline/exceptions.py",
                          base_class="Exception", docstring="Base pipeline error"),
            ExceptionSpec(name="StageError", file_path="pipeline/exceptions.py",
                          base_class="PipelineError", docstring="Stage processing error"),
            ExceptionSpec(name="ValidationError", file_path="pipeline/exceptions.py",
                          base_class="PipelineError", docstring="Record validation error"),
        ],
        external_dependencies=["pydantic", "pydantic-settings"],
        entry_point="main.py",
    )


# ---------------------------------------------------------------------------
# Mock LLM for Pass 2
# ---------------------------------------------------------------------------

def _mock_llm_fn(system: str, user: str, model: str) -> str:
    """
    Mock LLM that generates valid Python implementation files.
    Parses the file path from the user prompt and returns appropriate code.
    """
    # Extract file path from prompt
    path_match = re.search(r"Path:\s*(\S+)", user)
    file_path = path_match.group(1) if path_match else "unknown.py"

    # Generate mock implementations that import from the correct modules
    implementations = {
        "pipeline/stage_ingest.py": '''
"""Stage 1: Data ingestion."""
from pipeline.base import BaseStage
from models.data import RawRecord

class IngestStage(BaseStage):
    """Ingests raw data from source."""
    stage_name: str = "ingest"

    async def process(self, record: RawRecord) -> RawRecord:
        """Process a single record through ingestion."""
        return record

    def validate_input(self, record: RawRecord) -> bool:
        """Validate input record."""
        return bool(record.id and record.source)
''',
        "pipeline/stage_transform.py": '''
"""Stage 2: Data transformation."""
from pipeline.base import BaseStage
from models.data import RawRecord, TransformedRecord

class TransformStage(BaseStage):
    """Transforms raw records."""
    stage_name: str = "transform"

    async def process(self, record: RawRecord) -> RawRecord:
        """Process a single record through transformation."""
        return TransformedRecord(
            id=record.id,
            source=record.source,
            raw_data=record.raw_data,
            transformed_data=record.raw_data,
        )

    def validate_input(self, record: RawRecord) -> bool:
        """Validate input record."""
        return bool(record.raw_data)
''',
        "pipeline/stage_validate.py": '''
"""Stage 3: Data validation."""
from pipeline.base import BaseStage
from models.data import RawRecord, TransformedRecord, ValidatedRecord

class ValidateStage(BaseStage):
    """Validates transformed records."""
    stage_name: str = "validate"

    async def process(self, record: RawRecord) -> RawRecord:
        """Process a single record through validation."""
        if isinstance(record, TransformedRecord):
            return ValidatedRecord(
                id=record.id,
                source=record.source,
                raw_data=record.raw_data,
                transformed_data=record.transformed_data,
                is_valid=True,
            )
        return record

    def validate_input(self, record: RawRecord) -> bool:
        """Validate input record."""
        return isinstance(record, TransformedRecord)
''',
        "pipeline/stage_output.py": '''
"""Stage 4: Output writing."""
from pipeline.base import BaseStage
from models.data import RawRecord, ValidatedRecord, OutputRecord

class OutputStage(BaseStage):
    """Writes validated records to output."""
    stage_name: str = "output"

    async def process(self, record: RawRecord) -> RawRecord:
        """Process a single record through output."""
        if isinstance(record, ValidatedRecord):
            return OutputRecord(
                id=record.id,
                source=record.source,
                raw_data=record.raw_data,
                transformed_data=record.transformed_data,
                is_valid=record.is_valid,
                validation_errors=record.validation_errors,
                output_path="./data/output",
            )
        return record

    def validate_input(self, record: RawRecord) -> bool:
        """Validate input record."""
        return isinstance(record, ValidatedRecord) and record.is_valid
''',
        "pipeline/runner.py": '''
"""Pipeline runner."""
from models.config import PipelineConfig
from pipeline.base import BaseStage
from pipeline.exceptions import PipelineError

class PipelineRunner:
    """Orchestrates pipeline stages."""

    def __init__(self, config: PipelineConfig, stages: list[BaseStage]):
        self.config = config
        self.stages = stages

    async def run(self, records):
        """Run all stages on records."""
        for stage in self.stages:
            processed = []
            for record in records:
                try:
                    result = await stage.process(record)
                    processed.append(result)
                except Exception as e:
                    raise PipelineError(f"Stage {stage.stage_name} failed: {e}")
            records = processed
        return records
''',
        "main.py": '''
"""CLI entry point."""
import asyncio
from pipeline.runner import PipelineRunner
from models.config import PipelineConfig

def main():
    """Run the data pipeline."""
    config = PipelineConfig()
    print(f"Pipeline configured: batch_size={config.batch_size}")

if __name__ == "__main__":
    main()
''',
    }

    code = implementations.get(file_path, f'"""Auto-generated: {file_path}"""\npass\n')
    return code.strip()


# ===========================================================================
# Tests
# ===========================================================================

class TestArchitectValidation:
    def test_valid_skeleton_passes(self):
        skeleton = _make_data_pipeline_skeleton()
        issues = validate_skeleton(skeleton)
        assert issues == [], f"Expected no issues, got: {issues}"

    def test_cycle_detection(self):
        skeleton = _make_data_pipeline_skeleton()
        # Add a cycle: models/data.py → pipeline/base.py (already has reverse)
        skeleton.dependency_edges.append(
            DependencyEdge(source="models/data.py", target="pipeline/base.py",
                           kind=DependencyKind.IMPORTS, symbols=["BaseStage"])
        )
        issues = validate_skeleton(skeleton)
        assert any("cycle" in i.lower() for i in issues)

    def test_missing_file_reference(self):
        skeleton = _make_data_pipeline_skeleton()
        skeleton.dependency_edges.append(
            DependencyEdge(source="nonexistent.py", target="models/data.py",
                           kind=DependencyKind.IMPORTS, symbols=[])
        )
        issues = validate_skeleton(skeleton)
        assert any("nonexistent.py" in i for i in issues)

    def test_skeleton_flag_consistency(self):
        skeleton = _make_data_pipeline_skeleton()
        # Mark an impl file as skeleton — should flag
        skeleton.file_tree[4] = FileTreeEntry(
            path="pipeline/stage_ingest.py", role=FileRole.IMPLEMENTATION,
            description="Stage 1", skeleton=True  # Wrong!
        )
        issues = validate_skeleton(skeleton)
        assert any("skeleton=true" in i.lower() for i in issues)


class TestPass1Generation:
    def test_all_skeleton_files_generated(self):
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        files = generate_all_skeletons(skeleton, registry)
        assert len(files) == 4
        assert "models/data.py" in files
        assert "models/config.py" in files
        assert "pipeline/base.py" in files
        assert "pipeline/exceptions.py" in files

    def test_all_skeleton_files_parse(self):
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        files = generate_all_skeletons(skeleton, registry)
        for path, code in files.items():
            ast.parse(code)  # Should not raise

    def test_model_chain_complete(self):
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        files = generate_all_skeletons(skeleton, registry)
        model_code = files["models/data.py"]
        assert "class RawRecord" in model_code
        assert "class TransformedRecord" in model_code
        assert "class ValidatedRecord" in model_code
        assert "class OutputRecord" in model_code

    def test_registry_has_all_symbols(self):
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        generate_all_skeletons(skeleton, registry)

        # Check all model names are resolvable
        for name in ["RawRecord", "TransformedRecord", "ValidatedRecord", "OutputRecord"]:
            result = registry.resolve_symbol(name)
            assert result is not None, f"Symbol {name} not found in registry"
            assert result[0] == "models.data"

        # Check ABC
        result = registry.resolve_symbol("BaseStage")
        assert result is not None
        assert result[0] == "pipeline.base"

        # Check config
        result = registry.resolve_symbol("PipelineConfig")
        assert result is not None


class TestPass2WithMockLLM:
    def test_all_impl_files_generated(self):
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        skeleton_files = generate_all_skeletons(skeleton, registry)

        impl_files = build_all_implementations(
            skeleton=skeleton,
            registry=registry,
            skeleton_files=skeleton_files,
            llm_fn=_mock_llm_fn,
        )

        # All 6 implementation files should succeed
        assert len(impl_files) == 6
        assert all(v is not None for v in impl_files.values())

    def test_all_impl_files_parse(self):
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        skeleton_files = generate_all_skeletons(skeleton, registry)

        impl_files = build_all_implementations(
            skeleton=skeleton,
            registry=registry,
            skeleton_files=skeleton_files,
            llm_fn=_mock_llm_fn,
        )

        for path, code in impl_files.items():
            if code is not None:
                ast.parse(code)

    def test_impl_files_import_from_skeletons(self):
        """The core test: implementation files import symbols defined in skeletons."""
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        skeleton_files = generate_all_skeletons(skeleton, registry)

        impl_files = build_all_implementations(
            skeleton=skeleton,
            registry=registry,
            skeleton_files=skeleton_files,
            llm_fn=_mock_llm_fn,
        )

        # Collect all symbols from skeleton files
        skeleton_symbols = set()
        for fs in registry.all_files():
            skeleton_symbols.update(fs.all_symbol_names())

        # Check that implementation files reference skeleton symbols
        for path, code in impl_files.items():
            if code is None:
                continue
            tree = ast.parse(code)
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.name)

            # Each impl file should import at least one skeleton symbol
            overlap = imported_names & skeleton_symbols
            assert overlap, (
                f"{path} doesn't import any skeleton symbols. "
                f"Imported: {imported_names}, Available: {skeleton_symbols}"
            )


class TestImportResolution:
    """
    The critical validation: all imports across all files resolve
    to symbols that actually exist in other generated files.
    """

    def test_cross_file_import_resolution(self):
        """Every import in every file resolves to a symbol in another file."""
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        skeleton_files = generate_all_skeletons(skeleton, registry)

        impl_files = build_all_implementations(
            skeleton=skeleton,
            registry=registry,
            skeleton_files=skeleton_files,
            llm_fn=_mock_llm_fn,
        )

        # Combine all files
        all_files = {**skeleton_files}
        for k, v in impl_files.items():
            if v is not None:
                all_files[k] = v

        # Build a map: module_path → set of exported symbol names
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

        # Check every import resolves
        unresolved = []
        for path, code in all_files.items():
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    # Only check project-internal imports
                    module = node.module
                    if module in module_exports:
                        for alias in node.names:
                            if alias.name not in module_exports[module]:
                                unresolved.append(
                                    f"{path}: 'from {module} import {alias.name}' "
                                    f"— {alias.name} not found in {module} "
                                    f"(available: {module_exports[module]})"
                                )

        assert not unresolved, (
            f"Unresolved imports:\n" + "\n".join(unresolved)
        )


class TestGraphNodes:
    """Test the LangGraph node functions work with state dicts."""

    def test_pass1_node(self):
        skeleton = _make_data_pipeline_skeleton()
        state = {"skeleton_artifact": skeleton.model_dump()}

        result = skeleton_pass1_node(state)
        assert "skeleton_files" in result
        assert len(result["skeleton_files"]) == 4
        assert "symbol_registry" in result

    def test_builder_node(self):
        skeleton = _make_data_pipeline_skeleton()
        registry = SymbolRegistry()
        skeleton_files = generate_all_skeletons(skeleton, registry)

        state = {
            "skeleton_artifact": skeleton.model_dump(),
            "skeleton_files": skeleton_files,
            "symbol_registry": registry,
        }

        result = skeleton_builder_node(state, llm_fn=_mock_llm_fn)
        assert result["builder_done"] is True
        assert len(result["code_files"]) == 10  # 4 skeleton + 6 impl
        assert len(result["builder_failures"]) == 0


class TestFullPipeline:
    """End-to-end test with mock LLM."""

    def test_run_full_build(self):
        skeleton = _make_data_pipeline_skeleton()

        # Mock architect LLM to return our pre-built skeleton
        call_count = {"n": 0}

        def mock_llm(system: str, user: str, model: str) -> str:
            call_count["n"] += 1
            if "architect" in system.lower() or "skeleton" in system.lower():
                return skeleton.model_dump_json()
            return _mock_llm_fn(system, user, model)

        result = run_full_skeleton_build(
            goal="build a 4-stage data pipeline with progressive Pydantic models",
            llm_fn=mock_llm,
        )

        assert len(result["all_files"]) == 10
        assert len(result["failures"]) == 0
        assert "data_pipeline" in result["report"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
