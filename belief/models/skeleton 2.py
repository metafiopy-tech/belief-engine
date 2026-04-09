"""
Skeleton Architecture Models — Milestone 1

The SkeletonArtifact replaces FileManifestPlan as the Architect's output.
It captures the full typed interface layer of a project before any
implementation logic is written.

The two-pass build flow:
  Architect → SkeletonArtifact
  SkeletonBuilder (Haiku) → skeleton files (ABCs, models, types, configs)
  Builder (Sonnet) → implementation files (against skeleton contracts)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FileRole(str, Enum):
    """Categorizes a file's role in the skeleton vs implementation split."""
    MODEL = "model"              # Pydantic models, dataclasses
    ABC = "abc"                  # Abstract base classes
    PROTOCOL = "protocol"        # typing.Protocol definitions
    TYPE_ALIAS = "type_alias"    # Type alias modules
    CONFIG = "config"            # Config / settings schemas
    EXCEPTION = "exception"      # Exception hierarchies
    IMPLEMENTATION = "impl"      # Concrete logic (Pass 2)
    ENTRY_POINT = "entry_point"  # CLI, main, server entry
    TEST = "test"                # Test files


class DependencyKind(str, Enum):
    """Edge type in the dependency DAG."""
    IMPORTS = "imports"          # A imports symbols from B
    INHERITS = "inherits"        # A's class inherits from B's class
    IMPLEMENTS = "implements"    # A implements B's ABC/Protocol
    USES = "uses"                # A calls/instantiates something from B


# ---------------------------------------------------------------------------
# Dependency Edge
# ---------------------------------------------------------------------------

class DependencyEdge(BaseModel):
    """A directed edge in the file dependency graph: source depends on target."""
    source: str = Field(description="File path that has the dependency (e.g. 'pipeline/runner.py')")
    target: str = Field(description="File path being depended on (e.g. 'models/lead.py')")
    kind: DependencyKind = Field(description="Nature of the dependency")
    symbols: list[str] = Field(
        default_factory=list,
        description="Specific symbols imported/used (e.g. ['RawLead', 'EnrichedLead'])"
    )


# ---------------------------------------------------------------------------
# Model Chain
# ---------------------------------------------------------------------------

class ModelFieldSpec(BaseModel):
    """A single field in a Pydantic model."""
    name: str
    type_annotation: str = Field(description="Python type as string, e.g. 'Optional[str]', 'list[float]'")
    default: Optional[str] = Field(
        default=None,
        description="Default value as string, or None if required"
    )
    description: Optional[str] = None


class ModelSpec(BaseModel):
    """Specification for a single Pydantic model."""
    name: str = Field(description="Class name, e.g. 'RawLead'")
    file_path: str = Field(description="Where this model lives, e.g. 'models/lead.py'")
    base_class: str = Field(
        default="BaseModel",
        description="Parent class (BaseModel, or a prior model in the chain)"
    )
    fields: list[ModelFieldSpec] = Field(default_factory=list)
    validators: list[str] = Field(
        default_factory=list,
        description="Validator method names to generate (bodies will be `pass`)"
    )
    docstring: Optional[str] = None


class ModelChain(BaseModel):
    """
    A progressive model chain where each model extends or transforms the previous.
    Example: RawLead → EnrichedLead → ScoredLead → OutreachLead
    """
    name: str = Field(description="Chain name, e.g. 'LeadProgression'")
    models: list[ModelSpec] = Field(description="Models in dependency order (base first)")


# ---------------------------------------------------------------------------
# ABC / Protocol Definitions
# ---------------------------------------------------------------------------

class MethodSignature(BaseModel):
    """A method signature for an ABC or Protocol."""
    name: str
    params: str = Field(
        description="Parameter string including type annotations, e.g. 'self, lead: RawLead, *, timeout: float = 30.0'"
    )
    return_type: str = Field(default="None", description="Return type annotation as string")
    is_async: bool = False
    is_abstract: bool = True
    docstring: Optional[str] = None


class ABCDefinition(BaseModel):
    """An abstract base class that implementation files will inherit from."""
    name: str = Field(description="Class name, e.g. 'BaseEnricher'")
    file_path: str = Field(description="Where this ABC lives")
    base_classes: list[str] = Field(
        default_factory=lambda: ["ABC"],
        description="Parent classes, e.g. ['ABC'] or ['ABC', 'Generic[T]']"
    )
    methods: list[MethodSignature] = Field(default_factory=list)
    class_attributes: list[ModelFieldSpec] = Field(
        default_factory=list,
        description="Class-level attributes with type annotations"
    )
    docstring: Optional[str] = None


class ProtocolDefinition(BaseModel):
    """A typing.Protocol definition for structural subtyping."""
    name: str
    file_path: str
    methods: list[MethodSignature] = Field(default_factory=list)
    attributes: list[ModelFieldSpec] = Field(default_factory=list)
    docstring: Optional[str] = None


# ---------------------------------------------------------------------------
# Config & Exception Schemas
# ---------------------------------------------------------------------------

class ConfigSchema(BaseModel):
    """A configuration/settings class."""
    name: str = Field(description="Class name, e.g. 'PipelineSettings'")
    file_path: str
    fields: list[ModelFieldSpec] = Field(default_factory=list)
    env_prefix: Optional[str] = Field(
        default=None,
        description="Pydantic Settings env prefix, e.g. 'PIPELINE_'"
    )
    docstring: Optional[str] = None


class ExceptionSpec(BaseModel):
    """A custom exception class."""
    name: str
    file_path: str
    base_class: str = Field(default="Exception")
    message_template: Optional[str] = None
    docstring: Optional[str] = None


# ---------------------------------------------------------------------------
# File Tree Entry
# ---------------------------------------------------------------------------

class FileTreeEntry(BaseModel):
    """A single file in the project's file tree."""
    path: str = Field(description="Relative file path, e.g. 'pipeline/runner.py'")
    role: FileRole
    description: str = Field(description="One-line description of what this file does")
    skeleton: bool = Field(
        default=False,
        description="True if this file is generated in Pass 1 (skeleton). "
                    "Models, ABCs, Protocols, configs, exceptions → True. "
                    "Implementations, entry points → False."
    )
    estimated_lines: Optional[int] = Field(
        default=None,
        description="Rough estimate of implementation line count"
    )


# ---------------------------------------------------------------------------
# SkeletonArtifact — the main output
# ---------------------------------------------------------------------------

class SkeletonArtifact(BaseModel):
    """
    The Architect's complete output for a project.

    Contains everything needed to generate skeleton files (Pass 1)
    and then implementation files (Pass 2) against those skeletons.

    This replaces FileManifestPlan.
    """
    project_name: str = Field(description="Snake_case project name, e.g. 'lead_gen_pipeline'")
    description: str = Field(description="One-paragraph project description")

    # Structure
    file_tree: list[FileTreeEntry] = Field(
        description="Every file in the project, in dependency order"
    )
    dependency_edges: list[DependencyEdge] = Field(
        default_factory=list,
        description="All file-to-file dependency edges"
    )

    # Typed interfaces (Pass 1 content)
    model_chains: list[ModelChain] = Field(
        default_factory=list,
        description="Progressive Pydantic model chains"
    )
    abc_definitions: list[ABCDefinition] = Field(
        default_factory=list,
        description="Abstract base classes"
    )
    protocol_definitions: list[ProtocolDefinition] = Field(
        default_factory=list,
        description="Protocol definitions for structural subtyping"
    )
    config_schemas: list[ConfigSchema] = Field(
        default_factory=list,
        description="Configuration/settings classes"
    )
    exception_specs: list[ExceptionSpec] = Field(
        default_factory=list,
        description="Custom exception hierarchy"
    )

    # Metadata
    entry_point: Optional[str] = Field(
        default=None,
        description="Main entry point file path"
    )
    external_dependencies: list[str] = Field(
        default_factory=list,
        description="Third-party pip packages needed (e.g. ['fastapi', 'httpx', 'pydantic'])"
    )

    # --- Derived helpers ---

    def skeleton_files(self) -> list[FileTreeEntry]:
        """Files to generate in Pass 1 (Haiku)."""
        return [f for f in self.file_tree if f.skeleton]

    def implementation_files(self) -> list[FileTreeEntry]:
        """Files to generate in Pass 2 (Sonnet)."""
        return [f for f in self.file_tree if not f.skeleton]

    def dependencies_for(self, file_path: str) -> list[DependencyEdge]:
        """Get all dependency edges where `file_path` is the source (depends on others)."""
        return [e for e in self.dependency_edges if e.source == file_path]

    def dependents_of(self, file_path: str) -> list[DependencyEdge]:
        """Get all files that depend on `file_path`."""
        return [e for e in self.dependency_edges if e.target == file_path]
