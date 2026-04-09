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
# API Contract (Move 5: Contract-first generation)
# ---------------------------------------------------------------------------

class EndpointContract(BaseModel):
    """A single API endpoint contract — the source of truth for both code and tests."""
    method: str = Field(description="HTTP method: GET, POST, PUT, DELETE, PATCH")
    path: str = Field(description="URL path, e.g. '/bookmarks/{id}'")
    description: str = Field(default="", description="What this endpoint does")
    request_model: Optional[str] = Field(default=None, description="Pydantic model name for request body")
    response_model: Optional[str] = Field(default=None, description="Pydantic model name for response")
    status_code: int = Field(default=200, description="Expected success status code")
    error_codes: list[int] = Field(default_factory=list, description="Expected error status codes, e.g. [404, 422]")


class CLIContract(BaseModel):
    """A single CLI command contract."""
    name: str = Field(description="Command name, e.g. 'add'")
    description: str = Field(default="", description="What this command does")
    arguments: list[str] = Field(default_factory=list, description="Required arguments")
    options: list[str] = Field(default_factory=list, description="Optional flags, e.g. ['--verbose', '--output FILE']")
    exit_codes: list[int] = Field(default_factory=lambda: [0], description="Expected exit codes")


class APIContract(BaseModel):
    """The complete API/CLI contract — source of truth for code and test generation.

    Move 5: Both the builder and tester reference this contract.
    - Builder generates code that implements every endpoint/command.
    - Tester generates tests that verify every endpoint/command.
    - The oracle problem dissolves because both reference the same spec.
    """
    endpoints: list[EndpointContract] = Field(default_factory=list, description="REST API endpoints")
    cli_commands: list[CLIContract] = Field(default_factory=list, description="CLI commands")
    base_url: str = Field(default="http://localhost:8000", description="Base URL for API testing")


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

    # API/CLI Contract (Move 5: source of truth for code AND tests)
    api_contract: Optional[APIContract] = Field(
        default=None,
        description="API endpoints or CLI commands — the contract both builder and tester reference"
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

    def format_contract(self) -> str:
        """Format the API/CLI contract as a readable spec for builder and tester.

        This is the single source of truth that both agents reference.
        """
        if not self.api_contract:
            return ""

        lines = ["## API CONTRACT (source of truth for code AND tests)"]

        if self.api_contract.endpoints:
            lines.append("\n### Endpoints:")
            for ep in self.api_contract.endpoints:
                req = f" ← {ep.request_model}" if ep.request_model else ""
                resp = f" → {ep.response_model}" if ep.response_model else ""
                lines.append(f"  {ep.method} {ep.path}{req}{resp} [{ep.status_code}]")
                if ep.description:
                    lines.append(f"    {ep.description}")
                if ep.error_codes:
                    lines.append(f"    Errors: {ep.error_codes}")

        if self.api_contract.cli_commands:
            lines.append("\n### CLI Commands:")
            for cmd in self.api_contract.cli_commands:
                args = " ".join(cmd.arguments)
                opts = " ".join(cmd.options)
                lines.append(f"  {cmd.name} {args} {opts}".strip())
                if cmd.description:
                    lines.append(f"    {cmd.description}")

        # Also list model contracts
        if self.model_chains:
            lines.append("\n### Data Models:")
            for chain in self.model_chains:
                for model in chain.models:
                    fields = ", ".join(f"{f.name}: {f.type_annotation}" for f in model.fields)
                    lines.append(f"  {model.name}({model.base_class}): {fields}")

        return "\n".join(lines)
