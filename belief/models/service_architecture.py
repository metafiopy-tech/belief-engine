"""
Service Architecture — Tier 5 Multi-Service Project Descriptor

For projects with complexity ≥ 4, the architect can output a ServiceArchitecture
instead of a single SkeletonArtifact. This defines:
  - Multiple services, each independently buildable
  - Shared models as the single source of truth
  - Inter-service communication contracts (HTTP, events)
  - Infrastructure (Docker Compose, gateway)

Each service generates its own SkeletonArtifact-like file tree, but they share
models and respect each other's API contracts.

Source: Research report — ServiceArchitecture YAML descriptor pattern
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CommunicationType(str, Enum):
    HTTP = "http"
    GRPC = "grpc"
    EVENT = "event"
    SHARED_DB = "shared_db"


class RouteSpec(BaseModel):
    """A single API route in a service."""

    path: str = Field(description="e.g., '/users/{id}'")
    method: str = Field(default="GET", description="HTTP method")
    request_model: Optional[str] = Field(
        default=None, description="Pydantic model for request body"
    )
    response_model: Optional[str] = Field(default=None, description="Pydantic model for response")
    description: str = ""


class EventSpec(BaseModel):
    """An event published or subscribed to by a service."""

    event_name: str = Field(description="e.g., 'user.created'")
    schema_model: Optional[str] = Field(
        default=None, description="Pydantic model for event payload"
    )
    description: str = ""


class ServiceDependency(BaseModel):
    """An inter-service dependency — service A calls service B."""

    target_service: str = Field(description="Name of the service being called")
    communication: CommunicationType = CommunicationType.HTTP
    calls: list[RouteSpec] = Field(
        default_factory=list, description="Specific routes this service calls on the target"
    )
    purpose: str = ""


class ServiceSpec(BaseModel):
    """Specification for a single service in a multi-service architecture."""

    name: str = Field(description="e.g., 'user-service'")
    package: str = Field(description="Package/directory name, e.g., 'user_service'")
    description: str = ""
    port: int = Field(default=8000, description="Port this service listens on")
    framework: str = Field(default="fastapi", description="Web framework")
    language: str = Field(
        default="python", description="Programming language: python, typescript, go"
    )

    # Files in this service
    files: list[str] = Field(
        default_factory=list, description="Relative file paths within the service package"
    )

    # API contract
    routes: list[RouteSpec] = Field(default_factory=list)

    # Dependencies on other services
    depends_on: list[ServiceDependency] = Field(default_factory=list)

    # Events
    publishes: list[EventSpec] = Field(default_factory=list)
    subscribes: list[EventSpec] = Field(default_factory=list)

    # Database
    database: Optional[str] = Field(default=None, description="e.g., 'postgresql', 'sqlite'")

    @property
    def entry_point(self) -> str:
        if self.language == "typescript":
            return f"{self.package}/src/index.ts"
        if self.language == "go":
            return f"{self.package}/main.go"
        return f"{self.package}/server.py"

    @property
    def all_file_paths(self) -> list[str]:
        """All file paths prefixed with the package name."""
        return [f"{self.package}/{f}" for f in self.files]


class SharedModelSpec(BaseModel):
    """A model shared across services — the single source of truth."""

    name: str = Field(description="Class name, e.g., 'User'")
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Field name → type annotation, e.g., {'id': 'uuid', 'email': 'str'}",
    )
    description: str = ""


class ServiceArchitecture(BaseModel):
    """Multi-service architecture descriptor for Tier 5 projects.

    This is the architect's output when complexity ≥ 4 and the project
    involves multiple services that communicate with each other.

    Each service is independently buildable from its own spec + shared_models.
    The descriptor generates Docker Compose, API contracts, and inter-service
    client code.
    """

    system_name: str = Field(description="e.g., 'task-management-platform'")
    description: str = ""

    # Services
    services: list[ServiceSpec] = Field(description="Individual services")

    # Shared across all services
    shared_models: list[SharedModelSpec] = Field(
        default_factory=list,
        description="Models shared by multiple services (single source of truth)",
    )
    shared_package: str = Field(default="shared", description="Package name for shared modules")

    # Infrastructure
    gateway: Optional[ServiceSpec] = Field(
        default=None, description="API gateway service (routes requests to backend services)"
    )

    @property
    def all_service_names(self) -> list[str]:
        names = [s.name for s in self.services]
        if self.gateway:
            names.append(self.gateway.name)
        return names

    @property
    def all_packages(self) -> list[str]:
        packages = [s.package for s in self.services]
        if self.shared_models:
            packages.append(self.shared_package)
        if self.gateway:
            packages.append(self.gateway.package)
        return packages

    def get_service(self, name: str) -> Optional[ServiceSpec]:
        for s in self.services:
            if s.name == name:
                return s
        if self.gateway and self.gateway.name == name:
            return self.gateway
        return None

    def generate_file_tree(self) -> dict[str, str]:
        """Generate a complete file tree for the multi-service project.

        Returns a dict of filepath → description for all files across
        all services + shared modules + infrastructure.
        """
        tree: dict[str, str] = {}

        # Shared models
        if self.shared_models:
            tree[f"{self.shared_package}/__init__.py"] = "Shared package init"
            tree[f"{self.shared_package}/models.py"] = "Shared data models"
            tree[f"{self.shared_package}/schemas.py"] = "Shared Pydantic schemas"
            tree[f"{self.shared_package}/config.py"] = "Shared configuration"
            tree[f"{self.shared_package}/events.py"] = (
                "Event schemas for inter-service communication"
            )

        # Per-service files
        for service in self.services:
            pkg = service.package

            if service.language == "typescript":
                # TypeScript service structure
                tree[f"{pkg}/package.json"] = f"{service.name} npm package config"
                tree[f"{pkg}/tsconfig.json"] = f"{service.name} TypeScript config"
                tree[f"{pkg}/src/index.ts"] = f"{service.name} entry point"
                tree[f"{pkg}/src/types.ts"] = f"{service.name} TypeScript interfaces"
                tree[f"{pkg}/src/routes.ts"] = f"{service.name} route handlers"
                tree[f"{pkg}/src/service.ts"] = f"{service.name} business logic"
                if service.database:
                    tree[f"{pkg}/src/database.ts"] = f"{service.name} database setup"
                if service.depends_on:
                    tree[f"{pkg}/src/clients.ts"] = "HTTP clients for calling other services"
                tree[f"{pkg}/Dockerfile"] = f"{service.name} Docker build"
            else:
                # Python service structure (default)
                tree[f"{pkg}/__init__.py"] = f"{service.name} package init"
                tree[f"{pkg}/models.py"] = f"{service.name} data models"
                tree[f"{pkg}/schemas.py"] = f"{service.name} request/response schemas"
                tree[f"{pkg}/routes.py"] = f"{service.name} API route handlers"
                tree[f"{pkg}/service.py"] = f"{service.name} business logic"
                tree[f"{pkg}/server.py"] = f"{service.name} FastAPI application"
                if service.database:
                    tree[f"{pkg}/database.py"] = f"{service.name} database setup"
                if service.depends_on:
                    tree[f"{pkg}/clients.py"] = "HTTP clients for calling other services"

            # Service-specific files
            for f in service.files:
                full = f"{pkg}/{f}"
                if full not in tree:
                    tree[full] = f"{service.name} component"

        # Gateway
        if self.gateway:
            pkg = self.gateway.package
            tree[f"{pkg}/__init__.py"] = "Gateway package init"
            tree[f"{pkg}/main.py"] = "API gateway entry point"
            tree[f"{pkg}/middleware.py"] = "Gateway middleware (auth, rate limiting)"
            tree[f"{pkg}/routes.py"] = "Gateway route proxying"

        # Infrastructure
        tree["docker-compose.yml"] = "Docker Compose orchestration"
        tree["requirements.txt"] = "Python dependencies"
        tree[".env.example"] = "Environment variable template"

        return tree

    def generate_docker_compose(self) -> str:
        """Generate Docker Compose YAML from the architecture.

        Deterministic template — no LLM involved. This ensures correct
        port mappings, service names, and dependency ordering.
        """
        lines = ["version: '3.8'", "", "services:"]

        for service in self.services:
            lines.append(f"  {service.name}:")
            lines.append(f"    build: ./{service.package}")
            lines.append("    ports:")
            lines.append(f'      - "{service.port}:{service.port}"')

            # Dependencies
            dep_names = [d.target_service for d in service.depends_on]
            if dep_names:
                lines.append("    depends_on:")
                for dep in dep_names:
                    lines.append(f"      - {dep}")

            # Environment
            lines.append("    environment:")
            lines.append(f"      - PORT={service.port}")
            if service.database:
                lines.append("      - DATABASE_URL=${DATABASE_URL:-sqlite:///data.db}")
            for dep in service.depends_on:
                target = self.get_service(dep.target_service)
                if target:
                    env_name = dep.target_service.upper().replace("-", "_") + "_URL"
                    lines.append(f"      - {env_name}=http://{dep.target_service}:{target.port}")

            lines.append("")

        # Gateway
        if self.gateway:
            lines.append(f"  {self.gateway.name}:")
            lines.append(f"    build: ./{self.gateway.package}")
            lines.append("    ports:")
            lines.append(f'      - "{self.gateway.port}:{self.gateway.port}"')
            lines.append("    depends_on:")
            for s in self.services:
                lines.append(f"      - {s.name}")
            lines.append("    environment:")
            lines.append(f"      - PORT={self.gateway.port}")
            for s in self.services:
                env_name = s.name.upper().replace("-", "_") + "_URL"
                lines.append(f"      - {env_name}=http://{s.name}:{s.port}")
            lines.append("")

        return "\n".join(lines)


def service_architecture_to_file_manifest(arch: ServiceArchitecture):
    """Convert ServiceArchitecture to a FileManifestPlan for backward compat.

    This lets the existing builder + executor pipeline handle multi-service
    projects by treating them as a flat file tree with correct dependencies.
    """
    from belief.models.artifacts import FileManifest, FileManifestPlan

    files = []
    file_tree = arch.generate_file_tree()

    for filepath, description in file_tree.items():
        is_entry = filepath.endswith("/server.py") or filepath.endswith("/main.py")
        deps = []

        # Shared modules depend on nothing
        if filepath.startswith(arch.shared_package + "/"):
            deps = []
        else:
            # Service files depend on shared package
            if arch.shared_models:
                deps.append(f"{arch.shared_package}/models.py")

            # Route files depend on schemas and service
            pkg = filepath.split("/")[0] if "/" in filepath else ""
            if filepath.endswith("/routes.py"):
                deps.extend([f"{pkg}/schemas.py", f"{pkg}/service.py"])
            elif filepath.endswith("/server.py"):
                deps.extend([f"{pkg}/routes.py"])
            elif filepath.endswith("/service.py"):
                deps.extend([f"{pkg}/models.py"])

        files.append(
            FileManifest(
                filename=filepath,
                purpose=description,
                public_interface="",
                depends_on=deps,
                is_entry_point=is_entry,
            )
        )

    # Add docker-compose as a pre-generated file
    entry = (
        "gateway/main.py"
        if arch.gateway
        else (f"{arch.services[0].package}/server.py" if arch.services else "main.py")
    )

    return FileManifestPlan(
        files=files,
        architecture_notes=f"Multi-service: {arch.system_name} — {', '.join(arch.all_service_names)}",
        entry_point=entry,
    )
