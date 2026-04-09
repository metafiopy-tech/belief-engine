"""Contract Agent — Generate API contracts before code.

Takes a ServiceArchitecture from the Architect and generates:
1. OpenAPI 3.1 YAML per service (paths, schemas, operations)
2. Shared schemas YAML (models used across services)
3. Docker Compose YAML for multi-service orchestration

These contracts become the single source of truth — both the builder
and the integration tester reference them.

This agent is deterministic (no LLM calls) — the ServiceArchitecture
has enough structure to generate contracts programmatically.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("belief.agents.contract_agent")


async def contract_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate contracts from ServiceArchitecture.

    Reads: state["service_architecture"]
    Writes: state["openapi_specs"], state["shared_schemas"], state["docker_compose"]
    """
    result = dict(state)

    architecture = state.get("service_architecture")
    if not architecture:
        logger.info("Contract agent: no service_architecture in state, skipping")
        return result

    # Hydrate if dict
    if isinstance(architecture, dict):
        try:
            from belief.models.service_architecture import ServiceArchitecture
            architecture = ServiceArchitecture.model_validate(architecture)
        except Exception as e:
            logger.warning(f"Contract agent: failed to parse architecture: {e}")
            return result

    try:
        # 1. Generate OpenAPI specs per service
        from belief.models.openapi import generate_openapi_specs, generate_shared_schemas_yaml
        openapi_specs = generate_openapi_specs(architecture)
        result["openapi_specs"] = openapi_specs

        # 2. Generate shared schemas
        shared_schemas = generate_shared_schemas_yaml(architecture)
        result["shared_schemas"] = shared_schemas

        # 3. Generate docker-compose.yml
        compose = _generate_compose(architecture)
        result["docker_compose"] = compose

        logger.info(
            f"Contract agent: generated {len(openapi_specs)} OpenAPI specs, "
            f"shared schemas, and docker-compose.yml"
        )

    except Exception as e:
        logger.warning(f"Contract agent failed: {e}")

    return result


def _generate_compose(architecture) -> str:
    """Generate docker-compose.yml from ServiceArchitecture."""
    import yaml

    services = {}

    for svc in architecture.services:
        service_def = {
            "build": {
                "context": f"./{svc.package}",
                "dockerfile": "Dockerfile",
            },
            "ports": [f"{svc.port}:{svc.port}"],
            "environment": [
                f"PORT={svc.port}",
            ],
        }

        # Add health check
        service_def["healthcheck"] = {
            "test": ["CMD", "curl", "-f", f"http://localhost:{svc.port}/health"],
            "interval": "10s",
            "timeout": "5s",
            "retries": 3,
            "start_period": "10s",
        }

        # Add database dependency
        if svc.database and svc.database != "sqlite":
            service_def["depends_on"] = {
                "db": {"condition": "service_healthy"},
            }
            service_def["environment"].append(
                f"DATABASE_URL={_db_url(svc.database, svc.package)}"
            )

        # Add inter-service dependencies
        for dep in svc.depends_on:
            deps = service_def.setdefault("depends_on", {})
            target_svc = architecture.get_service(dep.target_service)
            if target_svc:
                deps[target_svc.package] = {"condition": "service_healthy"}
                service_def["environment"].append(
                    f"{dep.target_service.upper().replace('-', '_')}_URL="
                    f"http://{target_svc.package}:{target_svc.port}"
                )

        services[svc.package] = service_def

    # Add database service if needed
    db_services = {s.database for s in architecture.services if s.database and s.database != "sqlite"}
    if "postgresql" in db_services or "postgres" in db_services:
        services["db"] = {
            "image": "postgres:16-alpine",
            "environment": [
                "POSTGRES_USER=belief",
                "POSTGRES_PASSWORD=belief",
                "POSTGRES_DB=app",
            ],
            "ports": ["5432:5432"],
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U belief"],
                "interval": "5s",
                "timeout": "3s",
                "retries": 5,
            },
        }
    if "redis" in db_services or any(s.publishes or s.subscribes for s in architecture.services):
        services["redis"] = {
            "image": "redis:7-alpine",
            "ports": ["6379:6379"],
            "healthcheck": {
                "test": ["CMD", "redis-cli", "ping"],
                "interval": "5s",
                "timeout": "3s",
                "retries": 5,
            },
        }

    compose = {
        "version": "3.8",
        "services": services,
    }

    return yaml.dump(compose, default_flow_style=False, sort_keys=False)


def _db_url(db_type: str, package: str) -> str:
    """Generate a database URL for a service."""
    if db_type in ("postgresql", "postgres"):
        return "postgresql://belief:belief@db:5432/app"
    if db_type == "mysql":
        return "mysql://belief:belief@db:3306/app"
    return f"sqlite:///./{package}.db"
