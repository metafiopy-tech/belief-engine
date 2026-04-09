"""OpenAPI 3.1 Spec Generator — Contract-First Multi-Service Generation.

Converts a ServiceArchitecture's ServiceSpec + SharedModelSpec into valid
OpenAPI 3.1 YAML specs. These specs are generated BEFORE code exists and
serve as the single source of truth constraining both code generation and
contract testing.

Uses openapi-pydantic for Pydantic-native OpenAPI modeling with automatic
$ref resolution and JSON Schema 2020-12 compatibility (same draft Pydantic v2 uses).

Usage:
    from belief.models.openapi import generate_openapi_specs
    specs = generate_openapi_specs(service_architecture)
    # specs = {"user-service": "openapi: 3.1.0\\n...", "order-service": "..."}
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger("belief.models.openapi")


def generate_openapi_specs(
    architecture,  # ServiceArchitecture
) -> dict[str, str]:
    """Generate OpenAPI 3.1 YAML specs for each service in the architecture.

    Returns a dict of {service_name: openapi_yaml_string}.
    Each spec includes paths from the service's routes, shared model schemas,
    and inter-service dependency documentation.
    """
    specs = {}
    for service in architecture.services:
        spec = _build_spec_for_service(service, architecture)
        specs[service.name] = yaml.dump(spec, default_flow_style=False, sort_keys=False)
        logger.info(f"OpenAPI: generated spec for {service.name} ({len(service.routes)} routes)")

    return specs


def _build_spec_for_service(service, architecture) -> dict[str, Any]:
    """Build a complete OpenAPI 3.1.0 spec dict for one service."""
    spec: dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {
            "title": f"{service.name} API",
            "description": service.description or f"API for {service.name}",
            "version": "1.0.0",
        },
        "servers": [
            {"url": f"http://localhost:{service.port}", "description": "Local"},
        ],
        "paths": {},
        "components": {"schemas": {}},
    }

    # Build paths from routes
    for route in service.routes:
        path_item = spec["paths"].setdefault(route.path, {})
        method = route.method.lower()

        operation: dict[str, Any] = {
            "summary": route.description or f"{route.method} {route.path}",
            "operationId": _make_operation_id(route),
            "responses": {
                "200": {"description": "Successful response"},
            },
        }

        # Add request body if specified
        if route.request_model and route.method.upper() in ("POST", "PUT", "PATCH"):
            schema_name = route.request_model
            operation["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{schema_name}"},
                    },
                },
            }

        # Add response schema if specified
        if route.response_model:
            schema_name = route.response_model
            operation["responses"]["200"]["content"] = {
                "application/json": {
                    "schema": {"$ref": f"#/components/schemas/{schema_name}"},
                },
            }

        # Add path parameters
        path_params = _extract_path_params(route.path)
        if path_params:
            operation["parameters"] = [
                {
                    "name": param,
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
                for param in path_params
            ]

        path_item[method] = operation

    # Build component schemas from shared models
    for model in architecture.shared_models:
        schema = _model_to_schema(model)
        spec["components"]["schemas"][model.name] = schema

    # Add stub schemas for request/response models not in shared models
    shared_names = {m.name for m in architecture.shared_models}
    for route in service.routes:
        for model_name in (route.request_model, route.response_model):
            if model_name and model_name not in shared_names and model_name not in spec["components"]["schemas"]:
                spec["components"]["schemas"][model_name] = {
                    "type": "object",
                    "description": f"Schema for {model_name}",
                    "properties": {},
                }

    return spec


def _model_to_schema(model) -> dict[str, Any]:
    """Convert a SharedModelSpec to a JSON Schema object."""
    properties = {}
    for field_name, field_type in model.fields.items():
        properties[field_name] = _type_to_json_schema(field_type)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if model.description:
        schema["description"] = model.description

    # All fields with non-optional types are required
    required = [
        name for name, ftype in model.fields.items()
        if "optional" not in ftype.lower() and "none" not in ftype.lower()
    ]
    if required:
        schema["required"] = required

    return schema


def _type_to_json_schema(python_type: str) -> dict[str, Any]:
    """Convert a Python type annotation string to JSON Schema."""
    t = python_type.strip().lower()

    # Handle Optional
    if t.startswith("optional["):
        inner = python_type.strip()[9:-1]
        inner_schema = _type_to_json_schema(inner)
        return {"anyOf": [inner_schema, {"type": "null"}]}

    # Handle list
    if t.startswith("list["):
        inner = python_type.strip()[5:-1]
        return {"type": "array", "items": _type_to_json_schema(inner)}

    # Handle dict
    if t.startswith("dict["):
        return {"type": "object"}

    # Primitives
    type_map = {
        "str": {"type": "string"},
        "string": {"type": "string"},
        "int": {"type": "integer"},
        "integer": {"type": "integer"},
        "float": {"type": "number"},
        "number": {"type": "number"},
        "bool": {"type": "boolean"},
        "boolean": {"type": "boolean"},
        "uuid": {"type": "string", "format": "uuid"},
        "datetime": {"type": "string", "format": "date-time"},
        "date": {"type": "string", "format": "date"},
        "bytes": {"type": "string", "format": "binary"},
    }

    if t in type_map:
        return type_map[t]

    # Reference to another model
    return {"$ref": f"#/components/schemas/{python_type.strip()}"}


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from an OpenAPI path template."""
    import re
    return re.findall(r'\{(\w+)\}', path)


def _make_operation_id(route) -> str:
    """Generate a unique operation ID from method + path."""
    method = route.method.lower()
    path = route.path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
    return f"{method}_{path}" if path else method


def generate_shared_schemas_yaml(architecture) -> str:
    """Generate a standalone YAML file with just the shared schemas.

    Used as input to the builder for generating shared model code.
    """
    schemas = {}
    for model in architecture.shared_models:
        schemas[model.name] = _model_to_schema(model)

    doc = {
        "description": f"Shared schemas for {architecture.system_name}",
        "schemas": schemas,
    }
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)
