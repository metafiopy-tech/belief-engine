"""Cross-Language Type Pipeline — Tier 6.

Converts Pydantic models to TypeScript interfaces and Go structs
via JSON Schema as the intermediate representation.

Pipeline: Pydantic.model_json_schema() → JSON Schema → target language types

This ensures shared models in multi-service projects are consistent
across all languages.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("belief.languages.types")


# ── JSON Schema → TypeScript ─────────────────────────────────────────────────

_TS_TYPE_MAP = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "array": "any[]",
    "object": "Record<string, any>",
}


def json_schema_to_typescript(schema: dict[str, Any], name: str = "") -> str:
    """Convert a JSON Schema to a TypeScript interface.

    Handles:
    - Simple types (string, number, boolean)
    - Objects with properties → interfaces
    - Arrays with items → typed arrays
    - $ref references to definitions
    - Optional fields (not in required)
    - Enums → union types
    - allOf/anyOf for inheritance
    """
    lines = []
    defs = schema.get("$defs", schema.get("definitions", {}))

    # Process definitions first
    for def_name, def_schema in defs.items():
        lines.append(_schema_to_interface(def_name, def_schema, defs))
        lines.append("")

    # Process root schema
    root_name = name or schema.get("title", "Root")
    if schema.get("type") == "object" or "properties" in schema:
        lines.append(_schema_to_interface(root_name, schema, defs))

    return "\n".join(lines)


def _schema_to_interface(name: str, schema: dict, defs: dict) -> str:
    """Convert a single schema object to a TypeScript interface."""
    if "enum" in schema:
        values = " | ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in schema["enum"])
        return f"export type {name} = {values};"

    if schema.get("type") != "object" and "properties" not in schema:
        ts_type = _resolve_type(schema, defs)
        return f"export type {name} = {ts_type};"

    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    lines = [f"export interface {name} {{"]
    for prop_name, prop_schema in props.items():
        ts_type = _resolve_type(prop_schema, defs)
        optional = "" if prop_name in required else "?"
        description = prop_schema.get("description", "")
        comment = f"  // {description}" if description else ""
        lines.append(f"  {prop_name}{optional}: {ts_type};{comment}")
    lines.append("}")

    return "\n".join(lines)


def _resolve_type(schema: dict, defs: dict) -> str:
    """Resolve a JSON Schema type to a TypeScript type."""
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return ref

    if "anyOf" in schema:
        types = [_resolve_type(s, defs) for s in schema["anyOf"]]
        return " | ".join(types)

    if "allOf" in schema:
        types = [_resolve_type(s, defs) for s in schema["allOf"]]
        return " & ".join(types)

    if "enum" in schema:
        return " | ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in schema["enum"])

    schema_type = schema.get("type", "any")

    if schema_type == "array":
        items = schema.get("items", {})
        item_type = _resolve_type(items, defs) if items else "any"
        return f"{item_type}[]"

    if schema_type == "object":
        if "properties" in schema:
            # Inline object
            props = []
            for k, v in schema["properties"].items():
                t = _resolve_type(v, defs)
                props.append(f"{k}: {t}")
            return "{ " + "; ".join(props) + " }"
        return "Record<string, any>"

    if isinstance(schema_type, list):
        return " | ".join(_TS_TYPE_MAP.get(t, "any") for t in schema_type)

    return _TS_TYPE_MAP.get(schema_type, "any")


# ── JSON Schema → Go Struct ──────────────────────────────────────────────────

_GO_TYPE_MAP = {
    "string": "string",
    "integer": "int",
    "number": "float64",
    "boolean": "bool",
    "null": "interface{}",
    "array": "[]interface{}",
    "object": "map[string]interface{}",
}


def json_schema_to_go(schema: dict[str, Any], package: str = "models") -> str:
    """Convert a JSON Schema to Go struct definitions."""
    lines = [f"package {package}", "", ""]
    defs = schema.get("$defs", schema.get("definitions", {}))

    for def_name, def_schema in defs.items():
        lines.append(_schema_to_struct(def_name, def_schema, defs))
        lines.append("")

    root_name = schema.get("title", "Root")
    if schema.get("type") == "object" or "properties" in schema:
        lines.append(_schema_to_struct(root_name, schema, defs))

    return "\n".join(lines)


def _schema_to_struct(name: str, schema: dict, defs: dict) -> str:
    """Convert a single schema object to a Go struct."""
    if "enum" in schema:
        base_type = _GO_TYPE_MAP.get(schema.get("type", "string"), "string")
        lines = [f"type {name} {base_type}", "", "const ("]
        for i, v in enumerate(schema["enum"]):
            const_name = f"{name}{str(v).title().replace(' ', '')}"
            if isinstance(v, str):
                lines.append(f'\t{const_name} {name} = "{v}"')
            else:
                lines.append(f"\t{const_name} {name} = {v}")
        lines.append(")")
        return "\n".join(lines)

    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    lines = [f"type {name} struct {{"]
    for prop_name, prop_schema in props.items():
        go_type = _resolve_go_type(prop_schema, defs)
        field_name = _to_go_name(prop_name)
        json_tag = f'`json:"{prop_name}"`'
        if prop_name not in required:
            go_type = f"*{go_type}"
        lines.append(f"\t{field_name} {go_type} {json_tag}")
    lines.append("}")

    return "\n".join(lines)


def _resolve_go_type(schema: dict, defs: dict) -> str:
    """Resolve a JSON Schema type to a Go type."""
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]

    if "anyOf" in schema:
        return "interface{}"

    schema_type = schema.get("type", "interface{}")

    if schema_type == "array":
        items = schema.get("items", {})
        item_type = _resolve_go_type(items, defs) if items else "interface{}"
        return f"[]{item_type}"

    if schema_type == "object":
        return "map[string]interface{}"

    return _GO_TYPE_MAP.get(schema_type, "interface{}")


def _to_go_name(name: str) -> str:
    """Convert snake_case or camelCase to PascalCase for Go."""
    parts = re.split(r'[_\-]', name)
    return "".join(p.capitalize() for p in parts)


# ── Pydantic → Multi-Language ────────────────────────────────────────────────

def pydantic_to_typescript(model_code: str) -> str:
    """Convert Pydantic model Python code to TypeScript interfaces.

    Parses the Python code, extracts Pydantic models, generates JSON Schema,
    then converts to TypeScript.
    """
    import ast

    try:
        tree = ast.parse(model_code)
    except SyntaxError:
        return "// Could not parse Python model code"

    # Find all classes that inherit from BaseModel
    interfaces = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_pydantic = any(
                (isinstance(b, ast.Name) and b.id in ("BaseModel", "BaseSettings"))
                or (isinstance(b, ast.Attribute) and b.attr in ("BaseModel", "BaseSettings"))
                for b in node.bases
            )
            if not is_pydantic:
                continue

            fields = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_name = item.target.id
                    type_str = ast.unparse(item.annotation) if item.annotation else "any"
                    ts_type = _python_type_to_ts(type_str)
                    optional = item.value is not None  # Has default = optional
                    opt = "?" if optional else ""
                    fields.append(f"  {field_name}{opt}: {ts_type};")

            if fields:
                interfaces.append(f"export interface {node.name} {{\n" + "\n".join(fields) + "\n}")

    return "\n\n".join(interfaces) if interfaces else "// No Pydantic models found"


def _python_type_to_ts(type_str: str) -> str:
    """Convert a Python type annotation string to TypeScript."""
    mapping = {
        "str": "string",
        "int": "number",
        "float": "number",
        "bool": "boolean",
        "None": "null",
        "Any": "any",
        "dict": "Record<string, any>",
        "Dict": "Record<string, any>",
    }

    # Handle Optional[X] → X | null
    opt = re.match(r'Optional\[(.+)\]', type_str)
    if opt:
        inner = _python_type_to_ts(opt.group(1))
        return f"{inner} | null"

    # Handle list[X] → X[]
    lst = re.match(r'(?:list|List)\[(.+)\]', type_str)
    if lst:
        inner = _python_type_to_ts(lst.group(1))
        return f"{inner}[]"

    return mapping.get(type_str, type_str)
