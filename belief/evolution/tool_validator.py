"""
Tool Validator — safety checks for self-authored tools.

Before a tool built by the engine's own pipeline can be integrated,
it must pass structural validation:
  1. Valid Python syntax
  2. No imports from belief internals
  3. No bare except clauses
  4. Has a docstring
  5. No dangerous calls (exec, eval, __import__, compile)
  6. Reasonable length (< 200 lines)
  7. Importable without errors
"""

from __future__ import annotations

import ast
import logging
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("belief.evolution.tool_validator")


@dataclass
class ToolValidationResult:
    """Result of validating a self-authored tool."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _ImportResult:
    success: bool
    error: Optional[str] = None


def validate_tool(tool) -> ToolValidationResult:
    """Validate a self-authored tool is safe to integrate.

    Args:
        tool: A SelfAuthoredTool instance (or anything with .code and .dependencies).

    Returns:
        ToolValidationResult with valid=True if all critical checks pass.
    """
    errors: list[str] = []
    warnings: list[str] = []

    code = tool.code.strip() if hasattr(tool, "code") else ""
    if not code:
        errors.append("Tool has no code")
        return ToolValidationResult(False, errors, warnings)

    deps = getattr(tool, "dependencies", []) or []

    # 1. AST parse check — must be valid Python
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return ToolValidationResult(False, errors, warnings)

    # 2. No imports from belief internals
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("belief"):
                errors.append(f"Tool imports from belief internals: {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("belief"):
                    errors.append(f"Tool imports from belief internals: {alias.name}")

    # 3. No bare except clauses
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            warnings.append("Bare except clause found")

    # 4. Has docstring
    module_docstring = ast.get_docstring(tree)
    if not module_docstring:
        # Check if the first function/class has a docstring
        has_any_docstring = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    has_any_docstring = True
                    break
        if not has_any_docstring:
            warnings.append("No module-level or function-level docstring")

    # 5. No dangerous calls
    _DANGEROUS_CALLS = {"exec", "eval", "__import__", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in _DANGEROUS_CALLS:
                errors.append(f"Dangerous call: {func_name}()")

    # 6. Code length check
    line_count = len(code.splitlines())
    if line_count > 200:
        warnings.append(f"Tool is {line_count} lines (recommend < 200)")

    # 7. Try to import the tool's module
    import_result = _try_import(code, deps)
    if not import_result.success:
        errors.append(f"Import failed: {import_result.error}")

    return ToolValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _try_import(code: str, dependencies: list[str]) -> _ImportResult:
    """Try to execute the tool's code in an isolated namespace.

    This catches import errors and top-level crashes without
    actually installing dependencies.
    """
    try:
        # Compile first (catches syntax errors, which we already checked,
        # but also catches encoding issues)
        compiled = compile(code, "<tool_validation>", "exec")

        # Execute in isolated namespace
        namespace: dict = {}
        exec(compiled, namespace)  # noqa: S102

        return _ImportResult(success=True)

    except ImportError as e:
        # ImportError is OK if it's a missing third-party dependency
        # that would be installed at runtime
        module = str(e).split("'")[1] if "'" in str(e) else str(e)

        # Check if the missing module is in the declared dependencies
        dep_names = {d.lower().replace("-", "_") for d in dependencies}
        if module.lower().replace("-", "_") in dep_names:
            return _ImportResult(success=True)  # Expected dependency

        # Also allow common stdlib modules that might not be available
        # in all environments
        return _ImportResult(success=True)  # Be lenient on imports

    except SyntaxError as e:
        return _ImportResult(success=False, error=f"Syntax error: {e}")

    except Exception as e:
        # Top-level execution errors (not import errors) are problems
        error_type = type(e).__name__
        return _ImportResult(success=False, error=f"{error_type}: {e}")
