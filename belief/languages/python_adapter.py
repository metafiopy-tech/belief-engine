"""Python Language Adapter — wraps existing engine capabilities.

This adapter encapsulates all Python-specific logic that was previously
scattered across the executor, tester, builder, and skeleton builder.
No behavior changes — just consolidation into the adapter pattern.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from belief.languages import (
    Language,
    LanguageAdapter,
    VerificationResult,
    ExportedSymbol,
)

logger = logging.getLogger("belief.languages.python")


class PythonAdapter(LanguageAdapter):
    """Python language adapter — the engine's native language."""

    @property
    def language(self) -> Language:
        return Language.PYTHON

    @property
    def file_extensions(self) -> list[str]:
        return [".py"]

    @property
    def test_file_patterns(self) -> list[str]:
        return ["test_*.py", "*_test.py"]

    def scaffold_project(self, project_name: str, dependencies: list[str]) -> dict[str, str]:
        """Generate Python project scaffolding."""
        req_content = "\n".join(dependencies) + "\n" if dependencies else ""

        pyproject = f"""[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = {dependencies}
"""
        return {
            "requirements.txt": req_content,
            "pyproject.toml": pyproject,
        }

    def verify_code(self, code: str, filename: str) -> VerificationResult:
        """Verify Python code via ast.parse()."""
        try:
            ast.parse(code)
            return VerificationResult(success=True)
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                errors=[f"{filename} line {e.lineno}: {e.msg}"],
            )

    def parse_exports(self, code: str, filename: str) -> list[ExportedSymbol]:
        """Extract public classes, functions, and assignments via AST.

        This is the same logic from TesterAgent._extract_imports_and_exports(),
        now centralized in the adapter.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        exports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                # Extract method signatures for the class
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not item.name.startswith("_") or item.name == "__init__":
                            methods.append(item.name)

                sig = f"class {node.name}"
                if node.bases:
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(f"{_get_attr_name(base)}")
                    sig += f"({', '.join(bases)})"
                if methods:
                    sig += f"  # methods: {', '.join(methods[:5])}"

                exports.append(ExportedSymbol(
                    name=node.name,
                    kind="class",
                    file_path=filename,
                    line=node.lineno,
                    signature=sig,
                ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    # Build signature
                    params = []
                    for arg in node.args.args:
                        param = arg.arg
                        if arg.annotation:
                            param += f": {_unparse_annotation(arg.annotation)}"
                        params.append(param)

                    ret = ""
                    if node.returns:
                        ret = f" -> {_unparse_annotation(node.returns)}"

                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    sig = f"{prefix} {node.name}({', '.join(params)}){ret}"

                    exports.append(ExportedSymbol(
                        name=node.name,
                        kind="function",
                        file_path=filename,
                        line=node.lineno,
                        signature=sig,
                    ))

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        # Include app-like assignments and constants
                        if target.id[0].isupper() or target.id in ("app", "router"):
                            exports.append(ExportedSymbol(
                                name=target.id,
                                kind="variable",
                                file_path=filename,
                                line=node.lineno,
                                signature=f"{target.id} = ...",
                            ))

        return exports

    def get_system_prompt_additions(self) -> str:
        return (
            "Write Python 3.11+ code. Use type hints on all function signatures. "
            "Use Pydantic v2 BaseModel for data models. "
            "Handle errors with custom exception classes."
        )

    def get_import_statement(self, symbol: str, from_module: str) -> str:
        module = from_module.replace("/", ".").replace(".py", "")
        return f"from {module} import {symbol}"


def _get_attr_name(node: ast.Attribute) -> str:
    """Get dotted name from an Attribute node."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _unparse_annotation(node: ast.expr) -> str:
    """Convert an AST annotation node back to a string."""
    try:
        return ast.unparse(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return "Any"
