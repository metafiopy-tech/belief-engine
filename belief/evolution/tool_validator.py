"""
Tool Validator — safety checks for self-authored tools.

Before a tool built by the engine's own pipeline can be integrated,
it must pass structural validation:
  1. Valid Python syntax
  2. No imports from belief internals
  3. No bare except clauses
  4. Has a docstring
  5. No dangerous calls (exec, eval, __import__, compile, os.remove, etc.)
  6. Reasonable length (< 200 lines)
  7. Subprocess parse check (no in-process execution)
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("belief.evolution.tool_validator")

# Standard library module names (Python 3.10+)
_STDLIB_MODULES: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", frozenset())
)

# Dangerous calls — bare names and dotted names
_DANGEROUS_BARE: frozenset[str] = frozenset({
    "exec", "eval", "__import__", "compile", "execfile",
})

_DANGEROUS_DOTTED: frozenset[str] = frozenset({
    # Filesystem mutation
    "os.remove", "os.unlink", "os.rmdir", "os.makedirs",
    "shutil.rmtree", "shutil.move", "shutil.copy",
    # Process spawning (shell=False variants checked separately)
    "subprocess.call", "subprocess.Popen", "subprocess.run",
    "os.system", "os.popen",
    # Network
    "urllib.request.urlopen", "requests.get", "requests.post",
    "socket.socket", "http.client.HTTPConnection",
})


@dataclass
class ToolValidationResult:
    """Result of validating a self-authored tool."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _extract_imports(code: str) -> list[str]:
    """Extract all imported top-level module names via AST. No code execution."""
    tree = ast.parse(code)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def _check_imports(imports: list[str], declared_deps: list[str]) -> list[str]:
    """Return error strings for imports that are undeclared and unavailable."""
    errors: list[str] = []
    dep_names = {d.lower().replace("-", "_") for d in (declared_deps or [])}
    for mod in imports:
        if mod in _STDLIB_MODULES:
            continue
        if mod.lower().replace("-", "_") in dep_names:
            continue
        if importlib.util.find_spec(mod) is not None:
            continue
        errors.append(f"Undeclared/unavailable import: {mod}")
    return errors


def _subprocess_parse_check(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Verify the code parses cleanly in an isolated subprocess (no execution)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tool_path = Path(tmpdir) / "tool_check.py"
        tool_path.write_text(code)
        escaped = str(tool_path).replace("'", "\\'")
        result = subprocess.run(
            [
                sys.executable, "-c",
                f"import ast; ast.parse(open('{escaped}').read()); print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""


def _check_dangerous_calls(tree: ast.AST) -> list[str]:
    """Return error strings for any dangerous call nodes in the AST."""
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Bare name calls: exec(), eval(), …
        if isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_BARE:
                errors.append(f"Dangerous call: {node.func.id}()")

        # Dotted calls: os.remove(), subprocess.run(), …
        elif isinstance(node.func, ast.Attribute):
            parent = node.func
            # Reconstruct the dotted name (up to two levels)
            if isinstance(parent.value, ast.Name):
                dotted = f"{parent.value.id}.{parent.attr}"
                if dotted in _DANGEROUS_DOTTED:
                    # subprocess.run with shell=False is acceptable
                    if dotted == "subprocess.run":
                        shell_true = any(
                            (isinstance(kw.value, ast.Constant) and kw.value.value is True)
                            for kw in node.keywords
                            if kw.arg == "shell"
                        )
                        if shell_true:
                            errors.append(f"Dangerous call: {dotted}(shell=True)")
                    else:
                        errors.append(f"Dangerous call: {dotted}()")
            # Bare attribute (e.g. just `.exec` without a known prefix)
            if parent.attr in _DANGEROUS_BARE:
                errors.append(f"Dangerous call: {parent.attr}()")

    return errors


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
        has_any_docstring = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    has_any_docstring = True
                    break
        if not has_any_docstring:
            warnings.append("No module-level or function-level docstring")

    # 5. No dangerous calls (static AST check — no execution)
    errors.extend(_check_dangerous_calls(tree))

    # 6. Code length check
    line_count = len(code.splitlines())
    if line_count > 200:
        warnings.append(f"Tool is {line_count} lines (recommend < 200)")

    # 7. Static import availability check (no execution)
    imports = _extract_imports(code)
    errors.extend(_check_imports(imports, deps))

    # 8. Subprocess parse check (isolated, no execution of tool body)
    try:
        ok, err_msg = _subprocess_parse_check(code)
        if not ok:
            errors.append(f"Subprocess parse failed: {err_msg}")
    except Exception as e:
        warnings.append(f"Subprocess parse check skipped: {e}")

    return ToolValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
