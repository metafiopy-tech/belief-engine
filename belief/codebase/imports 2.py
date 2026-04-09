"""Static Import Verifier — implements Covenant #3.

"Before finalizing any multi-module Python project, statically verify
every cross-module import by tracing each `from X import Y` to confirm
Y is defined in X."

This runs before the executor to catch import errors that would cause
cascading failures. It's deterministic — no LLM calls needed.

Usage:
    issues = verify_imports(code_files)
    for issue in issues:
        print(f"{issue.source_file}: cannot import '{issue.symbol}' from '{issue.target_module}'")
        print(f"  Available: {issue.available_symbols}")
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("belief.codebase.imports")


@dataclass
class ImportIssue:
    """A cross-module import that can't be resolved."""
    source_file: str
    target_module: str
    symbol: str
    issue_type: str  # "missing_symbol", "missing_module", "circular"
    available_symbols: list[str] = field(default_factory=list)
    suggestion: str = ""


def verify_imports(code_files: dict[str, str]) -> list[ImportIssue]:
    """Statically verify all cross-module imports in a project.

    For each `from X import Y` statement:
    1. Resolve X to a file in code_files
    2. Parse that file's AST to find defined symbols
    3. Check if Y is defined in X
    4. If not, report the issue with available alternatives

    Returns a list of ImportIssues (empty = all imports valid).
    """
    # Step 1: Build module → file path mapping
    module_map = _build_module_map(code_files)

    # Step 2: Build symbol index (file → set of defined names)
    symbol_index = _build_symbol_index(code_files)

    # Step 3: Check each import
    issues = []
    for fpath, code in code_files.items():
        if not fpath.endswith(".py"):
            continue

        file_imports = _extract_from_imports(code)
        for module, symbols in file_imports:
            # Skip stdlib and third-party imports
            if _is_external_module(module, code_files):
                continue

            # Resolve module to file
            target_file = module_map.get(module)
            if not target_file:
                # Try partial resolution (e.g., "models" → "app/models.py")
                for mod, fpath2 in module_map.items():
                    if mod.endswith(f".{module}") or mod == module.split(".")[-1]:
                        target_file = fpath2
                        break

            if not target_file:
                # Module not found in project — might be external
                continue

            # Check each imported symbol
            available = symbol_index.get(target_file, set())
            for sym in symbols:
                if sym == "*":
                    continue  # Can't verify wildcard imports
                if sym not in available:
                    # Find suggestion
                    suggestion = _find_closest(sym, available)
                    issues.append(ImportIssue(
                        source_file=fpath,
                        target_module=module,
                        symbol=sym,
                        issue_type="missing_symbol",
                        available_symbols=sorted(available)[:10],
                        suggestion=suggestion,
                    ))

    return issues


def auto_fix_imports(
    code_files: dict[str, str], issues: list[ImportIssue]
) -> dict[str, str]:
    """Automatically fix import issues where possible.

    Fixes:
    - Wrong case: "pipeline" → "Pipeline" (case mismatch)
    - Wrong name: "Pipeline" → "DataPipeline" (similar name in target)

    Returns modified code_files dict.
    """
    fixed = dict(code_files)

    for issue in issues:
        if not issue.suggestion:
            continue

        source_code = fixed.get(issue.source_file, "")
        if not source_code:
            continue

        # Replace the wrong symbol with the correct one
        # Be careful to only replace in import statements
        old_import = f"import {issue.symbol}"
        new_import = f"import {issue.suggestion}"

        if old_import in source_code:
            new_code = source_code.replace(old_import, new_import)

            # Also replace uses of the symbol in the code body
            # Only replace whole-word matches
            new_code = re.sub(
                rf'\b{re.escape(issue.symbol)}\b',
                issue.suggestion,
                new_code,
            )

            # Validate
            try:
                ast.parse(new_code)
                fixed[issue.source_file] = new_code
                logger.info(
                    f"Import fix: {issue.source_file}: "
                    f"{issue.symbol} → {issue.suggestion}"
                )
            except SyntaxError:
                pass  # Skip invalid fixes

    return fixed


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_module_map(code_files: dict[str, str]) -> dict[str, str]:
    """Map module names to file paths.

    "app.models" → "app/models.py"
    "models" → "models.py"
    "app.services.task_service" → "app/services/task_service.py"
    """
    mapping = {}
    for fpath in code_files:
        if not fpath.endswith(".py"):
            continue
        # Full dotted path
        module = fpath.replace("/", ".").replace(".py", "")
        mapping[module] = fpath
        # Also map individual components for short imports
        parts = module.split(".")
        for i in range(len(parts)):
            partial = ".".join(parts[i:])
            if partial not in mapping:
                mapping[partial] = fpath
    return mapping


def _build_symbol_index(code_files: dict[str, str]) -> dict[str, set[str]]:
    """Build an index of defined symbols per file."""
    index = {}
    for fpath, code in code_files.items():
        if not fpath.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
            symbols = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbols.add(target.id)
                elif isinstance(node, ast.ImportFrom):
                    # Re-exported symbols (from X import Y → Y is available)
                    if node.names:
                        for alias in node.names:
                            name = alias.asname or alias.name
                            symbols.add(name)
            index[fpath] = symbols
        except SyntaxError:
            index[fpath] = set()
    return index


def _extract_from_imports(code: str) -> list[tuple[str, list[str]]]:
    """Extract `from X import Y, Z` statements.

    Returns list of (module_name, [symbol_names]).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            symbols = [alias.name for alias in node.names]
            imports.append((node.module, symbols))
    return imports


def _is_external_module(module: str, code_files: dict[str, str]) -> bool:
    """Check if a module is external (stdlib or third-party)."""
    # Common stdlib modules
    stdlib = {
        "os", "sys", "re", "json", "typing", "dataclasses", "enum", "pathlib",
        "collections", "functools", "itertools", "abc", "ast", "logging",
        "datetime", "time", "math", "hashlib", "uuid", "asyncio", "subprocess",
        "tempfile", "shutil", "copy", "io", "contextlib", "inspect",
    }
    root = module.split(".")[0]
    if root in stdlib:
        return True

    # Common third-party
    third_party = {
        "fastapi", "pydantic", "uvicorn", "httpx", "sqlalchemy", "alembic",
        "pytest", "starlette", "click", "typer", "rich", "dotenv",
        "chromadb", "anthropic", "langchain", "langgraph",
        "flask", "django", "celery", "redis", "requests",
    }
    if root in third_party:
        return True

    # Check if any file in code_files matches
    for fpath in code_files:
        mod = fpath.replace("/", ".").replace(".py", "")
        if mod == module or mod.endswith(f".{module}"):
            return False  # It's internal

    return True  # Assume external if not found


def _find_closest(name: str, available: set[str]) -> str:
    """Find the closest matching symbol name (case-insensitive, prefix match)."""
    # Exact case-insensitive match
    for sym in available:
        if sym.lower() == name.lower():
            return sym

    # Prefix/suffix match
    for sym in available:
        if name.lower() in sym.lower() or sym.lower() in name.lower():
            return sym

    return ""
