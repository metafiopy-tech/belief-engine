"""Change Impact Analysis — Test Selection After Patching.

Given a patch (set of changed files), determine which existing tests
need to be re-run for regression validation. Three strategies:

1. Static import analysis — parse test imports, select tests that
   import from changed modules. Fast, zero overhead, module-granularity.

2. pytest-testmon — coverage-based test selection. Precise (block-level),
   requires a coverage baseline run. ~2× CI speedup.

3. Full suite — run everything. Most accurate, most expensive.

The default strategy is static import analysis (fastest, no dependencies).
pytest-testmon is used when available and a baseline exists.

Research basis:
- pytest-testmon: ~2× CI speedup at Instawork (block-level coverage tracking)
- pytest-rts: coverage-based regression test selection (F-Secure)
- Static analysis: module-level granularity, fast but over-selects

Usage:
    from belief.codebase.change_impact import select_affected_tests
    tests = select_affected_tests(codebase, changed_files=["app/models.py"])
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

logger = logging.getLogger("belief.codebase.change_impact")


def select_affected_tests(
    codebase,
    changed_files: list[str],
    strategy: str = "auto",
) -> list[str]:
    """Select tests affected by changes to specific files.

    Args:
        codebase: Codebase object
        changed_files: list of file paths that were modified
        strategy: "static" (import analysis), "testmon" (coverage), "all", or "auto"

    Returns:
        List of test file paths to re-run
    """
    if strategy == "all":
        return list(codebase.test_files)

    if strategy == "auto":
        # Use testmon if available, else static
        strategy = "testmon" if _has_testmon() else "static"

    if strategy == "testmon":
        affected = _testmon_select(codebase, changed_files)
        if affected is not None:
            return affected
        # Fall through to static if testmon fails
        logger.debug("Testmon selection failed, falling back to static")

    return _static_import_select(codebase, changed_files)


def _static_import_select(codebase, changed_files: list[str]) -> list[str]:
    """Select tests by analyzing import statements.

    A test is affected if it imports from any changed module.
    Module-level granularity — may over-select.
    """
    # Build module names for changed files
    changed_modules = set()
    for fpath in changed_files:
        if fpath.endswith(".py"):
            module = fpath.replace("/", ".").replace(".py", "")
            changed_modules.add(module)
            # Also add the last component (e.g., "models" for "app/models.py")
            parts = module.split(".")
            for i in range(len(parts)):
                changed_modules.add(".".join(parts[i:]))

    if not changed_modules:
        return list(codebase.test_files)

    # Check each test file's imports
    affected = []
    for test_file in codebase.test_files:
        content = codebase.get_file_content(test_file)
        if not content:
            continue

        test_imports = _extract_import_modules(content)
        if test_imports & changed_modules:
            affected.append(test_file)

    # Also include tests that the codebase's dependency graph links
    for fpath in changed_files:
        graph_tests = codebase.get_affected_tests(fpath)
        for t in graph_tests:
            if t not in affected:
                affected.append(t)

    if not affected:
        # No direct imports found — return all tests as safety measure
        logger.debug("No import-linked tests found, returning all tests")
        return list(codebase.test_files)

    logger.info(
        f"Change impact: {len(affected)}/{len(codebase.test_files)} tests "
        f"affected by changes to {len(changed_files)} files"
    )
    return affected


def _extract_import_modules(code: str) -> set[str]:
    """Extract all imported module names from Python code."""
    modules = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
                # Add root module too
                root = node.module.split(".")[0]
                modules.add(root)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name)
                    modules.add(alias.name.split(".")[0])
    except SyntaxError:
        # Fallback to regex
        for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", code, re.MULTILINE):
            modules.add(m.group(1))
            modules.add(m.group(1).split(".")[0])
        for m in re.finditer(r"^\s*import\s+([\w.]+)", code, re.MULTILINE):
            modules.add(m.group(1))
            modules.add(m.group(1).split(".")[0])

    return modules


def _has_testmon() -> bool:
    """Check if pytest-testmon is installed."""
    try:
        import testmon  # noqa: F401

        return True
    except ImportError:
        return False


def _testmon_select(codebase, changed_files: list[str]) -> list[str] | None:
    """Use pytest-testmon for precise test selection.

    Requires a prior baseline run with --testmon to build the dependency database.
    Returns None if testmon DB doesn't exist or testmon fails.
    """
    import subprocess
    import sys
    from pathlib import Path

    testmon_db = Path(codebase.root) / ".testmondata"
    if not testmon_db.exists():
        logger.debug("No .testmondata found — testmon needs a baseline run first")
        return None

    try:
        # Run pytest --testmon --collect-only to get affected tests
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--testmon", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=codebase.root,
        )

        if proc.returncode != 0:
            return None

        # Parse collected test file paths
        affected = set()
        for line in proc.stdout.split("\n"):
            line = line.strip()
            if "::" in line:
                test_file = line.split("::")[0]
                affected.add(test_file)

        if affected:
            logger.info(f"Testmon: {len(affected)} tests selected for re-run")
            return list(affected)

        return None

    except Exception as e:
        logger.debug(f"Testmon selection failed: {e}")
        return None


def compute_change_scope(
    codebase,
    changed_files: list[str],
) -> dict[str, Any]:
    """Compute the scope and risk of a set of changes.

    Returns a summary dict useful for deciding how much validation to do.
    """
    affected_tests = _static_import_select(codebase, changed_files)

    # Count dependents (files that import from changed files)
    total_dependents = set()
    for fpath in changed_files:
        deps = codebase.get_dependents(fpath)
        total_dependents.update(deps)

    # Risk assessment
    high_risk = any(
        "model" in f.lower() or "schema" in f.lower() or "database" in f.lower()
        for f in changed_files
    )

    return {
        "changed_files": len(changed_files),
        "affected_tests": len(affected_tests),
        "total_tests": len(codebase.test_files),
        "dependents": len(total_dependents),
        "high_risk": high_risk,
        "recommended_strategy": "all" if high_risk else "static",
        "test_files": affected_tests,
    }
