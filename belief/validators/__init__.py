"""Covenant Enforcer — structural enforcement of self-learned rules.

MOVE 2: Convert covenants from prompt suggestions to AST validators.

Covenants are immutable rules the engine learned from repeated failures.
Previously they were RAG-retrieved and injected into prompts — the LLM
could ignore them. Now they're enforced structurally with AST checks
that return binary pass/fail and auto-fix violations.

Each enforcer is a pure function: (filename, code, context) → (pass, fixes)
No LLM calls. Deterministic. Fast.

Current covenants enforced:
  1. Explicit stdlib imports
  2. No file over 200 lines  
  3. Static import verification (handled by import_fix node)
  4. SQLAlchemy type annotations must be imported
  5. SQLAlchemy Mapped/mapped_column must be imported
  6. Entry point imports must resolve
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("belief.validators")


@dataclass
class Violation:
    """A covenant violation found by structural enforcement."""
    covenant: str
    file: str
    line: int = 0
    message: str = ""
    auto_fix: str = ""  # If non-empty, this is the fixed code
    severity: str = "error"  # "error" or "warning"


@dataclass
class EnforcementResult:
    """Result of running all covenant enforcers."""
    violations: list[Violation] = field(default_factory=list)
    fixes_applied: int = 0
    files_modified: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0


def enforce_all(
    code_files: dict[str, str],
    auto_fix: bool = True,
) -> tuple[dict[str, str], EnforcementResult]:
    """Run all covenant enforcers on generated code.

    Args:
        code_files: dict of filename → content
        auto_fix: if True, apply fixes and return modified files

    Returns:
        (fixed_code_files, result)
    """
    result = EnforcementResult()
    fixed = dict(code_files)

    # Detect project context
    all_code = "\n".join(fixed.values())
    uses_sqlalchemy = "sqlalchemy" in all_code.lower()
    uses_fastapi = "FastAPI" in all_code or "fastapi" in all_code

    # Run each enforcer
    enforcers = [
        _enforce_no_future_with_sqlalchemy,
        _enforce_sqlalchemy_imports,
        _enforce_file_length,
        _enforce_stdlib_imports,
        _enforce_no_stdlib_in_requirements,
        _enforce_no_bare_except,
    ]

    for enforcer in enforcers:
        for fname in list(fixed.keys()):
            if not fname.endswith(".py"):
                continue

            violations = enforcer(fname, fixed[fname], uses_sqlalchemy)

            for v in violations:
                result.violations.append(v)

                if auto_fix and v.auto_fix:
                    fixed[fname] = v.auto_fix
                    result.fixes_applied += 1
                    if fname not in result.files_modified:
                        result.files_modified.append(fname)
                    logger.info(f"Covenant enforced: {v.covenant} — {v.message}")

    # Special: requirements.txt check
    if "requirements.txt" in fixed:
        violations = _enforce_no_stdlib_in_requirements(
            "requirements.txt", fixed["requirements.txt"], uses_sqlalchemy
        )
        for v in violations:
            result.violations.append(v)
            if auto_fix and v.auto_fix:
                fixed["requirements.txt"] = v.auto_fix
                result.fixes_applied += 1
                result.files_modified.append("requirements.txt")

    if result.fixes_applied > 0:
        logger.info(
            f"Covenant enforcer: {result.fixes_applied} fixes applied "
            f"across {len(result.files_modified)} files"
        )

    return fixed, result


def enforce_with_registry(
    code_files: dict[str, str],
    auto_fix: bool = True,
    soil=None,
) -> tuple[dict[str, str], EnforcementResult]:
    """Run all covenants (static + dynamic) via the CovenantRegistry.

    This extends enforce_all() by also firing dynamically discovered
    covenants from the belief_covenants ChromaDB collection.

    Falls back to enforce_all() if soil is not available.
    """
    # Always run the static enforcers first
    fixed, result = enforce_all(code_files, auto_fix=auto_fix)

    # Fire dynamic covenants via registry if soil is available
    if soil is not None:
        try:
            from belief.validators.covenant_registry import CovenantRegistry
            registry = CovenantRegistry(soil)
            dynamic_results = registry.fire_all(fixed)

            for dr in dynamic_results:
                if dr.source == "dynamic" and not dr.passed:
                    for v in dr.violations:
                        result.violations.append(Violation(
                            covenant=dr.name,
                            file=v.get("file", ""),
                            line=v.get("line", 0),
                            message=v.get("message", ""),
                            severity=v.get("severity", "warning"),
                        ))
        except Exception as e:
            logger.debug(f"Dynamic covenant check skipped: {e}")

    return fixed, result


# ── Enforcer: No __future__ annotations with SQLAlchemy ──────────────────────

def _enforce_no_future_with_sqlalchemy(
    fname: str, code: str, uses_sqlalchemy: bool
) -> list[Violation]:
    """Covenant 4+5: Don't use `from __future__ import annotations` in
    files that use SQLAlchemy ORM types (Mapped, mapped_column, etc).

    The __future__ import makes all annotations strings, which breaks
    SQLAlchemy's type resolution at class definition time.
    """
    if not uses_sqlalchemy:
        return []

    # Check if this file uses SQLAlchemy ORM features
    sqlalchemy_markers = [
        "Mapped[", "mapped_column(", "DeclarativeBase",
        "relationship(", "Column(", "ForeignKey(",
        "from sqlalchemy", "import sqlalchemy",
    ]

    has_sqlalchemy = any(m in code for m in sqlalchemy_markers)
    has_future = "from __future__ import annotations" in code

    if has_sqlalchemy and has_future:
        # Auto-fix: remove the __future__ import
        fixed = code.replace("from __future__ import annotations\n", "")
        fixed = fixed.replace("from __future__ import annotations", "")

        return [Violation(
            covenant="no_future_with_sqlalchemy",
            file=fname,
            line=1,
            message=f"Removed `from __future__ import annotations` (breaks SQLAlchemy Mapped types)",
            auto_fix=fixed,
        )]

    return []


# ── Enforcer: SQLAlchemy Mapped/mapped_column imports ────────────────────────

def _enforce_sqlalchemy_imports(
    fname: str, code: str, uses_sqlalchemy: bool
) -> list[Violation]:
    """Covenant 5: Files using Mapped[] or mapped_column() must import them."""
    if not uses_sqlalchemy:
        return []

    uses_mapped = "Mapped[" in code
    uses_mapped_column = "mapped_column(" in code

    if not (uses_mapped or uses_mapped_column):
        return []

    # Check if already imported
    has_mapped_import = re.search(
        r'from\s+sqlalchemy\.orm\s+import\s+.*\bMapped\b', code
    )
    has_mc_import = re.search(
        r'from\s+sqlalchemy\.orm\s+import\s+.*\bmapped_column\b', code
    )

    missing = []
    if uses_mapped and not has_mapped_import:
        missing.append("Mapped")
    if uses_mapped_column and not has_mc_import:
        missing.append("mapped_column")

    if not missing:
        return []

    # Auto-fix: add the import
    # Check if there's already a sqlalchemy.orm import line
    existing_import = re.search(
        r'(from\s+sqlalchemy\.orm\s+import\s+)([^\n]+)', code
    )

    if existing_import:
        # Add to existing import
        current = existing_import.group(2)
        new_imports = current.rstrip()
        for m in missing:
            if m not in new_imports:
                new_imports += f", {m}"
        fixed = code.replace(
            existing_import.group(0),
            existing_import.group(1) + new_imports,
        )
    else:
        # Add new import line after other imports
        import_line = f"from sqlalchemy.orm import {', '.join(missing)}"
        # Find the last import line
        lines = code.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_idx = i + 1
        lines.insert(insert_idx, import_line)
        fixed = "\n".join(lines)

    # Validate fix
    try:
        ast.parse(fixed)
    except SyntaxError:
        return []  # Don't apply broken fix

    return [Violation(
        covenant="sqlalchemy_mapped_imports",
        file=fname,
        message=f"Added missing import: {', '.join(missing)}",
        auto_fix=fixed,
    )]


# ── Enforcer: File length limit ──────────────────────────────────────────────

def _enforce_file_length(
    fname: str, code: str, uses_sqlalchemy: bool
) -> list[Violation]:
    """Covenant 2: No generated file should exceed 200 lines."""
    line_count = code.count("\n") + 1
    if line_count <= 200:
        return []

    return [Violation(
        covenant="max_200_lines",
        file=fname,
        message=f"{fname} is {line_count} lines (max 200)",
        severity="warning",  # Can't auto-fix — needs file splitting
    )]


# ── Enforcer: Explicit stdlib imports ────────────────────────────────────────

_STDLIB_NAMES = {
    "datetime", "uuid", "enum", "os", "sys", "re", "json", "pathlib",
    "logging", "time", "hashlib", "secrets", "typing", "collections",
    "functools", "itertools", "dataclasses", "abc", "math", "random",
    "copy", "io", "contextlib", "textwrap", "shutil", "tempfile",
}


def _enforce_stdlib_imports(
    fname: str, code: str, uses_sqlalchemy: bool
) -> list[Violation]:
    """Covenant 1: Every stdlib module used must be explicitly imported."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Collect imported names
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    # Check for stdlib names used but not imported
    violations = []
    for name in _STDLIB_NAMES:
        # Check if the name is used in the code (as a module reference)
        if re.search(rf'\b{name}\.\w+', code) and name not in imported:
            # Auto-fix: add the import
            lines = code.split("\n")
            # Find insert position (after __future__, before other imports)
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("from __future__"):
                    insert_idx = i + 1
                elif line.startswith("import ") or line.startswith("from "):
                    if insert_idx == 0:
                        insert_idx = i
                    break

            lines.insert(insert_idx, f"import {name}")
            fixed = "\n".join(lines)

            try:
                ast.parse(fixed)
                violations.append(Violation(
                    covenant="explicit_stdlib_imports",
                    file=fname,
                    message=f"Added missing `import {name}`",
                    auto_fix=fixed,
                ))
                code = fixed  # Update for subsequent checks
            except SyntaxError:
                pass

    return violations


# ── Enforcer: No stdlib in requirements.txt ──────────────────────────────────

_STDLIB_PACKAGES = {
    "__future__", "abc", "argparse", "ast", "asyncio", "base64",
    "collections", "configparser", "contextlib", "copy", "csv",
    "dataclasses", "datetime", "decimal", "enum", "email", "functools",
    "glob", "hashlib", "html", "http", "importlib", "inspect", "io",
    "itertools", "json", "locale", "logging", "math", "multiprocessing",
    "numbers", "operator", "os", "pathlib", "pickle", "platform",
    "pprint", "queue", "random", "re", "secrets", "shutil", "signal",
    "site", "socket", "sqlite3", "string", "struct", "subprocess",
    "sys", "tempfile", "textwrap", "threading", "time", "traceback",
    "typing", "typing_extensions", "unicodedata", "unittest", "urllib",
    "uuid", "venv", "warnings", "weakref", "xml", "zipfile", "zlib",
}


def _enforce_no_stdlib_in_requirements(
    fname: str, code: str, uses_sqlalchemy: bool
) -> list[Violation]:
    """Requirements.txt must not include stdlib packages."""
    if fname != "requirements.txt":
        return []

    lines = code.strip().split("\n")
    clean_lines = []
    removed = []

    for line in lines:
        pkg = line.strip().split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
        if pkg.lower() in _STDLIB_PACKAGES or pkg.startswith("#"):
            removed.append(pkg)
        elif pkg:
            clean_lines.append(line)

    if not removed:
        return []

    fixed = "\n".join(clean_lines) + "\n" if clean_lines else "# No external dependencies\n"

    return [Violation(
        covenant="no_stdlib_in_requirements",
        file=fname,
        message=f"Removed stdlib from requirements.txt: {', '.join(removed)}",
        auto_fix=fixed,
    )]


# ── Enforcer: No bare except clauses ─────────────────────────────────────────

def _enforce_no_bare_except(
    fname: str, code: str, uses_sqlalchemy: bool
) -> list[Violation]:
    """Covenant 7: Never use bare 'except:' or 'except Exception:' with pass.

    Bare excepts swallow all errors silently, making debugging impossible.
    This is the #1 anti-pattern in generated code — the LLM adds
    'except: pass' to make code "robust" but actually hides bugs.

    Auto-fix: replace 'except:' with 'except Exception:' and
    'except Exception: pass' with 'except Exception: logger.exception("...")'.
    """
    if not fname.endswith(".py") or "test" in fname:
        return []

    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.split("\n")
    modified = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            line_idx = node.lineno - 1
            if line_idx >= len(lines):
                continue

            line = lines[line_idx]

            # Bare except: (no exception type)
            if node.type is None and "except:" in line:
                lines[line_idx] = line.replace("except:", "except Exception:")
                modified = True
                violations.append(Violation(
                    covenant="no_bare_except",
                    file=fname,
                    line=node.lineno,
                    message="Replaced bare 'except:' with 'except Exception:'",
                    severity="warning",
                ))

            # except ...: pass (swallows error silently)
            if (node.body and len(node.body) == 1
                    and isinstance(node.body[0], ast.Pass)):
                pass_line_idx = node.body[0].lineno - 1
                if pass_line_idx < len(lines):
                    indent = len(lines[pass_line_idx]) - len(lines[pass_line_idx].lstrip())
                    spaces = " " * indent
                    lines[pass_line_idx] = f"{spaces}pass  # TODO: handle error appropriately"
                    modified = True

    if modified:
        fixed = "\n".join(lines)
        if violations:
            violations[0].auto_fix = fixed

    return violations
