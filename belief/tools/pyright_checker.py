"""
Pyright Integration — Milestone 2

Runs basedpyright on generated project files and parses the JSON output
to identify import errors and type errors. Errors are mapped back to
specific files for the self-correction loop.

Usage:
    errors = run_pyright("/path/to/project")
    file_errors = group_errors_by_file(errors)
    # Feed file_errors into the self-correction loop
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

@dataclass
class PyrightError:
    """A single error from pyright."""
    file: str                    # Relative file path
    line: int                    # Line number (1-indexed)
    column: int                  # Column number (0-indexed)
    message: str                 # Error message
    severity: str                # "error", "warning", "information"
    rule: Optional[str] = None   # Pyright rule name (e.g. "reportMissingImports")

    @property
    def is_import_error(self) -> bool:
        """Check if this is an import resolution error."""
        import_rules = {
            "reportMissingImports",
            "reportMissingModuleSource",
            "reportMissingTypeStubs",
        }
        if self.rule and self.rule in import_rules:
            return True
        return "import" in self.message.lower() and "could not be resolved" in self.message.lower()

    @property
    def is_type_error(self) -> bool:
        """Check if this is a type annotation error."""
        type_rules = {
            "reportArgumentType",
            "reportReturnType",
            "reportAssignmentType",
            "reportAttributeAccessIssue",
            "reportIndexIssue",
        }
        return self.rule in type_rules if self.rule else False

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column} [{self.severity}] {self.message}"


@dataclass
class PyrightResult:
    """Results from a pyright run."""
    errors: list[PyrightError] = field(default_factory=list)
    success: bool = False
    error_count: int = 0
    warning_count: int = 0
    information_count: int = 0
    raw_output: str = ""

    @property
    def has_import_errors(self) -> bool:
        return any(e.is_import_error for e in self.errors)

    @property
    def import_errors(self) -> list[PyrightError]:
        return [e for e in self.errors if e.is_import_error]

    @property
    def type_errors(self) -> list[PyrightError]:
        return [e for e in self.errors if e.is_type_error]

    def errors_for_file(self, file_path: str) -> list[PyrightError]:
        """Get all errors for a specific file."""
        return [e for e in self.errors if e.file == file_path]


# ---------------------------------------------------------------------------
# Pyright runner
# ---------------------------------------------------------------------------

def run_pyright(
    project_dir: str | Path,
    files: Optional[list[str]] = None,
    python_version: str = "3.12",
) -> PyrightResult:
    """
    Run basedpyright on a project directory and parse the JSON output.

    Args:
        project_dir: Path to the project root.
        files: Optional list of specific files to check. If None, checks all.
        python_version: Python version for type checking.

    Returns:
        PyrightResult with parsed errors.
    """
    project_path = Path(project_dir)

    # Build command
    cmd = [
        "basedpyright",
        "--outputjson",
        "--pythonversion", python_version,
    ]

    if files:
        cmd.extend(str(project_path / f) for f in files)
    else:
        cmd.append(str(project_path))

    logger.info(f"Running pyright: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_path),
        )
    except FileNotFoundError:
        logger.warning("basedpyright not found — skipping type check")
        return PyrightResult(success=True, raw_output="basedpyright not installed")
    except subprocess.TimeoutExpired:
        logger.error("Pyright timed out after 120s")
        return PyrightResult(raw_output="timeout")

    return _parse_pyright_output(proc.stdout, project_path)


def _parse_pyright_output(raw_json: str, project_path: Path) -> PyrightResult:
    """Parse pyright's JSON output into structured errors."""
    result = PyrightResult(raw_output=raw_json)

    if not raw_json.strip():
        result.success = True
        return result

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse pyright JSON output: {e}")
        return result

    # Parse summary
    summary = data.get("summary", {})
    result.error_count = summary.get("errorCount", 0)
    result.warning_count = summary.get("warningCount", 0)
    result.information_count = summary.get("informationCount", 0)
    result.success = result.error_count == 0

    # Parse diagnostics
    diagnostics = data.get("generalDiagnostics", [])
    for diag in diagnostics:
        file_path = diag.get("file", "")
        # Make path relative to project
        try:
            rel_path = str(Path(file_path).relative_to(project_path))
        except ValueError:
            rel_path = file_path

        range_info = diag.get("range", {})
        start = range_info.get("start", {})

        error = PyrightError(
            file=rel_path,
            line=start.get("line", 0) + 1,  # pyright is 0-indexed
            column=start.get("character", 0),
            message=diag.get("message", ""),
            severity=diag.get("severity", "error"),
            rule=diag.get("rule", None),
        )
        result.errors.append(error)

    logger.info(
        f"Pyright: {result.error_count} errors, "
        f"{result.warning_count} warnings "
        f"({len(result.import_errors)} import errors)"
    )

    return result


# ---------------------------------------------------------------------------
# Project scaffolding for pyright
# ---------------------------------------------------------------------------

def write_project_for_pyright(
    files: dict[str, str],
    output_dir: str | Path,
    external_deps: Optional[list[str]] = None,
) -> Path:
    """
    Write generated files to disk in a structure pyright can analyze.

    Creates:
    - All Python files in their relative paths
    - __init__.py files for all packages
    - pyrightconfig.json with appropriate settings

    Args:
        files: Dict of {relative_path: source_code}.
        output_dir: Directory to write files to.
        external_deps: List of pip packages (for stub handling).

    Returns:
        Path to the project root.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Collect all directories that need __init__.py
    dirs_needing_init: set[Path] = set()

    for file_path, source_code in files.items():
        full_path = root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Track directories
        current = full_path.parent
        while current != root:
            dirs_needing_init.add(current)
            current = current.parent

        full_path.write_text(source_code)

    # Create __init__.py for all packages
    for dir_path in dirs_needing_init:
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")

    # Create pyrightconfig.json
    pyright_config = {
        "include": ["."],
        "pythonVersion": "3.12",
        "typeCheckingMode": "standard",
        "reportMissingImports": "error",
        "reportMissingModuleSource": "warning",
        "reportMissingTypeStubs": "none",  # Don't require stubs for third-party
        "reportUnusedImport": "none",
        "reportUnusedVariable": "none",
        "reportOptionalMemberAccess": "none",
        "executionEnvironments": [
            {
                "root": ".",
                "pythonVersion": "3.12",
            }
        ],
    }

    config_path = root / "pyrightconfig.json"
    config_path.write_text(json.dumps(pyright_config, indent=2))

    return root


# ---------------------------------------------------------------------------
# Error grouping helpers
# ---------------------------------------------------------------------------

def group_errors_by_file(result: PyrightResult) -> dict[str, list[PyrightError]]:
    """Group errors by file path for the self-correction loop."""
    grouped: dict[str, list[PyrightError]] = {}
    for error in result.errors:
        grouped.setdefault(error.file, []).append(error)
    return grouped


def format_errors_for_llm(errors: list[PyrightError], source_code: str) -> str:
    """
    Format pyright errors as context for the LLM self-correction prompt.

    Includes the error message, location, and surrounding source lines.
    """
    if not errors:
        return "No errors."

    lines = source_code.splitlines()
    sections = []

    for error in errors:
        section = [f"ERROR at line {error.line}: {error.message}"]
        if error.rule:
            section.append(f"  Rule: {error.rule}")

        # Add surrounding source context (3 lines before and after)
        start = max(0, error.line - 4)
        end = min(len(lines), error.line + 3)
        for i in range(start, end):
            marker = ">>>" if i == error.line - 1 else "   "
            section.append(f"  {marker} {i + 1:4d} | {lines[i]}")

        sections.append("\n".join(section))

    return "\n\n".join(sections)
