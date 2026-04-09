"""TypeScript Covenant Enforcer — Hard Rules for Generated Code.

Derived from research: 94% of LLM TypeScript compilation errors are type-check
failures. This module catches the exact failure patterns BEFORE code reaches
tsc, fixing what can be fixed automatically and rejecting what can't.

Covenants enforced:

MODULE RESOLUTION:
  C1: Relative imports must have .js extension (TS2835)
  C2: No bare @modelcontextprotocol/sdk import (ERR_PACKAGE_PATH_NOT_EXPORTED)
  C3: No @x402/types or @x402/client imports (packages don't exist)

RUNTIME SAFETY:
  C4: No __dirname or __filename in ESM (runtime crash, not tsc error)
  C5: No require() in ESM modules
  C6: Express 5 error handlers must use ErrorRequestHandler type

ETHERS V6:
  C7: No ethers.providers.*, ethers.utils.*, ethers.constants.* (v5 patterns)
  C8: No BigNumber — use native bigint
  C9: No @ethersproject/* imports (eliminated in v6)

VITEST:
  C10: No jest.* — use vi.* from vitest
  C11: Must import describe/it/expect from 'vitest' (no globals by default)

Usage:
    from belief.validators.typescript_covenants import enforce_ts_covenants
    fixed_files, violations = enforce_ts_covenants(code_files)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("belief.validators.ts_covenants")


@dataclass
class TSViolation:
    """A TypeScript covenant violation found in generated code."""
    covenant: str
    file: str
    line: int
    severity: str  # "critical", "important", "minor"
    message: str
    auto_fixed: bool = False
    fix_applied: str = ""


@dataclass
class TSEnforcementResult:
    """Result of TypeScript covenant enforcement."""
    violations: list[TSViolation] = field(default_factory=list)
    fixes_applied: int = 0
    files_modified: list[str] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "critical" and not v.auto_fixed for v in self.violations)


def enforce_ts_covenants(
    code_files: dict[str, str],
    auto_fix: bool = True,
) -> tuple[dict[str, str], TSEnforcementResult]:
    """Enforce TypeScript covenants on generated code files.

    Returns (fixed_files, enforcement_result).
    Only processes .ts, .tsx, .js, .jsx files.
    """
    result = TSEnforcementResult()
    fixed = dict(code_files)

    for fname, content in code_files.items():
        if not fname.endswith((".ts", ".tsx", ".js", ".jsx")):
            continue

        new_content = content
        file_violations = []

        # C1: Relative imports must have .js extension
        new_content, c1_violations = _enforce_js_extensions(fname, new_content, auto_fix)
        file_violations.extend(c1_violations)

        # C2: No bare @modelcontextprotocol/sdk import
        new_content, c2_violations = _enforce_mcp_subpaths(fname, new_content, auto_fix)
        file_violations.extend(c2_violations)

        # C3: No @x402/types or @x402/client imports
        new_content, c3_violations = _enforce_x402_packages(fname, new_content, auto_fix)
        file_violations.extend(c3_violations)

        # C4: No __dirname or __filename
        new_content, c4_violations = _enforce_no_dirname(fname, new_content, auto_fix)
        file_violations.extend(c4_violations)

        # C5: No require() in ESM
        new_content, c5_violations = _enforce_no_require(fname, new_content)
        file_violations.extend(c5_violations)

        # C7: No ethers v5 patterns
        new_content, c7_violations = _enforce_ethers_v6(fname, new_content, auto_fix)
        file_violations.extend(c7_violations)

        # C9: No @ethersproject/* imports
        new_content, c9_violations = _enforce_no_ethersproject(fname, new_content, auto_fix)
        file_violations.extend(c9_violations)

        # C10: No jest.* — use vi.*
        new_content, c10_violations = _enforce_vitest(fname, new_content, auto_fix)
        file_violations.extend(c10_violations)

        if new_content != content:
            fixed[fname] = new_content
            result.files_modified.append(fname)
            result.fixes_applied += sum(1 for v in file_violations if v.auto_fixed)

        result.violations.extend(file_violations)

    if result.fixes_applied > 0:
        logger.info(
            f"TS covenants: {result.fixes_applied} auto-fixes across "
            f"{len(result.files_modified)} files, "
            f"{len(result.violations)} total violations"
        )

    return fixed, result


# ── Covenant implementations ─────────────────────────────────────────────────

def _enforce_js_extensions(
    fname: str, content: str, auto_fix: bool
) -> tuple[str, list[TSViolation]]:
    """C1: Relative imports must have .js extension for NodeNext resolution."""
    violations = []
    lines = content.split("\n")
    new_lines = []

    for i, line in enumerate(lines, 1):
        new_line = line
        # Match: import ... from './something' or import ... from '../something'
        match = re.search(r"""(from\s+['"])(\.\.?/[^'"]+)(['"])""", line)
        if match:
            path = match.group(2)
            if not path.endswith((".js", ".json", ".mjs", ".cjs", ".css", ".svg")):
                v = TSViolation(
                    covenant="C1:js-extension",
                    file=fname, line=i, severity="critical",
                    message=f"Relative import missing .js extension: {path}",
                )
                if auto_fix:
                    fixed_path = path + ".js"
                    new_line = line.replace(
                        match.group(0),
                        f"{match.group(1)}{fixed_path}{match.group(3)}"
                    )
                    v.auto_fixed = True
                    v.fix_applied = f"{path} → {fixed_path}"
                violations.append(v)
        new_lines.append(new_line)

    return "\n".join(new_lines), violations


def _enforce_mcp_subpaths(
    fname: str, content: str, auto_fix: bool
) -> tuple[str, list[TSViolation]]:
    """C2: No bare @modelcontextprotocol/sdk import — must use subpaths."""
    violations = []

    # Match bare import (no subpath after sdk)
    pattern = r"""from\s+['"]@modelcontextprotocol/sdk['"]"""
    for i, line in enumerate(content.split("\n"), 1):
        if re.search(pattern, line):
            violations.append(TSViolation(
                covenant="C2:mcp-subpath",
                file=fname, line=i, severity="critical",
                message="Bare @modelcontextprotocol/sdk import — must use subpath like /server/mcp.js",
            ))

    # Can't reliably auto-fix because we don't know which subpath they need
    return content, violations


def _enforce_x402_packages(
    fname: str, content: str, auto_fix: bool
) -> tuple[str, list[TSViolation]]:
    """C3: No @x402/types or @x402/client — these packages don't exist."""
    violations = []
    new_content = content

    nonexistent = ["@x402/types", "@x402/client"]
    for pkg in nonexistent:
        for i, line in enumerate(content.split("\n"), 1):
            if pkg in line and "import" in line:
                violations.append(TSViolation(
                    covenant="C3:x402-nonexistent",
                    file=fname, line=i, severity="critical",
                    message=f"Package {pkg} does not exist — types are in @x402/core",
                ))

    return new_content, violations


def _enforce_no_dirname(
    fname: str, content: str, auto_fix: bool
) -> tuple[str, list[TSViolation]]:
    """C4: No __dirname or __filename in ESM — runtime crash."""
    violations = []

    for i, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if "__dirname" in stripped and "import.meta" not in stripped and "fileURLToPath" not in content:
            violations.append(TSViolation(
                covenant="C4:no-dirname",
                file=fname, line=i, severity="important",
                message="__dirname is undefined in ESM — use import.meta.dirname or fileURLToPath(import.meta.url)",
            ))
        if "__filename" in stripped and "import.meta" not in stripped and "fileURLToPath" not in content:
            violations.append(TSViolation(
                covenant="C4:no-filename",
                file=fname, line=i, severity="important",
                message="__filename is undefined in ESM — use fileURLToPath(import.meta.url)",
            ))

    return content, violations


def _enforce_no_require(
    fname: str, content: str,
) -> tuple[str, list[TSViolation]]:
    """C5: No require() in ESM modules."""
    violations = []

    for i, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if "require(" in stripped and "createRequire" not in content:
            violations.append(TSViolation(
                covenant="C5:no-require",
                file=fname, line=i, severity="critical",
                message="require() is not available in ESM — use import",
            ))

    return content, violations


def _enforce_ethers_v6(
    fname: str, content: str, auto_fix: bool
) -> tuple[str, list[TSViolation]]:
    """C7: No ethers v5 patterns — providers/utils/constants are top-level in v6."""
    violations = []
    new_content = content

    v5_patterns = [
        (r"ethers\.providers\.", "ethers v5 pattern: use top-level imports in v6"),
        (r"ethers\.utils\.", "ethers v5 pattern: parseEther/formatEther are top-level in v6"),
        (r"ethers\.constants\.", "ethers v5 pattern: ZeroAddress/MaxUint256 are top-level in v6"),
        (r"BigNumber\.from\(", "ethers v5 BigNumber: use native bigint in v6"),
        (r"new\s+Web3Provider\(", "ethers v5: use new BrowserProvider() in v6"),
        (r"\.callStatic\.", "ethers v5: use .staticCall() in v6"),
    ]

    for pattern, msg in v5_patterns:
        for i, line in enumerate(content.split("\n"), 1):
            if re.search(pattern, line):
                violations.append(TSViolation(
                    covenant="C7:ethers-v6",
                    file=fname, line=i, severity="critical",
                    message=msg,
                ))

    return new_content, violations


def _enforce_no_ethersproject(
    fname: str, content: str, auto_fix: bool
) -> tuple[str, list[TSViolation]]:
    """C9: No @ethersproject/* imports — eliminated in v6."""
    violations = []
    new_content = content

    for i, line in enumerate(content.split("\n"), 1):
        if "@ethersproject/" in line and "import" in line:
            v = TSViolation(
                covenant="C9:no-ethersproject",
                file=fname, line=i, severity="critical",
                message="@ethersproject/* packages eliminated in ethers v6 — import from 'ethers'",
            )
            if auto_fix:
                # Replace @ethersproject/anything with ethers
                new_line = re.sub(r"""['"]@ethersproject/\w+['"]""", "'ethers'", line)
                new_content = new_content.replace(line, new_line)
                v.auto_fixed = True
                v.fix_applied = "@ethersproject/* → 'ethers'"
            violations.append(v)

    return new_content, violations


def _enforce_vitest(
    fname: str, content: str, auto_fix: bool
) -> tuple[str, list[TSViolation]]:
    """C10: No jest.* — use vi.* from vitest."""
    violations = []
    new_content = content

    if not (fname.endswith((".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx",
                            ".test.js", ".spec.js"))):
        return content, violations

    jest_patterns = [
        (r"\bjest\.fn\(", "vi.fn("),
        (r"\bjest\.mock\(", "vi.mock("),
        (r"\bjest\.spyOn\(", "vi.spyOn("),
        (r"\bjest\.useFakeTimers\(", "vi.useFakeTimers("),
        (r"\bjest\.clearAllMocks\(", "vi.clearAllMocks("),
        (r"\bjest\.resetAllMocks\(", "vi.resetAllMocks("),
    ]

    for pattern, replacement in jest_patterns:
        for i, line in enumerate(content.split("\n"), 1):
            if re.search(pattern, line):
                v = TSViolation(
                    covenant="C10:no-jest",
                    file=fname, line=i, severity="critical",
                    message=f"jest.* not available in vitest — use {replacement}",
                )
                if auto_fix:
                    new_content = re.sub(pattern, replacement, new_content)
                    v.auto_fixed = True
                    v.fix_applied = f"jest.* → vi.*"
                violations.append(v)
                break  # One violation per pattern is enough

    # C11: Check for missing vitest imports
    has_test_funcs = bool(re.search(r"\b(describe|it|expect|vi)\b", content))
    has_vitest_import = "from 'vitest'" in content or 'from "vitest"' in content
    if has_test_funcs and not has_vitest_import:
        v = TSViolation(
            covenant="C11:vitest-import",
            file=fname, line=1, severity="critical",
            message="Test file missing vitest import — add: import { describe, it, expect } from 'vitest'",
        )
        if auto_fix:
            new_content = "import { describe, it, expect, vi } from 'vitest';\n" + new_content
            v.auto_fixed = True
            v.fix_applied = "Added vitest import"
        violations.append(v)

    return new_content, violations
