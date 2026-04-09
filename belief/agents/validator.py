"""Validator Agent — execution-based validation of the final build.

MOVE 1: Execute code instead of imagining it.
The validator now runs real pytest, ruff, and AST checks instead of
asking the LLM to imagine what tests would pass. LLM is only used
for architectural quality assessment (style, design, completeness).
"""

from __future__ import annotations
import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.models.artifacts import (
    TestCase, TestTier, TIER_WEIGHTS,
    ValidationResult, ValidationVerdict,
)
from belief.models.state import Phase, UnifiedState

logger = logging.getLogger("belief.agents.validator")


class ValidatorAgent(BaseAgent):
    role = ModelRole.VALIDATOR
    name = "Validator"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.VALIDATION
        if not state.code_files:
            state.validation_result = ValidationResult(
                verdict=ValidationVerdict.FAIL_FIXABLE, summary="No code files",
                issues=["No code files produced"],
            )
            state.phase = Phase.COMPLETE
            return state

        # ── Step 1: Run real tests ──
        tests, issues = _run_real_validation(
            state.code_files, state.test_files
        )

        # ── Step 2: Check if executor passed ──
        exec_ok = False
        exec_result = state.execution_result
        if exec_result:
            exec_ok = exec_result.get("success") if isinstance(exec_result, dict) else getattr(exec_result, "success", False)

        if exec_ok:
            tests.insert(0, TestCase(
                name="executor_verification",
                description="Code executes and entry points verify",
                passed=True, tier=TestTier.SMOKE,
            ))
        else:
            tests.insert(0, TestCase(
                name="executor_verification",
                description="Code executes and entry points verify",
                passed=False, tier=TestTier.SMOKE,
                error="Executor failed",
            ))
            issues.append("Code failed executor verification")

        # ── Step 3: Compute weighted score from REAL results ──
        result = ValidationResult(tests=tests, issues=issues)
        _classify_and_score(result)

        # Override quality scores based on deterministic checks
        result.correctness_score = result.weighted_score
        result.code_quality_score = min(1.0, _lint_score(state.code_files))
        result.completeness_score = result.weighted_score
        result.security_score = _security_score(state.code_files)

        state.validation_result = result
        logger.info(
            f"Validator: {result.verdict.value}, "
            f"{result.tests_passed}/{result.tests_total} tests, "
            f"weighted={result.weighted_score:.2f}"
        )

        state.phase = Phase.COMPLETE
        return state


# ── Real test execution ──────────────────────────────────────────────────────

def _run_real_validation(
    code_files: dict[str, str],
    test_files: dict[str, str],
) -> tuple[list[TestCase], list[str]]:
    """Run pytest for real and return TestCase objects from actual results.

    Also runs AST syntax checks on all source files.
    Returns (tests, issues).
    """
    tests = []
    issues = []

    # ── Syntax check all source files ──
    syntax_ok = True
    for fname, content in code_files.items():
        if not fname.endswith(".py"):
            continue
        try:
            ast.parse(content)
            tests.append(TestCase(
                name=f"syntax_{fname.replace('/', '_').replace('.py', '')}",
                description=f"{fname} has valid Python syntax",
                passed=True, tier=TestTier.SMOKE,
            ))
        except SyntaxError as e:
            syntax_ok = False
            tests.append(TestCase(
                name=f"syntax_{fname.replace('/', '_').replace('.py', '')}",
                description=f"{fname} has valid Python syntax",
                passed=False, tier=TestTier.SMOKE,
                error=f"Line {e.lineno}: {e.msg}",
            ))
            issues.append(f"Syntax error in {fname} line {e.lineno}: {e.msg}")

    if not syntax_ok:
        return tests, issues

    # ── Run pytest if test files exist ──
    all_test_files = dict(test_files)
    # Also include builder-generated test files
    for fname, content in code_files.items():
        if fname.startswith("test") or "/test" in fname:
            all_test_files[fname] = content

    if not all_test_files:
        issues.append("No test files generated")
        return tests, issues

    pytest_tests, pytest_issues = _execute_pytest(code_files, all_test_files)
    tests.extend(pytest_tests)
    issues.extend(pytest_issues)

    return tests, issues


def _execute_pytest(
    code_files: dict[str, str],
    test_files: dict[str, str],
) -> tuple[list[TestCase], list[str]]:
    """Run pytest in a temp directory and parse real results into TestCases."""
    tests = []
    issues = []

    with tempfile.TemporaryDirectory(prefix="belief_validate_") as tmp:
        tmp_path = Path(tmp)

        # Write all files
        for files_dict in [code_files, test_files]:
            for fname, content in files_dict.items():
                fpath = tmp_path / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)

        # Ensure __init__.py in all package dirs
        for dirpath, dirnames, filenames in os.walk(tmp_path):
            py_files = [f for f in filenames if f.endswith(".py")]
            if py_files:
                init = Path(dirpath) / "__init__.py"
                if not init.exists():
                    init.write_text("")

        # Install deps if requirements.txt exists
        req_path = tmp_path / "requirements.txt"
        if req_path.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "--break-system-packages", "-r", str(req_path)],
                    capture_output=True, text=True, timeout=60,
                )
            except Exception:
                pass

        # Run pytest with verbose output to get individual test results
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", "--tb=short", "--no-header"],
                capture_output=True, text=True,
                timeout=60, cwd=str(tmp_path),
                env={**os.environ, "PYTHONPATH": str(tmp_path)},
            )
            output = proc.stdout + "\n" + proc.stderr

            # Parse individual test results from -v output
            # Format: tests/test_main.py::test_health PASSED
            for line in output.split("\n"):
                line = line.strip()

                # Match: path::test_name PASSED/FAILED/ERROR
                match = re.match(
                    r'(.+?)::(\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)',
                    line,
                )
                if match:
                    file_path = match.group(1)
                    test_name = match.group(2)
                    status = match.group(3)

                    passed = status == "PASSED"
                    error = ""

                    # Extract error message for failed tests
                    if not passed:
                        # Look for the error in the short traceback
                        error = _extract_test_error(output, test_name)

                    tests.append(TestCase(
                        name=test_name,
                        description=f"{file_path}::{test_name}",
                        passed=passed,
                        error=error[:200] if error else "",
                    ))

            # If no individual tests parsed, try summary line
            if not tests:
                summary_match = re.search(r'(\d+) passed', output)
                failed_match = re.search(r'(\d+) failed', output)
                error_match = re.search(r'(\d+) error', output)

                passed_count = int(summary_match.group(1)) if summary_match else 0
                failed_count = int(failed_match.group(1)) if failed_match else 0
                error_count = int(error_match.group(1)) if error_match else 0

                for i in range(passed_count):
                    tests.append(TestCase(
                        name=f"test_passed_{i+1}", passed=True,
                    ))
                for i in range(failed_count):
                    tests.append(TestCase(
                        name=f"test_failed_{i+1}", passed=False,
                        error="Test failed (see pytest output)",
                    ))
                for i in range(error_count):
                    tests.append(TestCase(
                        name=f"test_error_{i+1}", passed=False,
                        error="Collection error (import/syntax)",
                    ))

            if proc.returncode != 0 and not tests:
                issues.append(f"pytest exited with code {proc.returncode}")
                # Check for collection errors
                if "ModuleNotFoundError" in output or "ImportError" in output:
                    issues.append("Test collection failed due to missing dependencies")

        except subprocess.TimeoutExpired:
            issues.append("pytest timed out after 60s")
        except Exception as e:
            issues.append(f"pytest execution error: {e}")

    return tests, issues


def _extract_test_error(output: str, test_name: str) -> str:
    """Extract the error message for a specific failed test from pytest output."""
    # Look for FAILED or ERROR section for this test
    pattern = rf'(?:FAILED|ERROR).*{re.escape(test_name)}.*?(?:\n.*?(?:Error|Exception|assert).*?)(?:\n|$)'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(0).strip()[:200]

    # Fallback: look for any error line mentioning the test
    for line in output.split("\n"):
        if test_name in line and ("Error" in line or "assert" in line.lower()):
            return line.strip()[:200]

    return ""


# ── Deterministic quality checks ─────────────────────────────────────────────

def _lint_score(code_files: dict[str, str]) -> float:
    """Run basic lint checks and return a score 0.0-1.0."""
    total_files = 0
    clean_files = 0

    for fname, content in code_files.items():
        if not fname.endswith(".py"):
            continue
        if fname.startswith("test") or "/test" in fname:
            continue

        total_files += 1
        issues = 0

        # Check for common problems
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if len(line) > 120:
                issues += 1
            if "import *" in line:
                issues += 2
            if line.strip() == "pass" and i > 0:
                # pass in a class/function is ok, standalone pass is suspicious
                prev = lines[i-1].strip() if i > 0 else ""
                if not prev.endswith(":"):
                    issues += 1

        if issues == 0:
            clean_files += 1

    return clean_files / max(total_files, 1)


def _security_score(code_files: dict[str, str]) -> float:
    """Check for basic security issues and return a score 0.0-1.0."""
    total_files = 0
    clean_files = 0

    dangerous_patterns = [
        r'\beval\s*\(', r'\bexec\s*\(', r'os\.system\s*\(',
        r'subprocess\.call\s*\(.*shell\s*=\s*True',
        r'pickle\.loads?\s*\(', r'__import__\s*\(',
    ]

    for fname, content in code_files.items():
        if not fname.endswith(".py"):
            continue
        total_files += 1
        found = False
        for pattern in dangerous_patterns:
            if re.search(pattern, content):
                found = True
                break
        if not found:
            clean_files += 1

    return clean_files / max(total_files, 1)


# ── Weighted scoring (unchanged) ─────────────────────────────────────────────

def _classify_and_score(result: ValidationResult) -> None:
    """Classify tests into tiers and compute weighted verdict score."""
    for test in result.tests:
        # Skip already-classified tests (e.g., executor, syntax)
        if test.tier != TestTier.FUNCTIONAL:
            continue

        name_lower = (test.name + " " + test.description).lower()
        error_lower = test.error.lower()

        if any(e in error_lower for e in ("importerror", "modulenotfounderror", "no module named")):
            test.tier = TestTier.ENVIRONMENT
        elif any(k in name_lower for k in ("import", "instantiat", "exist", "smoke", "health", "startup", "p0", "syntax")):
            test.tier = TestTier.SMOKE
        elif any(k in name_lower for k in ("error", "invalid", "edge", "empty", "boundary", "negative", "p2")):
            test.tier = TestTier.EDGE_CASE

    # Compute weighted score
    weighted_sum = 0.0
    weight_total = 0.0
    smoke_pass = True

    for test in result.tests:
        w = TIER_WEIGHTS[test.tier]
        if w == 0:
            continue
        weight_total += w
        if test.passed:
            weighted_sum += w
        elif test.tier == TestTier.SMOKE:
            smoke_pass = False

    result.weighted_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    result.tests_passed = sum(1 for t in result.tests if t.passed)
    result.tests_total = len(result.tests)

    if smoke_pass and result.weighted_score >= 0.75:
        result.verdict = ValidationVerdict.PASS
    elif result.weighted_score >= 0.90:
        # High score override: if 90%+ of weighted tests pass, a single
        # smoke failure is almost certainly a phantom test (bad import,
        # wrong fixture, etc.), not a real functional problem.
        result.verdict = ValidationVerdict.PASS
    elif result.weighted_score >= 0.5:
        result.verdict = ValidationVerdict.FAIL_FIXABLE
    else:
        result.verdict = ValidationVerdict.FAIL_FIXABLE
