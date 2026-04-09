"""Reproduction Tester — Kimi-Dev TestWriter Role.

Generates pytest tests that:
1. FAIL on the current (buggy) code — proving the bug exists
2. PASS after the fix is applied — proving the fix works

This is the TestWriter in Kimi-Dev's Duo framework. The key finding:
3 patches × 3 reproduction tests beats 40 patches with majority voting
(48.0% → 60.4% on SWE-bench Verified).

The reproduction test is different from regular tests:
- Regular tests: verify correct behavior
- Reproduction tests: demonstrate the INCORRECT behavior first,
  then verify the fix makes it correct

Usage:
    from belief.agents.reproduction_tester import generate_reproduction_tests
    tests = await generate_reproduction_tests(issue, localized_code, llm)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("belief.agents.reproduction_tester")


REPRO_SYSTEM = """You are a reproduction test writer for an automated bug-fixing system.
Your job is to write a pytest test that DEMONSTRATES a bug described in the issue.

CRITICAL RULES:
1. The test must FAIL on the current buggy code — this proves the bug exists.
2. The test must PASS once the bug is fixed — this proves the fix works.
3. Write exactly ONE focused test function that isolates the bug.
4. Import only from modules that exist in the codebase.
5. The test should be deterministic — no randomness, no timing dependencies.
6. Include a clear comment: # This test FAILS because: <reason>

Output format:
```python
<complete test file with imports and one test function>
```"""


REPRO_PROMPT = """## Bug Report
{issue}

## Relevant Code
{code_context}

## File Being Fixed
{target_file}

Write a reproduction test that FAILS on this code due to the reported bug.
The test should pass once the bug is fixed."""


async def generate_reproduction_tests(
    issue: str,
    target_file: str,
    code_context: str,
    llm=None,
    n: int = 3,
) -> list[str]:
    """Generate N reproduction tests for a bug.

    Each test is generated independently (different temperatures) to
    maximize diversity for self-play ranking.

    Args:
        issue: Bug report or issue description
        target_file: The file being patched
        code_context: Relevant source code (localized)
        llm: LLM client
        n: Number of tests to generate (default 3 for Kimi-Dev pattern)

    Returns:
        List of test file contents (Python source code strings)
    """
    if llm is None:
        return []

    import asyncio

    prompt = REPRO_PROMPT.format(
        issue=issue[:2000],
        code_context=code_context[:3000],
        target_file=target_file,
    )

    # Generate n tests at different temperatures for diversity
    temps = [0.2 + i * 0.2 for i in range(n)]  # [0.2, 0.4, 0.6]

    async def _gen_one(temp: float) -> str | None:
        try:
            raw = await llm.generate_text(
                role="tester",
                system=REPRO_SYSTEM,
                prompt=prompt,
                temperature=temp,
                max_tokens=1500,
            )
            return _extract_test_code(raw)
        except Exception as e:
            logger.debug(f"Reproduction test generation failed at temp={temp}: {e}")
            return None

    results = await asyncio.gather(*[_gen_one(t) for t in temps])

    # Filter out None and invalid tests
    valid = []
    for test_code in results:
        if test_code and _validate_test(test_code):
            valid.append(test_code)

    logger.info(f"Reproduction tester: generated {len(valid)}/{n} valid tests")
    return valid


def _extract_test_code(raw: str) -> str | None:
    """Extract Python code from LLM response."""
    import re

    # Try to extract from code block
    match = re.search(r'```python\n(.*?)```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to extract from bare code
    match = re.search(r'((?:import|from|def test_).*)', raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _validate_test(code: str) -> bool:
    """Check that the test code is valid Python with at least one test function."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Must have at least one test function
    has_test = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )

    return has_test


async def verify_reproduction_test(
    test_code: str,
    code_files: dict[str, str],
    should_fail: bool = True,
) -> bool:
    """Run a reproduction test against code and verify it fails/passes as expected.

    Args:
        test_code: The reproduction test source
        code_files: The codebase to test against
        should_fail: If True, the test should FAIL (pre-fix). If False, it should PASS (post-fix).

    Returns:
        True if the test behaved as expected
    """
    import asyncio
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    def _run():
        with tempfile.TemporaryDirectory(prefix="belief_repro_") as tmp:
            tmp_path = Path(tmp)

            # Write code files
            for fname, content in code_files.items():
                fpath = tmp_path / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)

            # Write the reproduction test
            test_path = tmp_path / "test_reproduction.py"
            test_path.write_text(test_code)

            # Run pytest
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "test_reproduction.py",
                     "-x", "--tb=short", "-q"],
                    capture_output=True, text=True,
                    timeout=30, cwd=str(tmp_path),
                    env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)},
                )

                test_passed = proc.returncode == 0

                if should_fail:
                    return not test_passed  # We WANT it to fail
                else:
                    return test_passed  # We WANT it to pass

            except subprocess.TimeoutExpired:
                return False
            except Exception:
                return False

    return await asyncio.to_thread(_run)
