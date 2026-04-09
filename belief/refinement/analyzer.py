"""Analyze Failures — verbal self-reflection on test output.

The critical insight from Reflexion: raw test output (AssertionError: expected 5, got 6)
is insufficient. The model needs semantic interpretation of what went wrong.

This node:
1. Parses pytest output to extract failing test names and tracebacks
2. Calls the LLM to generate a natural language diagnosis
3. Identifies which file and function to fix
4. Returns the diagnosis for the fixer node

Without this step, Reflexion's ablation showed ZERO improvement.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from belief.refinement import RefinementState

logger = logging.getLogger("belief.refinement.analyzer")

# ── pytest output parsing ────────────────────────────────────────────────────

def parse_test_results(output: str) -> tuple[int, int, list[str], list[str]]:
    """Parse pytest output to extract pass/fail counts and details.
    
    Returns: (passed, total, failed_test_ids, failure_details)
    """
    # Extract summary line: "5 passed, 15 failed"
    passed = 0
    failed = 0
    
    summary = re.search(r'(\d+) passed', output)
    if summary:
        passed = int(summary.group(1))
    
    fail_match = re.search(r'(\d+) failed', output)
    if fail_match:
        failed = int(fail_match.group(1))
    
    error_match = re.search(r'(\d+) error', output)
    if error_match:
        failed += int(error_match.group(1))
    
    total = passed + failed
    
    # Extract individual failing test names
    failed_ids = re.findall(r'FAILED\s+(\S+)', output)
    if not failed_ids:
        # Try alternative format: "test_file.py::test_name"
        failed_ids = re.findall(r'(?:ERROR|FAIL)\s+(\S+::\S+)', output)
    
    # Extract failure details (tracebacks)
    failure_blocks = []
    # Split on FAILED or ERROR markers
    sections = re.split(r'_{5,}\s+(?:FAILED|ERROR)', output)
    for section in sections[1:]:  # Skip preamble
        # Take first 500 chars of each failure
        failure_blocks.append(section[:500].strip())
    
    # If no structured failures found, take the last 1000 chars
    if not failure_blocks and failed > 0:
        failure_blocks = [output[-1000:]]
    
    return passed, total, failed_ids, failure_blocks[:5]  # Max 5 failures


# ── Analyzer node ────────────────────────────────────────────────────────────

ANALYZER_SYSTEM = """You are a test failure analyst for an autonomous code generation system.
Your job is to read pytest output and produce a precise diagnosis of ONE failure.

You must identify:
1. Which specific test failed and what it expected vs what it got
2. WHETHER THE BUG IS IN THE CODE OR THE TEST. Use these rules:
   - ImportError/ModuleNotFoundError in a test file → TEST BUG (wrong import path)
   - 404 status code in assertion → TEST BUG (wrong URL/endpoint name)
   - 405 status code → TEST BUG (wrong HTTP method)
   - 422 status code → TEST BUG (wrong request payload format)
   - 500 status code → CODE BUG (server crash)
   - AttributeError in test file → TEST BUG (wrong attribute/method name)
   - NameError in test file → TEST BUG (wrong variable/function name)
   - AssertionError where actual value is reasonable → TEST BUG (wrong assertion)
3. Which file (source OR test) contains the bug
4. A clear, actionable explanation of what's wrong and how to fix it

Focus on the MOST IMPACTFUL failure — the one that, if fixed, would likely fix other related failures.

Respond ONLY with valid JSON:
{
    "test_name": "test_create_task",
    "bug_location": "test",
    "target_file": "tests/test_main.py",
    "target_function": "test_create_task",
    "diagnosis": "The test posts to /tasks but the API route is /api/tasks. Fix the URL in the test.",
    "expected_behavior": "POST to correct endpoint creates a task",
    "actual_behavior": "404 because the URL is wrong in the test"
}"""


ANALYZER_PROMPT = """## Test Output
```
{test_output}
```

## Source Files Available
{file_list}

## Test Files Available
{test_file_list}

## Previous Fix Attempts (DO NOT repeat these)
{previous_fixes}

Analyze the MOST IMPACTFUL test failure. Determine whether the bug is in SOURCE code
or in TEST code. If the test is testing the wrong endpoint, using the wrong import,
or asserting the wrong value — the bug is in the TEST, not the code."""


async def analyze_failures(state: RefinementState, llm) -> dict:
    """Analyze test failures and generate a verbal diagnosis.
    
    Returns dict with 'diagnosis', 'target_file', 'target_function', 'bug_location'.
    bug_location is 'code' or 'test' — tells the fixer which file pool to target.
    """
    # Build file list (source files only, not tests)
    file_list = "\n".join(
        f"  {f} ({len(c)} chars, {c.count(chr(10))+1} lines)"
        for f, c in sorted(state.code_files.items())
        if f.endswith(".py") and "/test" not in f and not f.startswith("test")
    )
    
    # Build test file list
    test_file_list = "\n".join(
        f"  {f} ({len(c)} chars, {c.count(chr(10))+1} lines)"
        for f, c in sorted(state.test_files.items())
        if f.endswith(".py")
    ) if state.test_files else "  (no test files)"
    
    previous = "\n".join(f"  - {p}" for p in state.previous_fixes) if state.previous_fixes else "  (none — first cycle)"
    
    prompt = ANALYZER_PROMPT.format(
        test_output=state.test_output[-3000:],  # Last 3K chars of test output
        file_list=file_list,
        test_file_list=test_file_list,
        previous_fixes=previous,
    )
    
    try:
        from belief.config import ModelRouter
        from belief.llm import LLMClient
        
        if llm is None:
            router = ModelRouter()
            llm = LLMClient(router)
        
        result = await llm.generate_structured(
            role="debugger",
            system=ANALYZER_SYSTEM,
            prompt=prompt,
            response_schema=_AnalysisResult,
            temperature=0.2,
            max_tokens=1000,
        )
        
        diagnosis = result.diagnosis
        target_file = result.target_file
        target_function = result.target_function or ""
        bug_location = result.bug_location or "code"
        
        # Validate target file exists — search BOTH code and test files
        all_files = dict(state.code_files)
        if state.test_files:
            all_files.update(state.test_files)
        
        if target_file not in all_files:
            # Try to find a close match in all files
            for f in all_files:
                if f.endswith(target_file) or target_file.endswith(f):
                    target_file = f
                    break
            else:
                # Fall back to heuristic based on bug_location
                if bug_location == "test" and state.test_files:
                    target_file = list(state.test_files.keys())[0]
                    logger.warning(f"Analyzer: target '{result.target_file}' not found, using first test file")
                else:
                    logger.warning(f"Analyzer: target '{result.target_file}' not found, using heuristic")
                    target_file = _guess_target_file(state.test_output, state.code_files)
        
        logger.info(f"Analyzer: [{bug_location}] {target_file}::{target_function} — {diagnosis[:80]}...")
        
        return {
            "diagnosis": diagnosis,
            "target_file": target_file,
            "target_function": target_function,
            "test_name": result.test_name,
            "bug_location": bug_location,
        }
        
    except Exception as e:
        logger.warning(f"Analyzer failed: {e}")
        # Fallback: guess from test output
        target_file = _guess_target_file(state.test_output, state.code_files)
        return {
            "diagnosis": f"Test failures detected. Primary issue likely in {target_file}.",
            "target_file": target_file,
            "target_function": "",
            "test_name": "",
            "bug_location": "code",
        }


def _guess_target_file(test_output: str, code_files: dict[str, str]) -> str:
    """Heuristic: find the source file most mentioned in test output."""
    counts = {}
    for fname in code_files:
        if fname.endswith(".py") and "/test" not in fname and not fname.startswith("test"):
            # Count mentions of the filename in test output
            base = fname.split("/")[-1].replace(".py", "")
            counts[fname] = test_output.lower().count(base.lower())
    
    if counts:
        return max(counts, key=counts.get)
    return list(code_files.keys())[0]


# ── Pydantic model for structured LLM output ────────────────────────────────

from pydantic import BaseModel, Field

class _AnalysisResult(BaseModel):
    test_name: str = ""
    bug_location: str = "code"  # "code" or "test"
    target_file: str = ""
    target_function: str = ""
    diagnosis: str = ""
    expected_behavior: str = ""
    actual_behavior: str = ""
