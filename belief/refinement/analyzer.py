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
2. Which SOURCE file (not test file) contains the bug
3. Which specific function or class in that file needs fixing
4. A clear, actionable explanation of what's wrong and how to fix it

Focus on the MOST IMPACTFUL failure — the one that, if fixed, would likely fix other related failures.

Respond ONLY with valid JSON:
{
    "test_name": "test_create_task",
    "target_file": "app/service.py",
    "target_function": "create_task",
    "diagnosis": "The create_task function doesn't handle the case where project_id is None. The route passes it as an optional parameter but the SQLAlchemy model requires it. Add a default value or validation.",
    "expected_behavior": "Should create a task even without a project_id",
    "actual_behavior": "Raises IntegrityError because project_id is NOT NULL in the model"
}"""


ANALYZER_PROMPT = """## Test Output
```
{test_output}
```

## Source Files Available
{file_list}

## Previous Fix Attempts (DO NOT repeat these)
{previous_fixes}

Analyze the MOST IMPACTFUL test failure and identify the source file to fix.
Focus on failures in source code, not test code — we don't modify tests."""


async def analyze_failures(state: RefinementState, llm) -> dict:
    """Analyze test failures and generate a verbal diagnosis.
    
    Returns dict with 'diagnosis', 'target_file', 'target_function'.
    """
    # Build file list (source files only, not tests)
    file_list = "\n".join(
        f"  {f} ({len(c)} chars, {c.count(chr(10))+1} lines)"
        for f, c in sorted(state.code_files.items())
        if f.endswith(".py") and "/test" not in f and not f.startswith("test")
    )
    
    previous = "\n".join(f"  - {p}" for p in state.previous_fixes) if state.previous_fixes else "  (none — first cycle)"
    
    prompt = ANALYZER_PROMPT.format(
        test_output=state.test_output[-3000:],  # Last 3K chars of test output
        file_list=file_list,
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
        
        # Validate target file exists
        if target_file not in state.code_files:
            # Try to find a close match
            for f in state.code_files:
                if f.endswith(target_file) or target_file.endswith(f):
                    target_file = f
                    break
            else:
                # Fall back to most likely file
                logger.warning(f"Analyzer: target file '{target_file}' not found, using heuristic")
                target_file = _guess_target_file(state.test_output, state.code_files)
        
        logger.info(f"Analyzer: {target_file}::{target_function} — {diagnosis[:80]}...")
        
        return {
            "diagnosis": diagnosis,
            "target_file": target_file,
            "target_function": target_function,
            "test_name": result.test_name,
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
    target_file: str = ""
    target_function: str = ""
    diagnosis: str = ""
    expected_behavior: str = ""
    actual_behavior: str = ""
