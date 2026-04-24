"""Run a goal through a raw LLM with no engine infrastructure.

No soil, no covenants, no skeleton, no debugger loop, no memory.
Just: system prompt + goal → code → execute tests → score.
This is the control group for A/B experiments.

Deliberately has zero imports from belief.*  so the control group
can never accidentally inherit engine behaviour.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# System prompt given to the raw model
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Python code generator. Given a goal, produce ALL files needed
as a working project. Return your response as a series of file blocks
in this exact format:

### FILE: filename.py
```python
code here
```

### FILE: requirements.txt
```
requirements here
```

### FILE: test_main.py
```python
import pytest
# tests here
```

Rules:
- Include a main entry point
- Include requirements.txt (list third-party deps only; never stdlib)
- Include at least 3 pytest tests in test_main.py or test_*.py
- All code must be syntactically valid Python
- Use only well-known packages (fastapi, click, pydantic, requests, etc.)
- Do NOT include explanatory text outside of file blocks
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RawRunResult:
    goal: str
    model: str
    code_files: dict[str, str] = field(default_factory=dict)
    tests_passed: int = 0
    tests_total: int = 0
    weighted_score: float = 0.0
    time_seconds: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_raw(
    goal: str,
    model: str = "qwen2.5-coder:14b",
    base_url: str = "http://localhost:11434",
    timeout: int = 300,
) -> RawRunResult:
    """Run a goal through a raw model with no engine scaffolding.

    Returns a RawRunResult regardless of whether the model or tests
    succeed — failures are recorded in .error and .weighted_score=0.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "httpx is required for the raw runner. Install it with: pip install httpx"
        ) from exc

    start = time.time()

    # 1. Generate code from the raw model
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": goal},
                    ],
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 4096,
                        "seed": 42,  # determinism (matches engine)
                        "num_gpu": 99,  # full Metal offload (matches engine)
                        "num_thread": 6,  # matches engine
                        "num_batch": 256,  # matches engine
                        "num_keep": 512,  # KV-cache pinning (matches engine)
                        "mirostat": 0,  # greedy sampling (matches engine)
                    },
                },
            )
            resp.raise_for_status()
            raw_output = resp.json()["message"]["content"]
    except Exception as exc:
        return RawRunResult(
            goal=goal,
            model=model,
            time_seconds=time.time() - start,
            error=f"Model call failed: {exc}",
        )

    # 2. Parse file blocks from response
    files = parse_file_blocks(raw_output)
    if not files:
        return RawRunResult(
            goal=goal,
            model=model,
            code_files={},
            time_seconds=time.time() - start,
            error="No ### FILE: blocks found in model output",
        )

    # 3. Write files to temp dir and run tests
    tests_passed = 0
    tests_total = 0
    run_error: Optional[str] = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, content in files.items():
            path = Path(tmpdir) / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        # Install requirements if present
        req_file = Path(tmpdir) / "requirements.txt"
        if req_file.exists():
            try:
                subprocess.run(
                    ["pip3", "install", "-r", str(req_file), "-q", "--break-system-packages"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=60,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # Non-fatal — tests may still pass if deps already installed

        # Run pytest
        try:
            proc = subprocess.run(
                ["python3", "-m", "pytest", ".", "-q", "--tb=no", "--timeout=30", "--no-header"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                timeout=120,
            )
            tests_passed, tests_total = parse_pytest_output(proc.stdout)
            if tests_total == 0 and proc.returncode not in (0, 1):
                run_error = f"pytest failed to run (rc={proc.returncode}): {proc.stderr[:200]}"
        except subprocess.TimeoutExpired:
            run_error = "pytest timed out after 120s"
        except FileNotFoundError:
            run_error = "python3/pytest not found"

    weighted = tests_passed / max(tests_total, 1) if tests_total > 0 else 0.0

    return RawRunResult(
        goal=goal,
        model=model,
        code_files=files,
        tests_passed=tests_passed,
        tests_total=tests_total,
        weighted_score=weighted,
        time_seconds=time.time() - start,
        error=run_error,
    )


# ---------------------------------------------------------------------------
# Parsers (pure functions, easily unit-tested)
# ---------------------------------------------------------------------------


def parse_file_blocks(text: str) -> dict[str, str]:
    """Extract ### FILE: <name> / ``` ... ``` blocks from model output.

    Matches the format requested in the system prompt:
        ### FILE: filename.py
        ```python
        code here
        ```

    The language tag after the opening fence is optional and ignored.
    Returns {} if no blocks are found.
    """
    files: dict[str, str] = {}
    # Allow optional whitespace after FILE: and handle Windows CRLF
    pattern = re.compile(
        r"###\s*FILE:\s*(\S+)\s*\r?\n```[^\n]*\r?\n(.*?)```",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        filename = match.group(1).strip()
        content = match.group(2)
        # Normalise trailing whitespace but keep internal newlines
        content = content.rstrip()
        files[filename] = content
    return files


def parse_pytest_output(stdout: str) -> tuple[int, int]:
    """Extract (passed, total) from pytest -q summary line.

    Handles: "3 passed", "2 passed, 1 failed", "0 passed, 1 error",
    "1 passed, 2 failed, 1 error", etc.
    Returns (0, 0) when pytest produced no recognisable summary.
    """
    passed = 0
    total = 0

    m = re.search(r"(\d+) passed", stdout)
    if m:
        passed = int(m.group(1))
        total += passed

    for label_pattern in (r"failed", r"errors?"):
        m = re.search(rf"(\d+) {label_pattern}", stdout)
        if m:
            total += int(m.group(1))

    return passed, total
