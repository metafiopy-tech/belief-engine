"""FastAPI shim — wraps ``belief --mode local --goal <prompt>`` as an
OpenAI-compatible /v1/completions endpoint for the BigCode evaluation
harness.

Per request, the shim:

  1. Spawns ``belief --mode local --goal <prompt> --json-output``.
  2. Parses the JSON summary line at the end of stdout for the
     ``run_id``.
  3. Reads the first non-test ``.py`` file from
     ``$BELIEF_OUTPUT_DIR/<run_id>/`` and returns it as the completion
     text.

Run::

    pip install -e ".[bench]"
    uvicorn scripts.bigcode_shim:app --host 127.0.0.1 --port 8000

Then point the BigCode harness at ``http://localhost:8000/v1`` with a
model name of ``belief-engine-local``.

Token counts in the response are zeros — pass@k is computed by the
harness from the completion text, so token bookkeeping isn't load
bearing here. The shim deliberately does NOT propagate ``temperature``,
``top_p``, or ``seed`` from the harness request: the engine has its own
seeded local pipeline (Session 1 default seed=42), and overriding it
per request would defeat determinism.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("belief.shim.bigcode")

DEFAULT_TIMEOUT_S = 1800
DEFAULT_MODEL_NAME = "belief-engine-local"


# ---------------------------------------------------------------------------
# Request / response schemas (subset of the OpenAI completions surface)
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """Minimal OpenAI completions request shape.

    BigCode's ``ollama_chat`` / OpenAI-compatible drivers send fields
    we don't need (``temperature``, ``top_p``, ``logit_bias`` …);
    ``extra='allow'`` keeps them out of the way without 422-ing on
    every call.
    """

    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL_NAME
    prompt: str | list[str]
    max_tokens: int | None = None
    n: int = 1
    stop: list[str] | str | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: None = None
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage = Field(default_factory=CompletionUsage)


# ``from __future__ import annotations`` means Pydantic sees forward
# refs as strings. Resolve them now while the module's namespace is in
# scope — otherwise instantiation later fails with PydanticUserError
# when the shim is loaded via importlib (as the tests do).
CompletionRequest.model_rebuild()
CompletionResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_root() -> Path:
    """Resolve ``$BELIEF_OUTPUT_DIR`` (default ``./output``) to an
    absolute Path. Settings.output_dir uses the same default.
    """
    return Path(os.environ.get("BELIEF_OUTPUT_DIR", "./output")).resolve()


def _extract_first_code_file(run_dir: Path) -> str:
    """Read the first non-test ``.py`` file in ``run_dir``.

    Returns ``""`` if the directory doesn't exist or contains no Python
    files — the harness will mark that example as a failure but the
    shim itself stays up.
    """
    if not run_dir.is_dir():
        return ""
    py_files = sorted(run_dir.glob("*.py"))
    # Prefer non-test, non-conftest files first.
    for pf in py_files:
        name = pf.name.lower()
        if not (name.startswith("test_") or name == "tests.py" or name == "conftest.py"):
            try:
                return pf.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("read failed for %s: %s", pf, e)
                return ""
    if py_files:
        try:
            return py_files[0].read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("read failed for %s: %s", py_files[0], e)
            return ""
    return ""


def _parse_summary_line(stdout: str) -> dict[str, Any]:
    """Find the JSON summary line printed by ``belief --json-output``.

    The cli prints a single ``json.dumps(summary)`` line at the very
    end after the human-readable BUILD COMPLETE block. Search from the
    end for robustness against any future trailing chatter.
    """
    if not stdout:
        return {}
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                payload = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "run_id" in payload:
                return payload
    return {}


# ---------------------------------------------------------------------------
# HumanEval / MBPP-style code-stub adapter
# ---------------------------------------------------------------------------
#
# BigCode's HumanEval and MBPP tasks send the model a code stub: optional
# imports, then ``def NAME(args):`` with a docstring describing what the
# function should do. The expected completion is the *body* of that
# function — the lines that go after the docstring. The harness assembles
# ``prompt + completion`` and runs the test suite against the result.
#
# The Belief Engine's intake agent expects an English instruction, not a
# code prefix. So we:
#
#   1. Detect a stub-shaped prompt.
#   2. Rewrite it to "Implement the function below ... <stub>".
#   3. Run the engine and read its full Python file.
#   4. Extract just the body of the function with the matching name.
#   5. Return that body so harness-side ``prompt + completion`` is valid.
#
# When the prompt isn't stub-shaped (e.g. a natural-language goal sent
# directly), the adapter is a no-op and the existing whole-file path runs.


_DEF_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def _looks_like_code_stub(text: str) -> bool:
    """A prompt is a code stub if it parses as Python AND has at least
    one top-level function definition with a docstring as its first
    statement."""
    if not text or "def " not in text:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    return True
    return False


def _detect_function_name(stub: str) -> str | None:
    """Return the name of the *last* top-level def in the stub.

    HumanEval prompts often have helper imports, sometimes a small
    helper function, then the target function last. Picking the last
    def is more robust than picking the first.
    """
    try:
        tree = ast.parse(stub)
    except SyntaxError:
        m = list(_DEF_RE.finditer(stub))
        return m[-1].group(1) if m else None
    name: str | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
    return name


def _rewrite_stub_to_goal(stub: str) -> str:
    """Wrap a HumanEval-style stub in an English instruction the
    intake agent can act on."""
    return (
        "Implement the function described below. Return a single Python "
        "file containing the complete function definition (including the "
        "signature and docstring as given). Do not include __main__ "
        "blocks, example usage, or test code outside the function.\n\n"
        f"{stub.rstrip()}\n"
    )


def _extract_function_body(source: str, fn_name: str, *, indent: str = "    ") -> str:
    """Return the body of ``fn_name`` from ``source``, indented for
    insertion after the original stub.

    Drops the docstring (the harness's stub already has it). Falls back
    to returning the whole source if parsing fails or the function
    can't be found — better to give the harness *something* than a
    blank completion.
    """
    if not source:
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    target: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            target = node
            break
    if target is None:
        return source
    body = list(target.body)
    # Drop a leading docstring expression if present.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if not body:
        return f"{indent}pass\n"
    lines: list[str] = []
    for stmt in body:
        try:
            stmt_src = ast.unparse(stmt)
        except Exception:  # pragma: no cover — ast.unparse is robust on 3.9+
            continue
        for line in stmt_src.split("\n"):
            lines.append(indent + line if line else "")
    return "\n".join(lines) + "\n"


async def _run_engine(prompt: str, *, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Spawn the engine for a single prompt and return its summary +
    extracted code.

    All subprocess work runs in a worker thread via ``asyncio.to_thread``
    so the FastAPI event loop stays responsive while the build runs
    (typically 1–5 minutes on a 14B local model).
    """
    cmd = [
        "belief",
        "build",
        "--mode",
        "local",
        "--goal",
        prompt,
        "--json-output",
    ]
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"run_id": "", "verdict": "timeout", "code": "", "stderr": "timeout"}
    except FileNotFoundError:
        # belief CLI not on PATH — surface as 503 to the harness.
        raise HTTPException(
            status_code=503,
            detail="belief CLI not found on PATH; install with `pip install -e .`",
        )

    summary = _parse_summary_line(proc.stdout or "")
    run_id = str(summary.get("run_id") or "")
    code = _extract_first_code_file(_output_root() / run_id) if run_id else ""
    return {
        "run_id": run_id,
        "verdict": str(summary.get("verdict") or "unknown"),
        "code": code,
        "stderr": (proc.stderr or "")[-500:] if proc.returncode != 0 else "",
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(title="belief-engine BigCode shim", version="0.1.0")


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest) -> CompletionResponse:
    prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
    if not prompts:
        raise HTTPException(status_code=400, detail="prompt is required")

    choices: list[CompletionChoice] = []
    created = int(time.time())
    for i, p in enumerate(prompts):
        # Stub-shaped prompts (HumanEval/MBPP) get rewritten to an
        # English goal before the engine runs, and the engine's full
        # file is post-processed to just the target function body so
        # harness-side prompt+completion concatenation produces valid
        # Python. Non-stub prompts pass through untouched.
        is_stub = _looks_like_code_stub(p)
        if is_stub:
            fn_name = _detect_function_name(p)
            engine_input = _rewrite_stub_to_goal(p)
        else:
            fn_name = None
            engine_input = p

        result = await _run_engine(engine_input)
        engine_code = result["code"]
        if is_stub and fn_name and engine_code:
            completion_text = _extract_function_body(engine_code, fn_name)
        else:
            completion_text = engine_code

        finish_reason = "stop"
        if result["verdict"] == "timeout":
            finish_reason = "length"
        elif not completion_text:
            finish_reason = "content_filter"  # engine produced nothing usable
        choices.append(
            CompletionChoice(
                text=completion_text,
                index=i,
                finish_reason=finish_reason,
            )
        )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=created,
        model=req.model,
        choices=choices,
    )


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """OpenAI-compat: harnesses commonly ping /v1/models on startup."""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_NAME,
                "object": "model",
                "owned_by": "belief-engine",
            }
        ],
    }


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}
