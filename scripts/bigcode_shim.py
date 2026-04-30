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

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# scripts/ is not a regular package; pull the shared adapter helpers in
# whether we're invoked via "uvicorn scripts.bigcode_shim:app" (package
# import path) or via importlib (tests). Either way the module lives
# next to this file.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _humaneval_adapter import (  # noqa: E402
    detect_function_name,
    extract_function_body,
    looks_like_code_stub,
    rewrite_stub_to_goal,
)

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
# The actual logic lives in scripts/_humaneval_adapter.py so the
# generator (scripts/generate_humaneval_completions.py) can reuse it.
# These ``_``-prefixed wrappers preserve the shim's existing private
# test surface (``shim._looks_like_code_stub`` etc.).

_looks_like_code_stub = looks_like_code_stub
_detect_function_name = detect_function_name
_rewrite_stub_to_goal = rewrite_stub_to_goal
_extract_function_body = extract_function_body


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
