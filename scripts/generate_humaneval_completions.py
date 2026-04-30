"""Generate HumanEval completions in BigCode-evaluation-harness format.

The BigCode harness in this version (`bigcode-project/bigcode-evaluation-harness`)
loads models via Hugging Face transformers and has no HTTP/API driver,
so we cannot point it at the FastAPI shim over the network. Instead we
own the generation phase ourselves and feed pre-generated completions
to the harness via its ``--load_generations_path`` flag for scoring.

This script produces a JSON file in the harness's expected shape — a
list of lists of strings, indexed by task::

    [
        [completion_for_task_0_sample_0, completion_for_task_0_sample_1, ...],
        [completion_for_task_1_sample_0, ...],
        ...
    ]

with one entry per HumanEval problem. With ``--n-samples 1`` (the
default — engine is deterministic at temp=0+seed=42) each inner list
has length 1.

Two backends:

* ``--backend raw`` — direct Ollama HTTP call to ``/api/chat`` with the
  HumanEval prompt as the user message and a brief system prompt that
  asks for "the function body only". Returns the response text as the
  completion.

* ``--backend engine`` — detect the HumanEval stub, rewrite to an
  English goal, ``subprocess.run`` ``belief --mode local --goal
  ... --json-output``, parse the run's ``run_id``, read the first
  non-test ``.py`` from ``$BELIEF_OUTPUT_DIR/<run_id>/``, extract the
  body of the target function. Same shared adapter helpers the shim
  uses (``scripts/_humaneval_adapter.py``).

Resumable — partial completions are written to a per-run ``.partial``
JSONL file as we go, and ``--resume`` will pick up where a prior run
left off (skip task indices already present in ``.partial``).

Usage::

    # raw qwen baseline
    python3 scripts/generate_humaneval_completions.py \\
        --backend raw \\
        --limit 50 \\
        --output results/raw_humaneval_subset_50.json

    # belief engine
    python3 scripts/generate_humaneval_completions.py \\
        --backend engine \\
        --limit 50 \\
        --output results/engine_humaneval_subset_50.json

Then score with the harness::

    cd ~/bigcode-eval
    python3 main.py \\
        --tasks humaneval \\
        --load_generations_path /path/to/raw_humaneval_subset_50.json \\
        --metric_output_path /path/to/raw_humaneval_subset_50_metrics.json \\
        --allow_code_execution \\
        --n_samples 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Shared HumanEval/MBPP adapter helpers — same module the shim uses.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _humaneval_adapter import (  # noqa: E402
    detect_function_name,
    extract_function_body,
    looks_like_code_stub,
    rewrite_stub_to_goal,
)

logger = logging.getLogger("belief.generate_humaneval")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:14b"
DEFAULT_ENGINE_TIMEOUT_S = 1800
DEFAULT_DATASET = "openai_humaneval"
DEFAULT_DATASET_SPLIT = "test"

RAW_SYSTEM_PROMPT = (
    "You are a Python code completion assistant. Given a function "
    "signature and docstring, output ONLY the indented function body "
    "(the lines that follow the docstring). Do not repeat the "
    "signature, the docstring, or any imports. Do not include test "
    "code, __main__ blocks, or explanations. Output raw Python only."
)

_RUN_ID_RE = re.compile(r"Run ID:\s*(belief-[a-f0-9]+)")


# ---------------------------------------------------------------------------
# HumanEval loader
# ---------------------------------------------------------------------------


def load_humaneval_problems(
    *,
    limit: int | None = None,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_DATASET_SPLIT,
) -> list[dict[str, Any]]:
    """Load HumanEval problems via the ``datasets`` library.

    Returns a list of dicts with at least ``task_id``, ``prompt``, and
    ``entry_point`` (the function name to test against). The harness
    expects the prompt to be passed through unchanged when scoring.
    """
    try:
        from datasets import load_dataset  # imported lazily — heavy
    except ImportError as e:
        raise SystemExit(
            "datasets not installed. Run: pip3 install datasets --break-system-packages"
        ) from e

    ds = load_dataset(dataset_name, split=split)
    problems: list[dict[str, Any]] = []
    for row in ds:
        problems.append(
            {
                "task_id": row.get("task_id", ""),
                "prompt": row.get("prompt", ""),
                "entry_point": row.get("entry_point", ""),
            }
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems


# ---------------------------------------------------------------------------
# Backend: raw qwen via Ollama
# ---------------------------------------------------------------------------


def generate_raw_completion(
    prompt: str,
    *,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    seed: int = 42,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_s: int = 300,
    httpx_module: Any = None,  # injectable for tests
) -> str:
    """Send a single HumanEval prompt to Ollama, return the model's
    text response. The system prompt asks for "function body only" so
    the harness's ``prompt + completion`` concatenation produces valid
    Python. If the model returns the full function definition anyway,
    we run it through ``extract_function_body`` to slice off the
    duplicate signature.
    """
    httpx = httpx_module
    if httpx is None:
        import httpx as _httpx  # type: ignore

        httpx = _httpx
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": RAW_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed,
        },
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{ollama_url}/api/chat", json=body)
        resp.raise_for_status()
        payload = resp.json()
    text = ""
    msg = payload.get("message")
    if isinstance(msg, dict):
        text = str(msg.get("content") or "")
    if not text:
        text = str(payload.get("response") or "")
    return _maybe_extract_body_if_full_function(text, prompt)


def _maybe_extract_body_if_full_function(text: str, prompt: str) -> str:
    """If the raw model decided to emit a full ``def NAME(...)``
    redefinition (which Qwen does maybe a third of the time), slice
    out just the body so harness-side concatenation stays valid."""
    if not text:
        return ""
    if not looks_like_code_stub(text):
        return text
    fn_name = detect_function_name(prompt) or detect_function_name(text)
    if not fn_name:
        return text
    # If the function name appears as a `def` at the top of the
    # response, this is a full-redefinition case. Extract the body.
    if re.search(rf"^def\s+{re.escape(fn_name)}\s*\(", text, re.MULTILINE):
        return extract_function_body(text, fn_name)
    return text


# ---------------------------------------------------------------------------
# Backend: belief engine subprocess
# ---------------------------------------------------------------------------


def _output_root() -> Path:
    """Resolve ``$BELIEF_OUTPUT_DIR`` (default ``./output``)."""
    return Path(os.environ.get("BELIEF_OUTPUT_DIR", "./output")).resolve()


def _read_first_code_file(run_dir: Path) -> str:
    if not run_dir.is_dir():
        return ""
    py_files = sorted(run_dir.glob("*.py"))
    for pf in py_files:
        name = pf.name.lower()
        if not (name.startswith("test_") or name == "tests.py" or name == "conftest.py"):
            try:
                return pf.read_text(encoding="utf-8")
            except OSError:
                return ""
    if py_files:
        try:
            return py_files[0].read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def generate_engine_completion(
    prompt: str,
    *,
    timeout_s: int = DEFAULT_ENGINE_TIMEOUT_S,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    output_root_fn: Callable[[], Path] | None = None,
    code_reader: Callable[[Path], str] | None = None,
) -> str:
    """Drive the engine for one HumanEval problem.

    Detects the stub, rewrites to an English goal, spawns ``belief
    build --mode local --goal X``, locates the resulting code
    directory, reads the first non-test .py file, extracts the body of
    the target function. Returns the body (indented, ready to append
    to the harness's prompt).

    All side-effecting calls are injectable for tests.
    """
    sp_run = subprocess_run or subprocess.run
    out_root = output_root_fn() if output_root_fn else _output_root()
    read_code = code_reader or _read_first_code_file

    fn_name = detect_function_name(prompt)
    goal = rewrite_stub_to_goal(prompt) if looks_like_code_stub(prompt) else prompt

    cmd = [
        "belief",
        "build",
        "--mode",
        "local",
        "--goal",
        goal,
        "--json-output",
    ]
    try:
        proc = sp_run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("engine timed out on prompt; returning empty")
        return ""
    except FileNotFoundError as e:
        raise SystemExit(
            "belief CLI not on PATH; install with: pip3 install -e . --break-system-packages"
        ) from e

    stdout = getattr(proc, "stdout", "") or ""
    run_id = ""
    m = _RUN_ID_RE.search(stdout)
    if m:
        run_id = m.group(1)
    else:
        # Fall back to the JSON summary line.
        for line in reversed(stdout.splitlines()):
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    payload = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and "run_id" in payload:
                    run_id = str(payload["run_id"])
                    break

    if not run_id:
        logger.warning("no run_id parsed from engine stdout; returning empty")
        return ""

    code = read_code(out_root / run_id)
    if not code:
        logger.warning("engine produced no code for run %s; returning empty", run_id)
        return ""

    if fn_name:
        return extract_function_body(code, fn_name)
    return code


# ---------------------------------------------------------------------------
# Resumable I/O
# ---------------------------------------------------------------------------


def _partial_path_for(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".partial.jsonl")


def _load_partial(partial_path: Path) -> dict[int, str]:
    """Read previously-generated completions keyed by task index."""
    if not partial_path.exists():
        return {}
    out: dict[int, str] = {}
    with partial_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in row and "completion" in row:
                out[int(row["index"])] = str(row["completion"])
    return out


def _append_partial(partial_path: Path, index: int, completion: str) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a") as f:
        f.write(json.dumps({"index": index, "completion": completion}) + "\n")


def _write_final(output: Path, completions_by_index: dict[int, str], n_problems: int) -> None:
    """Assemble the BigCode list-of-lists JSON from per-problem
    completions and write atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: list[list[str]] = []
    for i in range(n_problems):
        payload.append([completions_by_index.get(i, "")])
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(output)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    backend: str,
    output: Path,
    limit: int | None,
    ollama_url: str,
    ollama_model: str,
    seed: int,
    temperature: float,
    engine_timeout_s: int,
    resume: bool,
) -> int:
    """Top-level driver. Returns 0 on full success, non-zero if any
    problem produced an empty completion."""
    problems = load_humaneval_problems(limit=limit)
    n = len(problems)
    if n == 0:
        logger.error("no problems loaded")
        return 1

    partial = _partial_path_for(output)
    done = _load_partial(partial) if resume else {}
    if done:
        logger.info("resume: skipping %d already-done tasks", len(done))

    n_empty = 0
    t_start = time.time()
    for i, prob in enumerate(problems):
        if i in done:
            continue
        prompt = prob["prompt"]
        logger.info("[%d/%d] %s — generating (%s)", i + 1, n, prob.get("task_id", "?"), backend)
        if backend == "raw":
            try:
                completion = generate_raw_completion(
                    prompt,
                    ollama_url=ollama_url,
                    model=ollama_model,
                    seed=seed,
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning("raw call failed on %s: %s", prob.get("task_id"), e)
                completion = ""
        elif backend == "engine":
            try:
                completion = generate_engine_completion(prompt, timeout_s=engine_timeout_s)
            except Exception as e:
                logger.warning("engine call failed on %s: %s", prob.get("task_id"), e)
                completion = ""
        else:
            raise SystemExit(f"unknown backend: {backend}")

        if not completion:
            n_empty += 1
        _append_partial(partial, i, completion)
        done[i] = completion
        elapsed = time.time() - t_start
        logger.info(
            "    [%d/%d] done in %.1fs (cumulative %.1fs, %d empty)",
            i + 1,
            n,
            elapsed,
            elapsed,
            n_empty,
        )

    _write_final(output, done, n)
    logger.info("wrote %d completions to %s (%d empty)", n, output, n_empty)
    return 0 if n_empty == 0 else 2


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--backend",
        choices=["raw", "engine"],
        required=True,
        help="raw: direct Ollama call. engine: belief CLI subprocess.",
    )
    p.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of HumanEval problems to run (default: all 164).",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default {DEFAULT_OLLAMA_URL}).",
    )
    p.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default {DEFAULT_OLLAMA_MODEL}).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    p.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (default 0.0)."
    )
    p.add_argument(
        "--engine-timeout-s",
        type=int,
        default=DEFAULT_ENGINE_TIMEOUT_S,
        help=f"Per-problem engine timeout (default {DEFAULT_ENGINE_TIMEOUT_S}s).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing .partial file and start over.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    return run(
        backend=args.backend,
        output=args.output,
        limit=args.limit,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        seed=args.seed,
        temperature=args.temperature,
        engine_timeout_s=args.engine_timeout_s,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
