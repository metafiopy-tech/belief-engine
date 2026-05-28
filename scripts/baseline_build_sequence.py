"""Run a fixed sequence of ``belief build`` invocations with logging.

Used by the substrate-transfer baseline-prep procedure to populate empty
soil up to a target build count (build_seq=5 or build_seq=15). The active
experiment condition is taken from the ``BELIEF_EXPERIMENT_CONDITION``
environment variable — set it before invoking this script.

Usage:

    BELIEF_EXPERIMENT_CONDITION=soil_only \\
        python3 scripts/baseline_build_sequence.py --count 4

    BELIEF_EXPERIMENT_CONDITION=soil_only \\
        python3 scripts/baseline_build_sequence.py --count 10 --offset 4

The 14 baseline-builder challenges are hardcoded (different set from the
15 measurement challenges to avoid contamination). ``--count N`` runs the
first N of them. ``--offset N`` skips the first N — so to run builds 5-14
after already running 1-4, use ``--count 10 --offset 4``.

Each build's output is appended to ``~/.belief-engine/baseline_prep.log``.
Failures do NOT halt the sequence; they're logged and the script moves on,
so a single transient timeout doesn't waste the whole run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# 14 challenges drawn from belief.benchmark.CHALLENGES, deliberately
# disjoint from the 15 in-distribution measurement challenges. Order is
# rough complexity-ascending so the soil grows progressively.
BASELINE_BUILDER_CHALLENGES: list[tuple[str, str]] = [
    (
        "t1-fizzbuzz",
        "Build a Python script that prints FizzBuzz from 1 to 100. "
        "Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for both.",
    ),
    (
        "t1-fibonacci",
        "Build a Python script that calculates the first 20 Fibonacci numbers "
        "and prints them as a JSON array.",
    ),
    (
        "t2-mcp-server",
        "Build a Model Context Protocol (MCP) server using FastMCP that exposes "
        "a single tool called 'echo' which echoes its input string back.",
    ),
    (
        "t2-websocket-echo",
        "Build a FastAPI server with a WebSocket endpoint at /ws that echoes "
        "any text message back to the client.",
    ),
    (
        "t3-api-wrapper",
        "Build a Python SDK wrapper around the public JSONPlaceholder API "
        "(https://jsonplaceholder.typicode.com) with methods for posts, users, "
        "and comments. Use httpx and expose a CLI via Click.",
    ),
    (
        "t3-config-manager",
        "Build a configuration manager library: load YAML configs with "
        "environment variable overrides, validate with Pydantic, expose CLI "
        "to print or get individual keys.",
    ),
    (
        "t3-log-analyzer",
        "Build a CLI tool that reads JSON-formatted logs from stdin or a "
        "file and prints summary statistics: counts by level, error rate, "
        "top 10 most frequent messages.",
    ),
    (
        "t7-add-pagination",
        "Given an existing FastAPI CRUD API for 'items', add pagination: "
        "GET /items?page=1&size=10 with total count in X-Total-Count header. "
        "Do not break existing endpoints.",
    ),
    (
        "t7-add-search",
        "Given an existing FastAPI notes API (CRUD for notes with title, "
        "content, created_at), add full-text search: "
        "GET /notes/search?q=keyword.",
    ),
    (
        "t7-fix-validation",
        "Given an existing FastAPI user registration API, fix validation: "
        "username 3-50 chars, valid email format, password min 8 chars, "
        "unique email constraint (409 on duplicate).",
    ),
    (
        "t7-add-auth",
        "Given an existing FastAPI todo API with no authentication, add "
        "JWT-based auth: POST /login returns {access_token}, task endpoints "
        "require Bearer token. Hardcoded test user (admin/admin123).",
    ),
    (
        "t7-add-export",
        "Given an existing FastAPI expense tracker, add CSV export: "
        "GET /expenses/export returns CSV with date range filtering "
        "(?start=2024-01-01&end=2024-12-31).",
    ),
    (
        "t4-poll-system",
        "Build a polling system API with FastAPI: create polls with multiple "
        "options, vote (one vote per session), view results with percentages, "
        "close polls. Real-time vote counts. SQLite.",
    ),
    (
        "t4-task-board",
        "Build a Kanban task board API with FastAPI: boards, columns "
        "(todo/in-progress/done), cards with title/description/assignee. "
        "Move cards between columns. SQLite.",
    ),
]

LOG_PATH = Path.home() / ".belief-engine" / "baseline_prep.log"


def _log(line: str) -> None:
    """Append a timestamped line to the prep log and echo to stdout."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    formatted = f"[{stamp}] {line}"
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(formatted + "\n")
    print(formatted, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential baseline-builder for substrate-transfer experiment",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="How many builds to run in this invocation (1-14).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first OFFSET challenges (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-build timeout in seconds (default: 1800 = 30 min).",
    )
    args = parser.parse_args()

    condition = os.environ.get("BELIEF_EXPERIMENT_CONDITION", "").strip().lower()
    if condition not in ("soil_only", "full"):
        print(
            "ERROR: BELIEF_EXPERIMENT_CONDITION must be set to 'soil_only' "
            "or 'full' before running this script. Got: "
            f"{condition!r}",
            file=sys.stderr,
        )
        return 1

    selected = BASELINE_BUILDER_CHALLENGES[args.offset : args.offset + args.count]
    if not selected:
        print(
            f"ERROR: offset={args.offset} + count={args.count} selects no "
            f"challenges (only {len(BASELINE_BUILDER_CHALLENGES)} available)",
            file=sys.stderr,
        )
        return 1

    local_model = os.environ.get("BELIEF_LOCAL_MODEL", "qwen2.5-coder:14b")
    _log(
        f"BEGIN baseline run: condition={condition} offset={args.offset} "
        f"count={args.count} mode=local model={local_model}"
    )

    # Force local-mode routing in the child process. Without these, the
    # 'belief build' CLI inherits the default routing — which can send the
    # heavy roles to Anthropic cloud. Belt + suspenders: --mode flag AND
    # the env var the engine consults internally.
    child_env = os.environ.copy()
    child_env["BELIEF_MODEL_MODE"] = "local"

    passed = 0
    failed = 0
    for i, (cid, goal) in enumerate(selected, start=args.offset + 1):
        _log(f"build {i}/{args.offset + args.count}: {cid}")
        start = time.time()
        try:
            # NOTE: --mode and --local-model are TOP-LEVEL belief flags
            # (per `belief --help`), so they must appear BEFORE any subcommand.
            # The 'build' subcommand is the default action when --goal is given,
            # so we omit it to keep the invocation minimal.
            result = subprocess.run(
                [
                    "belief",
                    "--mode",
                    "local",
                    "--local-model",
                    local_model,
                    "--goal",
                    goal,
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=child_env,
            )
            elapsed = time.time() - start
            # `belief build` exits 1 even on PASS verdicts when score < 1.0,
            # so we detect success from stdout markers rather than exit code.
            stdout = result.stdout or ""
            looks_passed = (
                "verdict=ValidationVerdict.PASS" in stdout
                or "Verdict: pass" in stdout
                or '"verdict": "pass"' in stdout
            )
            if looks_passed:
                passed += 1
                _log(f"  PASS {cid} in {elapsed:.0f}s (exit {result.returncode})")
            else:
                failed += 1
                tail = (result.stderr or stdout or "")[-300:].replace("\n", " | ")
                _log(f"  FAIL {cid} in {elapsed:.0f}s (exit {result.returncode}): {tail}")
        except subprocess.TimeoutExpired:
            failed += 1
            _log(f"  TIMEOUT {cid} after {args.timeout}s")
        except FileNotFoundError:
            _log("ERROR: 'belief' command not found. Run 'pip3 install -e .' first.")
            return 1

    _log(f"END baseline run: condition={condition} passed={passed} failed={failed}")
    print(f"\nLog: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
