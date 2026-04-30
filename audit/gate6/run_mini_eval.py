"""Gate 6 mini-eval runner.

Runs each task in tasks.json N_SEEDS times under both conditions
(engine_local, raw_local), records per-run metrics to SQLite.

Resumable: skips runs already in the DB.
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Config
N_SEEDS = 2
CONDITIONS = ("engine_local", "raw_local")
SEEDS = [42, 1337]  # first seed matches our determinism pin
DB_PATH = Path("audit/gate6/results.db")
TASKS_PATH = Path("audit/gate6/tasks.json")

# Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    task_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    condition TEXT NOT NULL,
    seed INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    wallclock_s REAL,
    tests_passed INTEGER,
    tests_total INTEGER,
    weighted_score REAL,
    n_llm_calls INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    verdict TEXT,
    error TEXT,
    PRIMARY KEY (task_id, condition, seed)
);
"""


def db_init():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def already_done(conn, task_id, condition, seed):
    cur = conn.execute(
        "SELECT wallclock_s FROM runs WHERE task_id=? AND condition=? AND seed=? AND finished_at IS NOT NULL",
        (task_id, condition, seed),
    )
    return cur.fetchone() is not None


def record_start(conn, task_id, tier, condition, seed):
    conn.execute(
        "INSERT OR REPLACE INTO runs (task_id, tier, condition, seed, started_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, tier, condition, seed, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def record_finish(conn, task_id, condition, seed, **fields):
    cols = ["finished_at"] + list(fields.keys())
    vals = [datetime.now(timezone.utc).isoformat()] + list(fields.values())
    sql = f"UPDATE runs SET {', '.join(c + '=?' for c in cols)} WHERE task_id=? AND condition=? AND seed=?"
    conn.execute(sql, vals + [task_id, condition, seed])
    conn.commit()


_RUN_ID_RE = re.compile(r"Run ID:\s*(belief-[a-f0-9]+)")
_VERDICT_RE = re.compile(r"Verdict:\s*([A-Za-z_]+)")
# Validator log line, e.g. "Validator: pass, 9/9 tests, weighted=1.00"
_VALIDATOR_RE = re.compile(
    r"Validator:\s*(?P<v>[A-Za-z_]+),\s*"
    r"(?P<tp>\d+)\s*/\s*(?P<tt>\d+)\s*tests,\s*"
    r"weighted=(?P<w>[\d.]+)"
)


def _lookup_outcome_in_archive(run_id: str) -> dict | None:
    """Read a BuildOutcome from the agent archive by run_id.

    Returns the outcome dict (verdict/weighted_score/tests_passed/
    tests_total) or None on any failure. Never raises — the runner
    falls back to stdout/stderr regex parsing if this returns None.
    """
    if not run_id:
        return None
    try:
        from belief.archive.outcome import BuildOutcome
        from belief.archive.store import AgentArchive

        arch = AgentArchive()
        arch._ensure()
        result = arch._collection.get(ids=[run_id], include=["metadatas"])
    except Exception:
        return None

    metadatas = (result or {}).get("metadatas") or []
    if not metadatas:
        return None
    meta = metadatas[0] or {}
    raw = meta.get("outcome_json")
    if not raw:
        # Fall back to the metadata fields if outcome_json wasn't stored.
        return {
            "verdict": str(meta.get("verdict") or "unknown"),
            "weighted_score": float(meta.get("weighted_score") or 0.0),
            "tests_passed": 0,
            "tests_total": 0,
        }
    try:
        o: Any = BuildOutcome.from_json(raw)
    except Exception:
        return None
    return {
        "verdict": str(getattr(o, "verdict", "unknown")),
        "weighted_score": float(getattr(o, "weighted_score", 0.0) or 0.0),
        "tests_passed": int(getattr(o, "tests_passed", 0) or 0),
        "tests_total": int(getattr(o, "tests_total", 0) or 0),
    }


def _parse_metrics_from_streams(stdout: str, stderr: str) -> dict:
    """Regex fallback: scan stdout and stderr for verdict + validator line.

    The validator line is emitted via logger.info() which writes to
    stderr by default — Bug 1 in the original parser only looked at
    stdout, so it never matched. This fallback scans both streams.
    """
    verdict = "unknown"
    weighted_score = 0.0
    tests_passed = 0
    tests_total = 0
    for stream in (stdout, stderr):
        if not stream:
            continue
        for line in stream.splitlines():
            m = _VERDICT_RE.search(line)
            if m:
                verdict = m.group(1).strip().lower()
            m = _VALIDATOR_RE.search(line)
            if m:
                tests_passed = int(m.group("tp"))
                tests_total = int(m.group("tt"))
                weighted_score = float(m.group("w"))
                # If verdict still unknown, take the validator's verdict.
                if verdict == "unknown":
                    verdict = m.group("v").strip().lower()
    return {
        "verdict": verdict,
        "weighted_score": weighted_score,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
    }


async def run_engine(goal: str, seed: int) -> dict:
    """Run the goal through the engine. Returns metrics dict."""
    # Override the session-1 default seed for this run
    from belief.llm import _SESSION1_OPTION_DEFAULTS, LOCAL_TRACKER
    orig_seed = _SESSION1_OPTION_DEFAULTS.get("seed")
    _SESSION1_OPTION_DEFAULTS["seed"] = seed

    try:
        # Clear tracker
        LOCAL_TRACKER.records.clear()
        LOCAL_TRACKER.fallback_count = 0

        start = time.time()
        # Invoke the CLI build via subprocess so it's isolated
        import subprocess
        proc = subprocess.run(
            [
                "caffeinate", "-dimsu",
                "belief", "--mode", "local", "--goal", goal,
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            stdin=subprocess.DEVNULL,
        )
        wallclock = time.time() - start
        stdout = proc.stdout
        stderr = proc.stderr

        # Bug 1 fix — score capture in three layers (most-trustworthy first):
        #
        #   1. Look up the BuildOutcome the engine itself persisted to the
        #      agent archive. The archive holds the validator's authoritative
        #      counts (post Bug 2 fix), so this is exact.
        #   2. Failing that, regex-parse stdout AND stderr for "Verdict:"
        #      and "Validator: <v>, <tp>/<tt> tests, weighted=<w>". The
        #      validator log goes to stderr by default; the original
        #      stdout-only parser is why every engine row read 0.0.
        #   3. Finally, surface whatever zeros we found so the row at
        #      least carries the verdict/error context.
        run_id_match = (
            _RUN_ID_RE.search(stdout) if stdout else None
        ) or (_RUN_ID_RE.search(stderr) if stderr else None)
        run_id = run_id_match.group(1) if run_id_match else ""

        archive_metrics = _lookup_outcome_in_archive(run_id)
        if archive_metrics is not None and archive_metrics.get("tests_total", 0) > 0:
            verdict = archive_metrics["verdict"]
            weighted_score = archive_metrics["weighted_score"]
            tests_passed = archive_metrics["tests_passed"]
            tests_total = archive_metrics["tests_total"]
        else:
            parsed = _parse_metrics_from_streams(stdout, stderr)
            verdict = parsed["verdict"]
            weighted_score = parsed["weighted_score"]
            tests_passed = parsed["tests_passed"]
            tests_total = parsed["tests_total"]
            # Even if the archive lookup didn't have test counts, prefer
            # its verdict/score when the regex fallback yielded nothing.
            if archive_metrics is not None:
                if verdict == "unknown" and archive_metrics["verdict"]:
                    verdict = archive_metrics["verdict"]
                if weighted_score == 0.0 and archive_metrics["weighted_score"]:
                    weighted_score = archive_metrics["weighted_score"]

        # Read tracker snapshot
        n_calls = LOCAL_TRACKER.total_calls()
        by_role = LOCAL_TRACKER.by_role()
        pt = sum(b.get("prompt_tokens", 0) for b in by_role.values())
        ct = sum(b.get("completion_tokens", 0) for b in by_role.values())

        return {
            "wallclock_s": wallclock,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "weighted_score": weighted_score,
            "n_llm_calls": n_calls,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "verdict": verdict,
            "error": stderr[-500:] if proc.returncode != 0 else None,
        }
    finally:
        _SESSION1_OPTION_DEFAULTS["seed"] = orig_seed


async def run_raw(goal: str, seed: int) -> dict:
    """Run the goal through the raw runner. Returns metrics dict."""
    # Temporarily override seed in raw_runner options
    from belief.experiments import raw_runner
    # raw_runner embeds seed=42 in options literal; we need to override
    # by patching the function. Simpler: import and call, but we need to
    # pass seed. The cleanest way is to monkey-patch the options block.
    
    # Actually simpler - just call run_raw directly, it uses the hardcoded options.
    # For seed=42 (first seed) this is fine. For seed=1337 we need to swap.
    import httpx
    from belief.experiments.raw_runner import (
        SYSTEM_PROMPT,
        parse_file_blocks,
        parse_pytest_output,
        RawRunResult,
    )
    import subprocess
    import tempfile
    from pathlib import Path as P
    
    start = time.time()
    try:
        async with httpx.AsyncClient(base_url="http://localhost:11434", timeout=300) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "model": "qwen2.5-coder:14b",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": goal},
                    ],
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 4096,
                        "seed": seed,
                        "num_gpu": 99,
                        "num_thread": 6,
                        "num_batch": 256,
                        "num_keep": 512,
                        "mirostat": 0,
                    },
                },
            )
            resp.raise_for_status()
            rj = resp.json()
            raw_output = rj["message"]["content"]
            prompt_tokens = int(rj.get("prompt_eval_count", 0))
            completion_tokens = int(rj.get("eval_count", 0))
    except Exception as exc:
        return {
            "wallclock_s": time.time() - start,
            "tests_passed": 0,
            "tests_total": 0,
            "weighted_score": 0.0,
            "n_llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "verdict": "fail_hard",
            "error": f"Model call failed: {exc}",
        }

    files = parse_file_blocks(raw_output)
    if not files:
        return {
            "wallclock_s": time.time() - start,
            "tests_passed": 0,
            "tests_total": 0,
            "weighted_score": 0.0,
            "n_llm_calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "verdict": "fail_hard",
            "error": "No file blocks in output",
        }

    tests_passed = 0
    tests_total = 0
    run_error = None
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, content in files.items():
            path = P(tmpdir) / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        req_file = P(tmpdir) / "requirements.txt"
        if req_file.exists():
            try:
                subprocess.run(
                    ["pip3", "install", "-r", str(req_file), "-q", "--break-system-packages"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=60,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        try:
            proc = subprocess.run(
                ["python3", "-m", "pytest", ".", "-q", "--tb=no", "--timeout=30", "--no-header"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                timeout=120,
                stdin=subprocess.DEVNULL,
            )
            tests_passed, tests_total = parse_pytest_output(proc.stdout)
        except subprocess.TimeoutExpired:
            run_error = "pytest timed out"

    weighted = tests_passed / max(tests_total, 1) if tests_total > 0 else 0.0
    verdict = "pass" if weighted >= 0.5 else "fail_hard"

    return {
        "wallclock_s": time.time() - start,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "weighted_score": weighted,
        "n_llm_calls": 1,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "verdict": verdict,
        "error": run_error,
    }


async def main():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    conn = db_init()

    # Flatten into (tier, task) pairs
    all_tasks = []
    for tier, task_list in tasks.items():
        for t in task_list:
            all_tasks.append((tier, t))

    total_runs = len(all_tasks) * len(CONDITIONS) * len(SEEDS)
    done = 0

    print(f"Starting Gate 6 mini-eval: {len(all_tasks)} tasks × {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {total_runs} runs")
    print(f"Estimated wall-clock: {total_runs * 5 // 60} hours (at ~5 min/run average)")
    print()

    for tier, task in all_tasks:
        for condition in CONDITIONS:
            for seed in SEEDS:
                done += 1
                if already_done(conn, task["id"], condition, seed):
                    print(f"[{done}/{total_runs}] SKIP  {task['id']} {condition} seed={seed} (already done)")
                    continue

                print(f"[{done}/{total_runs}] RUN   {task['id']} {condition} seed={seed} at {datetime.now().strftime('%H:%M:%S')}")
                record_start(conn, task["id"], tier, condition, seed)

                try:
                    if condition == "engine_local":
                        metrics = await run_engine(task["goal"], seed)
                    else:
                        metrics = await run_raw(task["goal"], seed)

                    record_finish(conn, task["id"], condition, seed, **metrics)
                    status = "PASS" if metrics["verdict"] == "pass" else "FAIL"
                    print(f"              → {status} score={metrics['weighted_score']:.2f} time={metrics['wallclock_s']:.0f}s tokens={metrics['prompt_tokens']+metrics['completion_tokens']}")
                except Exception as e:
                    record_finish(conn, task["id"], condition, seed, verdict="error", error=str(e))
                    print(f"              → ERROR {e}")

    print()
    print("=== COMPLETE ===")
    cur = conn.execute("SELECT COUNT(*) FROM runs WHERE finished_at IS NOT NULL")
    print(f"Total finished runs in DB: {cur.fetchone()[0]}")


if __name__ == "__main__":
    asyncio.run(main())
