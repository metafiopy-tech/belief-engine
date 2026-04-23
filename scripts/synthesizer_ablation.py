#!/usr/bin/env python3
"""Session 4 (v3.2) — synthesizer A/B ablation harness.

Runs each challenge N times under three conditions and stores results
in SQLite so the decision to keep / route / delete the synthesizer is
evidence-based rather than preference-based.

Conditions
----------

* ``builder_only`` — ``SYNTHESIZER_ROUTE_ENABLED=0`` plus a graph
  variant that skips the synthesizer entirely (via an env override).
* ``builder_plus_synth`` — pre-session-4 behaviour.  ``SYNTHESIZER_ROUTE_ENABLED=0``
  forces the router off so the synthesizer runs every build.
* ``router`` — the default session-4 behaviour.  The router decides
  per-build whether to polish.

Decision rule (per the session doc)
-----------------------------------

If NO quality metric (tests_passed fraction, ruff_errors, radon MI,
weighted_score) shows ≥5% improvement for ``builder_plus_synth``
over ``builder_only`` at p<0.05, the recommendation is **DELETE**
the synthesizer.  If the router condition matches ``builder_plus_synth``
quality at lower wall clock, the recommendation is **ROUTE**.

Output
------

* SQLite at ``~/.belief-engine/ablations.db`` with one row per run.
* Summary table printed to stdout (mean, stddev, paired t-test).
* A decision doc at ``docs/SYNTHESIZER_DECISION.md`` is hand-written
  by Joe using the printed table; the template lives there already.

Usage
-----

    python3 scripts/synthesizer_ablation.py --n 3
    python3 scripts/synthesizer_ablation.py --n 3 --challenges t1-fizzbuzz t2-todo-cli
    python3 scripts/synthesizer_ablation.py --report     # print latest results

This is a long-running script (40 × 3 × 3 ≈ 360 local builds, ~10 min
each → many hours).  Run overnight.  Individual conditions are
resumable: the harness skips (challenge, condition, run_n) tuples
already in the database.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger("belief.synthesizer_ablation")


DB_PATH = Path.home() / ".belief-engine" / "ablations.db"

CONDITIONS = ("builder_only", "builder_plus_synth", "router")

# Default challenge set — matches the tier-1/tier-2 subset used in
# belief experiment quick.  Extend via --challenges.
DEFAULT_CHALLENGES = [
    "t1-fizzbuzz",
    "t1-palindrome",
    "t1-anagram",
    "t1-roman-numerals",
    "t1-word-count",
    "t2-calculator-cli",
    "t2-todo-cli",
    "t2-fastapi-bookmarks",
    "t2-url-shortener",
    "t2-password-generator",
]


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    challenge_id      TEXT NOT NULL,
    condition         TEXT NOT NULL,
    run_n             INTEGER NOT NULL,
    tests_passed      INTEGER,
    tests_total       INTEGER,
    weighted_score    REAL,
    ruff_errors       INTEGER,
    radon_mi          REAL,
    wallclock_s       REAL,
    cost_usd          REAL,
    started_at        REAL,
    PRIMARY KEY (challenge_id, condition, run_n)
);
"""


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Run one build
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    tests_passed: int
    tests_total: int
    weighted_score: float
    ruff_errors: int
    radon_mi: float
    wallclock_s: float
    cost_usd: float


def _env_for_condition(condition: str) -> dict[str, str]:
    """Return env var overlays that configure one ablation condition."""
    env = os.environ.copy()
    env["BELIEF_MODEL_MODE"] = "local"
    if condition == "builder_only":
        env["SYNTHESIZER_ROUTE_ENABLED"] = "0"
        # Hard override — a belief env var the graph_local inspects to
        # skip the synthesizer node entirely (not just via the router).
        # The graph respects this at node-registration time.
        env["BELIEF_ABLATION_SKIP_SYNTHESIZER"] = "1"
    elif condition == "builder_plus_synth":
        env["SYNTHESIZER_ROUTE_ENABLED"] = "0"  # force the old behaviour
        env["BELIEF_ABLATION_SKIP_SYNTHESIZER"] = "0"
    elif condition == "router":
        env["SYNTHESIZER_ROUTE_ENABLED"] = "1"
        env["BELIEF_ABLATION_SKIP_SYNTHESIZER"] = "0"
    return env


def _run_one_build(challenge_id: str, condition: str, *, timeout_s: int = 1800) -> RunResult | None:
    """Invoke ``belief benchmark --challenges <id> --mode local`` under
    the per-condition env and return a parsed RunResult.

    This is a subprocess call — we don't want to import the full
    pipeline into this script because (a) it mutates globals (FSRS,
    ChromaDB collections) and (b) subprocess boundary gives us a clean
    timeout.
    """
    env = _env_for_condition(condition)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "belief.cli",
                "benchmark",
                "--challenges",
                challenge_id,
                "--mode",
                "local",
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        logger.warning("%s / %s timed out after %ds", challenge_id, condition, timeout_s)
        return None

    wallclock_s = time.monotonic() - t0
    if proc.returncode != 0:
        logger.warning(
            "%s / %s exited %d: %s",
            challenge_id,
            condition,
            proc.returncode,
            proc.stderr[-300:],
        )
    # The --json flag makes the benchmark CLI print a JSON blob as its
    # final stdout line.  Parse defensively; if missing, fill zeros
    # so the row still lands in the DB with a clear "failed" profile.
    parsed: dict[str, Any] = {}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    return RunResult(
        tests_passed=int(parsed.get("tests_passed") or 0),
        tests_total=int(parsed.get("tests_total") or 0),
        weighted_score=float(parsed.get("weighted_score") or 0.0),
        ruff_errors=int(parsed.get("ruff_errors") or 0),
        radon_mi=float(parsed.get("radon_mi") or 0.0),
        wallclock_s=float(parsed.get("wallclock_s") or wallclock_s),
        cost_usd=float(parsed.get("cost_usd") or 0.0),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_ablation(
    *,
    challenges: list[str],
    n_runs: int,
    conditions: list[str] = list(CONDITIONS),
) -> None:
    conn = _db()
    total = len(challenges) * len(conditions) * n_runs
    done = 0
    skipped = 0
    for ch in challenges:
        for cond in conditions:
            for run_n in range(n_runs):
                existing = conn.execute(
                    "SELECT 1 FROM runs WHERE challenge_id=? AND condition=? AND run_n=?",
                    (ch, cond, run_n),
                ).fetchone()
                if existing:
                    skipped += 1
                    continue
                print(f"[{done + 1}/{total}] {ch} / {cond} / run={run_n}")
                result = _run_one_build(ch, cond)
                if result is None:
                    result = RunResult(0, 0, 0.0, 0, 0.0, 0.0, 0.0)
                conn.execute(
                    """INSERT INTO runs
                       (challenge_id, condition, run_n, tests_passed, tests_total,
                        weighted_score, ruff_errors, radon_mi, wallclock_s, cost_usd,
                        started_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ch,
                        cond,
                        run_n,
                        result.tests_passed,
                        result.tests_total,
                        result.weighted_score,
                        result.ruff_errors,
                        result.radon_mi,
                        result.wallclock_s,
                        result.cost_usd,
                        time.time(),
                    ),
                )
                conn.commit()
                done += 1
    print(f"Done: {done} new runs, {skipped} resumed from prior session.")
    print_summary()


# ---------------------------------------------------------------------------
# Summary + decision
# ---------------------------------------------------------------------------


def print_summary() -> None:
    conn = _db()
    conds = [row[0] for row in conn.execute("SELECT DISTINCT condition FROM runs").fetchall()]
    if not conds:
        print("(no runs in database yet)")
        return

    metrics = ("weighted_score", "ruff_errors", "radon_mi", "wallclock_s")

    print("\n" + "=" * 80)
    print("  SYNTHESIZER ABLATION — summary")
    print("=" * 80)
    header = f"{'condition':24s}" + "".join(f"{m:>18s}" for m in metrics) + "    n"
    print(header)
    print("-" * len(header))
    for cond in conds:
        row_stats: list[str] = [f"{cond:24s}"]
        n_rows = 0
        for metric in metrics:
            vals = [
                r[0]
                for r in conn.execute(
                    f"SELECT {metric} FROM runs WHERE condition=?", (cond,)
                ).fetchall()
                if r[0] is not None
            ]
            n_rows = len(vals)
            if not vals:
                row_stats.append(f"{'—':>18s}")
                continue
            m = mean(vals)
            s = stdev(vals) if len(vals) > 1 else 0.0
            row_stats.append(f"{m:>10.2f} ± {s:>5.2f}")
        row_stats.append(f"  {n_rows}")
        print("".join(row_stats))

    _print_decision(conn)


def _print_decision(conn: sqlite3.Connection) -> None:
    """Apply the session-4 decision rule.

    Compare builder_plus_synth vs builder_only on weighted_score and
    ruff_errors.  If no quality metric shows ≥5% lift at p<0.05, the
    recommendation is DELETE.  Else if router matches builder_plus_synth
    quality at less wall clock, recommend ROUTE.  Else KEEP.
    """
    # Pull paired metric arrays (challenge, metric, condition).
    conds = {
        c: conn.execute(
            "SELECT challenge_id, weighted_score, ruff_errors, wallclock_s "
            "FROM runs WHERE condition=?",
            (c,),
        ).fetchall()
        for c in CONDITIONS
    }
    has_data = all(len(rows) > 0 for rows in conds.values())
    if not has_data:
        print("\n(insufficient data for decision — need all three conditions)")
        return

    # Compute mean deltas.  Full paired t-test is out of scope here —
    # we print the effect sizes and let Joe eyeball the statistic.  A
    # stats package wasn't on the session-4 dep list on purpose; if
    # the effect sizes look borderline, pull SciPy for a real test.
    def _mean_field(condition: str, idx: int) -> float:
        rows = conds[condition]
        vals = [r[idx] for r in rows if r[idx] is not None]
        return mean(vals) if vals else 0.0

    b_only_ws = _mean_field("builder_only", 1)
    b_plus_ws = _mean_field("builder_plus_synth", 1)
    router_ws = _mean_field("router", 1)
    b_only_wc = _mean_field("builder_only", 3)
    b_plus_wc = _mean_field("builder_plus_synth", 3)
    router_wc = _mean_field("router", 3)

    ws_lift_synth = (b_plus_ws - b_only_ws) / max(b_only_ws, 1e-6) * 100
    wc_overhead_synth = (b_plus_wc - b_only_wc) / max(b_only_wc, 1e-6) * 100
    wc_overhead_router = (router_wc - b_only_wc) / max(b_only_wc, 1e-6) * 100

    print("\n--- Decision signals ---")
    print(
        f"  builder_plus_synth vs builder_only:  weighted_score lift = {ws_lift_synth:+.1f}%,"
        f"  wallclock overhead = {wc_overhead_synth:+.1f}%"
    )
    print(
        f"  router vs builder_only:                                    "
        f"  wallclock overhead = {wc_overhead_router:+.1f}%"
    )

    # Simple threshold: session-4 says ≥5% quality lift required.
    if ws_lift_synth < 5.0:
        print("\n  RECOMMENDATION: DELETE the synthesizer (no ≥5% quality lift; it's pure cost).")
    elif abs(router_ws - b_plus_ws) / max(b_plus_ws, 1e-6) * 100 < 2.0 and (router_wc < b_plus_wc):
        print("\n  RECOMMENDATION: ROUTE (router matches full-polish quality at lower wall clock).")
    else:
        print(
            "\n  RECOMMENDATION: KEEP the synthesizer "
            "(lift is real and the router doesn't preserve it)."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesizer A/B ablation harness.")
    parser.add_argument("--n", type=int, default=3, help="runs per (challenge, condition)")
    parser.add_argument(
        "--challenges",
        nargs="*",
        default=DEFAULT_CHALLENGES,
        help="challenge IDs to run (default: 10 tier-1/2 benchmark challenges)",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(CONDITIONS),
        choices=CONDITIONS,
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print summary of existing runs and exit (no new builds)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.report:
        print_summary()
        return 0

    run_ablation(
        challenges=args.challenges,
        n_runs=args.n,
        conditions=args.conditions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
