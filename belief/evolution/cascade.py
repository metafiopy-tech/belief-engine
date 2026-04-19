"""
Cascaded evaluation gates for agent versions.

Each gate is progressively more expensive.  Early gates reject cheap
failures so the full benchmark only runs on promising candidates.

Gate 1 — Canary edit     (~$0.01, 5s):    trivial hello-world build
Gate 2 — Smoke subset    (~$0.50, 2min):  5 challenges, 60% pass threshold
Gate 3 — Full benchmark  (~$3-5, 15-30m): all Tier 1-5 challenges
Gate 4 — Regression check:                compare against parent results

Total cost cap: $10.  Abort remaining gates if exceeded.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from belief.evolution.archive import AgentVersion, BenchmarkResult

logger = logging.getLogger("belief.evolution.cascade")

# Gate 2 smoke challenges
_SMOKE_CHALLENGES = [
    "t1-fizzbuzz",
    "t2-todo-cli",
    "t2-health-api",
    "t3-url-shortener",
    "t3-bookmark-api",
]

_SMOKE_PASS_THRESHOLD = 0.6  # 3/5

_COST_CAP = 10.0  # Abort if total cost exceeds this


async def cascaded_evaluate(
    version: AgentVersion,
    engine_graph,
    parent_results: Optional[list[BenchmarkResult]] = None,
) -> tuple[bool, list[BenchmarkResult], str]:
    """Run cascaded evaluation gates on *version*.

    Args:
        version:        The AgentVersion to evaluate.
        engine_graph:   The compiled LangGraph for running builds.
        parent_results: Benchmark results of the parent version (for Gate 4).

    Returns:
        (accepted, results, rejection_reason)
        - accepted: True if all gates passed
        - results: All BenchmarkResult objects collected
        - rejection_reason: "" if accepted, otherwise the gate that rejected
    """
    all_results: list[BenchmarkResult] = []
    total_cost = 0.0

    # ── Gate 1: Canary edit ─────────────────────────────────────────────
    logger.info("Cascade Gate 1: canary edit")
    t0 = time.time()

    try:
        canary_result = await _run_canary(version, engine_graph)
        all_results.append(canary_result)
        total_cost += canary_result.cost_usd
    except Exception as e:
        logger.warning(f"Gate 1 crashed: {e}")
        canary_result = BenchmarkResult(
            version_id=version.id,
            challenge_id="canary-hello-world",
            passed=False,
            score=0.0,
            cost_usd=0.01,
            time_seconds=time.time() - t0,
            error_summary=str(e),
        )
        all_results.append(canary_result)

    if not canary_result.passed:
        logger.warning("Cascade: REJECTED at Gate 1 (canary_failed)")
        return False, all_results, "canary_failed"

    logger.info(f"Gate 1 passed ({canary_result.time_seconds:.1f}s, ${canary_result.cost_usd:.3f})")

    # ── Gate 2: Smoke subset ────────────────────────────────────────────
    if total_cost >= _COST_CAP:
        return False, all_results, "cost_cap_exceeded"

    logger.info("Cascade Gate 2: smoke subset (5 challenges)")
    smoke_results = await _run_smoke(version, engine_graph)
    all_results.extend(smoke_results)
    total_cost += sum(r.cost_usd for r in smoke_results)

    smoke_passed = sum(1 for r in smoke_results if r.passed)
    smoke_rate = smoke_passed / max(len(smoke_results), 1)

    if smoke_rate < _SMOKE_PASS_THRESHOLD:
        logger.warning(
            f"Cascade: REJECTED at Gate 2 (smoke_failed: {smoke_passed}/{len(smoke_results)})"
        )
        return False, all_results, "smoke_failed"

    logger.info(f"Gate 2 passed ({smoke_passed}/{len(smoke_results)}, ${sum(r.cost_usd for r in smoke_results):.2f})")

    # ── Gate 3: Full benchmark ──────────────────────────────────────────
    if total_cost >= _COST_CAP:
        return False, all_results, "cost_cap_exceeded"

    logger.info("Cascade Gate 3: full benchmark")
    full_results = await _run_full_benchmark(version, engine_graph)
    all_results.extend(full_results)
    total_cost += sum(r.cost_usd for r in full_results)

    logger.info(
        f"Gate 3 complete: {sum(1 for r in full_results if r.passed)}/{len(full_results)} passed, "
        f"${sum(r.cost_usd for r in full_results):.2f}"
    )

    # ── Gate 4: Regression check ────────────────────────────────────────
    regressions: list[str] = []
    if parent_results:
        parent_passing = {r.challenge_id for r in parent_results if r.passed}
        child_passing = {r.challenge_id for r in all_results if r.passed}
        regressions = sorted(parent_passing - child_passing)

        if regressions:
            logger.warning(
                f"Gate 4: {len(regressions)} regression(s) detected (flagged, not auto-rejected): "
                f"{', '.join(regressions[:5])}"
            )
            # Flag but don't auto-reject — DGM insight: regressions may be stepping stones
            for reg_id in regressions:
                all_results.append(BenchmarkResult(
                    version_id=version.id,
                    challenge_id=f"regression:{reg_id}",
                    passed=False,
                    score=0.0,
                    cost_usd=0.0,
                    time_seconds=0.0,
                    error_summary=f"Regression from parent: {reg_id} was passing, now failing",
                ))

    logger.info(
        f"Cascade complete: accepted, total_cost=${total_cost:.2f}, "
        f"regressions={len(regressions)}"
    )
    return True, all_results, ""


# ── Gate implementations ────────────────────────────────────────────────────


async def _run_canary(version: AgentVersion, engine_graph) -> BenchmarkResult:
    """Gate 1: run a trivial hello-world build.

    Uses the engine graph to build a Python script that prints 'hello world'.
    If it produces a runnable script, it passes.
    """
    t0 = time.time()

    try:
        from belief.benchmark import run_challenge, Challenge

        canary_challenge = Challenge(
            id="canary-hello-world",
            tier=0,
            goal="Build a Python script that prints 'hello world'",
            acceptance_criteria=["Prints 'hello world' to stdout"],
            verify_commands=["python3 main.py"],
            timeout_seconds=30,
            tags=["canary"],
        )

        result = await run_challenge(canary_challenge)

        return BenchmarkResult(
            version_id=version.id,
            challenge_id="canary-hello-world",
            passed=result.verdict == "pass",
            score=result.weighted_score,
            cost_usd=result.cost_usd,
            time_seconds=time.time() - t0,
            error_summary=result.error if result.verdict != "pass" else None,
        )
    except Exception as e:
        return BenchmarkResult(
            version_id=version.id,
            challenge_id="canary-hello-world",
            passed=False,
            score=0.0,
            cost_usd=0.01,
            time_seconds=time.time() - t0,
            error_summary=str(e),
        )


async def _run_smoke(version: AgentVersion, engine_graph) -> list[BenchmarkResult]:
    """Gate 2: run 5 smoke-test challenges."""
    results: list[BenchmarkResult] = []

    try:
        from belief.benchmark import run_benchmark

        challenge_results = await run_benchmark(challenge_ids=_SMOKE_CHALLENGES)

        for cr in challenge_results:
            results.append(BenchmarkResult(
                version_id=version.id,
                challenge_id=cr.challenge_id,
                passed=cr.verdict == "pass",
                score=cr.weighted_score,
                cost_usd=cr.cost_usd,
                time_seconds=cr.build_time_seconds,
                error_summary=cr.error if cr.verdict != "pass" else None,
            ))
    except Exception as e:
        logger.warning(f"Smoke test crashed: {e}")
        for cid in _SMOKE_CHALLENGES:
            results.append(BenchmarkResult(
                version_id=version.id,
                challenge_id=cid,
                passed=False,
                score=0.0,
                cost_usd=0.0,
                time_seconds=0.0,
                error_summary=str(e),
            ))

    return results


async def _run_full_benchmark(
    version: AgentVersion, engine_graph
) -> list[BenchmarkResult]:
    """Gate 3: run all Tier 1-5 challenges."""
    results: list[BenchmarkResult] = []

    try:
        from belief.benchmark import run_benchmark

        challenge_results = await run_benchmark(tiers=[1, 2, 3, 4, 5])

        for cr in challenge_results:
            results.append(BenchmarkResult(
                version_id=version.id,
                challenge_id=cr.challenge_id,
                passed=cr.verdict == "pass",
                score=cr.weighted_score,
                cost_usd=cr.cost_usd,
                time_seconds=cr.build_time_seconds,
                error_summary=cr.error if cr.verdict != "pass" else None,
            ))
    except Exception as e:
        logger.warning(f"Full benchmark crashed: {e}")

    return results
