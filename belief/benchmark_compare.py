"""Side-by-side cloud-vs-local benchmark comparison renderer.

Pure presentation + dispatch glue. The actual scoring logic lives in
belief/benchmark.py and is left untouched (project constraint).

The `run_benchmark_compare` async function drives the benchmark runner
twice — once in cloud mode, once in local mode — and renders a per-
challenge comparison table. When Ollama isn't available, the local
run is skipped with a clear note instead of crashing.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


logger = logging.getLogger("belief.benchmark_compare")


@dataclass
class ChallengeComparison:
    challenge_id: str
    cloud_verdict: str = ""
    cloud_score: float = 0.0
    cloud_cost: float = 0.0
    local_verdict: str = ""
    local_score: float = 0.0
    local_cost: float = 0.0

    @property
    def delta_score(self) -> float:
        return self.local_score - self.cloud_score

    @property
    def delta_cost(self) -> float:
        return self.local_cost - self.cloud_cost


@dataclass
class CompareReport:
    rows: list[ChallengeComparison]
    local_available: bool
    local_skipped_reason: str = ""
    cloud_score_overall: float = 0.0
    local_score_overall: float = 0.0
    cloud_cost_overall: float = 0.0
    local_cost_overall: float = 0.0


# ---------------------------------------------------------------------------
# Row -> line renderer
# ---------------------------------------------------------------------------


def format_row(row: ChallengeComparison) -> str:
    cv = _cell(row.cloud_verdict, row.cloud_score)
    lv = _cell(row.local_verdict, row.local_score)
    delta = f"{row.delta_score:+.2f}" if row.cloud_verdict else "n/a"
    return f"{row.challenge_id:<22} | {cv:<10} | {lv:<10} | {delta}"


def _cell(verdict: str, score: float) -> str:
    if not verdict:
        return "-"
    label = verdict.upper()[:4]
    return f"{label} {score:.2f}"


def format_report(report: CompareReport) -> str:
    header = (
        f"{'Challenge':<22} | {'Cloud':<10} | {'Local':<10} | Delta"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in report.rows:
        lines.append(format_row(row))
    lines.append(sep)

    def pct(s: float) -> str:
        return f"{s * 100:.0f}%"

    if report.rows:
        n = len(report.rows)
        cloud_pct = report.cloud_score_overall / n if n else 0.0
        local_pct = report.local_score_overall / n if n else 0.0
        overall_delta = local_pct - cloud_pct
        lines.append(
            f"{'Overall':<22} | {pct(cloud_pct):<10} | "
            f"{pct(local_pct):<10} | {overall_delta * 100:+.0f}%"
        )
    cloud_cost = report.cloud_cost_overall
    local_cost = report.local_cost_overall
    cost_delta = local_cost - cloud_cost
    lines.append(
        f"{'Cost':<22} | ${cloud_cost:<9.2f} | ${local_cost:<9.2f} | ${cost_delta:+.2f}"
    )
    if not report.local_available:
        lines.append("")
        lines.append(
            f"(local run skipped: {report.local_skipped_reason or 'Ollama unavailable'})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------


async def run_benchmark_compare(
    *,
    challenge_ids: Optional[list[str]] = None,
    tiers: Optional[list[int]] = None,
    cloud_runner: Optional[Callable[[list[Any]], Awaitable[list[Any]]]] = None,
    local_runner: Optional[Callable[[list[Any]], Awaitable[list[Any]]]] = None,
    ollama_probe: Optional[Callable[[], Awaitable[bool]]] = None,
) -> CompareReport:
    """Run the benchmark in cloud and local modes and assemble a report.

    `cloud_runner` / `local_runner` are injected to keep the function
    testable — each takes a list of Challenge and returns the matching
    list of ChallengeResult. When not provided, they default to the
    real runners from belief.benchmark.
    """
    from belief.benchmark import CHALLENGES

    # Select challenges
    selected = list(CHALLENGES)
    if challenge_ids:
        ids = set(challenge_ids)
        selected = [c for c in selected if c.id in ids]
    if tiers:
        selected = [c for c in selected if c.tier in set(tiers)]

    if cloud_runner is None:
        from belief.benchmark import run_challenge  # real runner

        async def _default_cloud(challenges: list[Any]) -> list[Any]:
            return [await run_challenge(c) for c in challenges]

        cloud_runner = _default_cloud
    if local_runner is None:
        local_runner = cloud_runner  # same shape; env vars flip routing

    # Probe Ollama availability
    if ollama_probe is None:
        async def _default_probe() -> bool:
            try:
                from belief.llm import AsyncOllamaClient
            except ImportError:
                return False
            ollama = AsyncOllamaClient()
            try:
                return await ollama.is_available()
            finally:
                await ollama.close()

        ollama_probe = _default_probe

    # Run cloud side
    os.environ["BELIEF_MODEL_MODE"] = "cloud"
    cloud_results = await cloud_runner(selected)

    # Run local side — only if Ollama responds
    local_available = await ollama_probe()
    local_results: list[Any] = []
    skipped_reason = ""
    if local_available:
        os.environ["BELIEF_MODEL_MODE"] = "hybrid"
        try:
            local_results = await local_runner(selected)
        except Exception as exc:
            logger.warning("local benchmark run failed: %s", exc)
            local_available = False
            skipped_reason = f"local run raised: {exc!r}"
    else:
        skipped_reason = "Ollama not responding on configured URL"

    # Align by challenge id
    cloud_by_id = {r.challenge_id: r for r in cloud_results}
    local_by_id = {r.challenge_id: r for r in local_results}

    rows: list[ChallengeComparison] = []
    for c in selected:
        cr = cloud_by_id.get(c.id)
        lr = local_by_id.get(c.id)
        rows.append(
            ChallengeComparison(
                challenge_id=c.id,
                cloud_verdict=_verdict(cr),
                cloud_score=_score(cr),
                cloud_cost=_cost(cr),
                local_verdict=_verdict(lr),
                local_score=_score(lr),
                local_cost=_cost(lr),
            )
        )

    report = CompareReport(
        rows=rows,
        local_available=local_available,
        local_skipped_reason=skipped_reason,
        cloud_score_overall=sum(r.cloud_score for r in rows),
        local_score_overall=sum(r.local_score for r in rows),
        cloud_cost_overall=sum(r.cloud_cost for r in rows),
        local_cost_overall=sum(r.local_cost for r in rows),
    )
    return report


def _verdict(result: Any) -> str:
    if result is None:
        return ""
    v = getattr(result, "verdict", "")
    return str(v) if v else ""


def _score(result: Any) -> float:
    if result is None:
        return 0.0
    return float(getattr(result, "weighted_score", 0.0) or 0.0)


def _cost(result: Any) -> float:
    if result is None:
        return 0.0
    return float(getattr(result, "cost_usd", 0.0) or 0.0)


__all__ = [
    "ChallengeComparison",
    "CompareReport",
    "format_report",
    "format_row",
    "run_benchmark_compare",
]
