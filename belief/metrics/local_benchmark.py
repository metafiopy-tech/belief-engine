"""
Local-mode benchmark reporter (Session 17, Task 3).

Wraps the existing benchmark runner so callers can record the four
fields the spec asks for — pass rate, weighted score, build time,
and soil-nutrients-deposited — into a JSON report suitable for
A/B comparing local-mode against cloud-mode runs.

Does **not** touch ``belief/benchmark.py``'s scoring logic (the
project-wide CLAUDE.md constraint).  We consume whatever the
existing runner returns and re-shape it into a report value.

Typical use::

    from belief.metrics.local_benchmark import (
        LocalBenchmarkReport, run_local_benchmark, compare_reports,
    )

    report = await run_local_benchmark(run_benchmark_callable,
                                        tiers=[1, 2], soil=soil)
    report.write_json(Path("reports/local-2026-04-21.json"))

The runner is injected (spec: "run the full benchmark in local mode"
implies BELIEF_MODEL_MODE=local has already been set by the caller).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("belief.metrics.local_benchmark")


DEFAULT_REPORT_DIR = Path("~/.belief-engine/reports").expanduser()


# ── Report shape ──────────────────────────────────────────────────────────


@dataclass
class LocalBenchmarkReport:
    """Snapshot of a single benchmark run, suitable for JSON storage."""

    mode: str = "local"
    pass_rate: float = 0.0
    weighted_score: float = 0.0
    total_challenges: int = 0
    passed_challenges: int = 0
    build_time_s: float = 0.0
    soil_before: int = 0
    soil_after: int = 0
    soil_deposited: int = 0
    cost_usd: float = 0.0
    escalations: int = 0  # Session 17 probe-gated escalations
    started_at: float = 0.0
    finished_at: float = 0.0
    tiers: list[int] = field(default_factory=list)
    notes: str = ""
    # Per-challenge rows for deeper debugging.
    challenges: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        """Persist the report as indented JSON.  Never raises on write
        errors — they are logged so callers can keep moving."""
        out = Path(path).expanduser()
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(f"local_benchmark: write_json({out}) failed: {exc}")
        return out


# ── Helpers ────────────────────────────────────────────────────────────────


def _soil_count(soil: Any) -> int:
    """Duck-typed ``soil.count()`` — zero when soil is missing or errors."""
    if soil is None:
        return 0
    try:
        return int(soil.count())
    except Exception as exc:
        logger.debug(f"local_benchmark: soil.count() failed: {exc}")
        return 0


def _coerce_challenges(raw: Any) -> list[dict]:
    """Normalise whatever the runner returns into a list of dicts."""
    if not raw:
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "id": item.get("id") or item.get("challenge_id") or "",
                    "passed": bool(item.get("passed", False)),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "cost_usd": float(item.get("cost_usd", 0.0) or 0.0),
                    "time_s": float(item.get("time_s", item.get("duration_s", 0.0)) or 0.0),
                    "error": item.get("error") or "",
                }
            )
        else:
            out.append(
                {
                    "id": getattr(item, "challenge_id", ""),
                    "passed": bool(getattr(item, "passed", False)),
                    "score": float(getattr(item, "score", 0.0) or 0.0),
                    "cost_usd": float(getattr(item, "cost_usd", 0.0) or 0.0),
                    "time_s": float(getattr(item, "time_s", 0.0) or 0.0),
                    "error": getattr(item, "error", "") or "",
                }
            )
    return out


def build_report(
    summary: dict,
    *,
    mode: str = "local",
    soil_before: int = 0,
    soil_after: int = 0,
    build_time_s: float = 0.0,
    started_at: float = 0.0,
    finished_at: float = 0.0,
    escalations: int = 0,
    tiers: Optional[list[int]] = None,
    notes: str = "",
) -> LocalBenchmarkReport:
    """Turn a benchmark ``summary`` dict into a :class:`LocalBenchmarkReport`.

    Accepts either the shape produced by the existing
    ``_run_benchmark`` helper (``pass_rate``, ``passing_ids``,
    ``cost``, etc.) or a simpler ``passed / total / score`` shape.
    Works with whatever subset of fields are present.
    """
    challenges = _coerce_challenges(summary.get("challenges"))
    total = int(summary.get("total") or summary.get("total_challenges") or len(challenges) or 0)
    passed_ids = list(summary.get("passing_ids") or [])
    passed = int(
        summary.get("passed")
        or summary.get("passed_challenges")
        or len(passed_ids)
        or sum(1 for c in challenges if c["passed"])
    )

    if summary.get("pass_rate") is not None:
        pass_rate = float(summary["pass_rate"])
    else:
        pass_rate = (passed / total) if total > 0 else 0.0

    if summary.get("weighted_score") is not None:
        weighted_score = float(summary["weighted_score"])
    else:
        weighted_score = (
            sum(c.get("score", 0.0) for c in challenges) / max(1, total)
            if challenges
            else pass_rate
        )

    return LocalBenchmarkReport(
        mode=mode,
        pass_rate=round(pass_rate, 4),
        weighted_score=round(weighted_score, 4),
        total_challenges=total,
        passed_challenges=passed,
        build_time_s=round(float(build_time_s), 3),
        soil_before=int(soil_before),
        soil_after=int(soil_after),
        soil_deposited=max(0, int(soil_after) - int(soil_before)),
        cost_usd=round(float(summary.get("cost", summary.get("cost_usd", 0.0)) or 0.0), 4),
        escalations=int(escalations),
        started_at=float(started_at),
        finished_at=float(finished_at),
        tiers=list(tiers or []),
        notes=notes,
        challenges=challenges,
    )


async def run_local_benchmark(
    runner: Callable[..., Awaitable[dict]],
    *,
    tiers: Optional[list[int]] = None,
    ids: Optional[list[str]] = None,
    soil: Any = None,
    mode: str = "local",
    notes: str = "",
    **runner_kwargs: Any,
) -> LocalBenchmarkReport:
    """Execute ``runner`` and build a report from its return value.

    The runner must be an async callable matching the existing
    ``_run_benchmark_cmd`` contract — it receives ``tiers`` and/or
    ``ids`` and returns a summary dict.  We wrap the call with
    wallclock timing and soil before/after counts so the report has
    everything the spec asks for.

    This function doesn't set ``BELIEF_MODEL_MODE=local`` on its own
    — the caller is expected to have done that (either via CLI flag
    or ``ModelRouter.set_mode("local")``).  We just record the
    resulting ``mode`` in the report.
    """
    started = time.time()
    before = _soil_count(soil)
    try:
        summary = await runner(tiers=tiers, ids=ids, **runner_kwargs)
    except TypeError:
        # Runner might take positional args.
        summary = await runner(tiers, ids)
    finished = time.time()
    after = _soil_count(soil)

    if not isinstance(summary, dict):
        logger.debug(
            f"local_benchmark: runner returned non-dict ({type(summary).__name__}); "
            f"wrapping as empty summary"
        )
        summary = {}

    escalations = 0
    router = runner_kwargs.get("router")
    if router is not None and hasattr(router, "escalation_count"):
        try:
            escalations = int(router.escalation_count)
        except Exception:
            escalations = 0

    return build_report(
        summary,
        mode=mode,
        soil_before=before,
        soil_after=after,
        build_time_s=finished - started,
        started_at=started,
        finished_at=finished,
        escalations=escalations,
        tiers=tiers,
        notes=notes,
    )


# ── Comparison (cloud vs local) ───────────────────────────────────────────


@dataclass
class ReportDelta:
    """Difference between two :class:`LocalBenchmarkReport` values."""

    left_mode: str = ""
    right_mode: str = ""
    pass_rate_delta: float = 0.0
    weighted_score_delta: float = 0.0
    build_time_delta_s: float = 0.0
    cost_delta_usd: float = 0.0
    soil_delta: int = 0
    challenges_diff: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compare_reports(
    left: LocalBenchmarkReport,
    right: LocalBenchmarkReport,
) -> ReportDelta:
    """Subtract ``left`` from ``right`` — field-by-field delta."""
    diff: dict[str, dict] = {}
    left_by_id = {c["id"]: c for c in left.challenges if c.get("id")}
    right_by_id = {c["id"]: c for c in right.challenges if c.get("id")}
    for cid in sorted(set(left_by_id) | set(right_by_id)):
        lp = left_by_id.get(cid, {}).get("passed")
        rp = right_by_id.get(cid, {}).get("passed")
        if lp != rp:
            diff[cid] = {
                "left_passed": lp,
                "right_passed": rp,
                "left_error": left_by_id.get(cid, {}).get("error", ""),
                "right_error": right_by_id.get(cid, {}).get("error", ""),
            }
    return ReportDelta(
        left_mode=left.mode,
        right_mode=right.mode,
        pass_rate_delta=round(right.pass_rate - left.pass_rate, 4),
        weighted_score_delta=round(
            right.weighted_score - left.weighted_score,
            4,
        ),
        build_time_delta_s=round(right.build_time_s - left.build_time_s, 3),
        cost_delta_usd=round(right.cost_usd - left.cost_usd, 4),
        soil_delta=right.soil_after - left.soil_after,
        challenges_diff=diff,
    )


def format_comparison(
    delta: ReportDelta,
) -> str:
    """Render a :class:`ReportDelta` as a short text block for the CLI."""
    lines = [
        "",
        "═" * 62,
        f"  Benchmark comparison: {delta.left_mode} → {delta.right_mode}",
        "═" * 62,
        f"  Δ pass rate      : {delta.pass_rate_delta:+.2%}",
        f"  Δ weighted score : {delta.weighted_score_delta:+.4f}",
        f"  Δ build time     : {delta.build_time_delta_s:+.1f}s",
        f"  Δ cost           : {delta.cost_delta_usd:+.4f} USD",
        f"  Δ soil count     : {delta.soil_delta:+d}",
    ]
    if delta.challenges_diff:
        lines.append(f"  Challenges that flipped ({len(delta.challenges_diff)}):")
        for cid, rec in list(delta.challenges_diff.items())[:10]:
            lp = "✓" if rec["left_passed"] else "✗"
            rp = "✓" if rec["right_passed"] else "✗"
            lines.append(f"    {cid:<30} {lp} → {rp}")
    lines.append("═" * 62)
    return "\n".join(lines)
