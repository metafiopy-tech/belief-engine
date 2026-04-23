"""Session 7: per-domain progression tracking + related helpers.

Kept hermetic — no soil/ChromaDB/benchmark runner touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from belief.benchmark_compare import (
    ChallengeComparison,
    CompareReport,
    format_report,
    format_row,
    run_benchmark_compare,
)
from belief.evolution.progression import (
    DOMAINS,
    GENERAL_DOMAIN,
    ProgressionMetrics,
    _progress_fraction,
    _progress_note,
    compute_all_domains,
    compute_progression,
    detect_domain,
    format_all_domains_report,
)
from belief.memory.recomposer import reorder_by_domain


# ---------------------------------------------------------------------------
# detect_domain
# ---------------------------------------------------------------------------


class TestDetectDomain:
    def test_fastapi(self) -> None:
        assert detect_domain("Build a FastAPI REST API") == "fastapi"

    def test_cli(self) -> None:
        assert detect_domain("Build a Click CLI tool") == "cli"

    def test_mcp(self) -> None:
        assert detect_domain("Build an MCP server") == "mcp"

    def test_data(self) -> None:
        assert detect_domain("Build a CSV data pipeline") == "data"

    def test_async(self) -> None:
        assert detect_domain("Build an asyncio task queue") == "async"

    def test_library(self) -> None:
        assert detect_domain("Build a Python SDK wrapper") == "library"

    def test_script(self) -> None:
        assert detect_domain("Build a FizzBuzz script") == "script"

    def test_general_fallback(self) -> None:
        assert detect_domain("Brew a cup of coffee") == GENERAL_DOMAIN

    def test_tags_fallback(self) -> None:
        # Goal text says nothing, but tags carry the signal
        assert detect_domain("Build it", tags=["mcp", "server"]) == "mcp"

    def test_empty_input(self) -> None:
        assert detect_domain("") == GENERAL_DOMAIN
        assert detect_domain(None) == GENERAL_DOMAIN  # type: ignore[arg-type]

    def test_case_insensitive(self) -> None:
        assert detect_domain("FASTAPI service") == "fastapi"


# ---------------------------------------------------------------------------
# compute_progression domain routing
# ---------------------------------------------------------------------------


@dataclass
class _FakeTool:
    name: str
    description: str = ""
    created_by: str = "human"
    tags: list[str] = field(default_factory=list)


class _FakeRegistry:
    def __init__(self, tools: list[_FakeTool]) -> None:
        self._tools = tools

    def get_active_tools(self) -> list[_FakeTool]:
        return list(self._tools)


class _FakeSoil:
    """Minimal stand-in — progression helpers treat missing collections as empty."""

    def __init__(self) -> None:
        self._collections: dict[str, Any] = {}


def test_compute_progression_general_sees_all_tools() -> None:
    tools = [
        _FakeTool("fastapi_linter", tags=["fastapi"]),
        _FakeTool("click_helper", tags=["cli"]),
    ]
    metrics = compute_progression(_FakeSoil(), _FakeRegistry(tools), [], domain=GENERAL_DOMAIN)
    assert metrics.domain == GENERAL_DOMAIN
    assert metrics.total_tool_count == 2


def test_compute_progression_domain_filters_tools() -> None:
    tools = [
        _FakeTool("fastapi_linter", tags=["fastapi"]),
        _FakeTool("click_helper", tags=["cli"]),
        _FakeTool("mcp_probe", tags=["mcp"]),
    ]
    metrics = compute_progression(_FakeSoil(), _FakeRegistry(tools), [], domain="cli")
    assert metrics.domain == "cli"
    assert metrics.total_tool_count == 1


def test_compute_all_domains_includes_each_bucket() -> None:
    tools = [_FakeTool("x", tags=["fastapi"])]
    by_domain = compute_all_domains(_FakeSoil(), _FakeRegistry(tools), [])
    expected = set(DOMAINS) | {GENERAL_DOMAIN}
    assert set(by_domain.keys()) == expected
    for m in by_domain.values():
        assert isinstance(m, ProgressionMetrics)


# ---------------------------------------------------------------------------
# format_all_domains_report rendering
# ---------------------------------------------------------------------------


def test_report_contains_every_domain_and_stage() -> None:
    tools = [_FakeTool("x", tags=["fastapi"])]
    by_domain = compute_all_domains(_FakeSoil(), _FakeRegistry(tools), [])
    text = format_all_domains_report(by_domain)
    for domain in DOMAINS:
        assert f"  {domain:<9}" in text
    assert "Stage" in text


def test_progress_fraction_for_empty_domain_is_zero() -> None:
    m = ProgressionMetrics(total_tool_count=0)
    assert _progress_fraction(m) == 0.0
    assert _progress_note(m) == "no builds"


def test_progress_fraction_stage2_uses_coverage() -> None:
    m = ProgressionMetrics(total_tool_count=5, current_stage=2, coverage_fraction=0.85)
    assert _progress_fraction(m) == 0.85
    assert "coverage" in _progress_note(m)


# ---------------------------------------------------------------------------
# Recomposer domain re-ranking
# ---------------------------------------------------------------------------


@dataclass
class _FakeNutrient:
    content: str = ""
    framework: str = ""
    tags: list[str] = field(default_factory=list)


def test_reorder_same_domain_goes_first() -> None:
    nuts = [
        _FakeNutrient(content="click cli pattern", tags=["cli"]),
        _FakeNutrient(content="fastapi pattern", tags=["fastapi"]),
        _FakeNutrient(content="general helper", tags=[]),
    ]
    ordered = reorder_by_domain(nuts, "fastapi")
    assert ordered[0] is nuts[1]  # fastapi first


def test_reorder_general_domain_is_noop() -> None:
    nuts = [
        _FakeNutrient(content="a", tags=["cli"]),
        _FakeNutrient(content="b", tags=["fastapi"]),
    ]
    ordered = reorder_by_domain(nuts, GENERAL_DOMAIN)
    assert ordered[0] is nuts[0]
    assert ordered[1] is nuts[1]


def test_reorder_preserves_stability_on_ties() -> None:
    nuts = [
        _FakeNutrient(content="a", tags=["fastapi"]),
        _FakeNutrient(content="b", tags=["fastapi"]),
        _FakeNutrient(content="c", tags=["fastapi"]),
    ]
    ordered = reorder_by_domain(nuts, "fastapi")
    # All have the same score, original order preserved
    assert [n.content for n in ordered] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# benchmark_compare renderer + driver
# ---------------------------------------------------------------------------


def test_format_row_shows_delta() -> None:
    row = ChallengeComparison(
        challenge_id="t1-fizzbuzz",
        cloud_verdict="pass",
        cloud_score=1.0,
        cloud_cost=0.0,
        local_verdict="pass",
        local_score=0.9,
        local_cost=0.0,
    )
    line = format_row(row)
    assert "t1-fizzbuzz" in line
    assert "PASS" in line
    assert "-0.10" in line


def test_format_report_renders_overall() -> None:
    rows = [
        ChallengeComparison("t1-a", "pass", 1.0, 1.0, "pass", 0.8, 0.0),
        ChallengeComparison("t1-b", "pass", 1.0, 1.5, "fail", 0.2, 0.0),
    ]
    report = CompareReport(
        rows=rows,
        local_available=True,
        cloud_score_overall=2.0,
        local_score_overall=1.0,
        cloud_cost_overall=2.5,
        local_cost_overall=0.0,
    )
    text = format_report(report)
    assert "Overall" in text
    assert "Cost" in text
    # +? 100% cloud vs 50% local = -50% delta
    assert "-50%" in text
    assert "$-2.50" in text


def test_format_report_shows_skip_note_when_local_unavailable() -> None:
    report = CompareReport(
        rows=[],
        local_available=False,
        local_skipped_reason="Ollama not running",
    )
    text = format_report(report)
    assert "local run skipped" in text


@pytest.mark.asyncio
async def test_driver_skips_local_when_probe_returns_false() -> None:
    async def fake_probe() -> bool:
        return False

    async def fake_cloud(challenges: list[Any]) -> list[Any]:
        @dataclass
        class R:
            challenge_id: str
            verdict: str = "pass"
            weighted_score: float = 1.0
            cost_usd: float = 0.5

        return [R(c.id) for c in challenges]

    async def fake_local(challenges: list[Any]) -> list[Any]:
        raise AssertionError("local runner must not be invoked when probe is False")

    report = await run_benchmark_compare(
        tiers=[1],
        cloud_runner=fake_cloud,
        local_runner=fake_local,
        ollama_probe=fake_probe,
    )
    assert report.local_available is False
    assert len(report.rows) > 0
    for row in report.rows:
        assert row.cloud_verdict == "pass"
        assert row.local_verdict == ""


@pytest.mark.asyncio
async def test_driver_runs_both_when_probe_true() -> None:
    async def fake_probe() -> bool:
        return True

    @dataclass
    class R:
        challenge_id: str
        verdict: str = ""
        weighted_score: float = 0.0
        cost_usd: float = 0.0

    async def fake_cloud(cs: list[Any]) -> list[Any]:
        return [R(c.id, verdict="pass", weighted_score=1.0, cost_usd=0.50) for c in cs]

    async def fake_local(cs: list[Any]) -> list[Any]:
        return [R(c.id, verdict="pass", weighted_score=0.85, cost_usd=0.0) for c in cs]

    report = await run_benchmark_compare(
        tiers=[1],
        cloud_runner=fake_cloud,
        local_runner=fake_local,
        ollama_probe=fake_probe,
    )
    assert report.local_available is True
    assert all(row.cloud_verdict == "pass" for row in report.rows)
    assert all(row.local_verdict == "pass" for row in report.rows)
    # cost delta is negative (local is free)
    assert report.local_cost_overall == 0.0
    assert report.cloud_cost_overall > 0
