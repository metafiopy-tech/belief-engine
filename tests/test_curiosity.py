"""Hermetic tests for belief.ecology.curiosity (v3.3 Session 5).

Uses FakeSoil + FakeChromaCollection + FakeEconomist + FakeLLM stubs.
No ChromaDB or Anthropic API needed.

Run:
    python3 -m pytest tests/test_curiosity.py -q --timeout=60
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from belief.ecology._information_gain import (
    COVENANT_SPARSE_THRESHOLD,
    Gap,
    estimate_info_gain,
    gaps_summary,
    identify_gaps,
)
from belief.ecology.curiosity import (
    CURIOSITY_ACTION_KEY,
    CURIOSITY_PROPOSE_USD,
    DEFAULT_SUGGEST_N,
    CuriosityResult,
    GoalCandidate,
    auto_build,
    cli_format_result,
    result_to_dict,
)
from belief.ecology.curiosity import suggest as curiosity_suggest


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeChromaCollection:
    def __init__(self, name: str, items: list[tuple[str, dict]] | None = None):
        self.name = name
        self.items = list(items or [])  # list of (id, metadata)

    def count(self) -> int:
        return len(self.items)

    def get(self, include=None, limit=None, ids=None):
        rows = self.items
        if limit is not None:
            rows = rows[:limit]
        return {
            "ids": [r[0] for r in rows],
            "metadatas": [r[1] for r in rows],
            "documents": [""] * len(rows),
        }


class FakeSoil:
    def __init__(
        self,
        tools: list[dict] | None = None,
        episodes: list[dict] | None = None,
        covenants: list[dict] | None = None,
    ):
        self._collections = {
            "belief_tools": FakeChromaCollection(
                "belief_tools",
                [(f"t{i}", m) for i, m in enumerate(tools or [])],
            ),
            "belief_episodes": FakeChromaCollection(
                "belief_episodes",
                [(f"e{i}", m) for i, m in enumerate(episodes or [])],
            ),
            "belief_covenants": FakeChromaCollection(
                "belief_covenants",
                [(f"c{i}", m) for i, m in enumerate(covenants or [])],
            ),
        }


class _FakeQuote:
    def __init__(self, approved: bool, reason: str = ""):
        self.approved = approved
        self.reason = reason or ("approved" if approved else "rejected")


@dataclass
class FakeEconomist:
    approve: bool = True
    quotes: list[tuple[str, float]] = field(default_factory=list)
    commits: list[tuple[str, float]] = field(default_factory=list)

    def quote(self, action: str, estimated_usd: float):
        self.quotes.append((action, estimated_usd))
        return _FakeQuote(self.approve)

    def commit(self, action: str, actual_usd: float) -> None:
        self.commits.append((action, actual_usd))


@dataclass
class FakeLLM:
    """Minimal LLM stub. ``generate`` returns a static response per call."""

    canned_response: str = ""
    raise_on_call: bool = False
    calls: int = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("simulated llm failure")
        return self.canned_response


# ── Helpers ───────────────────────────────────────────────────────────────


def _run(
    soil: FakeSoil,
    econ: FakeEconomist,
    *,
    n: int = DEFAULT_SUGGEST_N,
    llm=None,
    dry_run: bool = False,
    budget_usd: float = 1.0,
    tmp_path: Path | None = None,
) -> CuriosityResult:
    sp = (tmp_path / "curiosity_state.json") if tmp_path else None
    ap = (tmp_path / "audit.jsonl") if tmp_path else None
    return asyncio.run(
        curiosity_suggest(
            n=n,
            soil=soil,
            economist=econ,
            llm=llm,
            state_path=sp,
            audit_path=ap,
            dry_run=dry_run,
            budget_usd=budget_usd,
        )
    )


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── 1. Gap identifier ─────────────────────────────────────────────────────


def test_empty_soil_returns_full_gap_list() -> None:
    """An empty soil has every tracked extension + framework absent + sparse covenants."""
    gaps = identify_gaps(FakeSoil())
    cats = {g.category for g in gaps}
    assert "file_ext" in cats
    assert "framework" in cats
    assert "covenant_sparse" in cats


def test_well_covered_soil_has_fewer_gaps() -> None:
    """If soil shows fastapi heavy + pytest covered, those frameworks aren't flagged."""
    tools = [
        {"framework": "fastapi", "tags": "fastapi"},
        {"framework": "pytest", "tags": "pytest"},
    ]
    gaps = identify_gaps(FakeSoil(tools=tools))
    framework_gaps = {g.name for g in gaps if g.category == "framework"}
    assert "fastapi" not in framework_gaps
    assert "pytest" not in framework_gaps
    # Other frameworks still flagged.
    assert "starlette" in framework_gaps


def test_covenant_sparse_disabled_when_threshold_met() -> None:
    covs = [{"tags": "crystallized"} for _ in range(COVENANT_SPARSE_THRESHOLD + 1)]
    gaps = identify_gaps(FakeSoil(covenants=covs))
    cats = {g.category for g in gaps}
    assert "covenant_sparse" not in cats


def test_dockerfile_episode_satisfies_dockerfile_gap() -> None:
    """An episode that recorded has_dockerfile=True should clear the .dockerfile gap."""
    eps = [{"has_dockerfile": True}]
    gaps = identify_gaps(FakeSoil(episodes=eps))
    file_ext_gaps = {g.name for g in gaps if g.category == "file_ext"}
    assert ".dockerfile" not in file_ext_gaps


def test_gaps_summary_handles_empty() -> None:
    assert gaps_summary([]) == "no gaps"


def test_gaps_summary_groups_by_category() -> None:
    gaps = [
        Gap(category="file_ext", name=".toml"),
        Gap(category="file_ext", name=".yaml"),
        Gap(category="framework", name="fastmcp"),
    ]
    text = gaps_summary(gaps)
    assert "file_ext: 2" in text
    assert "framework: 1" in text


# ── 2. Info gain estimator ────────────────────────────────────────────────


def test_info_gain_zero_when_no_gaps() -> None:
    gain, addressed = estimate_info_gain("build a fastapi app", [])
    assert gain == 0.0
    assert addressed == []


def test_info_gain_higher_with_more_addressed_gaps() -> None:
    gaps = [
        Gap(category="file_ext", name=".toml"),
        Gap(category="framework", name="fastmcp"),
    ]
    one, _ = estimate_info_gain("build a toml parser", gaps)
    two, _ = estimate_info_gain("build a fastmcp service that parses toml", gaps)
    assert two > one


def test_info_gain_matches_extension_with_or_without_dot() -> None:
    gaps = [Gap(category="file_ext", name=".toml")]
    gain, addressed = estimate_info_gain("write a toml validator", gaps)
    assert gain > 0
    assert len(addressed) == 1


# ── 3. Curiosity.suggest end-to-end ──────────────────────────────────────


def test_suggest_empty_soil_no_candidates_or_basic_exploration(tmp_path: Path) -> None:
    """Empty soil ⇒ many gaps but stub mode generates a goal per gap."""
    res = _run(FakeSoil(), FakeEconomist(), tmp_path=tmp_path)
    assert res.gaps_identified > 0
    assert len(res.candidates) > 0
    assert res.selected is not None
    assert res.llm_stub is True


def test_suggest_ranking_respects_bang_per_buck(tmp_path: Path) -> None:
    """Top candidate should have the highest bang_per_buck, descending after."""
    res = _run(FakeSoil(), FakeEconomist(), tmp_path=tmp_path)
    bpb = [c.bang_per_buck for c in res.candidates]
    assert bpb == sorted(bpb, reverse=True)


def test_stub_mode_is_deterministic(tmp_path: Path) -> None:
    """Same soil should produce the same stub goals across runs."""
    soil1 = FakeSoil()
    soil2 = FakeSoil()
    res1 = _run(soil1, FakeEconomist(), tmp_path=tmp_path / "a")
    res2 = _run(soil2, FakeEconomist(), tmp_path=tmp_path / "b")
    g1 = [c.goal for c in res1.candidates]
    g2 = [c.goal for c in res2.candidates]
    # Same gap-set produces same synthesized goals (modulo top-N truncation order).
    assert sorted(g1) == sorted(g2)


def test_suggest_n_caps_candidate_count(tmp_path: Path) -> None:
    res = _run(FakeSoil(), FakeEconomist(), n=2, tmp_path=tmp_path)
    assert len(res.candidates) <= 2


def test_dry_run_skips_state_and_commit(tmp_path: Path) -> None:
    econ = FakeEconomist()
    _run(FakeSoil(), econ, dry_run=True, tmp_path=tmp_path)
    assert econ.commits == []
    assert not (tmp_path / "curiosity_state.json").exists()


def test_real_run_writes_state_and_commits(tmp_path: Path) -> None:
    econ = FakeEconomist()
    _run(FakeSoil(), econ, tmp_path=tmp_path)
    assert econ.commits == [(CURIOSITY_ACTION_KEY, 0.0)]  # stub mode → 0 cost
    state = json.loads((tmp_path / "curiosity_state.json").read_text())
    assert state["candidates_proposed"] >= 1


# ── 4. LLM integration ───────────────────────────────────────────────────


def test_real_llm_response_parsed_into_candidates(tmp_path: Path) -> None:
    """When llm is provided and responds, its goals override the stubs."""
    llm = FakeLLM(
        canned_response=(
            "Build a TOML config validator service with a healthcheck endpoint\n"
            "Build a fastmcp-based notification gateway\n"
            "Build a starlette WebSocket echo server\n"
        )
    )
    res = _run(FakeSoil(), FakeEconomist(), llm=llm, tmp_path=tmp_path)
    assert res.llm_stub is False
    assert llm.calls == 1
    # All canned goals should appear in the candidate list.
    goals = [c.goal for c in res.candidates]
    assert any("TOML config validator" in g for g in goals)
    assert any("fastmcp" in g for g in goals)


def test_llm_failure_falls_back_to_stub_goals(tmp_path: Path) -> None:
    """When the real LLM raises, we fall back to deterministic stubs and
    set llm_stub=True (the result reflects what was actually used)."""
    llm = FakeLLM(raise_on_call=True)
    res = _run(FakeSoil(), FakeEconomist(), llm=llm, tmp_path=tmp_path)
    assert res.llm_stub is True  # fallback ⇒ end-state was stub mode
    assert len(res.candidates) > 0
    # cost_usd is 0 since the LLM call failed before we incremented it.
    assert res.cost_usd == 0.0


def test_real_llm_cost_recorded(tmp_path: Path) -> None:
    """A successful real-LLM run charges CURIOSITY_PROPOSE_USD to the Economist."""
    econ = FakeEconomist()
    llm = FakeLLM(canned_response="Build a TOML parser\nBuild a YAML validator")
    res = _run(FakeSoil(), econ, llm=llm, tmp_path=tmp_path)
    assert res.cost_usd == CURIOSITY_PROPOSE_USD
    assert econ.commits == [(CURIOSITY_ACTION_KEY, CURIOSITY_PROPOSE_USD)]


def test_unparseable_llm_response_falls_back(tmp_path: Path) -> None:
    llm = FakeLLM(canned_response="     \n\n   ")  # whitespace-only
    res = _run(FakeSoil(), FakeEconomist(), llm=llm, tmp_path=tmp_path)
    assert len(res.candidates) > 0  # stub fallback kicked in


# ── 5. Economist contract ────────────────────────────────────────────────


def test_quote_called_at_proposed_price(tmp_path: Path) -> None:
    econ = FakeEconomist()
    _run(FakeSoil(), econ, tmp_path=tmp_path)
    assert econ.quotes == [(CURIOSITY_ACTION_KEY, CURIOSITY_PROPOSE_USD)]


def test_economist_rejection_short_circuits(tmp_path: Path) -> None:
    res = _run(FakeSoil(), FakeEconomist(approve=False), tmp_path=tmp_path)
    assert res.economist_approved is False
    assert res.candidates == []
    assert res.gaps_identified == 0


def test_local_budget_under_proposal_price_skips(tmp_path: Path) -> None:
    """If --budget is below the per-proposal price, suggest bails before working."""
    res = _run(
        FakeSoil(),
        FakeEconomist(),
        budget_usd=CURIOSITY_PROPOSE_USD / 2,
        tmp_path=tmp_path,
    )
    assert res.candidates == []
    assert res.gaps_identified == 0


# ── 6. Audit + state ─────────────────────────────────────────────────────


def test_audit_records_gaps_candidates_and_summary(tmp_path: Path) -> None:
    _run(FakeSoil(), FakeEconomist(), tmp_path=tmp_path)
    events = _read_audit(tmp_path / "audit.jsonl")
    types = [e["event"] for e in events]
    assert "gaps_identified" in types
    assert "candidate" in types
    assert "run_summary" in types


def test_audit_records_economist_rejection(tmp_path: Path) -> None:
    _run(FakeSoil(), FakeEconomist(approve=False), tmp_path=tmp_path)
    events = _read_audit(tmp_path / "audit.jsonl")
    assert any(e["event"] == "rejected_by_economist" for e in events)


# ── 7. auto_build deferral ────────────────────────────────────────────────


def test_auto_build_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError) as exc:
        asyncio.run(auto_build())
    assert "Session 5b" in str(exc.value)


# ── 8. Misc ───────────────────────────────────────────────────────────────


def test_result_round_trips_through_json(tmp_path: Path) -> None:
    res = _run(FakeSoil(), FakeEconomist(), tmp_path=tmp_path)
    encoded = json.dumps(result_to_dict(res))
    decoded = json.loads(encoded)
    assert decoded["gaps_identified"] == res.gaps_identified


def test_cli_format_smoke(tmp_path: Path) -> None:
    res = _run(FakeSoil(), FakeEconomist(), tmp_path=tmp_path)
    text = cli_format_result(res)
    assert "Curiosity" in text
    assert "selected" in text
    assert "stub LLM" in text


def test_constants_pinned() -> None:
    assert CURIOSITY_ACTION_KEY == "curiosity.propose"
    assert CURIOSITY_PROPOSE_USD == 0.05
    assert DEFAULT_SUGGEST_N == 5


def test_goal_candidate_dataclass_shape() -> None:
    gc = GoalCandidate(
        goal="x",
        estimated_info_gain=0.5,
        estimated_cost_usd=1.0,
        rationale="r",
        coverage_gaps_addressed=["a:b"],
        bang_per_buck=0.5,
    )
    assert gc.goal == "x"
    assert gc.bang_per_buck == 0.5
