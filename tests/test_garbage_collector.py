"""Hermetic tests for belief.ecology.garbage_collector (v3.3 Session 4).

Uses FakeSoil + FakeEconomist + FakeNutrient stubs. No ChromaDB needed.

Run:
    python3 -m pytest tests/test_garbage_collector.py -q --timeout=60
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from belief.ecology.garbage_collector import (
    GC_ACTION_KEY,
    REASON_BROKEN_TOOL,
    REASON_DUPLICATE_TOOL,
    REASON_INVALID_COVENANT,
    GCReport,
    _find_broken_tools,
    _find_duplicate_tools,
    _find_invalid_covenants,
    _is_parseable,
    _normalized_source,
    cli_format_result,
    report_to_dict,
)
from belief.ecology.garbage_collector import run as gc_run


# ── Fakes ──────────────────────────────────────────────────────────────────


@dataclass
class FakeNutrient:
    nutrient_id: str
    nutrient_type: str = (
        "pattern"  # "pattern" carries a tool's code; "covenant" carries covenant code_sample
    )
    code: str = ""
    code_sample: str = ""
    reinforcement_count: int = 0
    created_at: float = 0.0


@dataclass
class FakeSoil:
    nutrients: list[FakeNutrient] = field(default_factory=list)
    invalidated: list[tuple[str, str]] = field(default_factory=list)
    raise_on_invalidate: set[str] = field(default_factory=set)

    def iter_all_nutrients(self, include_invalidated: bool = False, as_of=None):
        invalid_ids = {nid for nid, _ in self.invalidated}
        for n in self.nutrients:
            if n.nutrient_id in invalid_ids and not include_invalidated:
                continue
            yield n

    def invalidate_nutrient(self, nutrient_id: str, reason: str, now=None) -> bool:
        if nutrient_id in self.raise_on_invalidate:
            raise RuntimeError(f"simulated soil failure on {nutrient_id}")
        if any(nid == nutrient_id for nid, _ in self.invalidated):
            return False
        self.invalidated.append((nutrient_id, reason))
        return True


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


def _tool(nid: str, code: str, **kw) -> FakeNutrient:
    return FakeNutrient(nutrient_id=nid, nutrient_type="pattern", code=code, **kw)


def _covenant(nid: str, code_sample: str) -> FakeNutrient:
    return FakeNutrient(nutrient_id=nid, nutrient_type="covenant", code_sample=code_sample)


def _run(
    soil: FakeSoil,
    econ: FakeEconomist,
    *,
    check_only=False,
    dry_run=False,
    tmp_path: Path | None = None,
) -> GCReport:
    sp = (tmp_path / "gc_state.json") if tmp_path else None
    ap = (tmp_path / "audit.jsonl") if tmp_path else None
    return asyncio.run(
        gc_run(
            check_only=check_only,
            dry_run=dry_run,
            soil=soil,
            economist=econ,
            state_path=sp,
            audit_path=ap,
        )
    )


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── 1. Source normalization helpers ───────────────────────────────────────


def test_normalized_source_strips_comments_and_whitespace() -> None:
    a = "def f():\n    # this comment is fine\n    return 1\n"
    b = "def f():\n        return  1"
    assert _normalized_source(a) == _normalized_source(b)


def test_normalized_source_returns_none_on_syntax_error() -> None:
    assert _normalized_source("def f(:  # broken") is None


def test_normalized_source_distinguishes_semantic_change() -> None:
    a = "def f():\n    return 1"
    b = "def f():\n    return 2"
    assert _normalized_source(a) != _normalized_source(b)


def test_is_parseable_happy_path() -> None:
    ok, msg = _is_parseable("x = 1")
    assert ok is True
    assert msg == ""


def test_is_parseable_syntax_error() -> None:
    ok, msg = _is_parseable("def (x:")
    assert ok is False
    assert "SyntaxError" in msg


# ── 2. Detection: broken_tool ─────────────────────────────────────────────


def test_find_broken_tools_happy_path() -> None:
    tools = [_tool("good", "x = 1"), _tool("bad", "def (broken")]
    found = _find_broken_tools(tools)
    assert [nid for nid, _ in found] == ["bad"]
    assert REASON_BROKEN_TOOL in found[0][1]


def test_find_broken_tools_empty_code_is_not_broken() -> None:
    found = _find_broken_tools([_tool("empty", "")])
    assert found == []


# ── 3. Detection: invalid_covenant ────────────────────────────────────────


def test_find_invalid_covenants_happy_path() -> None:
    covs = [_covenant("v", "if True: pass"), _covenant("x", "def (((broken")]
    found = _find_invalid_covenants(covs)
    assert [nid for nid, _ in found] == ["x"]


def test_find_invalid_covenants_no_code_skipped() -> None:
    """Sparse covenants with no code_sample are not 'invalid' — just sparse."""
    covs = [_covenant("nocode", "")]
    assert _find_invalid_covenants(covs) == []


# ── 4. Detection: duplicate_tool ──────────────────────────────────────────


def test_find_duplicates_keeps_higher_reinforcement() -> None:
    tools = [
        _tool("low", "def f():\n    return 1", reinforcement_count=1, created_at=100.0),
        _tool("high", "def f():\n    return 1", reinforcement_count=10, created_at=200.0),
    ]
    pairs = _find_duplicate_tools(tools)
    assert pairs == [("high", "low")]


def test_find_duplicates_tie_breaker_older_wins() -> None:
    tools = [
        _tool("newer", "def f():\n    return 1", reinforcement_count=5, created_at=200.0),
        _tool("older", "def f():\n    return 1", reinforcement_count=5, created_at=100.0),
    ]
    pairs = _find_duplicate_tools(tools)
    assert pairs == [("older", "newer")]


def test_find_duplicates_ignores_whitespace_and_comments() -> None:
    tools = [
        _tool("a", "def f():\n    return 1"),
        _tool("b", "def f():\n    # different comment\n    return  1   "),
    ]
    pairs = _find_duplicate_tools(tools)
    assert len(pairs) == 1


def test_find_duplicates_distinguishes_semantic_difference() -> None:
    tools = [
        _tool("a", "def f():\n    return 1"),
        _tool("b", "def f():\n    return 2"),
    ]
    assert _find_duplicate_tools(tools) == []


def test_find_duplicates_skips_broken_code() -> None:
    """Broken tools fall through to broken_tool category, not duplicate."""
    tools = [_tool("a", "def (broken"), _tool("b", "def (broken")]
    assert _find_duplicate_tools(tools) == []


def test_find_duplicates_three_way_keeps_one_invalidates_two() -> None:
    tools = [
        _tool("a", "x = 1", reinforcement_count=1, created_at=100.0),
        _tool("b", "x = 1", reinforcement_count=5, created_at=200.0),
        _tool("c", "x = 1", reinforcement_count=2, created_at=300.0),
    ]
    pairs = _find_duplicate_tools(tools)
    # b kept (highest reinforcement); a and c both removed.
    assert {removed for _, removed in pairs} == {"a", "c"}
    assert all(kept == "b" for kept, _ in pairs)


# ── 5. Full run integration ───────────────────────────────────────────────


def test_empty_soil_clean_report(tmp_path: Path) -> None:
    res = _run(FakeSoil(), FakeEconomist(), tmp_path=tmp_path)
    assert res.examined == 0
    assert res.cleaned == 0
    assert res.broken_tools == res.invalid_covenants == res.duplicate_tools == []


def test_run_invalidates_each_finding_category(tmp_path: Path) -> None:
    soil = FakeSoil(
        nutrients=[
            _tool("good", "x = 1"),
            _tool("broken1", "def (broken syntax"),
            _covenant("cov_ok", "if True: pass"),
            _covenant("cov_bad", "def !!!"),
            _tool("dup_a", "def g():\n    return 2", reinforcement_count=1),
            _tool("dup_b", "def g():\n    return 2", reinforcement_count=5),
        ]
    )
    res = _run(soil, FakeEconomist(), tmp_path=tmp_path)
    invalidated_ids = {nid for nid, _ in soil.invalidated}
    assert "broken1" in invalidated_ids
    assert "cov_bad" in invalidated_ids
    assert "dup_a" in invalidated_ids  # lower reinforcement loses
    assert "dup_b" not in invalidated_ids
    assert "good" not in invalidated_ids
    assert "cov_ok" not in invalidated_ids
    assert res.cleaned == 3


def test_dry_run_does_not_invalidate(tmp_path: Path) -> None:
    soil = FakeSoil(nutrients=[_tool("broken1", "def (broken")])
    res = _run(soil, FakeEconomist(), dry_run=True, tmp_path=tmp_path)
    assert res.broken_tools == ["broken1"]
    assert soil.invalidated == []
    assert res.cleaned == 0
    assert res.dry_run is True


def test_check_only_does_not_invalidate_and_skips_state(tmp_path: Path) -> None:
    soil = FakeSoil(nutrients=[_tool("broken1", "def (broken")])
    res = _run(soil, FakeEconomist(), check_only=True, tmp_path=tmp_path)
    assert res.check_only is True
    assert soil.invalidated == []
    # State file should not be written under check_only.
    assert not (tmp_path / "gc_state.json").exists()


def test_state_file_written_after_real_run(tmp_path: Path) -> None:
    soil = FakeSoil(nutrients=[_tool("broken1", "def (broken")])
    _run(soil, FakeEconomist(), tmp_path=tmp_path)
    state = json.loads((tmp_path / "gc_state.json").read_text())
    assert state["cleaned"] == 1
    assert state["broken_tools"] == 1


def test_already_invalidated_nutrient_skipped(tmp_path: Path) -> None:
    """If a tool was already tombstoned, GC should not re-process it."""
    soil = FakeSoil(nutrients=[_tool("broken1", "def (broken")])
    soil.invalidated.append(("broken1", "previously: predator"))
    res = _run(soil, FakeEconomist(), tmp_path=tmp_path)
    # iter_all_nutrients filters out invalidated, so GC sees nothing.
    assert res.broken_tools == []
    assert res.examined == 0


def test_invalidate_failure_does_not_crash(tmp_path: Path) -> None:
    soil = FakeSoil(
        nutrients=[_tool("broken1", "def (broken"), _tool("broken2", "if [")],
        raise_on_invalidate={"broken1"},
    )
    res = _run(soil, FakeEconomist(), tmp_path=tmp_path)
    # broken2 succeeds; broken1 fails but doesn't crash the run.
    assert res.cleaned == 1


# ── 6. Economist contract ─────────────────────────────────────────────────


def test_quote_called_with_zero_dollars(tmp_path: Path) -> None:
    econ = FakeEconomist()
    _run(FakeSoil(), econ, tmp_path=tmp_path)
    assert econ.quotes == [(GC_ACTION_KEY, 0.0)]


def test_commit_only_when_not_dry_run(tmp_path: Path) -> None:
    econ = FakeEconomist()
    _run(FakeSoil(), econ, dry_run=True, tmp_path=tmp_path)
    assert econ.commits == []


def test_commit_only_when_not_check_only(tmp_path: Path) -> None:
    econ = FakeEconomist()
    _run(FakeSoil(), econ, check_only=True, tmp_path=tmp_path)
    assert econ.commits == []


def test_economist_rejection_short_circuits(tmp_path: Path) -> None:
    soil = FakeSoil(nutrients=[_tool("broken", "def (broken")])
    res = _run(soil, FakeEconomist(approve=False), tmp_path=tmp_path)
    assert res.economist_approved is False
    assert soil.invalidated == []
    assert res.examined == 0  # never even iterated


# ── 7. Audit log ──────────────────────────────────────────────────────────


def test_audit_one_finding_per_category(tmp_path: Path) -> None:
    soil = FakeSoil(
        nutrients=[
            _tool("broken", "def (broken"),
            _covenant("badcov", "def !!!"),
            _tool("a", "x = 1", reinforcement_count=1),
            _tool("b", "x = 1", reinforcement_count=5),
        ]
    )
    _run(soil, FakeEconomist(), tmp_path=tmp_path)
    events = _read_audit(tmp_path / "audit.jsonl")
    cats = [e.get("category") for e in events if e["event"] == "finding"]
    assert "broken_tool" in cats
    assert "invalid_covenant" in cats
    assert "duplicate_tool" in cats
    assert any(e["event"] == "run_summary" for e in events)


def test_orphan_episodes_field_documented_inactive(tmp_path: Path) -> None:
    """Spec ships orphan_episodes in GCReport but detector is inactive
    in this codebase (see module docstring). Result is always [] today."""
    soil = FakeSoil(nutrients=[_tool("ok", "x = 1")])
    res = _run(soil, FakeEconomist(), tmp_path=tmp_path)
    assert res.orphan_episodes == []


# ── 8. Misc ───────────────────────────────────────────────────────────────


def test_report_round_trips_through_json(tmp_path: Path) -> None:
    soil = FakeSoil(nutrients=[_tool("broken", "def (broken")])
    res = _run(soil, FakeEconomist(), tmp_path=tmp_path)
    encoded = json.dumps(report_to_dict(res))
    decoded = json.loads(encoded)
    assert decoded["cleaned"] == 1
    # tuples become lists via json — assert that's tolerated downstream.
    assert isinstance(decoded["duplicate_tools"], list)


def test_cli_format_smoke(tmp_path: Path) -> None:
    soil = FakeSoil(
        nutrients=[
            _tool("broken", "def (broken"),
            _tool("a", "x = 1", reinforcement_count=1),
            _tool("b", "x = 1", reinforcement_count=5),
        ]
    )
    res = _run(soil, FakeEconomist(), tmp_path=tmp_path)
    text = cli_format_result(res)
    assert "GC" in text
    assert "broken tools" in text
    assert "duplicate" in text


def test_constants_pinned() -> None:
    assert GC_ACTION_KEY == "gc.run"
    assert "broken_tool" in REASON_BROKEN_TOOL
    assert "invalid_covenant" in REASON_INVALID_COVENANT
    assert "duplicate_tool" in REASON_DUPLICATE_TOOL


def test_covenant_nutrient_treated_as_covenant_not_tool(tmp_path: Path) -> None:
    """A nutrient typed as 'covenant' should never be checked as a tool,
    even if its code_sample happens to parse cleanly."""
    soil = FakeSoil(
        nutrients=[
            _covenant("c1", "def f(): return 1"),  # parseable code_sample
            _covenant("c2", "def f(): return 1"),  # identical
        ]
    )
    res = _run(soil, FakeEconomist(), tmp_path=tmp_path)
    # No duplicate detection across covenants — only tools.
    assert res.duplicate_tools == []
    assert soil.invalidated == []
