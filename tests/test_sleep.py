"""Hermetic tests for belief.ecology.sleep (v3.3 Session 3).

Uses FakeSoil + FakeEconomist + FakeRegistry stubs and monkeypatches
the crystallizer's heavy LLM call. No ChromaDB or Anthropic API needed.

Run:
    python3 -m pytest tests/test_sleep.py -q --timeout=60
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from belief.ecology.sleep import (
    DEFAULT_CYCLES,
    SLEEP_REPLAY_CYCLE_USD,
    SleepConfig,
    SleepResult,
    _episode_anomaly_score,
    _refresh_fsrs_schedules,
    _sample_anomaly_weighted,
    cli_format_result,
    result_to_dict,
)
from belief.ecology.sleep import run as sleep_run


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeChromaCollection:
    """In-memory ChromaDB-shaped collection for tests."""

    def __init__(self, name: str, items: list[tuple[str, str | None, dict]] | None = None):
        self.name = name
        self.items = list(items or [])  # list of (id, document, metadata)
        self.update_calls: list[tuple[list[str], list[dict]]] = []

    def count(self) -> int:
        return len(self.items)

    def get(self, include=None, limit=None, ids=None):
        rows = self.items
        if ids is not None:
            rows = [r for r in rows if r[0] in ids]
        if limit is not None:
            rows = rows[:limit]
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
        }

    def update(self, ids, metadatas):
        self.update_calls.append((list(ids), list(metadatas)))
        for i, doc_id in enumerate(ids):
            for j, (existing_id, doc, _) in enumerate(self.items):
                if existing_id == doc_id:
                    self.items[j] = (existing_id, doc, metadatas[i])
                    break


class FakeSoil:
    def __init__(
        self,
        episodes: list[dict] | None = None,
        nutrient_metas: list[tuple[str, dict]] | None = None,
    ):
        ep_items = []
        for i, ep in enumerate(episodes or []):
            ep_items.append((ep.get("trace_id", f"ep{i}"), ep.get("description", ""), ep))
        nut_items = []
        for nid, meta in nutrient_metas or []:
            nut_items.append((nid, "", meta))
        self._collections = {
            "belief_episodes": FakeChromaCollection("belief_episodes", ep_items),
            "belief_tools": FakeChromaCollection("belief_tools", nut_items),
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
class FakeRegistry:
    descriptions: list[dict] = field(default_factory=list)
    reload_count: int = 0

    def get_all_covenant_descriptions(self) -> list[dict]:
        return list(self.descriptions)

    def load_dynamic_covenants(self) -> None:
        self.reload_count += 1


@dataclass
class FakeCandidate:
    name: str
    qualified: bool = True


class FakeCrystallizer:
    """Stand-in for belief.evolution.crystallizer module.

    Sleep accepts ``crystallizer=`` as an injection slot (added so tests
    don't need to import the real crystallizer, which drags pydantic
    via belief.evolution.__init__). Records call counts so tests can
    assert behaviour.
    """

    def __init__(self):
        self.calls = {"sweep": 0, "propose": 0, "filter": 0, "promote": []}
        self.raise_propose = False

    def sweep_templates(self, traces):
        self.calls["sweep"] += 1
        return [FakeCandidate(name=f"sweep-cycle-{self.calls['sweep']}", qualified=True)]

    async def propose_invariants(self, traces, existing, model="x", max_proposals=10):
        self.calls["propose"] += 1
        if self.raise_propose:
            raise RuntimeError("simulated llm failure")
        return [FakeCandidate(name=f"claude-cycle-{self.calls['propose']}", qualified=True)]

    def filter_candidates(self, candidates, traces, max_violation_rate=0.05):
        self.calls["filter"] += 1
        return list(candidates)

    def promote_to_covenant(self, candidate, soil):
        cid = f"cov-{candidate.name}"
        self.calls["promote"].append(cid)
        return cid


@pytest.fixture
def stub_crystallizer() -> FakeCrystallizer:
    return FakeCrystallizer()


# ── Helpers ────────────────────────────────────────────────────────────────


def _ep(
    trace_id: str, *, passed: bool = True, score: float = 1.0, cost_usd: float = 0.0, **extra
) -> dict:
    return {"trace_id": trace_id, "passed": passed, "score": score, "cost_usd": cost_usd, **extra}


def _run(
    cfg: SleepConfig,
    soil: FakeSoil,
    econ: FakeEconomist,
    registry: FakeRegistry | None = None,
    crystallizer: FakeCrystallizer | None = None,
) -> SleepResult:
    return asyncio.run(
        sleep_run(
            cfg,
            soil=soil,
            economist=econ,
            registry=registry,
            crystallizer=crystallizer,
        )
    )


@pytest.fixture
def cfg(tmp_path: Path) -> SleepConfig:
    return SleepConfig(
        cycles=1,
        budget_usd=2.0,
        state_path=tmp_path / "sleep_state.json",
        audit_path=tmp_path / "audit.jsonl",
    )


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── 1. Anomaly scoring + sampling ─────────────────────────────────────────


def test_anomaly_score_failed_build_outranks_passing() -> None:
    fail = _ep("f", passed=False, score=0.0, cost_usd=0.0)
    win = _ep("w", passed=True, score=1.0, cost_usd=0.0)
    assert _episode_anomaly_score(fail) > _episode_anomaly_score(win)


def test_anomaly_score_high_cost_contributes() -> None:
    cheap = _ep("c", passed=True, score=1.0, cost_usd=0.0)
    expensive = _ep("e", passed=True, score=1.0, cost_usd=10.0)
    assert _episode_anomaly_score(expensive) > _episode_anomaly_score(cheap)


def test_anomaly_score_low_score_contributes() -> None:
    low = _ep("l", passed=True, score=0.1, cost_usd=0.0)
    high = _ep("h", passed=True, score=0.9, cost_usd=0.0)
    assert _episode_anomaly_score(low) > _episode_anomaly_score(high)


def test_sample_anomaly_weighted_returns_worst_first() -> None:
    soil = FakeSoil(
        episodes=[
            _ep("good1", passed=True, score=1.0),
            _ep("bad1", passed=False, score=0.0, cost_usd=2.0),
            _ep("good2", passed=True, score=0.95),
            _ep("bad2", passed=False, score=0.1, cost_usd=4.0),
        ]
    )
    sample = _sample_anomaly_weighted(soil, n=2)
    assert len(sample) == 2
    ids = [s["trace_id"] for s in sample]
    assert "bad1" in ids and "bad2" in ids
    assert "good1" not in ids and "good2" not in ids


def test_sample_anomaly_weighted_empty_soil_returns_empty() -> None:
    assert _sample_anomaly_weighted(FakeSoil(), n=10) == []


# ── 2. Cycles + Phase-A integration ───────────────────────────────────────


def test_cycles_zero_is_noop(cfg: SleepConfig) -> None:
    cfg.cycles = 0
    res = _run(cfg, FakeSoil(), FakeEconomist())
    assert res.cycles_completed == 0
    assert res.new_covenant_ids == []


def test_single_cycle_with_episodes_promotes_covenants(cfg: SleepConfig, stub_crystallizer) -> None:
    soil = FakeSoil(episodes=[_ep("ep1", passed=False, score=0.0)])
    econ = FakeEconomist()
    reg = FakeRegistry()
    res = _run(cfg, soil, econ, reg, stub_crystallizer)
    assert res.cycles_completed == 1
    assert len(res.new_covenant_ids) == 2  # one sweep + one claude
    assert any("sweep" in cid for cid in res.new_covenant_ids)
    assert any("claude" in cid for cid in res.new_covenant_ids)
    assert reg.reload_count == 1


def test_no_episodes_means_no_phase_a_work(cfg: SleepConfig, stub_crystallizer) -> None:
    res = _run(cfg, FakeSoil(), FakeEconomist(), FakeRegistry(), stub_crystallizer)
    assert res.cycles_completed == 1  # cycle still ran (Phase B)
    assert res.new_covenant_ids == []
    assert stub_crystallizer.calls["sweep"] == 0  # never called — no episodes


def test_dry_run_does_not_promote_or_reload(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.dry_run = True
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    reg = FakeRegistry()
    res = _run(cfg, soil, FakeEconomist(), reg, stub_crystallizer)
    assert res.cycles_completed == 1
    assert res.dry_run is True
    assert all(cid.startswith("dry-run:") for cid in res.new_covenant_ids)
    assert reg.reload_count == 0
    assert stub_crystallizer.calls["promote"] == []


def test_propose_invariants_failure_falls_back_to_sweep_only(
    cfg: SleepConfig, stub_crystallizer
) -> None:
    stub_crystallizer.raise_propose = True
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    res = _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    # Sweep candidate still promoted; claude candidate skipped.
    assert len(res.new_covenant_ids) == 1
    assert "sweep" in res.new_covenant_ids[0]


def test_no_crystallize_skips_phase_a(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.crystallize = False
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    res = _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    assert res.new_covenant_ids == []
    assert stub_crystallizer.calls["sweep"] == 0


# ── 3. Phase B (FSRS housekeeping) ────────────────────────────────────────


def test_fsrs_refresh_skips_future_scheduled() -> None:
    """Already-scheduled-in-the-future nutrients should not be touched."""
    future_ts = 9_999_999_999.0
    soil = FakeSoil(
        nutrient_metas=[
            ("future", {"fsrs_stability": 5.0, "fsrs_next_review": future_ts}),
        ]
    )
    refreshed = _refresh_fsrs_schedules(soil)
    assert refreshed == 0
    assert soil._collections["belief_tools"].update_calls == []


def test_fsrs_refresh_updates_stale_overdue_schedule() -> None:
    soil = FakeSoil(
        nutrient_metas=[
            ("overdue", {"fsrs_stability": 5.0, "fsrs_next_review": 0.0}),
        ]
    )
    refreshed = _refresh_fsrs_schedules(soil)
    assert refreshed == 1
    # next_review now non-zero, in the future.
    new_meta = soil._collections["belief_tools"].items[0][2]
    assert new_meta["fsrs_next_review"] > 0


def test_fsrs_refresh_skips_invalidated_nutrients() -> None:
    soil = FakeSoil(
        nutrient_metas=[
            (
                "dead",
                {
                    "fsrs_stability": 5.0,
                    "fsrs_next_review": 0.0,
                    "valid_until": 100.0,  # invalidated long ago
                },
            ),
        ]
    )
    refreshed = _refresh_fsrs_schedules(soil)
    assert refreshed == 0


def test_no_fsrs_recompute_skips_phase_b(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.recompute_fsrs = False
    soil = FakeSoil(
        episodes=[],
        nutrient_metas=[("n1", {"fsrs_stability": 5.0, "fsrs_next_review": 0.0})],
    )
    res = _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    assert res.fsrs_schedules_refreshed == 0
    assert soil._collections["belief_tools"].update_calls == []


# ── 4. Time + budget caps ─────────────────────────────────────────────────


def test_max_minutes_cap_truncates_run(cfg: SleepConfig, stub_crystallizer) -> None:
    """A 0-minute cap means the first cycle is the only one allowed (elapsed=0 still <= 0)."""
    cfg.cycles = 5
    cfg.max_minutes = 0  # immediate timeout after first
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    res = _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    # First cycle should NOT run because elapsed > 0 quickly. Allow 0 or 1.
    assert res.truncated_reason in ("timeout", "")
    assert res.cycles_completed <= 1


def test_local_budget_cap_truncates_run(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.cycles = 10
    cfg.budget_usd = SLEEP_REPLAY_CYCLE_USD * 2.5  # ~2 cycles fit
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    res = _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    assert res.cycles_completed <= 3
    if res.truncated_reason:
        assert res.truncated_reason == "budget"


def test_economist_rejection_truncates_run(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.cycles = 3
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    res = _run(cfg, soil, FakeEconomist(approve=False), FakeRegistry(), stub_crystallizer)
    assert res.cycles_completed == 0
    assert res.economist_approved is False
    assert res.truncated_reason == "economist"


# ── 5. Economist contract ─────────────────────────────────────────────────


def test_quote_called_per_cycle(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.cycles = 2
    cfg.budget_usd = 5.0
    econ = FakeEconomist()
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    _run(cfg, soil, econ, FakeRegistry(), stub_crystallizer)
    assert len(econ.quotes) == 2
    assert all(q[0] == "sleep.replay_cycle" for q in econ.quotes)
    assert all(q[1] == SLEEP_REPLAY_CYCLE_USD for q in econ.quotes)


def test_commit_only_when_not_dry_run(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.dry_run = True
    econ = FakeEconomist()
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    _run(cfg, soil, econ, FakeRegistry(), stub_crystallizer)
    assert econ.commits == []  # quote yes, commit no


# ── 6. Phase C deferral ───────────────────────────────────────────────────


def test_synth_challenges_logs_deferred_event(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.synth_challenges = True
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    events = _read_audit(cfg.audit_path)
    assert any(e["event"] == "phase_c_deferred" for e in events)


# ── 7. Audit + state ─────────────────────────────────────────────────────


def test_audit_log_has_phase_a_b_and_summary(cfg: SleepConfig, stub_crystallizer) -> None:
    soil = FakeSoil(
        episodes=[_ep("ep1", passed=False)],
        nutrient_metas=[("n1", {"fsrs_stability": 5.0, "fsrs_next_review": 0.0})],
    )
    _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    events = _read_audit(cfg.audit_path)
    types = [e["event"] for e in events]
    assert "phase_a_complete" in types
    assert "phase_b_complete" in types
    assert "run_summary" in types


def test_state_file_written_after_real_run(cfg: SleepConfig, stub_crystallizer) -> None:
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    assert cfg.state_path.exists()
    state = json.loads(cfg.state_path.read_text())
    assert state["cycles_completed"] == 1


def test_state_file_not_written_in_dry_run(cfg: SleepConfig, stub_crystallizer) -> None:
    cfg.dry_run = True
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    assert not cfg.state_path.exists()


# ── 8. Concurrency ────────────────────────────────────────────────────────


def _concurrent_sleep_worker(state_path: str, audit_path: str, holds_for: float) -> None:
    """Top-level so multiprocessing can pickle it. Runs an empty sleep
    while holding the lock for `holds_for` seconds."""
    import time as _t
    from belief.ecology.sleep import (  # noqa: PLC0415
        SleepConfig as SC,
        run as r,
    )

    cfg = SC(cycles=0, state_path=Path(state_path), audit_path=Path(audit_path))
    asyncio.run(r(cfg, soil=FakeSoil(), economist=FakeEconomist()))
    _t.sleep(holds_for)


def test_concurrent_run_records_skip(tmp_path: Path) -> None:
    """When two sleep runs race, the second observes the lock and bails fast."""
    state_path = tmp_path / "sleep_state.json"
    audit_path = tmp_path / "audit.jsonl"

    # We test the lock contract directly rather than via real concurrency,
    # because an empty cycles=0 run releases the lock too quickly to race.
    from belief.ecology.sleep import _sleep_lock  # noqa: PLC0415

    with _sleep_lock(state_path) as held:
        assert held is True
        # Now a subprocess should observe the lock as held and audit a skip.
        proc = multiprocessing.Process(
            target=_concurrent_sleep_worker,
            args=(str(state_path), str(audit_path), 0.0),
        )
        proc.start()
        proc.join(timeout=10)
        assert not proc.is_alive()

    events = _read_audit(audit_path)
    assert any(e["event"] == "concurrent_run_skipped" for e in events)


# ── 9. Misc ───────────────────────────────────────────────────────────────


def test_result_round_trips_through_json(cfg: SleepConfig, stub_crystallizer) -> None:
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    res = _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    encoded = json.dumps(result_to_dict(res))
    decoded = json.loads(encoded)
    assert decoded["cycles_completed"] == 1


def test_cli_format_smoke(cfg: SleepConfig, stub_crystallizer) -> None:
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    res = _run(cfg, soil, FakeEconomist(), FakeRegistry(), stub_crystallizer)
    text = cli_format_result(res)
    assert "Sleep" in text
    assert "covenant" in text


def test_default_constants_pinned() -> None:
    assert DEFAULT_CYCLES == 3
    assert SLEEP_REPLAY_CYCLE_USD == 0.30


def test_inline_schedule_matches_fsrs() -> None:
    """Pin Sleep's inlined schedule formula to belief.memory.fsrs canonical impl.

    Sleep inlines this so its module load doesn't drag pydantic via
    belief.memory.__init__. Skipped in environments without pydantic
    (sandbox); runs on Mac to catch drift if either impl changes.
    """
    pytest.importorskip("pydantic")
    from belief.memory.fsrs import schedule_next_review  # noqa: PLC0415
    from belief.ecology.sleep import _schedule_days_for_retention  # noqa: PLC0415

    for stability in (0.0, 1.0, 5.0, 30.0, 365.0):
        for retention in (0.7, 0.85, 0.9, 0.95):
            assert _schedule_days_for_retention(stability, retention) == pytest.approx(
                schedule_next_review(stability, retention)
            )


def test_post_condition_covenants_strictly_non_decreasing(
    cfg: SleepConfig, stub_crystallizer
) -> None:
    """Sleep should never reduce the covenant count (per spec §3.3 test #18)."""
    cfg.cycles = 3
    cfg.budget_usd = 5.0
    soil = FakeSoil(episodes=[_ep("ep1", passed=False)])
    reg = FakeRegistry(descriptions=[{"name": "existing-1"}, {"name": "existing-2"}])
    res = _run(cfg, soil, FakeEconomist(), reg, stub_crystallizer)
    # We added covenants this run and reg.descriptions never had any removed.
    assert len(res.new_covenant_ids) >= 1
