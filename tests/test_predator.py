"""Hermetic tests for belief.ecology.predator (v3.3 Session 2).

Uses FakeSoil + FakeEconomist stubs so the suite runs without ChromaDB
or the full belief-engine dep stack. Tests against the real
``compute_utility`` and the real Predator orchestration logic — the only
thing mocked is the soil + economist storage layer.

Run:
    python3 -m pytest tests/test_predator.py -q --timeout=60
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from belief.ecology.predator import (
    DEFAULT_UTILITY_THRESHOLD,
    FIRST_RUN_HARD_CAP,
    PredatorConfig,
    PredatorResult,
    cli_format_result,
    result_to_dict,
)
from belief.ecology.predator import run as predator_run


# ── Fakes ──────────────────────────────────────────────────────────────────


@dataclass
class FakeNutrient:
    """Minimum surface Predator needs from a Nutrient.

    Predator reads: nutrient_id, nutrient_type, reinforcement_count,
    lapse_count, last_reinforced, created_at, retrievability().
    """

    nutrient_id: str
    nutrient_type: str = "pattern"
    reinforcement_count: int = 0
    lapse_count: int = 0
    last_reinforced: float = 0.0
    created_at: float = 0.0
    _retrievability: float = 0.5

    def retrievability(self) -> float:
        return self._retrievability


@dataclass
class FakeSoil:
    """Minimal soil that captures invalidations for assertion."""

    nutrients: list[FakeNutrient] = field(default_factory=list)
    invalidated: list[tuple[str, str, float | None]] = field(default_factory=list)
    raise_on_invalidate: set[str] = field(default_factory=set)

    def iter_all_nutrients(self, include_invalidated: bool = False, as_of=None):
        for n in self.nutrients:
            if n.nutrient_id in {nid for nid, _, _ in self.invalidated}:
                if not include_invalidated:
                    continue
            yield n

    def invalidate_nutrient(self, nutrient_id: str, reason: str, now=None) -> bool:
        if nutrient_id in self.raise_on_invalidate:
            raise RuntimeError(f"simulated soil failure on {nutrient_id}")
        # Don't allow double-invalidate.
        if any(nid == nutrient_id for nid, _, _ in self.invalidated):
            return False
        self.invalidated.append((nutrient_id, reason, now))
        return True


class _FakeQuote:
    def __init__(self, approved: bool, reason: str = ""):
        self.approved = approved
        self.reason = reason or ("approved; $5 headroom" if approved else "rejected")


@dataclass
class FakeEconomist:
    """Records quote/commit calls; deterministic approval policy."""

    approve: bool = True
    quotes: list[tuple[str, float]] = field(default_factory=list)
    commits: list[tuple[str, float]] = field(default_factory=list)

    def quote(self, action: str, estimated_usd: float):
        self.quotes.append((action, estimated_usd))
        return _FakeQuote(self.approve)

    def commit(self, action: str, actual_usd: float) -> None:
        self.commits.append((action, actual_usd))


# ── Helpers ────────────────────────────────────────────────────────────────


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _aged(days: float) -> float:
    return _now() - days * 86400.0


def _make_low_utility(nid: str, type_: str = "pattern", days_old: float = 30.0) -> FakeNutrient:
    """Returns a nutrient whose utility is well below 0.15."""
    return FakeNutrient(
        nutrient_id=nid,
        nutrient_type=type_,
        reinforcement_count=0,
        lapse_count=2,
        last_reinforced=_aged(60),
        created_at=_aged(days_old),
        _retrievability=0.05,
    )


def _make_high_utility(nid: str, type_: str = "pattern", days_old: float = 30.0) -> FakeNutrient:
    """Returns a nutrient whose utility is well above 0.15."""
    return FakeNutrient(
        nutrient_id=nid,
        nutrient_type=type_,
        reinforcement_count=8,
        lapse_count=0,
        last_reinforced=_aged(1),
        created_at=_aged(days_old),
        _retrievability=0.85,
    )


def _seeded_state(state_path: Path) -> None:
    """Pretend Predator has run before — disables first-run cap."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_run_ts": 0, "first_run_was": True}))


def _run(config: PredatorConfig, soil: FakeSoil, econ: FakeEconomist) -> PredatorResult:
    return asyncio.run(predator_run(config, soil=soil, economist=econ))


@pytest.fixture
def cfg(tmp_path: Path) -> PredatorConfig:
    """Default config that uses tmp_path for state + audit. Pre-seeded state
    so tests aren't surprise-clamped by the first-run safety cap."""
    state = tmp_path / "predator_state.json"
    _seeded_state(state)
    return PredatorConfig(
        state_path=state,
        audit_path=tmp_path / "audit.jsonl",
        weights_path=tmp_path / "weights.json",  # missing -> defaults
    )


# ── 1. Empty + no-op behavior ──────────────────────────────────────────────


def test_empty_soil_is_noop(cfg: PredatorConfig) -> None:
    res = _run(cfg, FakeSoil(), FakeEconomist())
    assert res.examined == 0
    assert res.tombstoned == 0
    assert res.items == []


def test_all_above_threshold_prunes_nothing(cfg: PredatorConfig) -> None:
    soil = FakeSoil(nutrients=[_make_high_utility(f"n{i}") for i in range(5)])
    res = _run(cfg, soil, FakeEconomist())
    assert res.examined == 5
    assert res.eligible == 5
    assert res.tombstoned == 0
    assert soil.invalidated == []


# ── 2. Threshold + selection ───────────────────────────────────────────────


def test_only_below_threshold_pruned(cfg: PredatorConfig) -> None:
    soil = FakeSoil(
        nutrients=[
            _make_high_utility("keep1"),
            _make_low_utility("kill1"),
            _make_high_utility("keep2"),
            _make_low_utility("kill2"),
        ]
    )
    res = _run(cfg, soil, FakeEconomist())
    assert res.examined == 4
    assert res.tombstoned == 2
    invalidated_ids = {nid for nid, _, _ in soil.invalidated}
    assert invalidated_ids == {"kill1", "kill2"}


def test_candidates_processed_worst_first(cfg: PredatorConfig) -> None:
    """When max_delete_per_run < candidates, the lowest-utility ones go first."""
    soil = FakeSoil(
        nutrients=[
            FakeNutrient(
                "worst",
                reinforcement_count=0,
                lapse_count=10,
                last_reinforced=_aged(90),
                created_at=_aged(60),
                _retrievability=0.0,
            ),
            FakeNutrient(
                "middle",
                reinforcement_count=0,
                lapse_count=1,
                last_reinforced=_aged(60),
                created_at=_aged(60),
                _retrievability=0.05,
            ),
            FakeNutrient(
                "nearmiss",
                reinforcement_count=1,
                lapse_count=0,
                last_reinforced=_aged(40),
                created_at=_aged(60),
                _retrievability=0.10,
            ),
        ]
    )
    cfg.max_delete_per_run = 2
    res = _run(cfg, soil, FakeEconomist())
    assert res.tombstoned == 2
    # The two lowest-utility (worst, middle) should be invalidated, not nearmiss.
    invalidated_ids = {nid for nid, _, _ in soil.invalidated}
    assert "worst" in invalidated_ids
    assert "middle" in invalidated_ids
    assert "nearmiss" not in invalidated_ids


# ── 3. Safety rails ────────────────────────────────────────────────────────


def test_min_age_days_respected(cfg: PredatorConfig) -> None:
    """Young low-utility nutrients are never tombstoned."""
    soil = FakeSoil(
        nutrients=[
            _make_low_utility("young", days_old=2),  # under default 7
            _make_low_utility("old", days_old=30),
        ]
    )
    res = _run(cfg, soil, FakeEconomist())
    assert res.examined == 2
    assert res.eligible == 1  # only "old" was eligible
    assert res.tombstoned == 1
    invalidated_ids = {nid for nid, _, _ in soil.invalidated}
    assert invalidated_ids == {"old"}


def test_covenant_nutrients_never_pruned(cfg: PredatorConfig) -> None:
    """Even a 0-utility covenant must survive — covenants are immutable."""
    soil = FakeSoil(
        nutrients=[
            _make_low_utility("c1", type_="covenant"),
            _make_low_utility("p1", type_="pattern"),
        ]
    )
    res = _run(cfg, soil, FakeEconomist())
    assert res.tombstoned == 1
    invalidated_ids = {nid for nid, _, _ in soil.invalidated}
    assert "c1" not in invalidated_ids
    assert "p1" in invalidated_ids


def test_max_delete_per_run_caps_tombstones(cfg: PredatorConfig) -> None:
    soil = FakeSoil(nutrients=[_make_low_utility(f"k{i}") for i in range(20)])
    cfg.max_delete_per_run = 5
    res = _run(cfg, soil, FakeEconomist())
    assert res.tombstoned == 5
    assert res.skipped_per_run_cap == 15
    assert len(soil.invalidated) == 5


def test_collections_filter_respected(cfg: PredatorConfig) -> None:
    """Limiting to 'antipattern' must skip 'pattern' nutrients entirely."""
    soil = FakeSoil(
        nutrients=[
            _make_low_utility("p1", type_="pattern"),
            _make_low_utility("a1", type_="antipattern"),
        ]
    )
    cfg.collections = ("antipattern",)
    res = _run(cfg, soil, FakeEconomist())
    assert res.tombstoned == 1
    invalidated_ids = {nid for nid, _, _ in soil.invalidated}
    assert invalidated_ids == {"a1"}


# ── 4. First-run safety cap ────────────────────────────────────────────────


def test_first_run_caps_at_ten_when_state_absent(tmp_path: Path) -> None:
    """No state file => effective cap clamps to 10 even if config says 50."""
    cfg = PredatorConfig(
        state_path=tmp_path / "predator_state.json",  # does NOT exist yet
        audit_path=tmp_path / "audit.jsonl",
        weights_path=tmp_path / "weights.json",
        max_delete_per_run=50,  # would normally allow 30
    )
    soil = FakeSoil(nutrients=[_make_low_utility(f"k{i}") for i in range(30)])
    res = _run(cfg, soil, FakeEconomist())
    assert res.first_run is True
    assert res.effective_cap == FIRST_RUN_HARD_CAP
    assert res.tombstoned == FIRST_RUN_HARD_CAP
    assert res.skipped_first_run_cap == 20


def test_confirm_first_run_bypasses_safety_cap(tmp_path: Path) -> None:
    cfg = PredatorConfig(
        state_path=tmp_path / "predator_state.json",
        audit_path=tmp_path / "audit.jsonl",
        weights_path=tmp_path / "weights.json",
        max_delete_per_run=50,
        confirm_first_run=True,
    )
    soil = FakeSoil(nutrients=[_make_low_utility(f"k{i}") for i in range(15)])
    res = _run(cfg, soil, FakeEconomist())
    assert res.first_run is True
    assert res.effective_cap == 50
    assert res.tombstoned == 15  # all of them


def test_state_file_is_written_after_real_run(tmp_path: Path) -> None:
    """A non-dry-run leaves a state file behind so the next call isn't first-run."""
    state = tmp_path / "predator_state.json"
    cfg = PredatorConfig(
        state_path=state,
        audit_path=tmp_path / "audit.jsonl",
        weights_path=tmp_path / "weights.json",
        confirm_first_run=True,
    )
    _run(cfg, FakeSoil(nutrients=[_make_low_utility("k1")]), FakeEconomist())
    assert state.exists()
    state2 = json.loads(state.read_text())
    assert state2["last_examined"] >= 1


def test_dry_run_does_not_write_state_file(tmp_path: Path) -> None:
    state = tmp_path / "predator_state.json"
    cfg = PredatorConfig(
        state_path=state,
        audit_path=tmp_path / "audit.jsonl",
        weights_path=tmp_path / "weights.json",
        dry_run=True,
    )
    _run(cfg, FakeSoil(nutrients=[_make_low_utility("k1")]), FakeEconomist())
    # First-run cap still applies, but no state written so it stays first-run.
    assert not state.exists()


# ── 5. Dry-run reporting ───────────────────────────────────────────────────


def test_dry_run_reports_but_does_not_invalidate(cfg: PredatorConfig) -> None:
    cfg.dry_run = True
    soil = FakeSoil(nutrients=[_make_low_utility(f"k{i}") for i in range(3)])
    res = _run(cfg, soil, FakeEconomist())
    assert res.dry_run is True
    assert len(res.items) == 3
    assert all(item["applied"] is False for item in res.items)
    assert soil.invalidated == []


# ── 6. Audit log ───────────────────────────────────────────────────────────


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_audit_writes_one_line_per_candidate_plus_summary(cfg: PredatorConfig) -> None:
    soil = FakeSoil(nutrients=[_make_low_utility(f"k{i}") for i in range(3)])
    _run(cfg, soil, FakeEconomist())
    events = _read_audit(cfg.audit_path)
    types = [e["event"] for e in events]
    assert types.count("tombstone_candidate") == 3
    assert "run_summary" in types


def test_audit_records_economist_rejection(cfg: PredatorConfig) -> None:
    soil = FakeSoil(nutrients=[_make_low_utility("k1")])
    _run(cfg, soil, FakeEconomist(approve=False))
    events = _read_audit(cfg.audit_path)
    assert any(e["event"] == "rejected_by_economist" for e in events)


# ── 7. Economist contract ──────────────────────────────────────────────────


def test_quote_called_before_work(cfg: PredatorConfig) -> None:
    econ = FakeEconomist()
    _run(cfg, FakeSoil(), econ)
    assert econ.quotes == [("predator.run", 0.0)]


def test_commit_happens_when_not_dry_run(cfg: PredatorConfig) -> None:
    econ = FakeEconomist()
    _run(cfg, FakeSoil(nutrients=[_make_low_utility("k")]), econ)
    assert econ.commits == [("predator.run", 0.0)]


def test_commit_skipped_in_dry_run(cfg: PredatorConfig) -> None:
    cfg.dry_run = True
    econ = FakeEconomist()
    _run(cfg, FakeSoil(nutrients=[_make_low_utility("k")]), econ)
    assert econ.commits == []  # quote yes, commit no


def test_economist_rejection_short_circuits_work(cfg: PredatorConfig) -> None:
    soil = FakeSoil(nutrients=[_make_low_utility(f"k{i}") for i in range(5)])
    res = _run(cfg, soil, FakeEconomist(approve=False))
    assert res.economist_approved is False
    assert res.examined == 0
    assert soil.invalidated == []


# ── 8. Weights + utility integration ───────────────────────────────────────


def test_custom_weights_change_pruning_decisions(tmp_path: Path) -> None:
    """Crank failure weight up so even one lapse pushes utility negative."""
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "usage": 0.1,
                "retrievability": 0.1,
                "recency": 0.1,
                "failure": 5.0,
            }
        )
    )
    cfg = PredatorConfig(
        state_path=tmp_path / "predator_state.json",
        audit_path=tmp_path / "audit.jsonl",
        weights_path=weights_path,
        confirm_first_run=True,
    )
    _seeded_state(cfg.state_path)
    soil = FakeSoil(
        nutrients=[
            FakeNutrient(
                "with_lapse",
                reinforcement_count=10,
                lapse_count=1,
                last_reinforced=_aged(1),
                created_at=_aged(60),
                _retrievability=0.9,
            ),
        ]
    )
    res = _run(cfg, soil, FakeEconomist())
    # Default weights would keep it (high retrievability dominates); custom
    # weights make failure penalty crush it.
    assert res.tombstoned == 1


def test_malformed_weights_fall_back_to_defaults(tmp_path: Path, cfg: PredatorConfig) -> None:
    cfg.weights_path.write_text("{ not valid json")
    # Should not raise, should still run with defaults.
    res = _run(cfg, FakeSoil(nutrients=[_make_high_utility("h")]), FakeEconomist())
    assert res.tombstoned == 0


# ── 9. Misc ────────────────────────────────────────────────────────────────


def test_invalidate_failure_does_not_crash_run(cfg: PredatorConfig) -> None:
    soil = FakeSoil(
        nutrients=[_make_low_utility("ok"), _make_low_utility("fail")],
        raise_on_invalidate={"fail"},
    )
    res = _run(cfg, soil, FakeEconomist())
    # 'fail' counted as a candidate but not tombstoned; 'ok' succeeded.
    assert res.tombstoned == 1
    invalidated_ids = {nid for nid, _, _ in soil.invalidated}
    assert invalidated_ids == {"ok"}
    # Both still appear in items, with applied flags reflecting outcome.
    by_id = {it["nutrient_id"]: it for it in res.items}
    assert by_id["ok"]["applied"] is True
    assert by_id["fail"]["applied"] is False


def test_result_round_trips_through_json(cfg: PredatorConfig) -> None:
    res = _run(cfg, FakeSoil(nutrients=[_make_low_utility("k")]), FakeEconomist())
    encoded = json.dumps(result_to_dict(res))
    decoded = json.loads(encoded)
    assert decoded["tombstoned"] == res.tombstoned


def test_cli_format_result_smoke(cfg: PredatorConfig) -> None:
    soil = FakeSoil(nutrients=[_make_low_utility("k1"), _make_low_utility("k2")])
    res = _run(cfg, soil, FakeEconomist())
    text = cli_format_result(res)
    assert "Predator" in text
    assert "tombstoned 2" in text
    assert "k1" in text or "k2" in text


def test_default_threshold_pinned() -> None:
    # Spec §3.1 + §6 — pinned for downstream consumers.
    assert DEFAULT_UTILITY_THRESHOLD == 0.15
    assert FIRST_RUN_HARD_CAP == 10
