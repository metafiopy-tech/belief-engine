"""Tests for succession modes (mycorrhizal Stage 7, Area 8)."""

from __future__ import annotations

import json
from pathlib import Path

from belief.lifecycle.succession import (
    MATURE_FLOOR,
    PIONEER_CEILING,
    SuccessionManager,
    SuccessionMode,
    consolidate_humus,
    policy_for,
)


class _FakeSoil:
    def __init__(self, n: int) -> None:
        self._n = n

    def count(self) -> int:
        return self._n


class _FakeHubs:
    def __init__(self, hubs: list[str]) -> None:
        self._hubs = hubs

    def current_hubs(self) -> list[str]:
        return self._hubs


class _FakeLedger:
    """Returns a controllable exchange-rate distribution."""

    def __init__(self, rates: list[float]) -> None:
        self._rates = rates

    def rank_agents(self, window: str = "30d"):
        from types import SimpleNamespace

        return [
            SimpleNamespace(agent_id=f"a{i}", exchange_rate=r) for i, r in enumerate(self._rates)
        ]


# ── Mode computation ─────────────────────────────────────────────────────────


def test_pioneer_when_soil_sparse(tmp_path: Path) -> None:
    mgr = SuccessionManager(
        soil=_FakeSoil(500),
        state_path=tmp_path / "s.json",
    )
    assert mgr.current_mode() is SuccessionMode.PIONEER


def test_mid_when_soil_grows(tmp_path: Path) -> None:
    mgr = SuccessionManager(
        soil=_FakeSoil(5_000),
        hub_registry=_FakeHubs(["h1"]),
        state_path=tmp_path / "s.json",
    )
    assert mgr.current_mode() is SuccessionMode.MID


def test_mid_when_dense_but_hubs_unstable(tmp_path: Path) -> None:
    """Even above the MATURE nutrient floor, no stable hubs → still MID."""
    mgr = SuccessionManager(
        soil=_FakeSoil(50_000),
        hub_registry=_FakeHubs([]),  # no hubs
        reciprocity_ledger=_FakeLedger([0.5, 0.51]),
        state_path=tmp_path / "s.json",
    )
    assert mgr.current_mode() is SuccessionMode.MID


def test_mature_when_all_conditions_met(tmp_path: Path) -> None:
    mgr = SuccessionManager(
        soil=_FakeSoil(50_000),
        hub_registry=_FakeHubs(["h1", "h2"]),
        reciprocity_ledger=_FakeLedger([0.50, 0.51, 0.50]),  # low variance
        state_path=tmp_path / "s.json",
    )
    assert mgr.current_mode() is SuccessionMode.MATURE


def test_mature_blocked_by_high_variance(tmp_path: Path) -> None:
    mgr = SuccessionManager(
        soil=_FakeSoil(50_000),
        hub_registry=_FakeHubs(["h1"]),
        reciprocity_ledger=_FakeLedger([0.1, 5.0, 0.2]),  # high variance
        state_path=tmp_path / "s.json",
    )
    assert mgr.current_mode() is SuccessionMode.MID


def test_thresholds_sane() -> None:
    assert PIONEER_CEILING < MATURE_FLOOR


# ── Policy ───────────────────────────────────────────────────────────────────


def test_policy_strictens_with_maturity() -> None:
    p = policy_for(SuccessionMode.PIONEER)
    m = policy_for(SuccessionMode.MID)
    mat = policy_for(SuccessionMode.MATURE)
    # Onboarding bar rises.
    assert p.onboarding_min_value <= m.onboarding_min_value <= mat.onboarding_min_value
    # Sanctions firmer.
    assert p.sanction_strength < mat.sanction_strength
    # Decomposition more selective.
    assert p.decomposition_aggressiveness > mat.decomposition_aggressiveness
    # Consolidation only in mature.
    assert p.consolidation_active is False
    assert mat.consolidation_active is True


# ── Transition logging ───────────────────────────────────────────────────────


def test_recompute_logs_transition(tmp_path: Path) -> None:
    state_path = tmp_path / "s.json"
    mgr = SuccessionManager(soil=_FakeSoil(500), state_path=state_path)
    status = mgr.recompute_and_log()
    assert status.mode is SuccessionMode.PIONEER
    # State persisted.
    with open(state_path) as f:
        assert json.load(f)["mode"] == "pioneer"


def test_default_no_deps_is_pioneer(tmp_path: Path) -> None:
    """With no soil/hubs/ledger wired, the engine is PIONEER (count 0) and
    can never accidentally reach MATURE."""
    mgr = SuccessionManager(state_path=tmp_path / "s.json")
    assert mgr.current_mode() is SuccessionMode.PIONEER


# ── Consolidation ────────────────────────────────────────────────────────────


def test_consolidate_humus_produces_manifest(tmp_path: Path) -> None:
    """With no real soil/tools, consolidation still emits a well-formed
    (empty) humus manifest — the mature-mode export is best-effort."""
    out = tmp_path / "humus.json"
    manifest = consolidate_humus(soil=None, out_path=out)
    assert out.exists()
    assert manifest["exported_count"] == 0
    assert "humus_version" in manifest
    with open(out) as f:
        loaded = json.load(f)
    assert loaded["exported_count"] == 0
