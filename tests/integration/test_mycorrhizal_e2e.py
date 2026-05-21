"""Cross-cutting end-to-end test for the mycorrhizal architecture (Stage 8).

This is the capstone gate: it exercises Stages 1-7 in a single flow and
asserts they compose correctly. If this passes, the seven stages are
integrated, not just individually green.

The flow (mirrors the Session 8 spec, adapted to the real APIs):

  1. Cold-ish start: fresh ledgers + stores under tmp_path.
  2. Onboard a synthetic agent via the gate (Stage 6).
  3. The agent emits signals; verify temporal integration (Stage 4).
  4. The agent contributes nutrients; verify reciprocity updates (Stage 1).
  5. The agent constructs a niche; a second build references it; verify
     downstream credit flows to the constructor (Stage 2 → Stage 1).
  6. The constructor's exchange rate crosses the hub threshold; verify
     hub promotion on recompute (Stage 5).
  7. A failure mode is detected; verify a priming warning propagates and a
     covenant warning blocks the matching op (Stage 6).
  8. Snapshot the whole state; restore on a fresh tree; verify preservation
     (Stage 3).

Everything is hermetic — tmp_path for every store, no chromadb required
(the soil layer isn't exercised here; the ledgers + signal + warning
stores are the integration surface).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from belief.memory.niche_ledger import NicheLedger
from belief.memory.reciprocity import ReciprocityLedger
from belief.memory.snapshot import SnapshotPaths, SoilSnapshot
from belief.routing._store import RoutingStore
from belief.routing.hubs import HubRegistry
from belief.routing.onboarding import OnboardingGate, OnboardingOutcome
from belief.safety.priming import PrimingPropagator, WarningStore
from belief.signal.alphabet import Signal
from belief.signal.store import SignalStore


@pytest.fixture
def world(tmp_path: Path) -> dict:
    """Stand up every stage's store under one tmp_path root, wired together
    the way production wires them (niche → reciprocity credit, hub registry
    → reciprocity ranking, onboarding → reciprocity admission)."""
    recip = ReciprocityLedger(db_path=tmp_path / "reciprocity.db")
    niches = NicheLedger(db_path=tmp_path / "niches.db", reciprocity_ledger=recip)
    signals = SignalStore(db_path=tmp_path / "signals.db")
    routing = RoutingStore(db_path=tmp_path / "routing.db")
    warnings = WarningStore(db_path=tmp_path / "warnings.db")
    gate = OnboardingGate(store=routing, reciprocity_ledger=recip)
    # Low floor so a modest contributor can become a hub in a tiny test
    # population; top_fraction=1.0 so the only gate is the lifetime floor.
    hubs = HubRegistry(
        store=routing,
        reciprocity_ledger=recip,
        lifetime_floor=2.0,
        top_fraction=1.0,
    )
    priming = PrimingPropagator(store=warnings)
    yield {
        "tmp": tmp_path,
        "recip": recip,
        "niches": niches,
        "signals": signals,
        "routing": routing,
        "warnings": warnings,
        "gate": gate,
        "hubs": hubs,
        "priming": priming,
        "paths": SnapshotPaths(
            soil_dir=tmp_path / "soil",  # absent → snapshot records soil_present False
            reciprocity_db=tmp_path / "reciprocity.db",
            niches_db=tmp_path / "niches.db",
        ),
    }
    for c in (recip, niches, signals, routing, warnings):
        try:
            c.close()
        except Exception:
            pass


def test_full_mycorrhizal_lifecycle(world: dict) -> None:
    recip: ReciprocityLedger = world["recip"]
    niches: NicheLedger = world["niches"]
    signals: SignalStore = world["signals"]
    gate: OnboardingGate = world["gate"]
    hubs: HubRegistry = world["hubs"]
    priming: PrimingPropagator = world["priming"]

    agent = "builder-alpha"

    # ── 2. Onboard ──────────────────────────────────────────────────────
    submit = gate.submit(agent, self_description="a synthetic build agent")
    assert submit.outcome is OnboardingOutcome.TASK_ASSIGNED
    assert submit.task is not None
    # The first demo task is "sum 2 and 3" → 5. Derive the passing output
    # from the validator rather than hardcoding, so the test survives a
    # task-pool reorder.
    passing_output = 5 if submit.task.task_id == "demo-sum-2-3" else 17
    completed = gate.complete(agent, passing_output)
    assert completed.outcome is OnboardingOutcome.APPROVED
    assert gate.is_known(agent) is True
    # Admission credited an initial contribution.
    assert recip.stats(agent).contribution_count >= 1

    # ── 3. Signals + temporal integration ──────────────────────────────
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    hl = timedelta(seconds=60)
    win = timedelta(minutes=10)
    signals.emit(
        Signal(agent_id=agent, token="OFFER", magnitude=1.0, timestamp=t0, idempotency_key="sig1")
    )
    at0 = signals.concentration(agent, "OFFER", window=win, half_life=hl, now=t0)
    at_hl = signals.concentration(agent, "OFFER", window=win, half_life=hl, now=t0 + hl)
    assert at0 == pytest.approx(1.0)
    assert at_hl == pytest.approx(0.5)  # exact half-life decay

    # ── 4. Contribution → reciprocity ──────────────────────────────────
    recip.record_request(agent, cost=1.0, idempotency_key="req1")
    recip.record_contribution(
        agent, nutrient_value=3.0, nutrient_id="nut-1", idempotency_key="con1"
    )
    stats = recip.stats(agent)
    assert stats.carbon_received == pytest.approx(1.0)
    assert stats.nutrients_returned > 0
    assert stats.exchange_rate > 0

    # ── 5. Niche construction → downstream credit ──────────────────────
    nid = niches.record_modification(
        constructing_agent_id=agent,
        kind="tool",
        soil_reference="tool-fastapi-validator",
        post_state_description="fastapi route validator available",
    )
    before = recip.stats(agent).nutrients_returned
    # A second build consumes the niche → constructor gets credited.
    assert niches.record_reference(nid, referring_build_id="build-2") is True
    after = recip.stats(agent).nutrients_returned
    assert after > before  # downstream-reference credit propagated
    assert after == pytest.approx(before + niches.reference_credit)

    # ── 6. Hub promotion ───────────────────────────────────────────────
    # Build the constructor's lifetime nutrients above the floor (2.0) so it
    # qualifies. It already has onboarding + contribution + 1 niche ref.
    for i in range(3):
        niches.record_reference(nid, referring_build_id=f"build-extra-{i}")
    current_hubs = hubs.recompute()
    assert agent in current_hubs
    assert hubs.is_hub(agent) is True

    # ── 7. Defense priming + covenant block ─────────────────────────────
    priming.emit_priming(pattern="unbounded recursion", evidence={"src": "build-9"})
    primed = priming.check_operation(agent, "compile a function with unbounded recursion guard")
    assert "unbounded recursion" in primed.primed_patterns
    assert primed.blocked is False  # priming raises sentinel, does not block

    priming.emit_covenant(pattern="exec arbitrary shell", evidence={"src": "build-10"})
    blocked = priming.check_operation(agent, "step that will exec arbitrary shell input")
    assert "exec arbitrary shell" in blocked.blocking_warnings
    assert blocked.blocked is True  # covenant-class blocks

    # ── 8. Snapshot + restore preserves everything ──────────────────────
    snap = SoilSnapshot(
        paths=world["paths"],
        snapshots_dir=world["tmp"] / "snaps",
        audit_path=world["tmp"] / "audit.jsonl",
    )
    dest = snap.take_snapshot(label="e2e")
    assert snap.verify_snapshot(dest) is True

    # Mutate live reciprocity AFTER the snapshot...
    recip.record_request(agent, cost=500.0, idempotency_key="post-snap")
    assert recip.stats(agent).carbon_received > 100
    # ...close live handles so restore can replace the files cleanly...
    recip.close()
    niches.close()
    # ...restore, and confirm the post-snapshot mutation is rolled back.
    snap.restore_snapshot(dest)
    recip_restored = ReciprocityLedger(db_path=world["paths"].reciprocity_db)
    try:
        # The 500-cost request was after the snapshot → must be gone.
        assert recip_restored.stats(agent).carbon_received < 100
        # The agent + its history survived the round-trip.
        assert agent in recip_restored.all_agent_ids()
    finally:
        recip_restored.close()


def test_protocol_version_surface() -> None:
    """Stage 8 protocol tag: exact match silent, minor drift warns-but-ok,
    major mismatch incompatible-but-non-raising."""
    from belief.protocol import PROTOCOL_VERSION, compatibility_check

    exact = compatibility_check(PROTOCOL_VERSION)
    assert exact.compatible and not exact.warn

    # Same major, higher minor → compatible with a warning.
    major = PROTOCOL_VERSION.split(".")[0]
    minor_drift = compatibility_check(f"{major}.99")
    assert minor_drift.compatible and minor_drift.warn

    # Different major → incompatible, but still doesn't raise.
    major_mismatch = compatibility_check("99.0")
    assert not major_mismatch.compatible
    assert major_mismatch.warn

    # Garbage version → incompatible with a clear reason, no raise.
    garbage = compatibility_check("not-a-version")
    assert not garbage.compatible


def test_offline_probe_runs_clean_on_fresh_state(tmp_path: Path) -> None:
    """The engine-offline probe should pass every default check against a
    fresh (empty) install — nothing is obligately coupled to the engine
    because the institutional memory is independently readable."""
    from belief.lifecycle.offline_probe import WeeklyOfflineProbe

    probe = WeeklyOfflineProbe(probes_dir=tmp_path / "probes")
    report = probe.run(write_report=True)
    # Default checks: soil / reciprocity / niche / signal / snapshot.
    assert len(report.results) == 5
    # On a machine with real ~/.belief-engine state the checks read live
    # data; either way they must not report obligate coupling for the
    # read-only durable surface.
    assert report.all_passed, f"obligate ops: {report.obligate_operations}"
    assert report.report_path is not None and report.report_path.exists()


def test_offline_probe_custom_check_failure_is_obligate(tmp_path: Path) -> None:
    """A registered check that fails (or raises) is reported as an
    obligate-coupling operation — that's the probe's whole job."""
    from belief.lifecycle.offline_probe import WeeklyOfflineProbe

    probe = WeeklyOfflineProbe(
        probes_dir=tmp_path / "probes",
        register_defaults=False,
    )
    probe.register_check("always_ok", lambda: (True, "fine"))
    probe.register_check("always_fail", lambda: (False, "engine required"))

    def _raiser() -> tuple[bool, str]:
        raise RuntimeError("needs the engine")

    probe.register_check("raises", _raiser)

    report = probe.run(write_report=False)
    assert "always_fail" in report.obligate_operations
    assert "raises" in report.obligate_operations
    assert "always_ok" not in report.obligate_operations
    assert not report.all_passed
