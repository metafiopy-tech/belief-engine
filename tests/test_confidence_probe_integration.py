"""Integration tests for the confidence probe — Session 8.5c.

The probe returns P(build-success) and maps it to one of three routing
decisions: proceed, escalate, abort.  The audit gap was: we didn't
have tests proving that a *genuinely low* confidence actually routes
to ``abort`` / ``escalate``, not just "proceed by default because the
probe returned 1.0 when untrained."

These tests install a fake model into the probe and verify the
(confidence → decision) mapping end-to-end, including:

* Untrained probe defaults to 1.0 → proceed.
* Low-confidence prediction routes to abort at the threshold boundary.
* Mid-confidence routes to escalate.
* High-confidence routes to proceed.
* predict_confidence failure returns 1.0 (fail-open — the probe is
  advisory, not authoritative).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from belief.safety.confidence_probe import (
    THRESH_ESCALATE,
    THRESH_PROCEED,
    ConfidenceProbe,
)


# ---------------------------------------------------------------------------
# Fake sklearn-shaped model (two-class predict_proba)
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for a sklearn ``CalibratedClassifierCV``.

    Returns a fixed ``[[1-p, p]]`` for every input — enough to let the
    probe's ``predict_proba`` path run without any real sklearn dep.
    """

    def __init__(self, success_prob: float) -> None:
        self._p = float(success_prob)

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        # Must return shape (n_samples, 2).  We ignore X and return
        # one row per input row.
        return [[1.0 - self._p, self._p] for _ in X]


def _make_probe_with_fake(tmp_path: Path, p_success: float) -> ConfidenceProbe:
    """Build a ConfidenceProbe that returns ``p_success`` for every input."""
    probe = ConfidenceProbe(model_path=tmp_path / "probe.pkl")
    probe.model = _FakeModel(p_success)
    return probe


# Representative feature dict — the probe's _coerce_features path
# projects this down to the feature vector.
_FEATURES = {
    "tokens_used": 5000,
    "budget_remaining": 5.0,
    "covenant_fires": 0,
    "step_index": 3,
}


# ---------------------------------------------------------------------------
# Untrained probe (model is None) — fail-open
# ---------------------------------------------------------------------------


class TestUntrainedProbe:
    def test_untrained_returns_one_point_zero(self, tmp_path: Path) -> None:
        probe = ConfidenceProbe(model_path=tmp_path / "probe.pkl")
        assert probe.model is None
        conf = probe.predict_confidence(_FEATURES)
        assert conf == 1.0

    def test_untrained_routes_to_proceed(self, tmp_path: Path) -> None:
        probe = ConfidenceProbe(model_path=tmp_path / "probe.pkl")
        decision = probe.should_escalate(probe.predict_confidence(_FEATURES))
        assert decision == "proceed"


# ---------------------------------------------------------------------------
# Threshold → decision mapping
# ---------------------------------------------------------------------------


class TestDecisionMapping:
    def test_high_confidence_proceeds(self, tmp_path: Path) -> None:
        probe = _make_probe_with_fake(tmp_path, p_success=0.95)
        conf = probe.predict_confidence(_FEATURES)
        assert conf > THRESH_PROCEED
        assert probe.should_escalate(conf) == "proceed"

    def test_mid_confidence_escalates(self, tmp_path: Path) -> None:
        probe = _make_probe_with_fake(tmp_path, p_success=0.55)
        conf = probe.predict_confidence(_FEATURES)
        assert THRESH_ESCALATE < conf <= THRESH_PROCEED
        assert probe.should_escalate(conf) == "escalate"

    def test_low_confidence_aborts(self, tmp_path: Path) -> None:
        probe = _make_probe_with_fake(tmp_path, p_success=0.2)
        conf = probe.predict_confidence(_FEATURES)
        assert conf < THRESH_ESCALATE
        assert probe.should_escalate(conf) == "abort"

    def test_exactly_at_proceed_threshold_escalates(self) -> None:
        """>0.8 proceeds.  Exactly 0.8 is NOT > 0.8, so it escalates."""
        probe = ConfidenceProbe(model_path=Path("/tmp/_ignored"))
        assert probe.should_escalate(THRESH_PROCEED) == "escalate"

    def test_exactly_at_escalate_threshold_aborts(self) -> None:
        """>0.4 escalates.  Exactly 0.4 is NOT > 0.4, so it aborts."""
        probe = ConfidenceProbe(model_path=Path("/tmp/_ignored"))
        assert probe.should_escalate(THRESH_ESCALATE) == "abort"


# ---------------------------------------------------------------------------
# Prediction-path failure recovery
# ---------------------------------------------------------------------------


class TestPredictionResilience:
    def test_model_raise_returns_one_point_zero(self, tmp_path: Path) -> None:
        """A broken model (predict_proba raises) must not propagate —
        the probe is advisory, and a crash here would block every
        build.  Fail-open to 1.0 (= proceed)."""

        class _BrokenModel:
            def predict_proba(self, X: Any) -> Any:
                raise RuntimeError("synthetic model failure")

        probe = ConfidenceProbe(model_path=tmp_path / "probe.pkl")
        probe.model = _BrokenModel()
        conf = probe.predict_confidence(_FEATURES)
        assert conf == 1.0
        assert probe.should_escalate(conf) == "proceed"

    def test_single_column_proba_returns_column_value(self, tmp_path: Path) -> None:
        """Edge case: some sklearn calibrators emit a single-column
        proba (one class only).  We should return that value verbatim
        instead of crashing.  Documented pickle-format variant."""

        class _OneClassModel:
            def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
                return [[0.42] for _ in X]

        probe = ConfidenceProbe(model_path=tmp_path / "probe.pkl")
        probe.model = _OneClassModel()
        conf = probe.predict_confidence(_FEATURES)
        assert conf == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Integration: low confidence → aborts a simulated build flow
# ---------------------------------------------------------------------------


class TestLowConfidenceRerouting:
    def test_low_confidence_aborts_simulated_pipeline(self, tmp_path: Path) -> None:
        """Simulate the pipeline step: extract features → predict →
        check routing.  A low-confidence result must produce 'abort',
        which in the real pipeline triggers the cloud-escalation
        pathway.  This test is the contract-level integration point."""
        probe = _make_probe_with_fake(tmp_path, p_success=0.1)

        # Simulated "pipeline" — in prod this is the gap-analyst /
        # debugger handoff; we stand in with just the probe call.
        def simulated_router(features: dict) -> str:
            conf = probe.predict_confidence(features)
            return probe.should_escalate(conf)

        assert simulated_router(_FEATURES) == "abort"

    def test_mid_confidence_escalates_simulated_pipeline(self, tmp_path: Path) -> None:
        probe = _make_probe_with_fake(tmp_path, p_success=0.5)

        def simulated_router(features: dict) -> str:
            return probe.should_escalate(probe.predict_confidence(features))

        assert simulated_router(_FEATURES) == "escalate"

    def test_routing_is_stable_across_calls(self, tmp_path: Path) -> None:
        """Same input → same decision — the probe must be pure when
        the model is deterministic."""
        probe = _make_probe_with_fake(tmp_path, p_success=0.3)
        d1 = probe.should_escalate(probe.predict_confidence(_FEATURES))
        d2 = probe.should_escalate(probe.predict_confidence(_FEATURES))
        d3 = probe.should_escalate(probe.predict_confidence(_FEATURES))
        assert d1 == d2 == d3 == "abort"
