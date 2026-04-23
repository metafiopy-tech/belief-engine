"""Tests for belief.safety.confidence_probe.

Every test is hermetic: uses a tmp model path and never touches
~/.belief-engine. sklearn-dependent tests skip cleanly when sklearn
isn't installed.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import pytest

from belief.safety.confidence_probe import (
    ConfidenceProbe,
    EXPECTED_TOTAL_STEPS,
    KNOWN_AGENTS,
    THRESH_ESCALATE,
    THRESH_PROCEED,
    extract_features,
    is_probe_routing_enabled,
    set_default_probe,
)


SK_AVAILABLE = True
try:
    import sklearn  # noqa: F401
except ImportError:
    SK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_one_hot_marks_known_agent(self) -> None:
        feats = extract_features(
            {
                "agent_name": "builder",
                "iteration": 1,
                "cost_so_far": 2.0,
                "output_summary": "ok",
                "step_index": 5,
            }
        )
        assert len(feats.agent_one_hot) == len(KNOWN_AGENTS)
        idx = KNOWN_AGENTS.index("builder")
        assert feats.agent_one_hot[idx] == 1
        # All other slots are zero
        assert sum(feats.agent_one_hot) == 1

    def test_unknown_agent_produces_all_zero_one_hot(self) -> None:
        feats = extract_features({"agent_name": "definitely_not_an_agent"})
        assert sum(feats.agent_one_hot) == 0

    def test_cost_ratio(self) -> None:
        feats = extract_features({"agent_name": "builder", "cost_so_far": 2.5}, max_budget=10.0)
        assert feats.cost_ratio == 0.25

    def test_error_keywords_detection_is_case_insensitive(self) -> None:
        pos = extract_features({"agent_name": "tester", "output_summary": "Traceback: Exception!"})
        assert pos.error_keywords == 1
        neg = extract_features({"agent_name": "tester", "output_summary": "all tests passed"})
        assert neg.error_keywords == 0

    def test_step_position_normalized(self) -> None:
        feats = extract_features({"agent_name": "builder", "step_index": 10})
        assert feats.step_position == pytest.approx(10 / EXPECTED_TOTAL_STEPS)

    def test_vector_length_matches_agents_plus_fixed(self) -> None:
        feats = extract_features({"agent_name": "intake"})
        # len(KNOWN_AGENTS) + 5 scalar features
        assert len(feats.to_vector()) == len(KNOWN_AGENTS) + 5


# ---------------------------------------------------------------------------
# should_escalate / predict_confidence defaults
# ---------------------------------------------------------------------------


class TestShouldEscalate:
    def test_high_confidence_proceeds(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        assert p.should_escalate(0.95) == "proceed"

    def test_mid_band_escalates(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        assert p.should_escalate(0.5) == "escalate"

    def test_low_confidence_aborts(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        assert p.should_escalate(0.2) == "abort"

    def test_boundaries(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        # > 0.8 is proceed; exactly 0.8 is escalate
        assert p.should_escalate(THRESH_PROCEED) == "escalate"
        # > 0.4 is escalate; exactly 0.4 is abort
        assert p.should_escalate(THRESH_ESCALATE) == "abort"


class TestUntrainedProbe:
    def test_predict_confidence_returns_max_when_no_model(
        self,
        tmp_path: Path,
    ) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        assert p.model is None
        conf = p.predict_confidence({"agent_name": "builder"})
        assert conf == 1.0
        assert p.should_escalate(conf) == "proceed"

    def test_evaluate_returns_untrained_marker(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        report = p.evaluate([])
        assert report["trained"] is False


# ---------------------------------------------------------------------------
# Training (sklearn-gated)
# ---------------------------------------------------------------------------


def _make_synthetic_rows(n: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """Build synthetic training rows where cost + error_keywords predict failure.

    Succeeding builds stay cheap and have clean output; failing ones
    accumulate cost and show error-ish text. The probe should learn the
    linear relationship easily.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        passed = i % 2 == 0
        build_id = f"b-{i}"
        for step in range(3):
            agent = rng.choice(["builder", "tester", "debugger", "validator"])
            rows.append(
                {
                    "build_id": build_id,
                    "step_index": step,
                    "agent_name": agent,
                    "output_summary": ("clean output" if passed else "traceback: exception raised"),
                    "edge_decision": "",
                    "cost_so_far": 0.1 if passed else 2.0,
                    "iteration": step,
                    "build_passed": passed,
                    "timestamp": 0.0,
                }
            )
    return rows


class TestTrain:
    def test_refuses_under_min_samples(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        meta = p.train(_make_synthetic_rows(5), min_samples=200)
        assert p.model is None
        assert meta.n_samples == 15  # 5 builds x 3 steps
        assert meta.min_samples_required == 200

    @pytest.mark.skipif(not SK_AVAILABLE, reason="sklearn not installed")
    def test_trains_and_saves(self, tmp_path: Path) -> None:
        out = tmp_path / "probe.pkl"
        p = ConfidenceProbe(out)
        meta = p.train(_make_synthetic_rows(80), min_samples=50)
        assert p.model is not None
        assert meta.n_samples == 240
        # File was persisted
        assert out.exists()
        payload = pickle.load(out.open("rb"))
        assert "model" in payload and "metadata" in payload

    @pytest.mark.skipif(not SK_AVAILABLE, reason="sklearn not installed")
    def test_trained_probe_predicts_reasonable(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        p.train(_make_synthetic_rows(120), min_samples=50)

        # A clearly-passing row: short output, low cost, clean text
        clean = {
            "agent_name": "builder",
            "iteration": 0,
            "cost_so_far": 0.1,
            "output_summary": "clean output",
            "step_index": 0,
        }
        # A clearly-failing row: high cost, error keywords
        dirty = {
            "agent_name": "debugger",
            "iteration": 3,
            "cost_so_far": 2.0,
            "output_summary": "traceback: exception raised",
            "step_index": 10,
        }
        clean_p = p.predict_confidence(clean)
        dirty_p = p.predict_confidence(dirty)
        assert 0.0 <= clean_p <= 1.0
        assert 0.0 <= dirty_p <= 1.0
        assert clean_p > dirty_p

    @pytest.mark.skipif(not SK_AVAILABLE, reason="sklearn not installed")
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "probe.pkl"
        p1 = ConfidenceProbe(path)
        p1.train(_make_synthetic_rows(80), min_samples=50)
        clean = {
            "agent_name": "builder",
            "iteration": 0,
            "cost_so_far": 0.1,
            "output_summary": "clean output",
            "step_index": 0,
        }
        before = p1.predict_confidence(clean)

        # Fresh instance should load the same model and produce same prediction
        p2 = ConfidenceProbe(path)
        assert p2.model is not None
        assert p2.metadata.n_samples == 240
        after = p2.predict_confidence(clean)
        assert before == pytest.approx(after)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.skipif(not SK_AVAILABLE, reason="sklearn not installed")
    def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        p = ConfidenceProbe(tmp_path / "probe.pkl")
        rows = _make_synthetic_rows(80)
        p.train(rows, min_samples=50)
        report = p.evaluate(rows)
        assert report["trained"] is True
        assert report["n"] == len(rows)
        assert 0.0 <= report["accuracy"] <= 1.0
        assert 0.0 <= report["brier"] <= 1.0
        cm = report["confusion_matrix"]
        assert len(cm) == 2 and all(len(r) == 2 for r in cm)


# ---------------------------------------------------------------------------
# Routing enablement
# ---------------------------------------------------------------------------


class TestProbeRoutingEnabled:
    def test_disabled_when_env_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("BELIEF_ENABLE_PROBE", raising=False)
        set_default_probe(ConfidenceProbe(tmp_path / "probe.pkl"))
        try:
            assert is_probe_routing_enabled() is False
        finally:
            set_default_probe(None)

    def test_disabled_when_env_on_but_no_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("BELIEF_ENABLE_PROBE", "1")
        set_default_probe(ConfidenceProbe(tmp_path / "probe.pkl"))
        try:
            # No model -> still disabled (safe default)
            assert is_probe_routing_enabled() is False
        finally:
            set_default_probe(None)

    @pytest.mark.skipif(not SK_AVAILABLE, reason="sklearn not installed")
    def test_enabled_when_env_on_and_model_trained(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("BELIEF_ENABLE_PROBE", "1")
        probe = ConfidenceProbe(tmp_path / "probe.pkl")
        probe.train(_make_synthetic_rows(80), min_samples=50)
        set_default_probe(probe)
        try:
            assert is_probe_routing_enabled() is True
        finally:
            set_default_probe(None)
