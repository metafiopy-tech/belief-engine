"""Tests for the cross-domain novelty gate (SE Session 6)."""

from __future__ import annotations

import json

from belief.photosynthesis.synthesis import prompts_cross_domain as prompts
from belief.photosynthesis.synthesis.cross_domain_novelty import (
    DEFAULT_NOVELTY_THRESHOLD,
    NoveltyVerdict,
    gate,
)
from belief.photosynthesis.synthesis.structural_mechanism import StructuralMechanism


def _few_shot(name: str) -> StructuralMechanism:
    txt = getattr(prompts, name).replace("{{", "{").replace("}}", "}")
    return StructuralMechanism.model_validate(json.loads(txt))


# ---------------------------------------------------------------------------
# Stub bio_store
# ---------------------------------------------------------------------------


class _FakeBioStore:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def novelty_score(self, mechanism) -> float:
        self.calls += 1
        return self.score


class _FailingBioStore:
    def novelty_score(self, mechanism) -> float:
        raise RuntimeError("simulated chromadb failure")


# ---------------------------------------------------------------------------
# Verdict construction
# ---------------------------------------------------------------------------


class TestNoveltyVerdict:
    def test_accepted_when_score_at_threshold(self) -> None:
        v = NoveltyVerdict(
            accepted=True,
            novelty_score=0.30,
            threshold=0.30,
            reason="novel",
        )
        assert v.accepted is True
        assert v.rejected is False

    def test_rejected_property_inverts(self) -> None:
        v = NoveltyVerdict(
            accepted=False,
            novelty_score=0.0,
            threshold=0.30,
            reason="cross_domain_redundant",
        )
        assert v.rejected is True

    def test_to_dict_round_trip(self) -> None:
        v = NoveltyVerdict(
            accepted=True,
            novelty_score=0.85,
            threshold=0.30,
            reason="novel",
        )
        d = v.to_dict()
        assert d["accepted"] is True
        assert d["novelty_score"] == 0.85
        assert d["threshold"] == 0.30


# ---------------------------------------------------------------------------
# gate() behavior
# ---------------------------------------------------------------------------


class TestGate:
    def test_default_threshold_is_0_30(self) -> None:
        assert DEFAULT_NOVELTY_THRESHOLD == 0.30

    def test_accepts_above_threshold(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        store = _FakeBioStore(score=0.95)
        v = gate(mech, bio_store=store)
        assert v.accepted is True
        assert v.novelty_score == 0.95
        assert v.reason == "novel"
        assert store.calls == 1

    def test_rejects_below_threshold(self) -> None:
        """SE acceptance: 'Novelty gate aborts a hand-constructed
        redundant mechanism.'"""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        store = _FakeBioStore(score=0.05)
        v = gate(mech, bio_store=store)
        assert v.accepted is False
        assert v.rejected is True
        assert v.reason == "cross_domain_redundant"

    def test_accepts_exactly_at_threshold(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        store = _FakeBioStore(score=0.30)
        v = gate(mech, bio_store=store)
        assert v.accepted is True

    def test_custom_threshold_applies(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        store = _FakeBioStore(score=0.50)
        # Below custom threshold -> reject even though it's above
        # the default.
        v = gate(mech, bio_store=store, threshold=0.75)
        assert v.accepted is False
        # Above custom threshold -> accept
        v2 = gate(mech, bio_store=store, threshold=0.40)
        assert v2.accepted is True

    def test_none_bio_store_accepts_by_default(self) -> None:
        """If the bio_store isn't wired (e.g. chromadb missing), the
        gate must not block synthesis. Accept with a tagged reason."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        v = gate(mech, bio_store=None)
        assert v.accepted is True
        assert v.reason == "bio_store_unavailable"

    def test_bio_store_failure_accepts_by_default(self) -> None:
        """A broken bio_store shouldn't poison synthesis; the gate
        accepts with a tagged reason so the audit log shows what
        happened."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        v = gate(mech, bio_store=_FailingBioStore())
        assert v.accepted is True
        assert "bio_store_error" in v.reason

    def test_score_clamped_to_unit_interval(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        v_high = gate(mech, bio_store=_FakeBioStore(score=99.0))
        assert v_high.novelty_score == 1.0
        v_low = gate(mech, bio_store=_FakeBioStore(score=-5.0))
        assert v_low.novelty_score == 0.0
