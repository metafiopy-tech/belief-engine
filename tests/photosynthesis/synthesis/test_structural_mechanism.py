"""Tests for the StructuralMechanism schema (Synthesis Engine Session 1)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from belief.photosynthesis.synthesis.structural_mechanism import (
    DomainEvidence,
    HigherOrderRelation,
    NearMiss,
    PredicateInstance,
    StructuralMechanism,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_predicate(**overrides) -> dict:
    """Return a fresh dict for a known-good predicate.

    The mantis-shrimp-meets-camera mechanism: both pre-classify a
    signal at the transducer level before any downstream processing.
    """
    base = {
        "name": "pre_classify_signal",
        "arity": 2,
        "roles": ["transducer", "signal"],
        "marr_level": "algorithmic",
    }
    base.update(overrides)
    return base


def _valid_mechanism(**overrides) -> dict:
    base = {
        "mechanism_id": "mantis-shrimp-camera-pre-classify",
        "source_domain": "mantis_shrimp",
        "target_domain": "digital_camera",
        "predicate_in_source": _valid_predicate(),
        "predicate_in_target": _valid_predicate(),
        "higher_order_relations": [
            {
                "name": "reduces_downstream_compute",
                "relates": ["pre_classify_signal", "downstream_compute"],
            }
        ],
        "near_miss": {
            "description": (
                "A bee's compound eye also has many photoreceptors but the signals "
                "are summed in the optic ganglia rather than pre-classified at the "
                "transducer."
            ),
            "breaks_at_argument": "predicate_in_source.argument[0]",
        },
        "considered_and_rejected_attributes": [
            "has_many_color_channels",
            "is_compact",
        ],
        "domain_evidence": [
            {
                "domain": "biology",
                "citation": "marshall_2007_mantis_shrimp_color_vision",
                "excerpt": "16 photoreceptor classes pre-classify at the retina.",
            },
            {
                "domain": "computing",
                "citation": "sony_imx_sensor_datasheet",
                "excerpt": "Pre-classification reduces ISP throughput requirements.",
            },
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# PredicateInstance
# ---------------------------------------------------------------------------


class TestPredicateInstance:
    def test_valid_predicate_constructs(self) -> None:
        p = PredicateInstance.model_validate(_valid_predicate())
        assert p.name == "pre_classify_signal"
        assert p.arity == 2
        assert p.roles == ["transducer", "signal"]
        assert p.marr_level == "algorithmic"

    def test_roles_length_must_match_arity(self) -> None:
        with pytest.raises(ValidationError, match="roles length"):
            PredicateInstance.model_validate(
                _valid_predicate(arity=3, roles=["transducer", "signal"])
            )

    def test_name_must_be_snake_case(self) -> None:
        with pytest.raises(ValidationError, match="snake_case"):
            PredicateInstance.model_validate(_valid_predicate(name="PreClassifySignal"))

    def test_name_must_start_with_letter(self) -> None:
        with pytest.raises(ValidationError, match="snake_case"):
            PredicateInstance.model_validate(_valid_predicate(name="2pre_classify"))

    def test_arity_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            PredicateInstance.model_validate(_valid_predicate(arity=0, roles=[]))

    def test_marr_level_constrained_to_three_values(self) -> None:
        with pytest.raises(ValidationError):
            PredicateInstance.model_validate(_valid_predicate(marr_level="biological"))


# ---------------------------------------------------------------------------
# HigherOrderRelation
# ---------------------------------------------------------------------------


class TestHigherOrderRelation:
    def test_valid_relation_constructs(self) -> None:
        r = HigherOrderRelation.model_validate({"name": "causes", "relates": ["pred_a", "pred_b"]})
        assert r.name == "causes"
        assert r.relates == ["pred_a", "pred_b"]

    def test_must_relate_at_least_two_predicates(self) -> None:
        with pytest.raises(ValidationError):
            HigherOrderRelation.model_validate({"name": "causes", "relates": ["pred_a"]})

    def test_must_relate_distinct_predicates(self) -> None:
        with pytest.raises(ValidationError, match="DISTINCT"):
            HigherOrderRelation.model_validate(
                {"name": "self_relates", "relates": ["pred_a", "pred_a"]}
            )


# ---------------------------------------------------------------------------
# NearMiss
# ---------------------------------------------------------------------------


class TestNearMiss:
    def test_valid_near_miss_constructs(self) -> None:
        nm = NearMiss.model_validate(
            {
                "description": "doesn't pre-classify",
                "breaks_at_argument": "predicate_in_target.argument[1]",
            }
        )
        assert nm.breaks_at_argument == "predicate_in_target.argument[1]"

    def test_breaks_at_argument_format_required(self) -> None:
        with pytest.raises(ValidationError, match="predicate_in_"):
            NearMiss.model_validate({"description": "x", "breaks_at_argument": "argument[0]"})

    def test_breaks_at_argument_must_reference_source_or_target(self) -> None:
        with pytest.raises(ValidationError):
            NearMiss.model_validate(
                {"description": "x", "breaks_at_argument": "predicate_in_other.argument[0]"}
            )

    def test_breaks_at_argument_must_have_index(self) -> None:
        with pytest.raises(ValidationError):
            NearMiss.model_validate(
                {"description": "x", "breaks_at_argument": "predicate_in_source.argument[]"}
            )


# ---------------------------------------------------------------------------
# DomainEvidence
# ---------------------------------------------------------------------------


class TestDomainEvidence:
    def test_valid_evidence_constructs(self) -> None:
        e = DomainEvidence.model_validate(
            {"domain": "biology", "citation": "paper_2024", "excerpt": "..."}
        )
        assert e.domain == "biology"

    def test_empty_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DomainEvidence.model_validate({"domain": "", "citation": "x", "excerpt": "y"})


# ---------------------------------------------------------------------------
# StructuralMechanism -- the load-bearing schema tests
# ---------------------------------------------------------------------------


class TestStructuralMechanism:
    def test_valid_mechanism_constructs(self) -> None:
        m = StructuralMechanism.model_validate(_valid_mechanism())
        assert m.mechanism_id == "mantis-shrimp-camera-pre-classify"
        assert m.predicate_in_source.name == m.predicate_in_target.name
        assert len(m.higher_order_relations) >= 1
        assert len(m.considered_and_rejected_attributes) >= 2

    def test_signature_name_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="name mismatch"):
            StructuralMechanism.model_validate(
                _valid_mechanism(
                    predicate_in_target=_valid_predicate(name="other_predicate"),
                )
            )

    def test_signature_arity_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="arity mismatch"):
            StructuralMechanism.model_validate(
                _valid_mechanism(
                    predicate_in_target=_valid_predicate(
                        arity=3, roles=["transducer", "signal", "context"]
                    ),
                )
            )

    def test_signature_roles_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="roles mismatch"):
            StructuralMechanism.model_validate(
                _valid_mechanism(
                    predicate_in_target=_valid_predicate(roles=["sensor", "input"]),
                )
            )

    def test_signature_marr_level_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Marr level mismatch"):
            StructuralMechanism.model_validate(
                _valid_mechanism(
                    predicate_in_target=_valid_predicate(marr_level="implementation"),
                )
            )

    def test_attribute_only_predicate_rejected(self) -> None:
        """The shared predicate name must appear in at least one higher-order
        relation; otherwise it's decorative and the mechanism is invalid."""
        with pytest.raises(ValidationError, match="attribute-only"):
            StructuralMechanism.model_validate(
                _valid_mechanism(
                    higher_order_relations=[
                        {
                            "name": "unrelated_relation",
                            "relates": ["other_pred_a", "other_pred_b"],
                        }
                    ],
                )
            )

    def test_missing_higher_order_relation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StructuralMechanism.model_validate(_valid_mechanism(higher_order_relations=[]))

    def test_near_miss_index_out_of_range_rejected(self) -> None:
        """breaks_at_argument index must be < predicate.arity (0-indexed)."""
        with pytest.raises(ValidationError, match="out of range"):
            StructuralMechanism.model_validate(
                _valid_mechanism(
                    near_miss={
                        "description": "x",
                        "breaks_at_argument": "predicate_in_source.argument[5]",
                    },
                )
            )

    def test_near_miss_index_at_arity_boundary_rejected(self) -> None:
        """For arity=2, valid indices are 0 and 1; index 2 must reject."""
        with pytest.raises(ValidationError, match="out of range"):
            StructuralMechanism.model_validate(
                _valid_mechanism(
                    near_miss={
                        "description": "x",
                        "breaks_at_argument": "predicate_in_target.argument[2]",
                    },
                )
            )

    def test_near_miss_index_zero_accepted(self) -> None:
        StructuralMechanism.model_validate(
            _valid_mechanism(
                near_miss={
                    "description": "x",
                    "breaks_at_argument": "predicate_in_target.argument[0]",
                },
            )
        )

    def test_too_few_considered_and_rejected_attributes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StructuralMechanism.model_validate(
                _valid_mechanism(considered_and_rejected_attributes=["only_one"])
            )

    def test_zero_considered_and_rejected_attributes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StructuralMechanism.model_validate(
                _valid_mechanism(considered_and_rejected_attributes=[])
            )

    def test_json_round_trip_preserves_all_fields(self) -> None:
        original = StructuralMechanism.model_validate(_valid_mechanism())
        as_json = original.model_dump_json()
        roundtripped = StructuralMechanism.model_validate(json.loads(as_json))
        assert roundtripped == original
        # Spot-check that nested objects also round-tripped
        assert roundtripped.predicate_in_source.roles == ["transducer", "signal"]
        assert roundtripped.near_miss.breaks_at_argument == "predicate_in_source.argument[0]"
        assert roundtripped.domain_evidence[0].domain == "biology"

    def test_domain_evidence_optional(self) -> None:
        """domain_evidence has a default and is not required at the schema
        level (Session 4 owns retrieval / grounding)."""
        m = StructuralMechanism.model_validate(_valid_mechanism(domain_evidence=[]))
        assert m.domain_evidence == []

    def test_unicode_in_excerpts_round_trips(self) -> None:
        m_dict = _valid_mechanism()
        m_dict["domain_evidence"][0]["excerpt"] = "16 photoreceptor classes — pre-classify"
        m = StructuralMechanism.model_validate(m_dict)
        as_json = m.model_dump_json()
        rt = StructuralMechanism.model_validate(json.loads(as_json))
        assert "—" in rt.domain_evidence[0].excerpt


# ---------------------------------------------------------------------------
# GoalSpec extension -- structural_mechanism field is optional and defaults
# to None, so existing photosynthesis cycles continue to work unchanged.
# ---------------------------------------------------------------------------


class TestGoalSpecExtension:
    def _good_goalspec_dict(self) -> dict:
        # Mirrors the fixture in tests/photosynthesis/synthesis/test_generator.py
        # so we don't drift from the existing baseline.
        return {
            "goal_id": "test-goal",
            "title": "A test goal",
            "one_paragraph_description": "Build something to exercise the schema.",
            "artifact_type": "api",
            "primary_libraries": ["fastapi"],
            "new_libraries_introduced": [],
            "acceptance_criteria": [
                {"kind": "endpoint", "spec": "GET /health returns 200"},
            ],
            "estimated_build_time_min": 30,
            "estimated_difficulty": 2,
            "prerequisite_skills": ["fastapi"],
            "relevance_rationale": "rationale",
            "novelty_rationale": "novelty",
            "source_citation": "src",
        }

    def test_goalspec_without_mechanism_unchanged(self) -> None:
        from belief.photosynthesis.synthesis.generator import GoalSpec

        spec = GoalSpec.model_validate(self._good_goalspec_dict())
        assert spec.structural_mechanism is None

    def test_goalspec_accepts_populated_mechanism(self) -> None:
        from belief.photosynthesis.synthesis.generator import GoalSpec

        d = self._good_goalspec_dict()
        d["structural_mechanism"] = _valid_mechanism()
        spec = GoalSpec.model_validate(d)
        assert spec.structural_mechanism is not None
        assert spec.structural_mechanism.mechanism_id == ("mantis-shrimp-camera-pre-classify")

    def test_goalspec_rejects_invalid_mechanism(self) -> None:
        """An invalid structural_mechanism payload must surface its
        validation error through GoalSpec, not silently fall through."""
        from belief.photosynthesis.synthesis.generator import GoalSpec

        d = self._good_goalspec_dict()
        bad = _valid_mechanism()
        bad["predicate_in_target"] = _valid_predicate(name="other_predicate")
        d["structural_mechanism"] = bad
        with pytest.raises(ValidationError, match="name mismatch"):
            GoalSpec.model_validate(d)

    def test_goalspec_json_round_trip_with_mechanism(self) -> None:
        from belief.photosynthesis.synthesis.generator import GoalSpec

        d = self._good_goalspec_dict()
        d["structural_mechanism"] = _valid_mechanism()
        spec = GoalSpec.model_validate(d)
        rt = GoalSpec.model_validate(json.loads(spec.model_dump_json()))
        assert rt.structural_mechanism is not None
        assert rt.structural_mechanism == spec.structural_mechanism
