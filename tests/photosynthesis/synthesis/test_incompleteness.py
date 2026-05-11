"""Tests for the incompleteness pass (SE Session 6)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from belief.photosynthesis.synthesis import prompts_cross_domain as prompts
from belief.photosynthesis.synthesis.incompleteness import (
    MAX_LOOPBACK_ITERATIONS,
    classify_probes,
    generate_probes,
    run_incompleteness,
)
from belief.photosynthesis.synthesis.structural_mechanism import (
    IncompletenessProbe,
    StructuralMechanism,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _few_shot(name: str) -> StructuralMechanism:
    txt = getattr(prompts, name).replace("{{", "{").replace("}}", "}")
    return StructuralMechanism.model_validate(json.loads(txt))


@dataclass
class FakeDoc:
    """Minimal doc shape for classify_probes tests."""

    title: str = ""
    summary: str = ""
    raw_excerpt: str = ""
    url: str = ""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Probe generation
# ---------------------------------------------------------------------------


class TestGenerateProbes:
    def test_arity_2_yields_14_probes(self) -> None:
        """2 sides * 2 args * 3 templates + 1 relation + 1 near_miss = 14."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")  # arity 2, 1 relation
        probes = generate_probes(mech)
        assert len(probes) == 14

    def test_arity_3_yields_20_probes(self) -> None:
        """SE plan target: 15-20 probes per mechanism. arity-3 hits 20."""
        mech = _few_shot("FEW_SHOT_MYCORRHIZAL_CDN")  # arity 3, 1 relation
        probes = generate_probes(mech)
        assert len(probes) == 20

    def test_every_probe_has_unique_id(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        ids = [p.probe_id for p in probes]
        assert len(set(ids)) == len(ids)

    def test_arg_probes_reference_predicate_arguments(self) -> None:
        """SE plan acceptance: 'Probes carry foreign-key references
        to specific predicate arguments.'"""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        arg_probes = [p for p in probes if "argument[" in p.references_field]
        # 2 sides * 2 args * 3 templates = 12 arg-targeted probes
        assert len(arg_probes) == 12
        for p in arg_probes:
            assert p.references_field.startswith("predicate_in_")
            assert "argument[" in p.references_field

    def test_relation_probe_present(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        rel_probes = [p for p in probes if p.references_field.startswith("higher_order_relations[")]
        assert len(rel_probes) == 1

    def test_near_miss_probe_present(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        nm_probes = [p for p in probes if p.references_field == "near_miss"]
        assert len(nm_probes) == 1

    def test_question_substitutes_role_and_side(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        # First arg's first probe should mention 'transducer' (role)
        # and either 'source' or 'target' (side).
        first_arg_probes = [p for p in probes if "argument[0]" in p.references_field]
        assert any("transducer" in p.question for p in first_arg_probes)
        assert any("source" in p.question or "target" in p.question for p in first_arg_probes)

    def test_all_probes_start_as_needs_research(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        assert all(p.classification == "needs_research" for p in probes)
        assert all(p.iteration == 0 for p in probes)

    def test_n_per_arg_clamps(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes_high = generate_probes(mech, n_per_arg=99)
        # Caps at template count (3)
        arg_probes_high = [p for p in probes_high if "argument[" in p.references_field]
        assert len(arg_probes_high) == 12
        probes_low = generate_probes(mech, n_per_arg=0)
        arg_probes_low = [p for p in probes_low if "argument[" in p.references_field]
        # Clamps up to 1 -> 2 sides * 2 args * 1 = 4
        assert len(arg_probes_low) == 4


# ---------------------------------------------------------------------------
# Probe classification
# ---------------------------------------------------------------------------


class TestClassifyProbes:
    def test_empty_corpus_leaves_all_needs_research(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        result = classify_probes(probes, corpus=[])
        assert all(p.classification == "needs_research" for p in result)

    def test_none_corpus_leaves_all_needs_research(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        result = classify_probes(probes, corpus=None)
        assert all(p.classification == "needs_research" for p in result)

    def test_resolves_probes_with_relevant_corpus(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        # Doc that mentions both a role token AND an implementation hint
        corpus = [
            FakeDoc(
                title="transducer implementation",
                summary="implementation of the transducer signal interface",
                url="http://example/x",
            )
        ]
        result = classify_probes(probes, corpus=corpus)
        resolved = [p for p in result if p.classification == "resolved_from_corpus"]
        assert len(resolved) > 0
        assert all(p.evidence_url == "http://example/x" for p in resolved)

    def test_classification_carries_iteration_index(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        result = classify_probes(probes, corpus=[], iteration=2)
        assert all(p.iteration == 2 for p in result)

    def test_already_resolved_probes_carry_forward(self) -> None:
        """Probes resolved on a prior iteration shouldn't be
        re-classified down to needs_research on a later pass."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probes = generate_probes(mech)
        # Manually mark one as resolved
        probes[0].classification = "resolved_from_corpus"
        probes[0].evidence_url = "http://manual"

        result = classify_probes(probes, corpus=[], iteration=1)
        resolved = next(p for p in result if p.probe_id == probes[0].probe_id)
        assert resolved.classification == "resolved_from_corpus"
        assert resolved.evidence_url == "http://manual"


# ---------------------------------------------------------------------------
# Loopback orchestration
# ---------------------------------------------------------------------------


class TestRunIncompleteness:
    def test_empty_dispatcher_unresolved_become_open_remainder(self) -> None:
        """SE acceptance: 'Loopback iterates max 2 times; third
        attempt forces emission with open probes propagated forward.'
        With no dispatcher there's no loopback at all; all
        unresolved probes immediately go to open_remainder."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        result = _run(run_incompleteness(mech, corpus=[]))
        assert result.iterations_used == 0
        assert result.resolved_count == 0
        assert result.open_count == 14
        assert all(p.classification == "open_remainder" for p in result.probes_open)

    def test_loopback_caps_at_max_iterations(self) -> None:
        """When the dispatcher never resolves anything, the loopback
        runs MAX_LOOPBACK_ITERATIONS times then gives up."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        call_count = {"n": 0}

        async def empty_dispatcher(prompts):
            call_count["n"] += 1
            return []

        result = _run(run_incompleteness(mech, corpus=[], dispatcher=empty_dispatcher))
        assert result.iterations_used == MAX_LOOPBACK_ITERATIONS
        assert call_count["n"] == MAX_LOOPBACK_ITERATIONS
        # Everything still unresolved -> open_remainder
        assert result.resolved_count == 0
        assert result.open_count == 14

    def test_loopback_resolves_on_dispatcher_corpus(self) -> None:
        """When the dispatcher returns relevant docs, probes get
        resolved on the next classification iteration."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")

        async def helpful_dispatcher(prompts):
            return [
                FakeDoc(
                    title="implementation",
                    summary="transducer implementation in signal processing",
                    url="http://example/loopback1",
                )
            ]

        result = _run(run_incompleteness(mech, corpus=[], dispatcher=helpful_dispatcher))
        # Iteration 1 fires once and resolves probes; iteration 2 may
        # fire too if any unresolved remain.
        assert result.iterations_used >= 1
        assert result.resolved_count > 0

    def test_dispatcher_failure_does_not_abort(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")

        async def boom(prompts):
            raise RuntimeError("simulated network failure")

        # Should not raise; failure is logged + treated as "no new
        # docs" so the loopback keeps iterating.
        result = _run(run_incompleteness(mech, corpus=[], dispatcher=boom))
        assert result.iterations_used == MAX_LOOPBACK_ITERATIONS
        assert result.open_count == 14

    def test_partial_resolution_propagates_open_remainder(self) -> None:
        """Some probes resolved, others not -- the open ones become
        open_remainder and propagate forward."""
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        # Corpus mentions the source-side roles but NOT implementation
        # hints for the relation/near_miss probes
        corpus = [
            FakeDoc(
                title="transducer implementation",
                summary="signal implementation in transducers",
                url="http://example/partial",
            )
        ]

        async def empty_dispatcher(prompts):
            return []

        result = _run(run_incompleteness(mech, corpus=corpus, dispatcher=empty_dispatcher))
        assert result.resolved_count > 0
        assert result.open_count > 0
        assert result.resolved_count + result.open_count == len(result.probes_generated)
        # Sanity: every probe in probes_open is classified open_remainder.
        assert all(p.classification == "open_remainder" for p in result.probes_open)


# ---------------------------------------------------------------------------
# IncompletenessProbe schema
# ---------------------------------------------------------------------------


class TestIncompletenessProbeSchema:
    def test_unknown_classification_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IncompletenessProbe(
                probe_id="p1",
                question="q",
                references_field="x",
                classification="invented_status",
            )

    def test_empty_references_field_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IncompletenessProbe(
                probe_id="p1",
                question="q",
                references_field="",
            )

    def test_iteration_must_be_nonnegative(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IncompletenessProbe(
                probe_id="p1",
                question="q",
                references_field="predicate_in_source.argument[0]",
                iteration=-1,
            )

    def test_round_trip_through_json(self) -> None:
        p = IncompletenessProbe(
            probe_id="p1",
            question="q",
            references_field="predicate_in_source.argument[0]",
            classification="resolved_from_corpus",
            evidence_url="http://example/x",
            iteration=1,
        )
        rt = IncompletenessProbe.model_validate_json(p.model_dump_json())
        assert rt == p


# ---------------------------------------------------------------------------
# StructuralMechanism propagation
# ---------------------------------------------------------------------------


class TestMechanismPropagation:
    def test_default_probes_open_is_empty(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        assert mech.incompleteness_probes_open == []

    def test_mechanism_carries_open_probes(self) -> None:
        mech = _few_shot("FEW_SHOT_MANTIS_SHRIMP_CAMERA")
        probe = IncompletenessProbe(
            probe_id="p1",
            question="q",
            references_field="predicate_in_source.argument[0]",
            classification="open_remainder",
            iteration=2,
        )
        mech_with = mech.model_copy(update={"incompleteness_probes_open": [probe]})
        assert len(mech_with.incompleteness_probes_open) == 1
        # Round-trip
        rt = StructuralMechanism.model_validate_json(mech_with.model_dump_json())
        assert len(rt.incompleteness_probes_open) == 1
        assert rt.incompleteness_probes_open[0].probe_id == "p1"
