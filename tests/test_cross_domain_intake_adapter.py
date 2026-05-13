"""Tests for the cross-domain intake adapter (Synthesis Engine S7).

Pure-function unit tests -- no LLM, no network, no async. Verifies
that ``apply_to`` injects the right strings into a RequirementSpec's
constraints + acceptance_criteria lists, in the right order, without
mutating the input.
"""

from __future__ import annotations

import pytest

from belief.agents.cross_domain_intake_adapter import apply_to
from belief.models.artifacts import RequirementSpec
from belief.photosynthesis.synthesis.structural_mechanism import (
    DomainEvidence,
    HigherOrderRelation,
    IncompletenessProbe,
    NearMiss,
    PredicateInstance,
    StructuralMechanism,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _baseline_spec(
    *,
    constraints: list[str] | None = None,
    acceptance_criteria: list[str] | None = None,
) -> RequirementSpec:
    return RequirementSpec(
        goal="ship a thing",
        goal_refined="ship a working thing",
        target_type="python",
        complexity_score=2,
        acceptance_criteria=acceptance_criteria
        if acceptance_criteria is not None
        else ["existing-criterion-1"],
        constraints=constraints if constraints is not None else ["existing-constraint-1"],
    )


def _predicate(name: str = "downsamples_at_source", arity: int = 2) -> PredicateInstance:
    return PredicateInstance(
        name=name,
        arity=arity,
        roles=["source", "downstream"][:arity],
        marr_level="algorithmic",
    )


def _mechanism(
    *,
    name: str = "downsamples_at_source",
    arity: int = 2,
    open_probes: list[IncompletenessProbe] | None = None,
) -> StructuralMechanism:
    pred = _predicate(name=name, arity=arity)
    return StructuralMechanism(
        mechanism_id="mantis_camera_001",
        source_domain="biology",
        target_domain="computing",
        predicate_in_source=pred,
        predicate_in_target=pred.model_copy(),
        higher_order_relations=[
            HigherOrderRelation(
                name="reduces_downstream_compute",
                relates=[name, "compresses_at_sensor"],
            ),
        ],
        near_miss=NearMiss(
            description="A naive RGB camera that streams raw bytes downstream",
            breaks_at_argument=f"predicate_in_source.argument[{arity - 1}]",
        ),
        considered_and_rejected_attributes=[
            "color_channels",
            "spectral_count",
        ],
        domain_evidence=[
            DomainEvidence(
                domain="biology",
                citation="https://example.org/mantis-shrimp",
                excerpt="The shrimp's eyes pre-process color before sending to brain.",
            ),
        ],
        incompleteness_probes_open=list(open_probes or []),
    )


def _probe(probe_id: str, references_field: str, question: str) -> IncompletenessProbe:
    return IncompletenessProbe(
        probe_id=probe_id,
        question=question,
        references_field=references_field,
        classification="open_remainder",
        iteration=2,
    )


# ---------------------------------------------------------------------------
# apply_to: input is not mutated
# ---------------------------------------------------------------------------


def test_apply_to_does_not_mutate_input_spec() -> None:
    spec = _baseline_spec()
    mech = _mechanism()
    original_constraints = list(spec.constraints)
    original_acceptance = list(spec.acceptance_criteria)
    _ = apply_to(spec, mech)
    assert spec.constraints == original_constraints
    assert spec.acceptance_criteria == original_acceptance


def test_apply_to_returns_a_new_object() -> None:
    spec = _baseline_spec()
    new_spec = apply_to(spec, _mechanism())
    assert new_spec is not spec


def test_apply_to_preserves_unmodified_fields() -> None:
    spec = _baseline_spec()
    new_spec = apply_to(spec, _mechanism())
    assert new_spec.goal == spec.goal
    assert new_spec.goal_refined == spec.goal_refined
    assert new_spec.target_type == spec.target_type
    assert new_spec.complexity_score == spec.complexity_score


# ---------------------------------------------------------------------------
# apply_to: constraint additions
# ---------------------------------------------------------------------------


def test_apply_to_keeps_existing_constraints_first() -> None:
    spec = _baseline_spec(constraints=["pre-existing-A", "pre-existing-B"])
    new_spec = apply_to(spec, _mechanism())
    assert new_spec.constraints[0] == "pre-existing-A"
    assert new_spec.constraints[1] == "pre-existing-B"


def test_apply_to_adds_predicate_constraint() -> None:
    spec = _baseline_spec(constraints=[])
    new_spec = apply_to(spec, _mechanism())
    pred_line = new_spec.constraints[0]
    assert "downsamples_at_source" in pred_line
    assert "arity=2" in pred_line
    assert "marr_level=algorithmic" in pred_line
    assert "source" in pred_line
    assert "downstream" in pred_line


def test_apply_to_adds_higher_order_relation_constraints() -> None:
    spec = _baseline_spec(constraints=[])
    new_spec = apply_to(spec, _mechanism())
    relation_line = next(c for c in new_spec.constraints if "reduces_downstream_compute" in c)
    assert "downsamples_at_source" in relation_line
    assert "compresses_at_sensor" in relation_line


def test_apply_to_adds_near_miss_constraint() -> None:
    spec = _baseline_spec(constraints=[])
    new_spec = apply_to(spec, _mechanism())
    near_miss_line = next(c for c in new_spec.constraints if "near-miss" in c)
    assert "predicate_in_source.argument[1]" in near_miss_line
    assert "naive RGB camera" in near_miss_line


def test_apply_to_appends_one_constraint_per_open_probe() -> None:
    probes = [
        _probe("probe_001", "predicate_in_source.argument[0]", "what tags the source role?"),
        _probe("probe_002", "higher_order_relations[0]", "what enforces the reduction?"),
    ]
    spec = _baseline_spec(constraints=[])
    new_spec = apply_to(spec, _mechanism(open_probes=probes))
    todo_lines = [c for c in new_spec.constraints if c.startswith("TODO (")]
    assert len(todo_lines) == 2
    assert "probe_001" in todo_lines[0]
    assert "predicate_in_source.argument[0]" in todo_lines[0]
    assert "what tags the source role?" in todo_lines[0]
    assert "probe_002" in todo_lines[1]
    assert "what enforces the reduction?" in todo_lines[1]


def test_apply_to_emits_no_todo_when_no_open_probes() -> None:
    spec = _baseline_spec(constraints=[])
    new_spec = apply_to(spec, _mechanism(open_probes=None))
    assert not any(c.startswith("TODO (") for c in new_spec.constraints)


# ---------------------------------------------------------------------------
# apply_to: acceptance-criteria additions
# ---------------------------------------------------------------------------


def test_apply_to_keeps_existing_acceptance_criteria_first() -> None:
    spec = _baseline_spec(acceptance_criteria=["pre-existing-AC-1", "pre-existing-AC-2"])
    new_spec = apply_to(spec, _mechanism())
    assert new_spec.acceptance_criteria[0] == "pre-existing-AC-1"
    assert new_spec.acceptance_criteria[1] == "pre-existing-AC-2"


def test_apply_to_appends_predicate_exhibition_acceptance() -> None:
    spec = _baseline_spec(acceptance_criteria=[])
    new_spec = apply_to(spec, _mechanism())
    exhibition_line = new_spec.acceptance_criteria[0]
    assert "downsamples_at_source" in exhibition_line
    assert "arity 2" in exhibition_line
    assert "algorithmic" in exhibition_line


def test_apply_to_appends_open_probes_acceptance_only_when_probes_exist() -> None:
    spec_empty = _baseline_spec(acceptance_criteria=[])
    new_empty = apply_to(spec_empty, _mechanism(open_probes=None))
    assert len(new_empty.acceptance_criteria) == 1  # only the predicate-exhibition AC

    probes = [_probe("p1", "predicate_in_source.argument[0]", "Q?")]
    spec_with = _baseline_spec(acceptance_criteria=[])
    new_with = apply_to(spec_with, _mechanism(open_probes=probes))
    assert len(new_with.acceptance_criteria) == 2
    assert "1 open implementation probes" in new_with.acceptance_criteria[1]


# ---------------------------------------------------------------------------
# apply_to: arity-3 mechanism (probe count + role formatting)
# ---------------------------------------------------------------------------


def test_apply_to_handles_arity_three_mechanism() -> None:
    pred = PredicateInstance(
        name="three_arg_mechanism",
        arity=3,
        roles=["alpha", "beta", "gamma"],
        marr_level="implementation",
    )
    mech = StructuralMechanism(
        mechanism_id="three_001",
        source_domain="biology",
        target_domain="computing",
        predicate_in_source=pred,
        predicate_in_target=pred.model_copy(),
        higher_order_relations=[
            HigherOrderRelation(name="orchestrates", relates=["three_arg_mechanism", "x_other"]),
        ],
        near_miss=NearMiss(
            description="The two-arg degenerate case",
            breaks_at_argument="predicate_in_source.argument[2]",
        ),
        considered_and_rejected_attributes=["c1", "c2"],
    )
    spec = _baseline_spec(constraints=[], acceptance_criteria=[])
    new_spec = apply_to(spec, mech)
    pred_line = new_spec.constraints[0]
    assert "three_arg_mechanism" in pred_line
    assert "arity=3" in pred_line
    assert "alpha" in pred_line and "beta" in pred_line and "gamma" in pred_line
    assert "implementation" in pred_line


# ---------------------------------------------------------------------------
# Integration with IntakeAgent (hermetic: spec-already-present path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intake_agent_applies_mechanism_when_spec_already_present() -> None:
    # When state.requirement_spec is already populated, IntakeAgent skips
    # the LLM call -- which lets us exercise the adapter wiring without
    # mocking an LLM client.
    from belief.agents.intake import IntakeAgent
    from belief.config.models import ModelRouter
    from belief.models.state import UnifiedState

    router = ModelRouter()
    agent = IntakeAgent(router)

    base_spec = _baseline_spec(
        constraints=["pre-existing"], acceptance_criteria=["pre-existing-AC"]
    )
    state = UnifiedState(
        run_id="test-run",
        user_goal="ship a thing",
        requirement_spec=base_spec,
        structural_mechanism=_mechanism(
            open_probes=[
                _probe("p_a", "predicate_in_source.argument[0]", "open question A?"),
            ]
        ),
    )
    out = await agent.run(state)
    assert out.requirement_spec is not None
    assert "pre-existing" in out.requirement_spec.constraints
    assert any("downsamples_at_source" in c for c in out.requirement_spec.constraints)
    assert any(c.startswith("TODO (p_a") for c in out.requirement_spec.constraints)
    assert "pre-existing-AC" in out.requirement_spec.acceptance_criteria


@pytest.mark.asyncio
async def test_intake_agent_no_mechanism_leaves_spec_untouched() -> None:
    from belief.agents.intake import IntakeAgent
    from belief.config.models import ModelRouter
    from belief.models.state import UnifiedState

    router = ModelRouter()
    agent = IntakeAgent(router)

    base_spec = _baseline_spec(constraints=["only-this"], acceptance_criteria=["only-this-AC"])
    state = UnifiedState(
        run_id="test-run",
        user_goal="ship a thing",
        requirement_spec=base_spec,
        structural_mechanism=None,
    )
    out = await agent.run(state)
    assert out.requirement_spec is not None
    assert out.requirement_spec.constraints == ["only-this"]
    assert out.requirement_spec.acceptance_criteria == ["only-this-AC"]
