"""Tests for CLI sidecar hydration (Synthesis Engine S7.6).

Hermetic. Tests the shared `hydrate_initial_state` helper that the
CLI's `belief build --sidecar PATH` flag invokes to wire sidecar
mechanisms into the build's initial state. Symmetric counterpart
to the Grinder daemon's S7.5 hydration.

The helper lives in `sidecar_loader.py` (not `cli.py`) so tests can
exercise it without importing the heavy CLI module (which transitively
pulls in langgraph and won't load in py3.10 sandboxes).

The CLI's full `run()` function is too entangled with HealthDaemon /
BuildStore / ModelRouter for direct testing; we cover the helper
and trust the wiring at the call site (which is one line in
`belief/cli.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

from belief.photosynthesis.synthesis.sidecar_loader import (
    hydrate_initial_state as hydrate_initial_state_from_sidecar,
)
from belief.photosynthesis.synthesis.generator import GoalSpec
from belief.photosynthesis.synthesis.renderer import write_session
from belief.photosynthesis.synthesis.structural_mechanism import (
    DomainEvidence,
    HigherOrderRelation,
    IncompletenessProbe,
    NearMiss,
    PredicateInstance,
    StructuralMechanism,
)


SAMPLE_SPEC = {
    "goal_id": "mantis-camera-cli",
    "title": "Build a downsampling camera mount via CLI",
    "one_paragraph_description": "FastAPI mount around a downsampling sensor.",
    "artifact_type": "api",
    "primary_libraries": ["fastapi"],
    "new_libraries_introduced": [],
    "acceptance_criteria": [{"kind": "endpoint", "spec": "POST /sample handles raw frames"}],
    "estimated_build_time_min": 60,
    "estimated_difficulty": 3,
    "prerequisite_skills": ["fastapi"],
    "relevance_rationale": "Cross-domain proof-point.",
    "novelty_rationale": "First mantis-shrimp inspired build via CLI.",
    "source_citation": "github.com/synth/test",
}


def _make_mechanism(*, with_open_probes: bool = True) -> StructuralMechanism:
    pred = PredicateInstance(
        name="downsamples_at_source",
        arity=2,
        roles=["source", "downstream"],
        marr_level="algorithmic",
    )
    open_probes: list[IncompletenessProbe] = []
    if with_open_probes:
        open_probes = [
            IncompletenessProbe(
                probe_id="probe_001",
                question="What protocol carries the downsampled signal?",
                references_field="predicate_in_source.argument[1]",
                classification="open_remainder",
                iteration=2,
            ),
        ]
    return StructuralMechanism(
        mechanism_id="mantis_camera_cli_001",
        source_domain="biology",
        target_domain="computing",
        predicate_in_source=pred,
        predicate_in_target=pred.model_copy(),
        higher_order_relations=[
            HigherOrderRelation(
                name="reduces_downstream_compute",
                relates=["downsamples_at_source", "compresses_at_sensor"],
            ),
        ],
        near_miss=NearMiss(
            description="Naive RGB camera streaming raw bytes",
            breaks_at_argument="predicate_in_source.argument[1]",
        ),
        considered_and_rejected_attributes=["color_channels", "spectral_count"],
        domain_evidence=[
            DomainEvidence(
                domain="biology",
                citation="https://example.org/mantis-cli",
                excerpt="Eyes pre-process before brain.",
            ),
        ],
        incompleteness_probes_open=open_probes,
    )


def _spec_with_mechanism(*, with_open_probes: bool = True) -> GoalSpec:
    payload = dict(SAMPLE_SPEC)
    payload["structural_mechanism"] = _make_mechanism(
        with_open_probes=with_open_probes
    ).model_dump()
    return GoalSpec.model_validate(payload)


# ---------------------------------------------------------------------------
# hydrate_initial_state_from_sidecar: trivial cases
# ---------------------------------------------------------------------------


def test_hydrate_returns_state_unchanged_when_path_is_none() -> None:
    state = {"user_goal": "x", "iteration": 0}
    out = hydrate_initial_state_from_sidecar(state, None)
    # Same content; helper does not require object identity for None case.
    assert out == state


def test_hydrate_returns_state_unchanged_for_missing_file(tmp_path: Path) -> None:
    state = {"user_goal": "x"}
    out = hydrate_initial_state_from_sidecar(state, tmp_path / "missing.json")
    assert out == state
    assert "structural_mechanism" not in out


def test_hydrate_returns_state_unchanged_for_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json {{{")
    state = {"user_goal": "x"}
    out = hydrate_initial_state_from_sidecar(state, p)
    assert "structural_mechanism" not in out


def test_hydrate_returns_state_unchanged_when_sidecar_lacks_mechanism(
    tmp_path: Path,
) -> None:
    p = tmp_path / "nomech.json"
    p.write_text(json.dumps({"goal_id": "x", "title": "y"}))
    state = {"user_goal": "x"}
    out = hydrate_initial_state_from_sidecar(state, p)
    assert "structural_mechanism" not in out


# ---------------------------------------------------------------------------
# hydrate_initial_state_from_sidecar: happy path
# ---------------------------------------------------------------------------


def test_hydrate_injects_mechanism_when_sidecar_has_one(tmp_path: Path) -> None:
    spec = _spec_with_mechanism(with_open_probes=True)
    _, json_path = write_session(spec, pending_dir=tmp_path)
    state = {"user_goal": "x", "iteration": 0}
    out = hydrate_initial_state_from_sidecar(state, json_path)
    assert "structural_mechanism" in out
    mech = out["structural_mechanism"]
    assert isinstance(mech, StructuralMechanism)
    assert mech.mechanism_id == "mantis_camera_cli_001"
    assert mech.predicate_in_source.name == "downsamples_at_source"
    assert len(mech.incompleteness_probes_open) == 1


def test_hydrate_does_not_mutate_input_state(tmp_path: Path) -> None:
    spec = _spec_with_mechanism()
    _, json_path = write_session(spec, pending_dir=tmp_path)
    state = {"user_goal": "x"}
    snapshot = dict(state)
    _ = hydrate_initial_state_from_sidecar(state, json_path)
    assert state == snapshot
    assert "structural_mechanism" not in state


def test_hydrate_accepts_string_path(tmp_path: Path) -> None:
    """The CLI argparse value comes in as a string -- make sure we handle it."""
    spec = _spec_with_mechanism()
    _, json_path = write_session(spec, pending_dir=tmp_path)
    state = {"user_goal": "x"}
    out = hydrate_initial_state_from_sidecar(state, str(json_path))
    assert "structural_mechanism" in out


def test_hydrate_preserves_other_state_keys(tmp_path: Path) -> None:
    spec = _spec_with_mechanism()
    _, json_path = write_session(spec, pending_dir=tmp_path)
    state = {
        "user_goal": "build a thing",
        "iteration": 0,
        "max_iterations": 3,
        "polarity": {"latios_coherence": 0.5},
    }
    out = hydrate_initial_state_from_sidecar(state, json_path)
    assert out["user_goal"] == "build a thing"
    assert out["iteration"] == 0
    assert out["max_iterations"] == 3
    assert out["polarity"] == {"latios_coherence": 0.5}
    assert "structural_mechanism" in out


# ---------------------------------------------------------------------------
# Round-trip: synth-emits -> CLI-build hydrates
# ---------------------------------------------------------------------------


def test_round_trip_synth_to_cli_build(tmp_path: Path) -> None:
    """Compose the full handoff: write_session emits the sidecar that the
    CLI's hydration helper picks up. Validates the contract end-to-end."""
    spec = _spec_with_mechanism(with_open_probes=True)
    _, json_path = write_session(spec, pending_dir=tmp_path)

    # CLI-side: argparse hands the path string to run(); the helper hydrates.
    initial_state = {"user_goal": "downsampling camera", "iteration": 0}
    hydrated = hydrate_initial_state_from_sidecar(initial_state, str(json_path))

    mech = hydrated["structural_mechanism"]
    assert isinstance(mech, StructuralMechanism)
    # Lossless round-trip on schema-significant fields.
    assert mech.model_dump() == spec.structural_mechanism.model_dump()
