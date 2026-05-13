"""Tests for the sidecar loader (Synthesis Engine S7.5).

Hermetic. Verifies that the disk -> state plumbing closes:

  - JSON sidecar with a `structural_mechanism` block hydrates into a
    validated StructuralMechanism.
  - Sidecar with the field absent (or None) returns None.
  - Sidecar with a malformed mechanism logs a warning and returns
    None -- never raises.
  - Round-trip via the renderer's write_session output is lossless:
    write -> read -> compare yields the same mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

from belief.photosynthesis.synthesis.generator import GoalSpec
from belief.photosynthesis.synthesis.renderer import write_session
from belief.photosynthesis.synthesis.sidecar_loader import (
    extract_structural_mechanism,
    load_sidecar_from_path,
)
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


SAMPLE_SPEC = {
    "goal_id": "mantis-camera",
    "title": "Build a downsampling camera mount",
    "one_paragraph_description": "A FastAPI mount around a downsampling sensor.",
    "artifact_type": "api",
    "primary_libraries": ["fastapi"],
    "new_libraries_introduced": [],
    "acceptance_criteria": [
        {"kind": "endpoint", "spec": "POST /sample handles raw frames"},
    ],
    "estimated_build_time_min": 60,
    "estimated_difficulty": 3,
    "prerequisite_skills": ["fastapi"],
    "relevance_rationale": "Core synthesis demo.",
    "novelty_rationale": "First mantis-shrimp inspired build.",
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
        mechanism_id="mantis_camera_001",
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
                citation="https://example.org/mantis",
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
# extract_structural_mechanism: trivial cases
# ---------------------------------------------------------------------------


def test_extract_returns_none_for_none_sidecar() -> None:
    assert extract_structural_mechanism(None) is None


def test_extract_returns_none_for_empty_dict() -> None:
    assert extract_structural_mechanism({}) is None


def test_extract_returns_none_when_field_missing() -> None:
    assert extract_structural_mechanism({"goal_id": "x", "title": "y"}) is None


def test_extract_returns_none_when_field_explicitly_null() -> None:
    assert extract_structural_mechanism({"structural_mechanism": None}) is None


def test_extract_returns_none_when_field_is_not_a_dict(caplog) -> None:
    caplog.set_level("WARNING")
    result = extract_structural_mechanism({"structural_mechanism": "not a dict"})
    assert result is None
    assert any("not a dict" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# extract_structural_mechanism: happy path
# ---------------------------------------------------------------------------


def test_extract_hydrates_valid_mechanism() -> None:
    mech = _make_mechanism()
    sidecar = {"structural_mechanism": mech.model_dump()}
    out = extract_structural_mechanism(sidecar)
    assert isinstance(out, StructuralMechanism)
    assert out.mechanism_id == "mantis_camera_001"
    assert out.predicate_in_source.name == "downsamples_at_source"
    assert len(out.incompleteness_probes_open) == 1


def test_extract_passes_through_already_typed_mechanism() -> None:
    mech = _make_mechanism()
    sidecar = {"structural_mechanism": mech}
    out = extract_structural_mechanism(sidecar)
    assert out is mech


# ---------------------------------------------------------------------------
# extract_structural_mechanism: malformed input is permissive
# ---------------------------------------------------------------------------


def test_extract_logs_and_returns_none_on_validation_failure(caplog) -> None:
    caplog.set_level("WARNING")
    # Missing required mechanism_id, source_domain, predicates etc.
    bad = {"structural_mechanism": {"mechanism_id": "x"}}
    out = extract_structural_mechanism(bad)
    assert out is None
    assert any("failed validation" in r.message for r in caplog.records)


def test_extract_does_not_raise_on_garbage() -> None:
    # Any garbage in the field should never raise -- the build must
    # keep going.
    for garbage in [123, [], 0.5, True, ["list", "items"]]:
        assert extract_structural_mechanism({"structural_mechanism": garbage}) is None


# ---------------------------------------------------------------------------
# load_sidecar_from_path: file IO wrapper
# ---------------------------------------------------------------------------


def test_load_sidecar_reads_valid_json(tmp_path: Path) -> None:
    p = tmp_path / "sample.json"
    p.write_text(json.dumps({"hello": "world"}))
    out = load_sidecar_from_path(p)
    assert out == {"hello": "world"}


def test_load_sidecar_returns_none_for_missing_file(tmp_path: Path, caplog) -> None:
    caplog.set_level("WARNING")
    out = load_sidecar_from_path(tmp_path / "missing.json")
    assert out is None
    assert any("could not read" in r.message for r in caplog.records)


def test_load_sidecar_returns_none_for_invalid_json(tmp_path: Path, caplog) -> None:
    caplog.set_level("WARNING")
    p = tmp_path / "bad.json"
    p.write_text("not json {{{")
    out = load_sidecar_from_path(p)
    assert out is None
    assert any("not valid JSON" in r.message for r in caplog.records)


def test_load_sidecar_returns_none_for_non_dict_top_level(tmp_path: Path, caplog) -> None:
    caplog.set_level("WARNING")
    p = tmp_path / "list.json"
    p.write_text(json.dumps(["a", "b"]))
    out = load_sidecar_from_path(p)
    assert out is None
    assert any("did not parse to a dict" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Round-trip: renderer write_session -> sidecar_loader read
# ---------------------------------------------------------------------------


def test_round_trip_via_write_session(tmp_path: Path) -> None:
    """The closing-the-loop test: write the sidecar via the renderer,
    read it back via the loader, recover the same mechanism."""
    spec = _spec_with_mechanism(with_open_probes=True)
    md_path, json_path = write_session(spec, pending_dir=tmp_path)
    assert json_path.exists()
    sidecar = load_sidecar_from_path(json_path)
    assert sidecar is not None
    out = extract_structural_mechanism(sidecar)
    assert isinstance(out, StructuralMechanism)
    assert out.mechanism_id == "mantis_camera_001"
    assert out.predicate_in_source.name == "downsamples_at_source"
    assert len(out.incompleteness_probes_open) == 1
    # Lossless round-trip on the schema-significant fields.
    assert out.model_dump() == spec.structural_mechanism.model_dump()


def test_round_trip_with_no_mechanism_yields_none(tmp_path: Path) -> None:
    """write -> read for a spec WITHOUT a mechanism returns None."""
    spec = GoalSpec.model_validate(SAMPLE_SPEC)
    _, json_path = write_session(spec, pending_dir=tmp_path)
    sidecar = load_sidecar_from_path(json_path)
    assert sidecar is not None  # the sidecar itself loads fine
    assert extract_structural_mechanism(sidecar) is None


# ---------------------------------------------------------------------------
# Daemon integration: the build runner now hydrates state.structural_mechanism
# (test the wiring without invoking the real pipeline)
# ---------------------------------------------------------------------------


def _install_fake_graph_module(monkeypatch, captured: dict[str, object]) -> None:
    """Replace `belief.graph` in sys.modules with a stub.

    The daemon's `_default_build_runner` does
    ``from belief.graph import build_pipeline`` lazily inside the
    function. Stuffing a fake module into ``sys.modules`` BEFORE that
    runtime import means we never trigger the real `belief.graph`
    (which transitively imports langgraph and would fail in the
    sandbox without that dep).
    """
    import sys
    import types

    class FakeGraph:
        async def ainvoke(self, state):  # noqa: ANN001
            captured.update(state)
            return {"phase": "complete"}

    fake = types.ModuleType("belief.graph")
    fake.build_pipeline = lambda *_a, **_kw: FakeGraph()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "belief.graph", fake)


def test_daemon_wires_mechanism_into_initial_state(monkeypatch, tmp_path: Path) -> None:
    """The default build runner should put structural_mechanism into the
    dict it hands to graph.ainvoke when the sidecar carries one."""
    import asyncio

    from belief.grinder.daemon import _default_build_runner
    from belief.grinder.goal_queue import GoalEnvelope

    captured: dict[str, object] = {}
    _install_fake_graph_module(monkeypatch, captured)

    spec = _spec_with_mechanism(with_open_probes=True)
    md_path, json_path = write_session(spec, pending_dir=tmp_path)
    sidecar = load_sidecar_from_path(json_path)

    env = GoalEnvelope(
        goal_id=spec.goal_id,
        goal_text="downsampling camera mount",
        priority=0.5,
        md_path=md_path,
        json_path=json_path,
        sidecar=sidecar,
        source="queue",
    )
    asyncio.run(_default_build_runner(env))
    assert "structural_mechanism" in captured
    mech = captured["structural_mechanism"]
    assert isinstance(mech, StructuralMechanism)
    assert mech.mechanism_id == "mantis_camera_001"


def test_daemon_omits_mechanism_when_sidecar_has_none(monkeypatch, tmp_path: Path) -> None:
    """When the sidecar lacks a mechanism, the runner does NOT inject
    a structural_mechanism key (default behavior preserved)."""
    import asyncio

    from belief.grinder.daemon import _default_build_runner
    from belief.grinder.goal_queue import GoalEnvelope

    captured: dict[str, object] = {}
    _install_fake_graph_module(monkeypatch, captured)
    spec = GoalSpec.model_validate(SAMPLE_SPEC)
    md_path, json_path = write_session(spec, pending_dir=tmp_path)
    sidecar = load_sidecar_from_path(json_path)

    env = GoalEnvelope(
        goal_id=spec.goal_id,
        goal_text="something",
        priority=0.5,
        md_path=md_path,
        json_path=json_path,
        sidecar=sidecar,
        source="queue",
    )
    asyncio.run(_default_build_runner(env))
    assert "structural_mechanism" not in captured
    assert captured.get("user_goal") == "something"
