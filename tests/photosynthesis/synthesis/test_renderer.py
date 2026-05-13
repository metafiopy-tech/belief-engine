"""Tests for the GoalSpec -> session .md renderer.

Includes a byte-exact golden-file comparison. If you intentionally
change render_session_markdown, regenerate EXPECTED_MD by printing
render_session_markdown(sample_spec()) and re-pasting it below.
"""

from __future__ import annotations

from pathlib import Path


from belief.photosynthesis.synthesis.generator import GoalSpec
from belief.photosynthesis.synthesis.renderer import (
    render_session_markdown,
    write_session,
)


SAMPLE_SPEC = {
    "goal_id": "mcp-echo-fastapi",
    "title": "Mount a FastMCP echo server on FastAPI with MCP tool",
    "one_paragraph_description": (
        "Build a FastAPI app that mounts a FastMCP server exposing an echo tool."
    ),
    "artifact_type": "api",
    "primary_libraries": ["fastapi", "fastmcp"],
    "new_libraries_introduced": ["fastmcp"],
    "acceptance_criteria": [
        {"kind": "endpoint", "spec": "POST /mcp handles MCP protocol"},
        {"kind": "test", "spec": "pytest verifies echo tool returns input"},
    ],
    "estimated_build_time_min": 60,
    "estimated_difficulty": 3,
    "prerequisite_skills": ["fastapi", "fastmcp-basics"],
    "relevance_rationale": "MCP is primary tool-use protocol the engine targets.",
    "novelty_rationale": "No FastMCP goals currently in archive.",
    "source_citation": "github.com/user/repo",
}


EXPECTED_MD = """\
# SESSION: mcp-echo-fastapi
# Estimated: 60 minutes, difficulty 3/5
# Source: github.com/user/repo

CONTEXT
Build a FastAPI app that mounts a FastMCP server exposing an echo tool.

Relevance: MCP is primary tool-use protocol the engine targets.
Novelty: No FastMCP goals currently in archive.

Artifact type: api
Primary libraries: fastapi, fastmcp
New libraries introduced: fastmcp
Prerequisite skills: fastapi, fastmcp-basics

TASK LIST

1. Produce a api named mcp-echo-fastapi that implements:
   Mount a FastMCP echo server on FastAPI with MCP tool

2. For each acceptance criterion below, implement the behavior and add
   a matching test (when kind=test) or endpoint/artifact verification.

3. Package the deliverable in the conventional Belief Engine layout
   (entry point, README, pyproject.toml) using the primary libraries.

CONSTRAINTS
- Python 3.14
- Target build time: 60 minutes
- Difficulty tier: 3/5
- Must be buildable in one autonomous session
- All acceptance criteria objectively checkable

ACCEPTANCE CRITERIA
1. [endpoint] POST /mcp handles MCP protocol
2. [test] pytest verifies echo tool returns input
"""


def _spec() -> GoalSpec:
    return GoalSpec.model_validate(SAMPLE_SPEC)


def test_render_matches_golden_file() -> None:
    rendered = render_session_markdown(_spec())
    assert rendered == EXPECTED_MD


def test_render_is_deterministic() -> None:
    spec = _spec()
    a = render_session_markdown(spec)
    b = render_session_markdown(spec)
    assert a == b


def test_write_session_produces_md_and_json(tmp_path: Path) -> None:
    md_path, json_path = write_session(_spec(), pending_dir=tmp_path)
    assert md_path.read_text() == EXPECTED_MD
    # Sidecar JSON has sorted keys + trailing newline
    sidecar = json_path.read_text()
    assert sidecar.endswith("\n")
    assert '"goal_id": "mcp-echo-fastapi"' in sidecar


def test_write_session_is_overwrite_safe(tmp_path: Path) -> None:
    spec = _spec()
    first_md, first_json = write_session(spec, pending_dir=tmp_path)
    before_md = first_md.read_text()
    before_json = first_json.read_text()
    # Re-run — byte-identical output
    second_md, second_json = write_session(spec, pending_dir=tmp_path)
    assert first_md == second_md
    assert first_json == second_json
    assert second_md.read_text() == before_md
    assert second_json.read_text() == before_json


# ---------------------------------------------------------------------------
# Synthesis Engine Session 7: structural_mechanism + open probes rendering
# ---------------------------------------------------------------------------


def _spec_with_mechanism(*, with_open_probes: bool = True) -> GoalSpec:
    """A GoalSpec carrying a populated structural_mechanism for S7 tests."""
    from belief.photosynthesis.synthesis.structural_mechanism import (
        DomainEvidence,
        HigherOrderRelation,
        IncompletenessProbe,
        NearMiss,
        PredicateInstance,
        StructuralMechanism,
    )

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
            IncompletenessProbe(
                probe_id="probe_002",
                question="What invariant prevents oversampling?",
                references_field="higher_order_relations[0]",
                classification="open_remainder",
                iteration=2,
            ),
        ]
    mechanism = StructuralMechanism(
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
    payload = dict(SAMPLE_SPEC)
    payload["structural_mechanism"] = mechanism.model_dump()
    return GoalSpec.model_validate(payload)


def test_render_omits_mechanism_section_when_no_mechanism() -> None:
    """The default sample spec has no mechanism -> output unchanged."""
    rendered = render_session_markdown(_spec())
    assert "## Structural Mechanism" not in rendered
    assert "## Open Implementation Probes" not in rendered


def test_render_appends_mechanism_section_when_mechanism_present() -> None:
    rendered = render_session_markdown(_spec_with_mechanism())
    assert "## Structural Mechanism" in rendered
    assert "mantis_camera_001" in rendered
    assert "downsamples_at_source/2 (algorithmic)" in rendered
    assert "source, downstream" in rendered
    assert "reduces_downstream_compute" in rendered
    assert "compresses_at_sensor" in rendered
    assert "near_miss" in rendered
    assert "Naive RGB camera streaming raw bytes" in rendered
    assert "color_channels, spectral_count" in rendered


def test_render_appends_open_probes_subsection_when_probes_exist() -> None:
    rendered = render_session_markdown(_spec_with_mechanism(with_open_probes=True))
    assert "## Open Implementation Probes" in rendered
    assert "[probe_001 @ predicate_in_source.argument[1]]" in rendered
    assert "What protocol carries the downsampled signal?" in rendered
    assert "[probe_002 @ higher_order_relations[0]]" in rendered


def test_render_omits_open_probes_subsection_when_no_probes() -> None:
    rendered = render_session_markdown(_spec_with_mechanism(with_open_probes=False))
    # Mechanism section is present
    assert "## Structural Mechanism" in rendered
    # ...but the open-probes header is suppressed (no header without items)
    assert "## Open Implementation Probes" not in rendered


def test_render_with_mechanism_is_deterministic() -> None:
    spec = _spec_with_mechanism()
    a = render_session_markdown(spec)
    b = render_session_markdown(spec)
    assert a == b


def test_render_mechanism_section_appears_after_acceptance_criteria() -> None:
    rendered = render_session_markdown(_spec_with_mechanism())
    ac_idx = rendered.index("ACCEPTANCE CRITERIA")
    mech_idx = rendered.index("## Structural Mechanism")
    assert ac_idx < mech_idx
