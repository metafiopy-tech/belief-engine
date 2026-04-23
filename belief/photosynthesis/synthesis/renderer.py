"""Render a validated GoalSpec to the Belief Engine v3.0 session format.

Output layout on disk:

    pending_sessions/{goal_id}.md      Claude-Code-style session prompt
    pending_sessions/{goal_id}.json    structured metadata sidecar

The .md file has four top-level sections in order:

    CONTEXT / TASK LIST / CONSTRAINTS / ACCEPTANCE CRITERIA

Sections match the uploaded session file's formatting exactly so the
Grinder's schema validator passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from belief.photosynthesis.synthesis.generator import GoalSpec


DEFAULT_PENDING_DIR = Path("/var/lib/photosynthesis/pending_sessions")


def render_session_markdown(spec: GoalSpec) -> str:
    """Deterministic GoalSpec -> Belief Engine v3.0 session .md text.

    Byte-exact determinism is a constraint: this function must never
    emit time-dependent text (no datestamps, no random UUIDs). The
    existing golden-file test relies on it.
    """
    new_libs = spec.new_libraries_introduced
    primary_libs = spec.primary_libraries

    ac_lines = []
    for i, ac in enumerate(spec.acceptance_criteria, start=1):
        ac_lines.append(f"{i}. [{ac.kind}] {ac.spec}")

    prereq = ", ".join(spec.prerequisite_skills) or "(none)"
    primary = ", ".join(primary_libs) or "(none)"
    new_libs_line = ", ".join(new_libs) or "(none)"

    lines = [
        f"# SESSION: {spec.goal_id}",
        f"# Estimated: {spec.estimated_build_time_min} minutes,"
        f" difficulty {spec.estimated_difficulty}/5",
        f"# Source: {spec.source_citation}",
        "",
        "CONTEXT",
        spec.one_paragraph_description,
        "",
        f"Relevance: {spec.relevance_rationale}",
        f"Novelty: {spec.novelty_rationale}",
        "",
        f"Artifact type: {spec.artifact_type}",
        f"Primary libraries: {primary}",
        f"New libraries introduced: {new_libs_line}",
        f"Prerequisite skills: {prereq}",
        "",
        "TASK LIST",
        "",
        f"1. Produce a {spec.artifact_type} named {spec.goal_id} that implements:",
        f"   {spec.title}",
        "",
        "2. For each acceptance criterion below, implement the behavior and add",
        "   a matching test (when kind=test) or endpoint/artifact verification.",
        "",
        "3. Package the deliverable in the conventional Belief Engine layout",
        "   (entry point, README, pyproject.toml) using the primary libraries.",
        "",
        "CONSTRAINTS",
        "- Python 3.14",
        f"- Target build time: {spec.estimated_build_time_min} minutes",
        f"- Difficulty tier: {spec.estimated_difficulty}/5",
        "- Must be buildable in one autonomous session",
        "- All acceptance criteria objectively checkable",
        "",
        "ACCEPTANCE CRITERIA",
    ]
    lines.extend(ac_lines)
    lines.append("")  # trailing newline
    return "\n".join(lines)


def write_session(
    spec: GoalSpec,
    *,
    pending_dir: Path = DEFAULT_PENDING_DIR,
) -> tuple[Path, Path]:
    """Write {goal_id}.md and {goal_id}.json to `pending_dir`.

    Returns (md_path, json_path). Idempotent: overwriting an existing
    pair yields byte-identical output for the same GoalSpec (renderer
    determinism + Pydantic's stable JSON ordering).
    """
    pending_dir = Path(pending_dir)
    pending_dir.mkdir(parents=True, exist_ok=True)

    md_path = pending_dir / f"{spec.goal_id}.md"
    json_path = pending_dir / f"{spec.goal_id}.json"

    md_text = render_session_markdown(spec)
    md_path.write_text(md_text, encoding="utf-8")

    sidecar: dict[str, Any] = json.loads(spec.model_dump_json())
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return md_path, json_path


__all__ = [
    "DEFAULT_PENDING_DIR",
    "render_session_markdown",
    "write_session",
]
