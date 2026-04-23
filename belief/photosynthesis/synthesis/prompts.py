"""Module-level prompt constants for Session-4 synthesis.

Per the spec constraint "All prompts must be embedded as module-level
constants (not f-strings inside functions) for auditability." Every
call site formats the constant with .format(...) — never rebuilds it.

The research doc's §4.2 generator template isn't included in the
uploaded material; GENERATOR_PROMPT is a spec-faithful stand-in that
covers every required output JSON field. Feel free to replace it with
the canonical version — no caller inspects internal structure.
"""


# ---------------------------------------------------------------------------
# Novelty — OMNI-EPIC interestingness judge (Haiku 4.5)
# ---------------------------------------------------------------------------

INTERESTINGNESS_PROMPT = """\
You are deciding whether a proposed Python build goal is INTERESTINGLY
NOVEL compared to an existing archive of built/attempted goals.
Domain: autonomous Python coding agent targeting LangGraph, MCP,
ChromaDB, asyncio, FastAPI, RAG pipelines, Pydantic v2.

PROPOSED GOAL:
Title: {title}
Summary: {summary}
Source: {source} ({source_id})
Tags: {domain_tags}

5 MOST SIMILAR EXISTING GOALS (cosine shown):
{neighbors_formatted}

INTERESTING if (a) new library/API surface, (b) combines >=2 archived
capabilities in a new way, (c) repairs a failed goal's documented
failure mode, or (d) reflects a new research pattern.
NOT INTERESTING if (x) parametric variation (different DB, model
name, endpoint path) with same control-flow shape, (y) top-1
subsumes it, (z) novelty is surface-level (rename, refactor).

Return strict JSON:
{{"interesting": bool,
  "category": "a"|"b"|"c"|"d"|"x"|"y"|"z",
  "nearest_archived_goal_id": str|null,
  "one_line_justification": str (<=30 words)}}
"""


# ---------------------------------------------------------------------------
# Difficulty — ZPD-fit predicted build time (Haiku 4.5, max_tokens=150)
# ---------------------------------------------------------------------------

DIFFICULTY_PROMPT = """\
You are estimating how long an autonomous Python coding agent would
take to build the following goal. The agent is competent at LangGraph,
MCP, FastAPI, Pydantic v2, asyncio, and ChromaDB.

GOAL:
Title: {title}
Summary: {summary}
Prerequisite skills (cosine>=0.6 against skill library): {skills_found}
Estimated required skills: {estimated_skills_needed}

Return strict JSON:
{{"pred_time_min": int,    // minutes, 5..240
  "reasoning": str (<=40 words)}}
"""


# ---------------------------------------------------------------------------
# Generator — SPEC-FAITHFUL STAND-IN for the research-doc template.
# Ships a complete GoalSpec JSON per the Session 4 task list.
# ---------------------------------------------------------------------------

GENERATOR_PROMPT = """\
You are the Goal Synthesizer for the Belief Engine. Your job is to
convert a filtered research signal into a precise, buildable goal spec
the engine's autonomous pipeline can execute.

The Belief Engine targets Python 3.14. Its agents are strong at
LangGraph, MCP / FastMCP, FastAPI, Pydantic v2, asyncio, ChromaDB,
httpx, Click/Typer. Brownfield sessions edit existing code; greenfield
sessions produce a complete new package.

RAW SIGNAL:
Title: {title}
Summary: {summary}
Raw excerpt: {raw_excerpt}
Source: {source} ({source_id})
Domain tags: {domain_tags}

NEAREST ARCHIVED GOALS (for framing contrast, do NOT copy):
{neighbors_formatted}

NOVELTY RATIONALE (from judge): {novelty_rationale}
PREDICTED BUILD TIME: ~{pred_time_min} minutes
ZPD FIT: {zpd_fit:.2f}

Produce a JSON object with EXACTLY these keys:

{{
  "goal_id":                       string, slugified title,
  "title":                         string, <=80 chars,
  "one_paragraph_description":     string, <=500 chars,
  "artifact_type":                 "cli" | "api" | "library" | "mcp_server" | "pipeline" | "script",
  "primary_libraries":             list[str],
  "new_libraries_introduced":      list[str],
  "acceptance_criteria":           list of {{"kind": "test"|"endpoint"|"behavior"|"artifact", "spec": str}},
  "estimated_build_time_min":      int, match the predicted time above,
  "estimated_difficulty":          int 1..5,
  "prerequisite_skills":           list[str],
  "relevance_rationale":           string, why this matters for the engine now,
  "novelty_rationale":             string, why this isn't a duplicate,
  "source_citation":               string, url or source_id
}}

Constraints:
- Goal must be buildable in one autonomous session (<= 4 hours).
- Every acceptance_criterion must be objectively checkable.
- Do not reference proprietary or nonexistent libraries.
- Return JSON only, no prose before or after.
"""


# ---------------------------------------------------------------------------
# A one-line formatter for neighbor rows injected into prompts.
# ---------------------------------------------------------------------------


def format_neighbors(neighbors: list[dict]) -> str:
    """Turn a list of {goal_id, title, cosine, ...} dicts into prompt text.

    Each line: "  goal_id  cosine=0.84  Title here".
    If neighbors is empty, returns "  (archive empty — no neighbors found)".
    """
    if not neighbors:
        return "  (archive empty — no neighbors found)"
    lines = []
    for n in neighbors[:5]:
        lines.append(
            f"  {n.get('goal_id', '?')}  cosine={float(n.get('cosine', 0)):.3f}  "
            f"{n.get('title', '(no title)')}"
        )
    return "\n".join(lines)


__all__ = [
    "DIFFICULTY_PROMPT",
    "GENERATOR_PROMPT",
    "INTERESTINGNESS_PROMPT",
    "format_neighbors",
]
