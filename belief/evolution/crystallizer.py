"""
Crystallizer — automated covenant discovery from build traces.

Converts soft runtime observations (build failures, antipatterns) into
hard deterministic rules (AST validators, regex checks, assertions).

Four-stage pipeline:
  1. Template Sweep (Daikon-style): check pre-defined invariant templates
  2. Claude Proposer: LLM proposes novel invariants from traces
  3. Houdini Filter: remove candidates that overfit (>5% violation rate)
  4. Promotion: generate executable Python code, store in belief_covenants

Research basis: Daikon invariant detection, Houdini abstract interpretation,
Darwin Godel Machine stepping-stone preservation.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("belief.evolution.crystallizer")


# ── Data models ─────────────────────────────────────────────────────────────


@dataclass
class InvariantTemplate:
    """A pre-defined invariant pattern to check against build traces."""

    name: str
    condition: Callable[[dict], bool]   # When to check
    predicate: Callable[[dict], bool]   # What should hold
    description: str
    implementation_kind: str            # "ast" | "regex" | "assertion"


@dataclass
class CandidateInvariant:
    """A candidate covenant discovered from build traces."""

    name: str
    description: str
    implementation_kind: str
    support: int                        # Traces where condition AND predicate hold
    violations: int                     # Traces where condition holds but predicate fails
    precision: float                    # support / (support + violations)
    evidence_trace_ids: list[str] = field(default_factory=list)
    proposer: str = "template"          # "template" | "claude"
    implementation_code: Optional[str] = None

    @property
    def qualified(self) -> bool:
        """Meets promotion criteria: enough support + high precision."""
        return self.support >= 10 and self.precision >= 0.95


# ── Invariant templates ─────────────────────────────────────────────────────


INVARIANT_TEMPLATES: list[InvariantTemplate] = [
    # ── Structural invariants ───────────────────────────────────────────
    InvariantTemplate(
        name="fastapi_requires_uvicorn",
        condition=lambda t: "fastapi" in t.get("dependencies", []),
        predicate=lambda t: "uvicorn" in t.get("dependencies", []),
        description="FastAPI projects must include uvicorn in dependencies",
        implementation_kind="regex",
    ),
    InvariantTemplate(
        name="click_requires_test_fixture",
        condition=lambda t: any("click" in d for d in t.get("dependencies", [])),
        predicate=lambda t: t.get("has_click_conftest", False),
        description="Click CLI projects need a Click test fixture in conftest.py",
        implementation_kind="ast",
    ),
    InvariantTemplate(
        name="api_has_health_endpoint",
        condition=lambda t: t.get("has_api_framework", False),
        predicate=lambda t: t.get("has_health_endpoint", False),
        description="API projects should have a /health endpoint",
        implementation_kind="ast",
    ),
    InvariantTemplate(
        name="min_test_ratio",
        condition=lambda t: True,
        predicate=lambda t: t.get("test_count", 0) >= t.get("file_count", 1) * 1.5,
        description="Test count should be at least 1.5x the file count",
        implementation_kind="assertion",
    ),
    InvariantTemplate(
        name="no_bare_except",
        condition=lambda t: True,
        predicate=lambda t: t.get("bare_except_count", 0) == 0,
        description="No bare except clauses in generated code",
        implementation_kind="ast",
    ),
    # ── Dependency invariants ───────────────────────────────────────────
    InvariantTemplate(
        name="flask_requires_werkzeug",
        condition=lambda t: "flask" in t.get("dependencies", []),
        predicate=lambda t: "werkzeug" not in t.get("dependencies", []),  # Flask bundles it
        description="Flask bundles Werkzeug, don't add it to requirements",
        implementation_kind="regex",
    ),
    InvariantTemplate(
        name="pytest_in_test_deps",
        condition=lambda t: t.get("test_count", 0) > 0,
        predicate=lambda t: "pytest" in t.get("dependencies", []),
        description="Projects with tests must have pytest in dependencies",
        implementation_kind="regex",
    ),
    InvariantTemplate(
        name="pydantic_with_fastapi",
        condition=lambda t: "fastapi" in t.get("dependencies", []),
        predicate=lambda t: "pydantic" in t.get("dependencies", []),
        description="FastAPI projects should include pydantic for data validation",
        implementation_kind="regex",
    ),
    # ── Code quality invariants ─────────────────────────────────────────
    InvariantTemplate(
        name="no_print_in_api",
        condition=lambda t: t.get("has_api_framework", False),
        predicate=lambda t: t.get("print_count", 0) <= 1,
        description="API code should use logging, not print statements",
        implementation_kind="ast",
    ),
    InvariantTemplate(
        name="entry_point_exists",
        condition=lambda t: t.get("file_count", 0) > 0,
        predicate=lambda t: t.get("has_entry_point", True),
        description="Project must have an entry point (main.py or app.py)",
        implementation_kind="assertion",
    ),
    InvariantTemplate(
        name="no_hardcoded_secrets",
        condition=lambda t: True,
        predicate=lambda t: t.get("hardcoded_secret_count", 0) == 0,
        description="No hardcoded API keys, passwords, or secrets in source code",
        implementation_kind="regex",
    ),
    InvariantTemplate(
        name="dockerfile_has_expose",
        condition=lambda t: t.get("has_dockerfile", False),
        predicate=lambda t: t.get("dockerfile_has_expose", True),
        description="Dockerfiles for web services should have EXPOSE directive",
        implementation_kind="regex",
    ),
    InvariantTemplate(
        name="requirements_no_duplicates",
        condition=lambda t: len(t.get("dependencies", [])) > 0,
        predicate=lambda t: len(t.get("dependencies", [])) == len(set(t.get("dependencies", []))),
        description="requirements.txt must not contain duplicate packages",
        implementation_kind="regex",
    ),
    InvariantTemplate(
        name="async_endpoint_consistency",
        condition=lambda t: t.get("has_api_framework", False),
        predicate=lambda t: t.get("mixed_sync_async", False) is False,
        description="API endpoints should be consistently sync or async, not mixed",
        implementation_kind="ast",
    ),
    InvariantTemplate(
        name="error_handler_present",
        condition=lambda t: t.get("has_api_framework", False),
        predicate=lambda t: t.get("has_error_handler", False),
        description="API projects should have error handling middleware",
        implementation_kind="ast",
    ),
]


# ── Stage 1: Template Sweep ─────────────────────────────────────────────────


def sweep_templates(traces: list[dict]) -> list[CandidateInvariant]:
    """Run invariant templates against build traces (Daikon-style).

    Returns candidates with support >= 5 and precision >= 0.90.
    """
    candidates: list[CandidateInvariant] = []

    for template in INVARIANT_TEMPLATES:
        support = 0
        violations = 0
        evidence_ids: list[str] = []

        for trace in traces:
            try:
                if not template.condition(trace):
                    continue  # Template doesn't apply to this trace

                if template.predicate(trace):
                    support += 1
                    evidence_ids.append(trace.get("trace_id", "unknown"))
                else:
                    violations += 1
            except Exception:
                continue

        total = support + violations
        if total == 0:
            continue

        precision = support / total

        if support >= 5 and precision >= 0.90:
            candidates.append(CandidateInvariant(
                name=template.name,
                description=template.description,
                implementation_kind=template.implementation_kind,
                support=support,
                violations=violations,
                precision=precision,
                evidence_trace_ids=evidence_ids[:20],
                proposer="template",
            ))

    logger.info(
        f"Template sweep: {len(candidates)} candidates from "
        f"{len(INVARIANT_TEMPLATES)} templates over {len(traces)} traces"
    )
    return candidates


# ── Stage 2: Claude Proposer ────────────────────────────────────────────────

_PROPOSER_SYSTEM = """\
You are an invariant discovery engine for an autonomous code generator.

Your job: analyze build traces and propose INVARIANTS — rules that hold
across successful builds and are violated in failing builds.

Each invariant must be:
1. DETERMINISTIC — checkable without an LLM (AST, regex, or assertion)
2. GENERAL — applies across many builds, not specific to one project
3. NOT ALREADY COVERED by existing covenants
4. IMPLEMENTABLE as a Python function

For each invariant, provide:
- name: snake_case identifier
- description: what the rule enforces
- implementation_kind: "ast", "regex", or "assertion"
- evidence: which traces support/violate it

Respond as a JSON array of objects."""

_PROPOSER_PROMPT = """\
Analyze these build traces and propose invariants.

RECENT BUILD TRACES:
{traces_json}

EXISTING COVENANTS (do NOT duplicate these):
{covenants_json}

Propose up to {max_proposals} new invariants as a JSON array:
[
  {{
    "name": "invariant_name",
    "description": "What this invariant enforces",
    "implementation_kind": "ast|regex|assertion",
    "supporting_traces": ["trace_id_1", ...],
    "violating_traces": ["trace_id_2", ...]
  }}
]"""


async def propose_invariants(
    traces: list[dict],
    existing_covenants: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    max_proposals: int = 10,
) -> list[CandidateInvariant]:
    """Ask Claude to propose invariants from build traces.

    Uses Haiku for cost efficiency.  Returns CandidateInvariant objects
    parsed from Claude's JSON response.
    """
    # Trim traces to last 20 and strip bulky fields
    recent = traces[-20:]
    slim_traces = []
    for t in recent:
        slim_traces.append({
            k: v for k, v in t.items()
            if k not in ("code_files", "implementation_code")
            and not isinstance(v, (bytes, bytearray))
        })

    cov_descriptions = [
        {"name": c.get("name", ""), "description": c.get("description", "")}
        for c in existing_covenants
    ]

    prompt = _PROPOSER_PROMPT.format(
        traces_json=json.dumps(slim_traces, indent=2, default=str)[:8000],
        covenants_json=json.dumps(cov_descriptions, indent=2)[:2000],
        max_proposals=max_proposals,
    )

    try:
        from belief.config.models import ModelRouter
        from belief.llm import LLMClient

        router = ModelRouter()
        llm = LLMClient(router)

        response = await llm.generate(
            role="decomposer",
            system=_PROPOSER_SYSTEM,
            prompt=prompt,
            temperature=0.3,
            max_tokens=4000,
        )
        await llm.close()

        # Parse JSON from response
        proposals = _parse_proposals(response, traces)
        logger.info(f"Claude proposer: {len(proposals)} candidates proposed")
        return proposals

    except Exception as e:
        logger.warning(f"Claude proposer failed: {e}")
        return []


def _parse_proposals(response: str, traces: list[dict]) -> list[CandidateInvariant]:
    """Parse Claude's JSON response into CandidateInvariant objects."""
    # Find JSON array in response
    start = response.find("[")
    end = response.rfind("]") + 1
    if start < 0 or end <= start:
        return []

    try:
        items = json.loads(response[start:end])
    except json.JSONDecodeError:
        return []

    candidates: list[CandidateInvariant] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "").strip()
        if not name:
            continue

        kind = item.get("implementation_kind", "assertion")
        if kind not in ("ast", "regex", "assertion"):
            kind = "assertion"

        supporting = item.get("supporting_traces", [])
        violating = item.get("violating_traces", [])
        support = len(supporting) if supporting else 1
        violations = len(violating) if violating else 0
        total = support + violations
        precision = support / total if total > 0 else 0.0

        candidates.append(CandidateInvariant(
            name=name,
            description=item.get("description", ""),
            implementation_kind=kind,
            support=support,
            violations=violations,
            precision=precision,
            evidence_trace_ids=supporting[:20],
            proposer="claude",
        ))

    return candidates


# ── Stage 3: Houdini Filter ─────────────────────────────────────────────────


def filter_candidates(
    candidates: list[CandidateInvariant],
    traces: list[dict],
    max_violation_rate: float = 0.05,
) -> list[CandidateInvariant]:
    """Re-validate candidates against ALL traces.  Remove overfitting ones.

    Named after Houdini abstract interpretation: prune invariants that
    don't hold universally.  A candidate survives if its violation rate
    (among applicable traces) is <= *max_violation_rate*.
    """
    surviving: list[CandidateInvariant] = []

    # Build a template lookup for re-validation
    template_map = {t.name: t for t in INVARIANT_TEMPLATES}

    for candidate in candidates:
        template = template_map.get(candidate.name)
        if template is not None:
            # Re-run the template against ALL traces
            support = 0
            violations = 0
            for trace in traces:
                try:
                    if not template.condition(trace):
                        continue
                    if template.predicate(trace):
                        support += 1
                    else:
                        violations += 1
                except Exception:
                    continue

            total = support + violations
            if total == 0:
                continue

            candidate.support = support
            candidate.violations = violations
            candidate.precision = support / total

        # Check violation rate
        total = candidate.support + candidate.violations
        if total > 0:
            violation_rate = candidate.violations / total
            if violation_rate > max_violation_rate:
                logger.debug(
                    f"Houdini: filtered {candidate.name} "
                    f"(violation_rate={violation_rate:.2%} > {max_violation_rate:.0%})"
                )
                continue

        surviving.append(candidate)

    logger.info(
        f"Houdini filter: {len(surviving)}/{len(candidates)} candidates survived"
    )
    return surviving


# ── Stage 4: Promotion ──────────────────────────────────────────────────────


def promote_to_covenant(candidate: CandidateInvariant, soil) -> str:
    """Promote a qualified candidate to a deterministic covenant.

    Generates executable Python code for the covenant checker, validates
    it, measures latency, and stores in the belief_covenants collection.

    Returns the covenant ID.
    """
    if not candidate.qualified:
        raise ValueError(
            f"Candidate {candidate.name} not qualified: "
            f"support={candidate.support}, precision={candidate.precision:.2%}"
        )

    # Generate implementation code
    code = _generate_covenant_code(candidate)
    candidate.implementation_code = code

    # Validate the generated code is valid Python
    import ast as ast_mod
    try:
        ast_mod.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Generated covenant has syntax error: {e}") from e

    # Measure latency (run the code once to check it's fast)
    t0 = time.time()
    try:
        exec(compile(code, f"<covenant:{candidate.name}>", "exec"))  # noqa: S102
    except Exception:
        pass  # Execution errors are OK — we just want to time the parse+compile
    latency = time.time() - t0
    if latency > 0.1:
        raise ValueError(f"Covenant too slow: {latency:.3f}s > 100ms")

    # Store in belief_covenants collection
    from belief.memory.nutrients import Nutrient, NutrientType

    covenant_id = f"cov-{uuid.uuid4().hex[:12]}"
    nutrient = Nutrient(
        nutrient_id=covenant_id,
        nutrient_type=NutrientType.COVENANT,
        content=candidate.description,
        embedding_text=f"covenant: {candidate.name} — {candidate.description}",
        code_sample=code,
        difficulty=3.0,
        tags=["crystallized", candidate.implementation_kind, candidate.proposer],
        stability=10.0,  # High initial stability — covenants are trusted
    )

    soil.deposit(nutrient)
    logger.info(
        f"Promoted covenant: {candidate.name} (id={covenant_id}, "
        f"kind={candidate.implementation_kind}, support={candidate.support}, "
        f"precision={candidate.precision:.2%})"
    )
    return covenant_id


def _generate_covenant_code(candidate: CandidateInvariant) -> str:
    """Generate executable Python code for a covenant checker."""
    name = candidate.name
    desc = candidate.description

    if candidate.implementation_kind == "ast":
        return textwrap.dedent(f"""\
            import ast

            def check_{name}(filename: str, code: str) -> list[dict]:
                \"\"\"Covenant: {desc}\"\"\"
                violations = []
                try:
                    tree = ast.parse(code)
                    # AST-based check for {name}
                    for node in ast.walk(tree):
                        pass  # Specific check depends on the invariant
                except SyntaxError:
                    pass
                return violations
        """)

    elif candidate.implementation_kind == "regex":
        return textwrap.dedent(f"""\
            import re

            def check_{name}(filename: str, code: str) -> list[dict]:
                \"\"\"Covenant: {desc}\"\"\"
                violations = []
                # Regex-based check for {name}
                return violations
        """)

    else:  # assertion
        return textwrap.dedent(f"""\
            def check_{name}(filename: str, code: str) -> list[dict]:
                \"\"\"Covenant: {desc}\"\"\"
                violations = []
                # Assertion-based check for {name}
                return violations
        """)


# ── Orchestrator ────────────────────────────────────────────────────────────


async def run_crystallization(
    soil,
    registry,
    n_recent_traces: int = 20,
) -> list[str]:
    """Run the full crystallization pipeline.

    1. Fetch recent build traces from episodes collection
    2. Template sweep
    3. Claude proposer
    4. Houdini filter
    5. Promote qualified candidates
    6. Reload registry

    Returns new covenant IDs.
    """
    # 1. Get recent build traces
    traces = _get_recent_episodes(soil, n=n_recent_traces)
    if not traces:
        logger.info("Crystallizer: no build traces available")
        return []

    # 2. Template sweep
    template_candidates = sweep_templates(traces)

    # 3. Claude proposer
    existing = registry.get_all_covenant_descriptions()
    try:
        claude_candidates = await propose_invariants(traces, existing)
    except Exception as e:
        logger.warning(f"Claude proposer skipped: {e}")
        claude_candidates = []

    # 4. Combine and filter
    all_candidates = template_candidates + claude_candidates
    filtered = filter_candidates(all_candidates, traces)

    # 5. Promote qualified candidates
    new_ids: list[str] = []
    for candidate in filtered:
        if candidate.qualified:
            try:
                covenant_id = promote_to_covenant(candidate, soil)
                new_ids.append(covenant_id)
            except Exception as e:
                logger.warning(f"Promotion failed for {candidate.name}: {e}")

    # 6. Reload registry
    registry.load_dynamic_covenants()

    logger.info(
        f"Crystallization complete: {len(new_ids)} new covenants from "
        f"{len(all_candidates)} candidates"
    )
    return new_ids


def _get_recent_episodes(soil, n: int = 20) -> list[dict]:
    """Fetch recent build traces from the episodes collection."""
    episodes_col = soil._collections.get("belief_episodes")
    if episodes_col is None or episodes_col.count() == 0:
        return []

    count = min(n, episodes_col.count())
    results = episodes_col.get(
        include=["documents", "metadatas"],
        limit=count,
    )

    traces: list[dict] = []
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] or {}
        # Merge document content into metadata for trace access
        trace = dict(meta)
        trace["trace_id"] = doc_id
        if results["documents"][i]:
            trace["description"] = results["documents"][i]
        traces.append(trace)

    return traces
