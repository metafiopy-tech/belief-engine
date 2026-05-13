"""Cross-domain intake adapter (Synthesis Engine Session 7).

Pure-function adapter that surfaces a :class:`StructuralMechanism`
into a :class:`RequirementSpec` so that downstream agents in the
Belief Engine build pipeline (research / planner / architect / builder)
see the mechanism's predicate signature, relational wiring, near-miss
disambiguation, and any unresolved incompleteness probes as plain
strings inside the spec's ``constraints`` and ``acceptance_criteria``
lists.

Why this lives in ``belief/agents/`` rather than next to the synthesis
modules: the adapter is part of the build pipeline's input-shaping
stage. The synthesis layer's job ends when it emits a
StructuralMechanism + GoalSpec; the adapter is the bridge piece the
intake agent invokes when it spots one in state.

Design notes:

  - **Pure function.** ``apply_to`` does not mutate its input spec.
    It returns a new ``RequirementSpec`` via ``model_copy(update=...)``.
    Callers that need the mutation in place must rebind.

  - **No LLM calls.** The adapter is deterministic string-formatting
    only. Tests can pin exact output without mocking anything.

  - **Idempotent within a build.** Re-applying the adapter to a spec
    that already has these constraints will duplicate them; the intake
    agent only invokes it once per build, so this is fine. If you want
    re-application safety, dedupe in the caller.

  - **Open probes become TODO constraints.** Each
    ``IncompletenessProbe`` in ``incompleteness_probes_open`` becomes a
    constraint string prefixed with ``"TODO (probe_id @ field):"`` so
    grep-style scans for unresolved work succeed.

Out of scope for Session 7:
  - Probe-aware test generation (probe -> test stub).
  - Auto-resolving probes by re-running the synthesis loop's research
    dispatcher with the spec in hand.
  - Adapter chaining for multiple structural mechanisms in one spec.
"""

from __future__ import annotations

from belief.models.artifacts import RequirementSpec
from belief.photosynthesis.synthesis.structural_mechanism import StructuralMechanism


# ---------------------------------------------------------------------------
# String-formatting helpers (kept module-private so the public API is one
# function and the format is easy to change in one place).
# ---------------------------------------------------------------------------


def _format_predicate_constraint(mechanism: StructuralMechanism) -> str:
    pred = mechanism.predicate_in_source  # source / target share signature
    roles = ", ".join(pred.roles)
    return (
        f"must implement predicate '{pred.name}' "
        f"(arity={pred.arity}, marr_level={pred.marr_level}) "
        f"with role arguments: {roles}"
    )


def _format_higher_order_constraint(name: str, relates: list[str]) -> str:
    related = ", ".join(relates)
    return f"must wire '{name}' relation across predicates: {related}"


def _format_near_miss_constraint(mechanism: StructuralMechanism) -> str:
    nm = mechanism.near_miss
    return f"must distinguish from near-miss (breaks at {nm.breaks_at_argument}): {nm.description}"


def _format_probe_constraint(probe_id: str, references_field: str, question: str) -> str:
    return f"TODO ({probe_id} @ {references_field}): {question}"


def _format_predicate_acceptance(mechanism: StructuralMechanism) -> str:
    pred = mechanism.predicate_in_source
    return (
        f"The implementation exhibits predicate '{pred.name}' as a "
        f"distinguishable callable or structure with arity {pred.arity}, "
        f"and the {pred.marr_level}-level role-arguments "
        f"({', '.join(pred.roles)}) are recoverable from the code."
    )


def _format_open_probes_acceptance(n_open: int) -> str:
    return (
        f"All {n_open} open implementation probes from the structural "
        "mechanism are either addressed in code, surfaced as failing "
        "tests, or explicitly deferred with a rationale comment."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_to(spec: RequirementSpec, mechanism: StructuralMechanism) -> RequirementSpec:
    """Return a new ``RequirementSpec`` with mechanism content injected.

    The returned spec is identical to the input except that
    ``constraints`` and ``acceptance_criteria`` have additional entries
    derived from the structural mechanism.

    Constraint additions, in order:
      1. Predicate signature constraint (one entry).
      2. One constraint per higher-order relation.
      3. Near-miss disambiguation constraint (one entry).
      4. One ``TODO (...)`` constraint per open incompleteness probe.

    Acceptance-criteria additions:
      1. Predicate-exhibition criterion (always added).
      2. Open-probes coverage criterion (only added when there is at
         least one probe in ``incompleteness_probes_open``).

    Does not mutate ``spec``. Does not call out to any I/O.
    """
    new_constraints: list[str] = list(spec.constraints)
    new_constraints.append(_format_predicate_constraint(mechanism))
    for hor in mechanism.higher_order_relations:
        new_constraints.append(_format_higher_order_constraint(hor.name, list(hor.relates)))
    new_constraints.append(_format_near_miss_constraint(mechanism))
    for probe in mechanism.incompleteness_probes_open:
        new_constraints.append(
            _format_probe_constraint(probe.probe_id, probe.references_field, probe.question)
        )

    new_acceptance: list[str] = list(spec.acceptance_criteria)
    new_acceptance.append(_format_predicate_acceptance(mechanism))
    if mechanism.incompleteness_probes_open:
        new_acceptance.append(
            _format_open_probes_acceptance(len(mechanism.incompleteness_probes_open))
        )

    return spec.model_copy(
        update={
            "constraints": new_constraints,
            "acceptance_criteria": new_acceptance,
        }
    )


__all__ = ["apply_to"]
