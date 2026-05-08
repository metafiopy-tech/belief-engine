"""StructuralMechanism schema -- Synthesis Engine Session 1.

This module defines the Pydantic payload that the cross-domain
synthesizer (Session 3) emits and that ``GoalSpec`` optionally carries
through to the existing Belief Engine build pipeline.

Design intent (from the Synthesis Engine plan):

  - ``PredicateInstance`` is the unit of cross-domain isomorphism: an
    n-ary predicate at a Marr level (computational / algorithmic /
    implementation).  Source and target instances must share the same
    abstract signature -- that is the structural-isomorphism claim.

  - ``HigherOrderRelation`` wires predicates to other predicates.  A
    mechanism whose predicate isn't wired into any higher-order relation
    is decorative; the predicate is attribute-only and the schema
    rejects it.

  - ``NearMiss`` carries a counterexample mechanism whose
    ``breaks_at_argument`` pinpoints the specific predicate slot where
    the analogy fails -- load-bearing for the incompleteness pass
    (Session 6).

  - ``considered_and_rejected_attributes`` records the surface-level
    attributes the synthesizer rejected in favor of the mechanism so
    the trace of "what we noticed but discarded" survives review.

The schema enforces structural-isomorphism at validation time.  No
storage / generation logic lives here -- those are Sessions 3 / 4.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Marr's three levels of analysis
# ---------------------------------------------------------------------------

MarrLevel = Literal["computational", "algorithmic", "implementation"]


# Argument reference format used by NearMiss.breaks_at_argument and (later)
# the incompleteness probes' references_field.  Format:
#   predicate_in_source.argument[N]  or  predicate_in_target.argument[N]
_ARG_REF_RE = re.compile(r"^predicate_in_(source|target)\.argument\[(\d+)\]$")

_PREDICATE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PredicateInstance(BaseModel):
    """An n-ary predicate at a Marr level.

    ``name`` is the abstract predicate identifier (snake_case, no
    spaces).  ``roles`` names each argument position; ``len(roles) ==
    arity`` is enforced.  ``marr_level`` ties the predicate to one of
    Marr's three levels of analysis -- source and target predicates
    must agree on this level (enforced at the StructuralMechanism
    layer, not here).
    """

    name: str
    arity: int = Field(ge=1)
    roles: list[str] = Field(min_length=1)
    marr_level: MarrLevel

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _PREDICATE_NAME_RE.match(v):
            raise ValueError(
                "predicate name must be snake_case (lowercase letters, digits, "
                "underscore; must start with a letter)"
            )
        return v

    @model_validator(mode="after")
    def _roles_match_arity(self) -> "PredicateInstance":
        if len(self.roles) != self.arity:
            raise ValueError(f"roles length ({len(self.roles)}) must equal arity ({self.arity})")
        return self


class DomainEvidence(BaseModel):
    """A citation tying a predicate's claim to a specific domain.

    The synthesizer should produce at least one evidence entry per
    {source, target} domain so the structural claim is grounded in
    retrievable text.  This is not enforced at the schema level
    (Session 4 owns retrieval and grounding); the field is present
    here so Session 3's generator has somewhere to attach citations.
    """

    domain: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class HigherOrderRelation(BaseModel):
    """A relation BETWEEN predicates (not between attributes).

    ``name`` describes the relation (e.g., ``reduces_downstream_compute``,
    ``causes``, ``constrains``).  ``relates`` lists the predicate names
    being related -- a higher-order relation must connect at least two
    distinct predicates; if it doesn't, the relation is a tautology
    and rejected.
    """

    name: str = Field(min_length=1)
    relates: list[str] = Field(min_length=2)

    @field_validator("relates")
    @classmethod
    def _distinct_predicates(cls, v: list[str]) -> list[str]:
        if len(set(v)) < 2:
            raise ValueError("higher-order relation must connect at least two DISTINCT predicates")
        return v


class NearMiss(BaseModel):
    """A counterexample mechanism that almost fits but breaks at a slot.

    ``breaks_at_argument`` points to a specific predicate argument
    position using the format
    ``predicate_in_(source|target).argument[N]`` so downstream consumers
    (incompleteness probes in Session 6) can reason about exactly which
    slot fails.  The index is validated against the predicate's arity
    at the StructuralMechanism layer.
    """

    description: str = Field(min_length=1)
    breaks_at_argument: str

    @field_validator("breaks_at_argument")
    @classmethod
    def _valid_arg_ref(cls, v: str) -> str:
        if not _ARG_REF_RE.match(v):
            raise ValueError(
                "breaks_at_argument must match the format "
                "'predicate_in_(source|target).argument[N]'"
            )
        return v


# ---------------------------------------------------------------------------
# Top-level structural mechanism
# ---------------------------------------------------------------------------


class StructuralMechanism(BaseModel):
    """The cross-domain isomorphism payload.

    Validation enforces, in order:

      1. Source and target predicates share name, arity, roles, and
         Marr level -- this is the structural-isomorphism claim.
      2. At least one ``HigherOrderRelation`` is present, and the
         shared predicate name appears in some relation's ``relates``
         list (catches attribute-only predicates that aren't wired
         into the relational structure).
      3. ``near_miss.breaks_at_argument`` indexes a real argument
         position (within the shared ``predicate.arity``).
      4. At least two ``considered_and_rejected_attributes`` are
         recorded so the trace of surface attributes the synthesizer
         saw and discarded survives review.
    """

    mechanism_id: str = Field(min_length=1)
    source_domain: str = Field(min_length=1)
    target_domain: str = Field(min_length=1)
    predicate_in_source: PredicateInstance
    predicate_in_target: PredicateInstance
    higher_order_relations: list[HigherOrderRelation] = Field(min_length=1)
    near_miss: NearMiss
    considered_and_rejected_attributes: list[str] = Field(min_length=2)
    domain_evidence: list[DomainEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _signatures_match(self) -> "StructuralMechanism":
        s, t = self.predicate_in_source, self.predicate_in_target
        if s.name != t.name:
            raise ValueError(
                f"predicate name mismatch: source='{s.name}' target='{t.name}' "
                "(structural isomorphism requires shared name)"
            )
        if s.arity != t.arity:
            raise ValueError(f"predicate arity mismatch: source={s.arity} target={t.arity}")
        if list(s.roles) != list(t.roles):
            raise ValueError(f"predicate roles mismatch: source={s.roles!r} target={t.roles!r}")
        if s.marr_level != t.marr_level:
            raise ValueError(
                f"predicate Marr level mismatch: source={s.marr_level} target={t.marr_level}"
            )
        return self

    @model_validator(mode="after")
    def _predicate_is_relationally_wired(self) -> "StructuralMechanism":
        """Reject attribute-only predicates.

        If neither source nor target predicate name appears in any
        higher-order relation's ``relates`` list, the predicate is
        decorative -- it might describe an attribute of the domain but
        it doesn't participate in the relational structure that makes
        the mechanism a mechanism.
        """
        related: set[str] = set()
        for hor in self.higher_order_relations:
            related.update(hor.relates)
        # Source and target share .name (enforced by _signatures_match).
        if self.predicate_in_source.name not in related:
            raise ValueError(
                f"predicate '{self.predicate_in_source.name}' must appear in at least "
                "one higher_order_relation.relates -- attribute-only predicates that "
                "aren't wired into the relational structure don't constitute a "
                "structural mechanism"
            )
        return self

    @model_validator(mode="after")
    def _near_miss_index_in_range(self) -> "StructuralMechanism":
        m = _ARG_REF_RE.match(self.near_miss.breaks_at_argument)
        if m is None:
            # field-level validator on NearMiss already raised; defensive.
            return self
        idx = int(m.group(2))
        # Source and target arities are equal once _signatures_match passes;
        # if signatures don't match, that earlier validator raises and we
        # never reach this code.
        arity = self.predicate_in_source.arity
        if idx >= arity:
            raise ValueError(
                f"near_miss.breaks_at_argument index {idx} is out of range for "
                f"predicate arity {arity} (valid range: 0..{arity - 1})"
            )
        return self


__all__ = [
    "DomainEvidence",
    "HigherOrderRelation",
    "MarrLevel",
    "NearMiss",
    "PredicateInstance",
    "StructuralMechanism",
]
