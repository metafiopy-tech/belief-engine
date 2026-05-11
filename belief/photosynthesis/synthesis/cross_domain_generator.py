"""Cross-domain synthesizer (SE Session 3).

Sibling to ``synthesis/generator.py``. Where the existing single-
domain generator turns a harvested signal into a ``GoalSpec``, this
generator turns a user-submitted word-set bundle into a ``GoalSpec``
*with* a populated ``structural_mechanism`` payload.

Internal four-pass design (see ``prompts_cross_domain``):

  1. FREEFORM brainstorm cross-domain analogies between the two
     domains. Free prose; no schema.
  2. PREDICATE-FORM FORCING: extract the deepest structural
     similarity as a typed predicate.
  3. ANTI-RATIONALIZATION: enumerate at least two surface attributes
     that LOOK shared but were rejected as decorative.
  4. STRUCTURER: emit the full StructuralMechanism JSON.

Then the candidate runs through the CoVe critic
(``cross_domain_critic.critique``) in an INDEPENDENT context. ACCEPT
verdicts produce a ``GeneratorResult`` with a populated GoalSpec;
REJECT verdicts return ``reason="critic_rejected"`` so the cycle can
mark the bundle's source rows as 'rejected'.

Out of scope for Session 3:
  - Biological-primitives retrieval (Session 4).
  - Atomization fan-out (Session 5).
  - Incompleteness probe loopback (Session 6).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from pydantic import ValidationError

from belief.photosynthesis.synthesis.cross_domain_critic import (
    CriticResult,
    critique,
)
from belief.photosynthesis.synthesis.generator import (
    AcceptanceCriterion,
    GeneratorResult,
    GoalSpec,
)
from belief.photosynthesis.synthesis.prompts_cross_domain import (
    ANTI_RATIONALIZATION_PROMPT,
    FREEFORM_PROMPT,
    PREDICATE_FORCING_PROMPT,
    STRUCTURER_PROMPT,
)
from belief.photosynthesis.synthesis.structural_mechanism import (
    PredicateInstance,
    StructuralMechanism,
)


logger = logging.getLogger("belief.photosynthesis.synthesis.cross_domain_generator")


SONNET_MODEL = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000
# Pass 4 (structurer) needs a bigger budget because it emits the full
# nested mechanism JSON plus citations and rationale.
STRUCTURER_MAX_TOKENS = 3500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class CrossDomainResult:
    """Wrapper for the cross-domain synthesizer's outcome.

    Mirrors ``GeneratorResult`` so the caller in ``cycle.py`` can
    treat both paths the same. Adds ``mechanism`` and ``critic`` so
    callers that want the validated mechanism / critic output can
    inspect them without re-parsing the GoalSpec.
    """

    spec: Optional[GoalSpec]
    mechanism: Optional[StructuralMechanism]
    critic: Optional[CriticResult]
    reason: str
    raw_passes: dict[str, str] = field(default_factory=dict)

    def as_generator_result(self) -> GeneratorResult:
        """Adapt to the existing GeneratorResult shape so the cycle
        can consume cross-domain and single-domain outputs uniformly."""
        return GeneratorResult(spec=self.spec, ranker=None, reason=self.reason)


async def synthesize_cross_domain(
    *,
    words: list[str],
    bundle_id: str,
    generator_client: Callable[..., Awaitable[str]],
    critic_client: Optional[Callable[..., Awaitable[str]]] = None,
    bio_store: Any = None,
    corpus: Any = None,
    research_dispatcher: Optional[Callable[..., Awaitable[Any]]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> CrossDomainResult:
    """Run the four-pass cross-domain synthesizer + critic.

    Args:
        words: at least 2 user-submitted concept words. The first two
            are treated as source / target; additional words are
            included in the freeform context but the predicate is
            forced over the source <-> target pair.
        bundle_id: the word_set bundle id, used for slug generation
            and audit linkage.
        generator_client: async LLM callable for the four passes.
        critic_client: async LLM callable for the critic pass. When
            None, the critic is skipped and the GoalSpec is emitted
            with ``reason="accepted_no_critic"`` -- useful for
            hermetic tests and live runs without a critic budget.
        bio_store: optional ``BiologicalPrimitiveStore`` (SE Session 4).
            When provided, the freeform pass receives a primer block
            listing the top 5 nearest existing mechanisms so the
            synthesizer doesn't re-derive what's already known.
            Accepted mechanisms are added to the store on success.
        temperature: sampling temperature for the synthesizer passes.
            The critic uses 0.0 internally.

    Returns:
        A :class:`CrossDomainResult`. ``spec`` is populated only when
        all four passes succeed AND (if a critic_client is provided)
        the critic returns ACCEPT. Otherwise ``reason`` carries the
        failure mode.
    """
    if len(words) < 2:
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="too_few_words",
        )

    source, target = words[0], words[1]

    # ------------------------------------------------------------------
    # Bio-store priming (SE Session 4) -- query the top 5 nearest
    # existing mechanisms BEFORE pass 1 so the freeform brainstorm
    # doesn't re-derive what's already known. The primer prepends the
    # freeform prompt as a "prior mechanisms in your library" section.
    # When bio_store is None, the primer is empty and pass 1 runs as
    # in pre-S4.
    # ------------------------------------------------------------------
    primer = _build_bio_store_primer(bio_store, source, target)

    # SE Session 5: optional retrieved-corpus block. When the
    # caller atomized + dispatched research before invoking us, the
    # corpus is a list of RetrievedDoc-like objects. We format it as
    # a "RETRIEVED CORPUS" preamble so the freeform pass reasons
    # over actual documents instead of training-data only.
    corpus_block = _build_corpus_block(corpus)

    # ------------------------------------------------------------------
    # Pass 1 — freeform brainstorm
    # ------------------------------------------------------------------
    freeform_prompt = FREEFORM_PROMPT.format(source=source, target=target)
    if corpus_block:
        freeform_prompt = corpus_block + "\n\n" + freeform_prompt
    if primer:
        freeform_prompt = primer + "\n\n" + freeform_prompt
    try:
        freeform = await generator_client(
            freeform_prompt,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("freeform pass raised: %s", exc)
        return CrossDomainResult(
            spec=None, mechanism=None, critic=None, reason="freeform_pass_error"
        )

    # ------------------------------------------------------------------
    # Pass 2 — predicate-form forcing
    # ------------------------------------------------------------------
    predicate_prompt = PREDICATE_FORCING_PROMPT.format(
        source=source,
        target=target,
        freeform=(freeform or "").strip()[:4000],
    )
    try:
        predicate_raw = await generator_client(
            predicate_prompt,
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("predicate pass raised: %s", exc)
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="predicate_pass_error",
            raw_passes={"freeform": freeform},
        )

    predicate_data = _extract_json(predicate_raw)
    if predicate_data is None or not isinstance(predicate_data, dict):
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="predicate_parse_error",
            raw_passes={"freeform": freeform, "predicate": predicate_raw},
        )

    try:
        predicate = PredicateInstance.model_validate(
            {k: predicate_data[k] for k in ("name", "arity", "roles", "marr_level")}
        )
    except (ValidationError, KeyError) as exc:
        logger.warning("predicate validation failed: %s", exc)
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="predicate_validation_error",
            raw_passes={"freeform": freeform, "predicate": predicate_raw},
        )

    # ------------------------------------------------------------------
    # Pass 3 — anti-rationalization
    # ------------------------------------------------------------------
    anti_prompt = ANTI_RATIONALIZATION_PROMPT.format(
        predicate_name=predicate.name,
        arity=predicate.arity,
        source=source,
        target=target,
    )
    try:
        anti_raw = await generator_client(
            anti_prompt,
            temperature=0.4,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("anti-rationalization pass raised: %s", exc)
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="anti_rationalization_pass_error",
            raw_passes={
                "freeform": freeform,
                "predicate": predicate_raw,
            },
        )

    anti_data = _extract_json(anti_raw)
    rejected_attrs: list[str] = []
    if isinstance(anti_data, dict):
        rejected_attrs = [
            str(a)
            for a in anti_data.get("considered_and_rejected_attributes", [])
            if isinstance(a, (str, int, float))
        ]
    if len(rejected_attrs) < 2:
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="anti_rationalization_too_few",
            raw_passes={
                "freeform": freeform,
                "predicate": predicate_raw,
                "anti": anti_raw,
            },
        )

    # ------------------------------------------------------------------
    # Pass 4 — structurer (full StructuralMechanism JSON)
    # ------------------------------------------------------------------
    structurer_prompt = STRUCTURER_PROMPT.format(
        source=source,
        target=target,
        predicate_json=predicate.model_dump_json(),
        rejected_attributes_json=json.dumps({"considered_and_rejected_attributes": rejected_attrs}),
        freeform=(freeform or "").strip()[:3000],
    )
    try:
        structurer_raw = await generator_client(
            structurer_prompt,
            temperature=0.2,
            max_tokens=STRUCTURER_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("structurer pass raised: %s", exc)
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="structurer_pass_error",
            raw_passes={
                "freeform": freeform,
                "predicate": predicate_raw,
                "anti": anti_raw,
            },
        )

    mechanism_data = _extract_json(structurer_raw)
    if mechanism_data is None or not isinstance(mechanism_data, dict):
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="structurer_parse_error",
            raw_passes={
                "freeform": freeform,
                "predicate": predicate_raw,
                "anti": anti_raw,
                "structurer": structurer_raw,
            },
        )

    try:
        mechanism = StructuralMechanism.model_validate(mechanism_data)
    except ValidationError as exc:
        logger.warning("structurer schema validation failed: %s", exc.errors()[:3])
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=None,
            reason="schema_invalid",
            raw_passes={
                "freeform": freeform,
                "predicate": predicate_raw,
                "anti": anti_raw,
                "structurer": structurer_raw,
            },
        )

    # ------------------------------------------------------------------
    # Incompleteness pass (SE Session 6) -- generates 15-20
    # implementation-focused probes against the candidate mechanism,
    # classifies each against the corpus, loops back through the
    # research_dispatcher up to MAX_LOOPBACK_ITERATIONS times for
    # needs_research probes. Open Remainder probes get attached to
    # mechanism.incompleteness_probes_open so the critic and the
    # downstream GoalSpec see them.
    # ------------------------------------------------------------------
    try:
        from belief.photosynthesis.synthesis.incompleteness import run_incompleteness

        inc_result = await run_incompleteness(
            mechanism,
            corpus=corpus,
            dispatcher=research_dispatcher,
        )
        if inc_result.probes_open:
            # Re-validate to keep the schema honest -- model_copy
            # with update= produces a fresh instance and forces the
            # validators to re-run.
            mechanism = mechanism.model_copy(
                update={"incompleteness_probes_open": list(inc_result.probes_open)}
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("incompleteness pass raised (skipping): %s", exc)

    # ------------------------------------------------------------------
    # Critic pass (independent context)
    # ------------------------------------------------------------------
    critic_result: Optional[CriticResult] = None
    if critic_client is not None:
        critic_result = await critique(mechanism, critic_client=critic_client)
        if not critic_result.accepted:
            return CrossDomainResult(
                spec=None,
                mechanism=mechanism,
                critic=critic_result,
                reason="critic_rejected",
                raw_passes={
                    "freeform": freeform,
                    "predicate": predicate_raw,
                    "anti": anti_raw,
                    "structurer": structurer_raw,
                },
            )

    # ------------------------------------------------------------------
    # Bio-store deposit (SE Session 4) -- accepted mechanisms get added
    # to the store so subsequent calls metabolize prior outputs. Failure
    # is non-fatal -- the GoalSpec still gets returned to the caller.
    # ------------------------------------------------------------------
    if bio_store is not None:
        try:
            bio_store.add(mechanism)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("bio_store.add failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Build the GoalSpec wrapping the validated mechanism
    # ------------------------------------------------------------------
    goal_spec = _goalspec_from_mechanism(
        mechanism=mechanism,
        words=words,
        bundle_id=bundle_id,
    )

    return CrossDomainResult(
        spec=goal_spec,
        mechanism=mechanism,
        critic=critic_result,
        reason="accepted" if critic_client is not None else "accepted_no_critic",
        raw_passes={
            "freeform": freeform,
            "predicate": predicate_raw,
            "anti": anti_raw,
            "structurer": structurer_raw,
        },
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _goalspec_from_mechanism(
    *,
    mechanism: StructuralMechanism,
    words: list[str],
    bundle_id: str,
) -> GoalSpec:
    """Wrap a validated mechanism into a buildable GoalSpec.

    Session 3 produces a GoalSpec where the mechanism is the artifact
    of interest. The downstream Belief Engine pipeline will read the
    structural_mechanism field via the cross_domain_intake_adapter
    that Session 7 introduces; for now we just emit a GoalSpec that
    survives schema validation.
    """
    word_summary = ", ".join(words)
    title = f"Cross-domain mechanism: {mechanism.source_domain} <-> {mechanism.target_domain}"
    title = title[:120]
    description = (
        f"Implement the cross-domain mechanism '{mechanism.predicate_in_source.name}/"
        f"{mechanism.predicate_in_source.arity}' relating {mechanism.source_domain} and "
        f"{mechanism.target_domain}. Concept words from the user: [{word_summary}]. "
        f"The mechanism's predicate roles are {mechanism.predicate_in_source.roles}. "
        "See structural_mechanism for the full payload."
    )[:1200]

    acceptance: list[AcceptanceCriterion] = [
        AcceptanceCriterion(
            kind="behavior",
            spec=(
                f"output exhibits the predicate "
                f"{mechanism.predicate_in_source.name}/"
                f"{mechanism.predicate_in_source.arity} "
                f"with role assignments matching {mechanism.predicate_in_source.roles}"
            ),
        ),
        AcceptanceCriterion(
            kind="test",
            spec="pytest covers the near_miss case and confirms it is rejected",
        ),
    ]

    goal_id = _slugify(
        f"cross-domain-{mechanism.source_domain}-{mechanism.target_domain}-"
        f"{mechanism.predicate_in_source.name}"
    )
    if not goal_id:
        goal_id = f"cross-domain-{bundle_id}"
    goal_id = goal_id[:60]

    return GoalSpec(
        goal_id=goal_id,
        title=title,
        one_paragraph_description=description,
        artifact_type="library",
        primary_libraries=[],
        new_libraries_introduced=[],
        acceptance_criteria=acceptance,
        estimated_build_time_min=120,
        estimated_difficulty=4,
        prerequisite_skills=[],
        relevance_rationale=(
            f"Cross-domain mechanism synthesis between {mechanism.source_domain} and "
            f"{mechanism.target_domain}. The structural isomorphism claim is enforced "
            "by the StructuralMechanism schema."
        ),
        novelty_rationale=(
            f"User-submitted concept set [{word_summary}] (bundle {bundle_id}); "
            "the synthesizer rejected the surface attributes "
            f"{mechanism.considered_and_rejected_attributes} in favor of the "
            f"{mechanism.predicate_in_source.name} predicate."
        ),
        source_citation=f"word_set:{bundle_id}",
        structural_mechanism=mechanism,
    )


def _build_corpus_block(corpus: Any) -> str:
    """Format a list of RetrievedDoc-like objects as a grounding preamble.

    Accepts any iterable whose elements expose ``title``, ``summary``,
    ``source``, and (optionally) ``url`` attributes. Empty / None
    corpora return an empty string. Each doc gets one line; the
    block tops out at ~30 docs to keep the prompt within budget.
    """
    if not corpus:
        return ""
    try:
        docs = list(corpus)
    except TypeError:
        return ""
    if not docs:
        return ""

    lines = [
        "RETRIEVED CORPUS (atomized research surfaced these "
        "documents; ground your brainstorm in them where they apply):"
    ]
    for i, d in enumerate(docs[:30], start=1):
        title = getattr(d, "title", "") or ""
        summary = getattr(d, "summary", "") or ""
        src = getattr(d, "source", "") or ""
        url = getattr(d, "url", "") or ""
        # Each line: rank index, source, title, then a clipped summary.
        head = f"  {i}. [{src}] {title[:160]}"
        if url:
            head += f" <{url[:120]}>"
        lines.append(head)
        if summary:
            lines.append(f"     {summary[:300]}")
    if len(docs) > 30:
        lines.append(f"  ... ({len(docs) - 30} more documents available)")
    return "\n".join(lines)


def _build_bio_store_primer(bio_store: Any, source: str, target: str) -> str:
    """Format the top-5 nearest existing mechanisms as a primer block.

    Returns an empty string when ``bio_store`` is None, the store is
    empty, or any retrieval error occurs -- the freeform pass then
    runs without priming. Format is plain prose so the LLM can reason
    over it without parsing structured input.
    """
    if bio_store is None:
        return ""
    try:
        query = f"{source} <-> {target}"
        neighbors = bio_store.query_nearest(query, top_k=5)
    except Exception as exc:
        logger.warning("bio_store.query_nearest failed (skipping primer): %s", exc)
        return ""
    if not neighbors:
        return ""

    lines = [
        "PRIOR MECHANISMS IN YOUR LIBRARY (closest first by combined "
        "embedding-similarity * FSRS-retrievability score):",
    ]
    for i, n in enumerate(neighbors, start=1):
        m = n.mechanism
        rels = ", ".join(r.name for r in m.higher_order_relations)
        lines.append(
            f"  {i}. {m.source_domain} <-> {m.target_domain}: "
            f"{m.predicate_in_source.name}/{m.predicate_in_source.arity} "
            f"({m.predicate_in_source.marr_level}); relations: {rels} "
            f"[score={n.weighted_score:.3f}]"
        )
    lines.append(
        "Build on these where they apply, but DO NOT re-derive a "
        "mechanism that's already in the library -- pick a deeper or "
        "different predicate if the listed ones already cover the "
        "structural claim."
    )
    return "\n".join(lines)


def _extract_json(raw: str) -> Any:
    """Best-effort JSON extraction. Handles fenced code blocks and
    surrounding prose. Returns the parsed value or None on failure."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _slugify(text: str) -> str:
    lowered = (text or "").lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or f"cross-domain-{uuid.uuid4().hex[:8]}"


__all__ = [
    "CrossDomainResult",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "SONNET_MODEL",
    "STRUCTURER_MAX_TOKENS",
    "synthesize_cross_domain",
]
