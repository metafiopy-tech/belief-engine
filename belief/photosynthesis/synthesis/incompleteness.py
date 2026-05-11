"""Incompleteness pass -- probe + classify + loopback (SE Session 6).

Closes the conceptual pyramid. Between the structurer and the critic,
the incompleteness pass generates 15-20 implementation-focused probes
against the candidate ``StructuralMechanism``, classifies each as
``resolved_from_corpus`` / ``needs_research`` / ``open_remainder``,
and dispatches the ``needs_research`` probes back through the
research dispatcher up to two times. Probes that remain unresolved
after the second loopback become ``open_remainder`` and propagate
into the GoalSpec via ``mechanism.incompleteness_probes_open``.

Probe references point at specific slots in the mechanism using the
same ``predicate_in_(source|target).argument[N]`` regex that
:class:`NearMiss.breaks_at_argument` uses. Higher-order relations and
near_miss can also be referenced as
``higher_order_relations[N]`` or ``near_miss``.

Out of scope for Session 6:
  - Auto-resolving open probes via additional research. The loopback
    only retries needs_research probes; resolution is the
    synthesizer's job (or a future session's).
  - LLM-augmented probe generation. The Session 6 path is template-
    only so tests stay hermetic; an ``llm`` kwarg is reserved on
    ``generate_probes`` for future Haiku augmentation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from belief.photosynthesis.synthesis.atomizer import ResearchPrompt
from belief.photosynthesis.synthesis.structural_mechanism import (
    IncompletenessProbe,
    StructuralMechanism,
)


logger = logging.getLogger("belief.photosynthesis.synthesis.incompleteness")


# Per the SE plan acceptance criterion ("loopback iterates max 2 times;
# third attempt forces emission with open probes propagated forward").
MAX_LOOPBACK_ITERATIONS = 2


# Per-axis probe templates targeted at predicate arguments.
# Format strings receive {role}, {arg_index}, {side}, {predicate}.
_ARG_PROBE_TEMPLATES: tuple[str, ...] = (
    "In the {side} domain, what concrete type or data structure "
    "occupies the '{role}' role of {predicate}/{arity} (argument {arg_index})?",
    "What invariant must the '{role}' argument of {predicate} preserve "
    "in the {side} domain for the mechanism to function?",
    "How is the '{role}' argument of {predicate} typically realized in "
    "the {side} domain -- by which actual entity, sensor, or process?",
)


# Mechanism-wide probes hit the higher-order relations + near_miss.
_RELATION_PROBE_TEMPLATE = (
    "How does the higher-order relation '{name}' translate to "
    "operational behavior in the target domain -- what observable "
    "outcome marks its presence?"
)
_NEAR_MISS_PROBE_TEMPLATE = (
    "What test would distinguish the candidate mechanism from the "
    "near-miss '{near_miss_desc}' in code or operational terms?"
)


# Tokens that indicate a corpus doc plausibly answers an
# implementation question. Cheap heuristic; the LLM-classifier path
# (when wired) replaces this for the LLM checks but keeps it as a
# deterministic pre-filter.
_IMPLEMENTATION_HINT_TOKENS: tuple[str, ...] = (
    "implementation",
    "implement",
    "type",
    "interface",
    "signature",
    "API",
    "function",
    "method",
    "class",
    "schema",
    "data structure",
    "algorithm",
    "protocol",
    "invariant",
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IncompletenessResult:
    """Output of the incompleteness pass."""

    probes_generated: list[IncompletenessProbe]
    probes_resolved: list[IncompletenessProbe]
    probes_open: list[IncompletenessProbe]
    iterations_used: int = 0
    corpus_size_seen: int = 0

    def __post_init__(self) -> None:
        # Sanity: the union should equal probes_generated (modulo
        # mutations during classification).
        pass

    @property
    def resolved_count(self) -> int:
        return len(self.probes_resolved)

    @property
    def open_count(self) -> int:
        return len(self.probes_open)


# ---------------------------------------------------------------------------
# Probe generation
# ---------------------------------------------------------------------------


def generate_probes(
    mechanism: StructuralMechanism,
    *,
    n_per_arg: int = 3,
    include_relation_probes: bool = True,
    include_near_miss_probe: bool = True,
) -> list[IncompletenessProbe]:
    """Generate implementation-focused probes for a candidate mechanism.

    For each predicate side (source, target) and each argument
    position (``arity`` slots), produce ``n_per_arg`` probes drawn
    from :data:`_ARG_PROBE_TEMPLATES`. Plus one probe per higher-order
    relation and one for the near-miss. The default settings produce
    ``2 * arity * 3 + len(relations) + 1`` probes -- for a typical
    arity-2 mechanism with 1 relation that's ``2*2*3 + 1 + 1 = 14``
    probes; arity-3 yields 20.
    """
    probes: list[IncompletenessProbe] = []
    n = max(1, min(n_per_arg, len(_ARG_PROBE_TEMPLATES)))

    for side, pred in (
        ("source", mechanism.predicate_in_source),
        ("target", mechanism.predicate_in_target),
    ):
        for arg_index, role in enumerate(pred.roles):
            for i in range(n):
                tpl = _ARG_PROBE_TEMPLATES[i]
                probes.append(
                    IncompletenessProbe(
                        probe_id=_make_probe_id(),
                        question=tpl.format(
                            role=role,
                            arg_index=arg_index,
                            side=side,
                            predicate=pred.name,
                            arity=pred.arity,
                        ),
                        references_field=f"predicate_in_{side}.argument[{arg_index}]",
                        classification="needs_research",
                        iteration=0,
                    )
                )

    if include_relation_probes:
        for i, rel in enumerate(mechanism.higher_order_relations):
            probes.append(
                IncompletenessProbe(
                    probe_id=_make_probe_id(),
                    question=_RELATION_PROBE_TEMPLATE.format(name=rel.name),
                    references_field=f"higher_order_relations[{i}]",
                    classification="needs_research",
                    iteration=0,
                )
            )

    if include_near_miss_probe:
        probes.append(
            IncompletenessProbe(
                probe_id=_make_probe_id(),
                question=_NEAR_MISS_PROBE_TEMPLATE.format(
                    near_miss_desc=mechanism.near_miss.description[:140]
                ),
                references_field="near_miss",
                classification="needs_research",
                iteration=0,
            )
        )

    return probes


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_probes(
    probes: list[IncompletenessProbe],
    *,
    corpus: Any = None,
    iteration: int = 0,
) -> list[IncompletenessProbe]:
    """Classify each probe against the retrieved corpus.

    Heuristic: a probe is ``resolved_from_corpus`` when at least one
    corpus doc's title+summary text contains BOTH (a) a substantive
    token from the probe's references_field (e.g. the role name or
    predicate name) AND (b) at least one implementation-hint token
    from :data:`_IMPLEMENTATION_HINT_TOKENS`.

    Unresolved probes stay ``needs_research`` until the iteration cap
    is hit; the caller (run_incompleteness) flips them to
    ``open_remainder`` when ``iteration >= MAX_LOOPBACK_ITERATIONS``.

    ``corpus`` accepts any iterable of objects exposing ``.title``,
    ``.summary``, and (optionally) ``.raw_excerpt`` attributes.
    ``None`` or empty corpus leaves every probe as ``needs_research``.
    """
    docs = _normalize_corpus(corpus)
    if not docs:
        return [_with_iteration(p, iteration) for p in probes]

    haystack_per_doc = [
        (
            (getattr(d, "title", "") or "")
            + " "
            + (getattr(d, "summary", "") or "")
            + " "
            + (getattr(d, "raw_excerpt", "") or "")
        ).lower()
        for d in docs
    ]
    urls_per_doc = [getattr(d, "url", "") or "" for d in docs]

    out: list[IncompletenessProbe] = []
    for probe in probes:
        if probe.classification == "resolved_from_corpus":
            # Already resolved on a prior iteration; carry forward.
            out.append(probe)
            continue

        question_tokens = _extract_probe_tokens(probe)
        if not question_tokens:
            out.append(_with_iteration(probe, iteration))
            continue

        matched_url: Optional[str] = None
        for hay, url in zip(haystack_per_doc, urls_per_doc):
            if any(t in hay for t in question_tokens) and any(
                h in hay for h in _IMPLEMENTATION_HINT_TOKENS
            ):
                matched_url = url
                break

        if matched_url is not None:
            out.append(
                _replace_probe(
                    probe,
                    classification="resolved_from_corpus",
                    evidence_url=matched_url,
                    iteration=iteration,
                )
            )
        else:
            out.append(_with_iteration(probe, iteration))

    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator with loopback
# ---------------------------------------------------------------------------


async def run_incompleteness(
    mechanism: StructuralMechanism,
    *,
    corpus: Any = None,
    dispatcher: Optional[Callable[..., Awaitable[Any]]] = None,
    n_per_arg: int = 3,
) -> IncompletenessResult:
    """Run the full incompleteness pipeline.

    1. ``generate_probes`` against ``mechanism``.
    2. ``classify_probes`` using the existing ``corpus``.
    3. If any probes are ``needs_research`` AND a ``dispatcher`` is
       provided, build :class:`ResearchPrompt` objects from each
       unresolved probe and dispatch. Reclassify with the augmented
       corpus. Iterate up to :data:`MAX_LOOPBACK_ITERATIONS` times.
    4. Anything still ``needs_research`` after the cap becomes
       ``open_remainder``.

    ``dispatcher`` is the same callable shape as
    :func:`research_dispatcher.dispatch` -- ``async (prompts, *,
    client, sources) -> list[RetrievedDoc]``. When None, the loopback
    is skipped and unresolved probes go straight to open_remainder.
    """
    probes = generate_probes(mechanism, n_per_arg=n_per_arg)
    docs = list(_normalize_corpus(corpus))
    iteration = 0
    probes = classify_probes(probes, corpus=docs, iteration=iteration)

    while iteration < MAX_LOOPBACK_ITERATIONS and dispatcher is not None:
        unresolved = [p for p in probes if p.classification == "needs_research"]
        if not unresolved:
            break

        iteration += 1
        try:
            new_docs = await _loopback_dispatch(unresolved, dispatcher)
        except Exception as exc:
            logger.warning("incompleteness loopback iter %d failed: %s", iteration, exc)
            new_docs = []

        if new_docs:
            # Append new docs to the running corpus, dedup-by-URL.
            seen = {getattr(d, "url", "") or "" for d in docs}
            for d in new_docs:
                if (getattr(d, "url", "") or "") in seen:
                    continue
                docs.append(d)
                seen.add(getattr(d, "url", "") or "")

        probes = classify_probes(probes, corpus=docs, iteration=iteration)

    # Anything still needs_research after the cap -> open_remainder.
    for p in probes:
        if p.classification == "needs_research":
            p.classification = "open_remainder"

    resolved = [p for p in probes if p.classification == "resolved_from_corpus"]
    open_remainder = [p for p in probes if p.classification == "open_remainder"]

    return IncompletenessResult(
        probes_generated=probes,
        probes_resolved=resolved,
        probes_open=open_remainder,
        iterations_used=iteration,
        corpus_size_seen=len(docs),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _make_probe_id() -> str:
    return f"probe_{uuid.uuid4().hex[:10]}"


def _normalize_corpus(corpus: Any) -> list:
    if corpus is None:
        return []
    try:
        return list(corpus)
    except TypeError:
        return []


def _extract_probe_tokens(probe: IncompletenessProbe) -> list[str]:
    """Pull the substantive tokens from a probe's references_field.

    For ``predicate_in_source.argument[2]`` -> ``["predicate"]``
    (we strip the side and the slot index since those carry no
    classification signal). For ``higher_order_relations[0]`` ->
    ``["higher_order_relation"]``. The role name is also drawn from
    the question text since the role string is more discriminating
    than the bare structural reference.
    """
    tokens: list[str] = []
    ref = probe.references_field
    if "predicate" in ref:
        tokens.append("predicate")
    if "higher_order" in ref:
        tokens.append("higher_order")
    if "near_miss" in ref:
        tokens.append("near_miss")

    # Pull single-quoted role / name tokens from the question text.
    # The templates emit them as ``'role'``, ``'name'``, etc.
    question = probe.question
    start = 0
    while True:
        i = question.find("'", start)
        if i < 0:
            break
        j = question.find("'", i + 1)
        if j < 0:
            break
        candidate = question[i + 1 : j]
        if 1 < len(candidate) < 80:
            tokens.append(candidate.lower())
        start = j + 1
    return tokens


def _with_iteration(probe: IncompletenessProbe, iteration: int) -> IncompletenessProbe:
    if probe.iteration == iteration:
        return probe
    return _replace_probe(probe, iteration=iteration)


def _replace_probe(probe: IncompletenessProbe, **overrides) -> IncompletenessProbe:
    data = probe.model_dump()
    data.update(overrides)
    return IncompletenessProbe.model_validate(data)


async def _loopback_dispatch(
    unresolved: list[IncompletenessProbe],
    dispatcher: Callable[..., Awaitable[Any]],
) -> list[Any]:
    """Convert unresolved probes to ResearchPrompts and dispatch them.

    Each probe becomes one prompt under the ``cross_domain`` axis
    (the broadest fan-out). The atomizer's _AXIS_SOURCE_HINTS still
    routes the prompt through the dispatcher; we don't need to
    duplicate that logic here.
    """
    prompts: list[ResearchPrompt] = []
    for probe in unresolved:
        prompts.append(
            ResearchPrompt(
                word=probe.probe_id,
                axis="cross_domain",
                query=probe.question,
                source_hints=("arxiv",),
            )
        )
    try:
        return await dispatcher(prompts)
    except TypeError:
        # Some callers may want positional-only or different kwargs.
        return []


__all__ = [
    "IncompletenessResult",
    "MAX_LOOPBACK_ITERATIONS",
    "classify_probes",
    "generate_probes",
    "run_incompleteness",
]
