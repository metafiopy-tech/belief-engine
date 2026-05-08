"""Atomizer -- expand each user-submitted word into research prompts (SE S5).

The cross-domain synthesizer's quality is gated by the breadth of the
base corpus it sees. Without atomization, the synthesizer reasons
from training data alone -- exactly the regime that produces
plausible-but-shallow analogies.

This module takes a list of words and emits a structured set of
research prompts spanning five axes:

  - ``mechanism``     -- "How does X work, mechanistically?"
  - ``constraint``    -- "What forces shape X? Where are the limits?"
  - ``tradeoff``      -- "What does X sacrifice for what gain?"
  - ``cross_domain``  -- "Where else does X-like structure appear?"
  - ``counterexample``-- "Where does X-like structure NOT appear /
                         when does X fail?"

Three templates per axis = 15 prompts per word; two input words
produce 30 prompts as the SE plan acceptance criterion specifies.

LLM augmentation is intentionally optional. The templates work
standalone (deterministic, hermetic-test-friendly); a future
``llm=`` parameter can route through Ollama qwen3:8b for richer
prompt phrasing once the dispatcher's quality is tuned.

Out of scope:
  - DSPy/GEPA optimization of the templates -- defer until >=10
    known-good word -> PRD examples exist.
  - Long-context summarization of retrieved corpora -- chunk and let
    the synthesizer's context window handle it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


# ---------------------------------------------------------------------------
# Axes and templates
# ---------------------------------------------------------------------------

AXES: tuple[str, ...] = (
    "mechanism",
    "constraint",
    "tradeoff",
    "cross_domain",
    "counterexample",
)


# Each axis has 3 templates. Format string with {word}.
AXIS_TEMPLATES: dict[str, tuple[str, ...]] = {
    "mechanism": (
        "How does {word} work mechanistically -- what is the underlying process step by step?",
        "What signals or inputs does {word} transform, and into what outputs?",
        "At what spatial and temporal scale does {word} operate?",
    ),
    "constraint": (
        "What physical, biological, or computational constraints shape how {word} can work?",
        "What energy, information, or material budget does {word} operate within?",
        "What invariants must {word} preserve to function?",
    ),
    "tradeoff": (
        "What does {word} sacrifice to achieve its function? What does it gain in return?",
        "When is {word} more efficient than its alternatives, and when less efficient?",
        "What design choices in {word} are inherent vs. accidental?",
    ),
    "cross_domain": (
        "What other natural or engineered systems exhibit a similar mechanism to {word}?",
        "What abstract pattern does {word} instantiate that appears across domains?",
        "If we abstracted {word} away from its substrate, what "
        "computational or causal structure remains?",
    ),
    "counterexample": (
        "Where does {word}-like structure fail to appear despite seemingly favorable conditions?",
        "When does {word} break down -- what are its known failure modes?",
        "What systems superficially resemble {word} but operate via "
        "fundamentally different mechanisms?",
    ),
}


_TEMPLATES_PER_AXIS = 3


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResearchPrompt:
    """One atomized prompt the dispatcher will issue against external sources.

    ``word`` -- the source concept word (e.g. "mantis_shrimp").
    ``axis`` -- one of :data:`AXES`.
    ``query`` -- the natural-language query string.
    ``source_hints`` -- ordered list of source names (``"arxiv"``,
    ``"github"``, ``"pypi"``, ...) the dispatcher should prefer for
    this prompt. Empty = dispatcher fans out to all configured
    sources.
    """

    word: str
    axis: str
    query: str
    source_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown axis: {self.axis!r}")


# Default per-axis source hints. arxiv biases toward
# scientific/biological mechanism content; github biases toward
# computing implementations; pypi biases toward software packages.
# These are hints, not hard routing -- the dispatcher fans out broadly
# and lets dedup handle overlap.
_AXIS_SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "mechanism": ("arxiv", "github"),
    "constraint": ("arxiv",),
    "tradeoff": ("arxiv", "github"),
    "cross_domain": ("arxiv",),
    "counterexample": ("arxiv", "github"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def atomize_word(
    word: str,
    *,
    n_per_axis: int = _TEMPLATES_PER_AXIS,
    llm: Optional[Callable[..., Awaitable[str]]] = None,
) -> list[ResearchPrompt]:
    """Expand a single word into ``5 * n_per_axis`` research prompts.

    ``llm`` is currently ignored (future hook for qwen3:8b-augmented
    phrasing); the deterministic template path is the only mode in
    Session 5. Future sessions may wire ``llm`` through the
    ModelRouter LATIOS role for cheap bulk generation.
    """
    _ = llm  # reserved for future LLM augmentation
    if not word or not word.strip():
        raise ValueError("word must be non-empty")
    n = max(1, min(n_per_axis, _TEMPLATES_PER_AXIS))

    out: list[ResearchPrompt] = []
    for axis in AXES:
        templates = AXIS_TEMPLATES[axis][:n]
        for tpl in templates:
            out.append(
                ResearchPrompt(
                    word=word,
                    axis=axis,
                    query=tpl.format(word=word),
                    source_hints=_AXIS_SOURCE_HINTS.get(axis, ()),
                )
            )
    return out


def atomize_words(
    words: list[str],
    *,
    n_per_axis: int = _TEMPLATES_PER_AXIS,
    llm: Optional[Callable[..., Awaitable[str]]] = None,
) -> list[ResearchPrompt]:
    """Expand each word independently and concatenate the results.

    ``len(words) * 5 * n_per_axis`` prompts total. Order is stable:
    word 1's prompts first, then word 2's, etc. Within each word the
    axes appear in :data:`AXES` order.
    """
    if not words:
        raise ValueError("at least one word is required")
    out: list[ResearchPrompt] = []
    for w in words:
        out.extend(atomize_word(w, n_per_axis=n_per_axis, llm=llm))
    return out


def prompts_per_word(n_per_axis: int = _TEMPLATES_PER_AXIS) -> int:
    """Convenience for callers sizing batches: returns 5 * n_per_axis."""
    return len(AXES) * max(1, min(n_per_axis, _TEMPLATES_PER_AXIS))


__all__ = [
    "AXES",
    "AXIS_TEMPLATES",
    "ResearchPrompt",
    "atomize_word",
    "atomize_words",
    "prompts_per_word",
]
