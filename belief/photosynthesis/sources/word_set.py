"""Word-set input adapter -- Synthesis Engine Session 2.

A *sibling* input path to the six photosynthesis harvesters. Where the
harvesters fetch from external APIs (arxiv, github, pypi, ...), this
adapter accepts a list of user-submitted words and emits ``raw_signal``
rows in the same shape, so the existing cascade filter / novelty gate /
ranker / heap treat them uniformly with harvested ones (no
special-cased branches).

Signal shape per submission:

  - **One synthetic per-word signal** for each word, so the cascade
    filter can score each concept independently. ``source_id`` carries
    the bundle id to disambiguate from prior submissions of the same
    word.
  - **One bundled signal** containing the joined word list, so
    cross-word context survives into the synthesizer (Session 3 reads
    this bundle to produce the structural-mechanism payload).

Word-set is *not* added to ``daemon.py::HARVESTERS`` -- the spec is
explicit that this path is intentional / manual, invoked from the
``belief synth words "x,y"`` CLI rather than scheduled. See the SE
session plan, Session 2 for the rationale.

Out of scope for Session 2:
  - Cross-domain synthesis (Session 3 owns ``cross_domain_generator``).
    For now, word inputs flow through the existing single-domain
    generator and produce ordinary ``GoalSpec``s with
    ``structural_mechanism = None``.
  - Atomization research fan-out (Session 5).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Optional

from belief.photosynthesis.state import CandidateSeed

if TYPE_CHECKING:  # pragma: no cover
    from belief.photosynthesis.config import PhotoConfig
    from belief.photosynthesis.state import PhotosynthesisState


logger = logging.getLogger("belief.photosynthesis.sources.word_set")


SOURCE_NAME = "word_set"

# Keep word validation lenient -- domain words like "mantis_shrimp" or
# "slime-mold" are plausible. Reject control characters, leading /
# trailing whitespace (after stripping), and obviously empty input.
_VALID_WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")

# Bundled signal id pattern: "<bundle_id>:bundle"
# Per-word signal id pattern: "<bundle_id>:word:<index>:<word>"
# Including the index keeps source_id unique even if the same word
# appears twice in a submission (which is unusual but possible).


def parse_words(raw: str) -> list[str]:
    """Split a comma-separated CLI argument into a clean word list.

    Strips whitespace from each token. Drops empty tokens. Rejects
    duplicates (case-sensitive) so a typo like ``"x,x"`` doesn't silently
    halve the apparent word count downstream.
    """
    if raw is None:
        raise ValueError("words argument is required")
    pieces = [p.strip() for p in raw.split(",")]
    pieces = [p for p in pieces if p]
    if not pieces:
        raise ValueError("at least one non-empty word is required")
    seen: set[str] = set()
    deduped: list[str] = []
    for w in pieces:
        if w in seen:
            continue
        seen.add(w)
        deduped.append(w)
    for w in deduped:
        if not _VALID_WORD_RE.match(w):
            raise ValueError(
                f"invalid word {w!r}: must start with [A-Za-z0-9] and contain only "
                "alphanumerics, underscore, or hyphen"
            )
    return deduped


def make_bundle_id(prefix: Optional[str] = None) -> str:
    """Return a fresh per-invocation id used as the source_id stem.

    A new bundle id per call means re-submitting the same word list
    produces fresh raw_signal rows rather than dedup'ing into the prior
    submission's row -- important when a previous run's signals were
    rejected by the cascade and the user wants to retry.
    """
    if prefix:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"
    return uuid.uuid4().hex[:12]


def _bundle_seed(words: list[str], bundle_id: str, captured_at: int) -> CandidateSeed:
    """Build the bundled signal that carries cross-word context."""
    joined = ", ".join(words)
    title = f"Word-set: {joined}"[:400]
    summary = (
        f"User-submitted concept set [{joined}]. Cross-domain mechanism candidate "
        "for the Synthesis Engine pipeline."
    )
    raw_excerpt = (
        f"Word-set submission carrying {len(words)} concept(s): {joined}. "
        "The Session 3 cross-domain generator reads this bundled row to produce "
        "a StructuralMechanism payload. The single-domain generator will treat it "
        "as an ordinary seed."
    )
    return CandidateSeed(
        source=SOURCE_NAME,
        source_id=f"{bundle_id}:bundle",
        title=title,
        summary=summary,
        raw_excerpt=raw_excerpt,
        captured_at=captured_at,
    )


def _per_word_seed(
    word: str, index: int, all_words: list[str], bundle_id: str, captured_at: int
) -> CandidateSeed:
    """Build the per-word synthetic signal."""
    other = [w for w in all_words if w != word]
    other_text = ", ".join(other) if other else "(none)"
    summary = f"User-submitted concept: {word}. Bundle peers: {other_text}."
    raw_excerpt = (
        f"Word-set probe for concept {word!r}. Submitted alongside: {other_text}. "
        "The cascade filter scores this row independently of the bundle."
    )
    return CandidateSeed(
        source=SOURCE_NAME,
        source_id=f"{bundle_id}:word:{index}:{word}",
        title=word[:400],
        summary=summary,
        raw_excerpt=raw_excerpt,
        captured_at=captured_at,
    )


async def emit(
    state: "PhotosynthesisState",
    config: "PhotoConfig",
    *,
    words: list[str],
    bundle_id: Optional[str] = None,
    captured_at: Optional[int] = None,
) -> list[CandidateSeed]:
    """Emit synthetic raw_signal rows for ``words``.

    ``state`` and ``config`` mirror the harvester signature minus the
    HTTP client (word-set has no upstream to fetch). The function is
    async to compose with the rest of the photosynthesis async layer
    (``run_synthesis_cycle`` is async) without forcing the caller to
    hop event loops.

    Returns the list of newly inserted ``CandidateSeed`` objects, in
    insertion order: per-word signals first, bundle signal last.
    Already-seen ``source_id`` values are skipped via the existing
    ``state.mark_if_new`` dedup path -- duplicates return an empty list
    rather than raising. ``config`` is accepted for harvester-shape
    parity even though Session 2 doesn't currently read fields from it;
    Session 5's atomization layer will.
    """
    # The unused-argument check below keeps lint happy until Session 5
    # wires atomization, which will read tracked_deps / arxiv_categories
    # off ``config`` to expand each word into research prompts.
    _ = config

    if not words:
        raise ValueError("emit() requires at least one word")
    bid = bundle_id or make_bundle_id()
    ts = captured_at if captured_at is not None else int(time.time())

    inserted: list[CandidateSeed] = []
    for i, word in enumerate(words):
        seed = _per_word_seed(word, i, words, bid, ts)
        if not state.mark_if_new(SOURCE_NAME, seed.source_id):
            continue
        if state.insert_signal(seed) is not None:
            inserted.append(seed)

    bundle_seed = _bundle_seed(words, bid, ts)
    if state.mark_if_new(SOURCE_NAME, bundle_seed.source_id):
        if state.insert_signal(bundle_seed) is not None:
            inserted.append(bundle_seed)

    logger.info(
        "word_set.emit: bundle_id=%s words=%d inserted=%d",
        bid,
        len(words),
        len(inserted),
    )
    return inserted


__all__ = [
    "SOURCE_NAME",
    "emit",
    "make_bundle_id",
    "parse_words",
]
