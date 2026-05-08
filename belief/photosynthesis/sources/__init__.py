"""Source harvesters for Photosynthesis.

Every source exports a single coroutine with a uniform signature::

    async def harvest(
        client: BreakerAsyncClient,
        state: PhotosynthesisState,
        config: PhotoConfig,
    ) -> list[CandidateSeed]

Responsibilities per source:

1. Read its watermark (or fall back to a sensible "first run" window).
2. Issue conditional GETs wherever the upstream supports ETag /
   If-Modified-Since / 304 (GitHub, Atom feeds, RSS).
3. Normalize every item to a CandidateSeed.
4. Call state.mark_if_new() — skip duplicates.
5. Call state.insert_signal() for genuinely new rows.
6. Advance the watermark at the end, before returning.

Failures in one source MUST NOT take down the daemon. The daemon
catches exceptions from harvest() at the scheduler callback layer;
each source should still guard its own loops so a single malformed
item doesn't abort the whole batch.

**word_set** (Synthesis Engine S2) is a sibling adapter that does NOT
follow this contract: it has no upstream to fetch, no watermark, and is
exposed as ``async def emit(state, config, *, words, ...)`` rather than
``async def harvest(client, state, config)``. It is intentionally NOT
registered in ``daemon.HARVESTERS`` -- the ``belief synth words`` CLI
invokes it directly. From the cascade filter onward, its raw_signal
rows are indistinguishable from the rest.
"""

from belief.photosynthesis.sources import (
    arxiv,
    github_releases,
    github_search,
    hackernews,
    pypi,
    stackoverflow,
    word_set,
)

__all__ = [
    "arxiv",
    "github_releases",
    "github_search",
    "hackernews",
    "pypi",
    "stackoverflow",
    "word_set",
]
