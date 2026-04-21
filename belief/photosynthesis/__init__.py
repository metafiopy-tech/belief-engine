"""Photosynthesis: autonomous signal harvester and goal synthesizer.

The daemon polls six sources (GitHub search, GitHub release atoms,
PyPI RSS, Stack Overflow, Show HN via Algolia, arXiv cs.AI/CL/SE)
on independent cadences, deduplicates, and funnels the stream
through a four-stage cascading relevance filter:

    stage 0  bloom blocklist (domains/authors we never want)
    stage 1  compiled keyword regex (~200-term allowlist)
    stage 2  TF-IDF cosine vs past-goal corpus
    stage 3  MiniLM embedding vs ChromaDB domain centroids

Only the top 20 survivors per pass reach LLM scoring in Session 4.

Heavy dependencies (apscheduler, sentence-transformers, scikit-learn,
pybloom-live, tenacity, pybreaker) are deliberately NOT imported at
package level — install the `[photosynthesis]` extra to enable. See
pyproject.toml.
"""

from belief.photosynthesis.state import CandidateSeed, PhotosynthesisState

__all__ = ["CandidateSeed", "PhotosynthesisState"]
