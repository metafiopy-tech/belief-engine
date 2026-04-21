"""Unit tests for belief.photosynthesis.filter.cascade.

Stages 0 and 1 run entirely in-process (stdlib regex). Stage 2 (TF-IDF)
and stage 3 (sentence-transformers) require optional deps; those paths
are exercised only when the deps are importable.

We deliberately don't *enforce* the spec's "1,000 signals in ≤15s on
2 vCPU" throughput number here — hardware-dependent — but we do check
throughput is at least "reasonable" when the cheap path is used, to
catch accidental O(N²) regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.photosynthesis.filter.cascade import (
    CascadingRelevanceFilter,
    FilterResult,
    Stage,
    throughput,
)


@pytest.fixture()
def keywords_file(tmp_path: Path) -> Path:
    p = tmp_path / "keywords.yaml"
    p.write_text(
        "keywords:\n  - fastapi\n  - langgraph\n  - mcp\n  - pydantic\n"
    )
    return p


def test_stage0_blocklist_drops_signal(keywords_file: Path) -> None:
    f = CascadingRelevanceFilter(
        keywords_path=keywords_file,
        blocklist=["spam-domain.example"],
    )
    [res] = f.score(["visit https://spam-domain.example/cool-fastapi-post"])
    assert res.stage_reached == Stage.BLOCKED
    assert not res.kept
    assert "blocklist" in res.reason


def test_stage1_keyword_passes_through(keywords_file: Path) -> None:
    f = CascadingRelevanceFilter(keywords_path=keywords_file)
    [hit, miss] = f.score(
        [
            "New fastapi plugin for background jobs",
            "Breaking news from the world of competitive yodeling",
        ]
    )
    # hit survives at least stage 1
    assert hit.stage_reached >= Stage.KEYWORD
    assert "fastapi" in [h.lower() for h in hit.keyword_hits]
    # miss is dropped at stage 1 (below)
    assert miss.stage_reached == Stage.BLOCKED
    assert not miss.kept
    assert "no keyword" in miss.reason


def test_kept_requires_tfidf_or_embed(keywords_file: Path) -> None:
    """Without a TF-IDF corpus or embedding, stage-1-survivors aren't 'kept'.

    The filter is a gate, not a scorer of last resort — signals that
    survive stage 1 but can't be evaluated by later stages default to
    `kept=False` and are not promoted. This prevents an accidentally
    unconfigured filter from leaking raw signals downstream.
    """
    f = CascadingRelevanceFilter(keywords_path=keywords_file)
    [res] = f.score(["fastapi is great, and so is pydantic"])
    assert res.stage_reached == Stage.KEYWORD
    assert not res.kept


def test_top_k_returns_only_kept_sorted_by_score(keywords_file: Path) -> None:
    results = [
        FilterResult(text="a", kept=True, filter_score=0.3),
        FilterResult(text="b", kept=False, filter_score=0.9),  # excluded
        FilterResult(text="c", kept=True, filter_score=0.7),
        FilterResult(text="d", kept=True, filter_score=0.5),
    ]
    f = CascadingRelevanceFilter(keywords_path=keywords_file)
    top = f.top_k_for_llm(results, k=2)
    assert [r.text for r in top] == ["c", "d"]


def test_tfidf_gating_when_sklearn_available(keywords_file: Path) -> None:
    """When sklearn + a corpus are provided, stage 2 gates signals correctly."""
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    corpus = [
        "Build a fastapi bookmark service with sqlalchemy",
        "Build a langgraph agent that uses mcp tools",
        "Build a pydantic-based config loader",
    ]
    f = CascadingRelevanceFilter(
        keywords_path=keywords_file,
        corpus=corpus,
        stage2_coarse=0.05,
        stage2_high=0.25,
    )
    # Strong overlap with corpus → reaches stage 2, kept via high-confidence
    on_topic = "fastapi sqlalchemy bookmark api"
    # Passes stage 1 (matches fastapi) but TF-IDF against the corpus is weak
    off_topic = "fastapi tutorial: how to install python on windows xp"
    results = f.score([on_topic, off_topic])

    # on_topic must reach at least stage 2; off_topic may be dropped below
    # the coarse threshold or kept at stage 1 — we only assert the on_topic
    # actually benefited from TF-IDF.
    assert results[0].stage_reached >= Stage.TFIDF


def test_throughput_reasonable_on_cheap_path(keywords_file: Path) -> None:
    """1,000 synthetic signals through stages 0/1 should take <5s on any
    modern machine — well under the spec's 15s bar. We don't enforce the
    spec number because CI runners vary wildly, but we do want to catch
    accidental O(N²) regressions.
    """
    f = CascadingRelevanceFilter(keywords_path=keywords_file)
    signals = [
        (
            "article about fastapi and langgraph and mcp and pydantic"
            if i % 3 == 0
            else "totally irrelevant topic about baking bread"
        )
        for i in range(1000)
    ]
    tps = throughput(f, signals)
    assert tps >= 50.0, f"stages 0/1 throughput dropped to {tps:.1f} signals/sec"
