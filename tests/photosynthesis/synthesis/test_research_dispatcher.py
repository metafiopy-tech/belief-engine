"""Tests for the research dispatcher (SE Session 5)."""

from __future__ import annotations

import asyncio


from belief.photosynthesis.synthesis.atomizer import atomize_words
from belief.photosynthesis.synthesis.research_dispatcher import (
    RetrievedDoc,
    SearchableSource,
    dispatch,
)


# ---------------------------------------------------------------------------
# Stub source -- used by all hermetic tests
# ---------------------------------------------------------------------------


class _StubSource:
    """In-memory source that returns canned docs per query.

    Tests can configure `return_per_query` and inspect `calls` after.
    """

    def __init__(self, name: str, *, docs_per_query: int = 1) -> None:
        self.name = name
        self.docs_per_query = docs_per_query
        self.calls: list[str] = []

    async def search(self, client, query: str) -> list[RetrievedDoc]:
        self.calls.append(query)
        idx = len(self.calls)
        # Use a per-source URL stem; same query twice -> same URL ->
        # dedup collapses them.
        return [
            RetrievedDoc(
                url=f"http://stub.example/{self.name}/{idx}/{i}",
                title=f"{self.name} #{idx}.{i}",
                summary=f"summary for: {query[:40]}",
                source=self.name,
                raw_excerpt=query,
            )
            for i in range(self.docs_per_query)
        ]


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestSearchableSource:
    def test_stub_satisfies_protocol(self) -> None:
        src = _StubSource("arxiv")
        assert isinstance(src, SearchableSource)


# ---------------------------------------------------------------------------
# Dispatch behavior
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_empty_prompts_returns_empty_list(self) -> None:
        result = _run(dispatch([], sources=[_StubSource("arxiv")]))
        assert result == []

    def test_basic_fan_out_two_sources(self) -> None:
        prompts = atomize_words(["mantis_shrimp", "camera"])
        arxiv = _StubSource("arxiv")
        github = _StubSource("github")
        pypi = _StubSource("pypi")
        docs = _run(dispatch(prompts, sources=[arxiv, github, pypi]))
        # Every prompt hints at least one source (arxiv is in all 5
        # axes via the source-hint defaults), so arxiv gets called for
        # all 30 prompts.
        assert len(arxiv.calls) == 30
        # github is hinted by mechanism, tradeoff, counterexample
        # axes: 3 axes * 3 templates * 2 words = 18.
        assert len(github.calls) == 18
        # pypi is not hinted by any default axis -- 0 calls.
        assert len(pypi.calls) == 0
        # Every doc has unique URL in the stub -- no dedup collapse.
        assert len(docs) == 30 + 18

    def test_dedup_by_url_collapses_same_doc_across_prompts(self) -> None:
        """When the same URL is returned by multiple prompts, dedup
        keeps a single doc and records all surfacing prompts."""
        prompts = atomize_words(["camera"])

        class FixedUrlSource:
            name = "arxiv"

            async def search(self, client, query):
                return [
                    RetrievedDoc(
                        url="http://stub.example/fixed",
                        title="fixed",
                        summary="...",
                        source=self.name,
                    )
                ]

        docs = _run(dispatch(prompts, sources=[FixedUrlSource()]))
        assert len(docs) == 1
        # All 15 prompts should be recorded as having surfaced this
        # one doc (the camera word produces 15 prompts, all hint arxiv).
        assert len(docs[0].prompts) == 15

    def test_source_hints_filter_routing(self) -> None:
        """When a prompt's source_hints excludes a source, that
        source should NOT be called for that prompt."""
        prompts = atomize_words(["camera"])
        # cross_domain axis hints only arxiv.
        # github should NOT be called for cross_domain prompts.
        arxiv = _StubSource("arxiv")
        github = _StubSource("github")
        _run(dispatch(prompts, sources=[arxiv, github]))
        # cross_domain has 3 prompts, all of which exclude github.
        # mechanism, tradeoff, counterexample each include github
        # (3 axes * 3 prompts = 9 github calls expected for one word).
        assert len(github.calls) == 9
        # arxiv is in every axis's hints -> 15 calls for one word.
        assert len(arxiv.calls) == 15

    def test_respect_source_hints_false_fans_to_all(self) -> None:
        prompts = atomize_words(["camera"])
        arxiv = _StubSource("arxiv")
        github = _StubSource("github")
        pypi = _StubSource("pypi")
        _run(
            dispatch(
                prompts,
                sources=[arxiv, github, pypi],
                respect_source_hints=False,
            )
        )
        # Every prompt goes to every source: 15 prompts * 3 = 45 each.
        assert len(arxiv.calls) == 15
        assert len(github.calls) == 15
        assert len(pypi.calls) == 15

    def test_source_failure_does_not_abort_others(self) -> None:
        """An exception from one source must not poison others."""
        prompts = atomize_words(["camera"])

        class BoomSource:
            name = "arxiv"

            async def search(self, client, query):
                raise RuntimeError("simulated network failure")

        github = _StubSource("github")
        docs = _run(dispatch(prompts, sources=[BoomSource(), github]))
        # github is still hinted by mechanism/tradeoff/counterexample
        # axes (9 prompts), so we should get 9 docs from github even
        # though the arxiv source is throwing.
        assert len(docs) == len(github.calls)
        assert all(d.source == "github" for d in docs)

    def test_doc_prompts_field_records_axis(self) -> None:
        """Caller can trace each doc back to which axis surfaced it."""
        prompts = atomize_words(["camera"])
        # One unique URL per call -- no collapsing.
        src = _StubSource("arxiv", docs_per_query=1)
        docs = _run(dispatch(prompts, sources=[src]))
        for d in docs:
            assert len(d.prompts) == 1
            assert d.prompts[0].axis in {
                "mechanism",
                "constraint",
                "tradeoff",
                "cross_domain",
                "counterexample",
            }


# ---------------------------------------------------------------------------
# RetrievedDoc.doc_key
# ---------------------------------------------------------------------------


class TestDocKey:
    def test_url_canonical_when_present(self) -> None:
        d = RetrievedDoc(url="http://example.com/x/", title="T", summary="S", source="arxiv")
        # Trailing slash gets stripped for stable dedup.
        assert d.doc_key() == "http://example.com/x"

    def test_falls_back_to_source_title(self) -> None:
        d = RetrievedDoc(url="", title="my title", summary="S", source="arxiv")
        assert d.doc_key() == "arxiv:my title"

    def test_dedup_treats_trailing_slash_consistently(self) -> None:
        a = RetrievedDoc(url="http://x/foo/", title="A", summary="", source="arxiv")
        b = RetrievedDoc(url="http://x/foo", title="B", summary="", source="github")
        assert a.doc_key() == b.doc_key()
