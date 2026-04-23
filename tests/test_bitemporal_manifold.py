"""Tests for Session 14: bi-temporal knowledge + domain manifold.

Covers:
  - Nutrient gains valid_from / valid_until / invalidation_reason
  - is_valid_at() handles creation-time, boundary, and post-invalidation
  - to_chromadb_metadata / from_chromadb round-trip preserves validity
  - Legacy records (no bi-temporal keys) back-fill sensibly
  - soil.invalidate_nutrient() soft-deletes and records the reason
  - invalidated nutrients are excluded from retrieve()
  - retrieve_as_of(past) reconstructs the pre-invalidation view
  - iter_all_nutrients(include_invalidated=False) filters invalidated
  - manifold.primary_domain / nutrient_domains classify correctly
  - manifold.build_manifold clusters, counts cross-edges, flags gaps
  - manifold.format_report + to_json produce valid output
  - manifold excludes invalidated nutrients from active view
"""

from __future__ import annotations

import json
from typing import Optional

import pytest


# ── Pure-stdlib tests (no chromadb) ────────────────────────────────────────


from belief.memory.nutrients import Nutrient, NutrientType, _now_ts


class TestNutrientBiTemporalFields:
    def test_defaults(self):
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="x",
            embedding_text="x",
        )
        assert n.valid_until == 0.0
        assert n.valid_from > 0.0
        assert n.invalidation_reason == ""
        assert n.is_active()
        assert n.is_valid_at()  # defaults to now

    def test_is_valid_at_before_creation_is_false(self):
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="x",
            embedding_text="x",
        )
        assert not n.is_valid_at(n.valid_from - 1)

    def test_is_valid_at_after_invalidation_is_false(self):
        past = _now_ts() - 1000
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="x",
            embedding_text="x",
            valid_from=past,
            valid_until=past + 500,
        )
        # Inside the valid window
        assert n.is_valid_at(past + 100)
        # At the upper boundary — exclusive (valid_until is *when* it
        # became invalid, so the point itself is no longer active).
        assert not n.is_valid_at(past + 500)
        assert not n.is_valid_at(past + 501)
        # is_active() now — it's been invalidated, so False.
        assert not n.is_active()

    def test_metadata_round_trip(self):
        past = _now_ts() - 1000
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="bad pattern",
            embedding_text="bad pattern",
            valid_from=past,
            valid_until=past + 500,
            invalidation_reason="superseded by pattern-v2",
        )
        meta = n.to_chromadb_metadata()
        assert meta["valid_from"] == past
        assert meta["valid_until"] == past + 500
        assert meta["invalidation_reason"] == "superseded by pattern-v2"
        n2 = Nutrient.from_chromadb(n.nutrient_id, n.embedding_text, meta)
        assert n2.valid_from == n.valid_from
        assert n2.valid_until == n.valid_until
        assert n2.invalidation_reason == n.invalidation_reason

    def test_legacy_metadata_backfills(self):
        """Records written before Session 14 lack the bi-temporal keys."""
        legacy = {
            "nutrient_type": "pattern",
            "tier": 1,
            "content": "legacy pattern",
            "created_at": 1000.0,
            "last_reinforced": 1000.0,
            "source_build_id": "",
            "framework": "",
            "tags": ["_none"],
            "lineage_parent_ids": ["_none"],
            "stability": 1.0,
            "difficulty": 5.0,
            "reinforcement_count": 0,
            "lapse_count": 0,
        }
        n = Nutrient.from_chromadb("legacy-id", "legacy pattern", legacy)
        # valid_from falls back to created_at; valid_until=0.0 (active).
        assert n.valid_from == 1000.0
        assert n.valid_until == 0.0
        assert n.invalidation_reason == ""
        assert n.is_active()

    def test_iso_string_timestamps_accepted(self):
        """ChromaDB round-trips sometimes produce ISO strings for timestamps."""
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="iso",
            embedding_text="iso",
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
        )
        assert isinstance(n.valid_from, float)
        assert isinstance(n.valid_until, float)
        assert n.valid_from < n.valid_until


# ── Manifold classification (no Soil needed) ──────────────────────────────


from belief.memory.manifold import (  # noqa: E402 — grouped with the tests that use it
    build_manifold,
    format_report,
    nutrient_domains,
    primary_domain,
)


class _FakeNutrient:
    """Duck-type match for Nutrient — only fields the manifold touches."""

    def __init__(
        self,
        nid: str = "n-0",
        content: str = "",
        framework: Optional[str] = None,
        tags=None,
        reinforcement_count: int = 0,
        lapse_count: int = 0,
        stability: float = 1.0,
        valid_from: float = 100.0,
        valid_until: float = 0.0,
    ):
        self.nutrient_id = nid
        self.content = content
        self.framework = framework
        self.tags = list(tags or [])
        self.reinforcement_count = reinforcement_count
        self.lapse_count = lapse_count
        self.stability = stability
        self.valid_from = valid_from
        self.valid_until = valid_until

    def is_valid_at(self, ts):
        if self.valid_from > ts:
            return False
        if self.valid_until > 0 and ts >= self.valid_until:
            return False
        return True


class _FakeSoil:
    def __init__(self, nutrients):
        self._ns = list(nutrients)

    def iter_all_nutrients(self, include_invalidated: bool = True, as_of: Optional[float] = None):
        ts = as_of if as_of is not None else _now_ts()
        for n in self._ns:
            if not include_invalidated:
                if n.valid_until > 0 and ts >= n.valid_until:
                    continue
            yield n


class TestDomainClassification:
    def test_framework_dominates(self):
        n = _FakeNutrient(framework="fastapi", content="irrelevant")
        assert primary_domain(n) == "fastapi"

    def test_tag_matches_when_framework_missing(self):
        n = _FakeNutrient(framework=None, content="", tags=["cli"])
        assert primary_domain(n) == "cli"

    def test_async_tag_matches_despite_trailing_space_keyword(self):
        """DOMAINS['async'] has 'async ' (with trailing space) — tag
        lookup must still match the bare 'async' tag."""
        n = _FakeNutrient(framework=None, content="", tags=["async"])
        assert primary_domain(n) == "async"

    def test_content_fallback(self):
        n = _FakeNutrient(framework=None, content="Build an MCP tool server")
        assert primary_domain(n) == "mcp"

    def test_general_when_nothing_matches(self):
        n = _FakeNutrient(framework=None, content="totally generic utility", tags=["helper"])
        assert primary_domain(n) == "general"

    def test_nutrient_domains_returns_union(self):
        """A nutrient tagged with multiple verticals appears in each."""
        n = _FakeNutrient(
            framework="fastapi",
            content="FastAPI async websocket pipeline",
            tags=["async"],
        )
        doms = nutrient_domains(n)
        assert "fastapi" in doms
        assert "async" in doms


class TestManifoldBuild:
    def _make_soil(self):
        return _FakeSoil(
            [
                _FakeNutrient(
                    "f1", "FastAPI users endpoint", framework="fastapi", reinforcement_count=5
                ),
                _FakeNutrient(
                    "f2", "FastAPI auth middleware", framework="fastapi", reinforcement_count=3
                ),
                _FakeNutrient(
                    "f3",
                    "FastAPI async websocket pipe",
                    framework="fastapi",
                    tags=["async"],
                    reinforcement_count=7,
                ),
                _FakeNutrient("c1", "Click CLI entry point", tags=["cli"]),
                _FakeNutrient("c2", "CLI argument parsing", tags=["cli"]),
                _FakeNutrient("m1", "MCP tool server scaffold", framework="mcp"),
                _FakeNutrient("x1", "Random generic utility", tags=["helper"]),
                # Invalidated — should not appear in active clusters.
                _FakeNutrient(
                    "inv", "Invalidated FastAPI thing", framework="fastapi", valid_until=100.0
                ),
            ]
        )

    def test_clusters_exclude_invalidated(self):
        report = build_manifold(self._make_soil(), gap_threshold=3)
        assert report.total_active == 7
        assert report.total_invalidated == 1
        fastapi = next(c for c in report.clusters if c.domain == "fastapi")
        assert fastapi.size == 3  # f1, f2, f3 (inv excluded)

    def test_cluster_sample_content_ordered_by_reuse(self):
        report = build_manifold(self._make_soil(), gap_threshold=3)
        fastapi = next(c for c in report.clusters if c.domain == "fastapi")
        # f3 has reinforcement_count=7 — should be first.
        assert "async websocket" in fastapi.sample_content[0].lower()

    def test_cross_edge_detected(self):
        report = build_manifold(self._make_soil(), gap_threshold=3)
        keys = {(e.domain_a, e.domain_b) for e in report.cross_edges}
        assert ("async", "fastapi") in keys

    def test_coverage_gaps(self):
        """Threshold 3 → everything except fastapi is sparse."""
        report = build_manifold(self._make_soil(), gap_threshold=3)
        assert "mcp" in report.coverage_gaps
        assert "async" in report.coverage_gaps
        assert "fastapi" not in report.coverage_gaps

    def test_empty_soil(self):
        report = build_manifold(_FakeSoil([]), gap_threshold=5)
        assert report.total_active == 0
        assert report.total_invalidated == 0
        # Every domain is a gap when there are zero nutrients.
        for dom in ["fastapi", "cli", "mcp", "data", "async", "library", "script", "general"]:
            assert dom in report.coverage_gaps

    def test_format_report_is_nonempty_string(self):
        report = build_manifold(self._make_soil(), gap_threshold=3)
        text = format_report(report, gap_threshold=3)
        assert isinstance(text, str)
        assert "Domain Manifold" in text
        assert "Coverage gaps" in text

    def test_to_json_parses_and_has_expected_keys(self):
        report = build_manifold(self._make_soil(), gap_threshold=3)
        payload = json.loads(report.to_json())
        assert payload["total_active"] == 7
        assert payload["total_invalidated"] == 1
        assert "clusters" in payload
        assert "cross_edges" in payload
        assert "coverage_gaps" in payload
        # Cluster shape
        first = payload["clusters"][0]
        for k in ("domain", "size", "mean_stability", "lapse_rate", "sample_content"):
            assert k in first


# ── Soil integration (needs chromadb) ──────────────────────────────────────

chromadb = pytest.importorskip("chromadb")

from belief.memory.soil import Soil  # noqa: E402


@pytest.fixture
def soil(tmp_path):
    return Soil(persist_dir=tmp_path / "soil")


class TestSoilInvalidation:
    def test_invalidate_soft_deletes(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="wrong pattern",
            embedding_text="wrong pattern",
        )
        soil.deposit(n)
        ok = soil.invalidate_nutrient(n.nutrient_id, "superseded")
        assert ok is True

        fetched = soil.get(n.nutrient_id)
        assert fetched is not None, "invalidated nutrient must still exist"
        assert fetched.valid_until > 0
        assert fetched.invalidation_reason == "superseded"
        assert not fetched.is_active()

    def test_invalidate_missing_returns_false(self, soil):
        assert soil.invalidate_nutrient("does-not-exist", "nope") is False

    def test_invalidate_idempotent(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="once",
            embedding_text="once",
        )
        soil.deposit(n)
        assert soil.invalidate_nutrient(n.nutrient_id, "first") is True
        # Second call is a no-op — preserves the original reason.
        assert soil.invalidate_nutrient(n.nutrient_id, "second") is False
        fetched = soil.get(n.nutrient_id)
        assert fetched.invalidation_reason == "first"

    def test_retrieve_excludes_invalidated(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="FastAPI CRUD pattern with user routes",
            embedding_text="FastAPI CRUD user route",
        )
        soil.deposit(n)
        before = soil.retrieve("FastAPI user route", n=5)
        assert any(x.nutrient_id == n.nutrient_id for x in before)

        soil.invalidate_nutrient(n.nutrient_id, "obsolete")
        after = soil.retrieve("FastAPI user route", n=5)
        assert not any(x.nutrient_id == n.nutrient_id for x in after)

    def test_retrieve_as_of_reconstructs_past(self, soil):
        past = _now_ts() - 3600  # One hour ago.
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="FastAPI user route",
            embedding_text="FastAPI user route",
            valid_from=past - 1000,  # created before our snapshot
        )
        soil.deposit(n)
        # Mutate stored valid_from so as_of sees it as already-created.
        col = soil._collections["belief_principles"]
        m = col.get(ids=[n.nutrient_id], include=["metadatas"])["metadatas"][0]
        m["valid_from"] = past - 1000
        col.update(ids=[n.nutrient_id], metadatas=[m])

        soil.invalidate_nutrient(n.nutrient_id, "bad pattern")

        # Now (live view): invalidated — excluded.
        live = soil.retrieve("FastAPI user route", n=5)
        assert not any(x.nutrient_id == n.nutrient_id for x in live)

        # Historical view: look at the state *before* invalidation.
        historical = soil.retrieve_as_of(past, query="FastAPI user route", n=5)
        assert any(x.nutrient_id == n.nutrient_id for x in historical)

    def test_iter_all_nutrients_filter(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="pattern",
            embedding_text="pattern A",
        )
        soil.deposit(n)
        soil.invalidate_nutrient(n.nutrient_id, "x")

        active = list(soil.iter_all_nutrients(include_invalidated=False))
        assert not any(x.nutrient_id == n.nutrient_id for x in active)

        all_ = list(soil.iter_all_nutrients(include_invalidated=True))
        assert any(x.nutrient_id == n.nutrient_id for x in all_)

    def test_count_helpers(self, soil):
        n = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="x",
            embedding_text="unique-for-count-test",
        )
        soil.deposit(n)
        assert soil.count_active() == 1
        assert soil.count_invalidated() == 0

        soil.invalidate_nutrient(n.nutrient_id, "reason")
        assert soil.count_active() == 0
        assert soil.count_invalidated() == 1


class TestManifoldAgainstRealSoil:
    def test_end_to_end(self, soil):
        """Populate a real soil with three domains, one invalidation,
        and verify the manifold reflects the live view."""
        # Use fastapi framework, unique contents to avoid 0.92-sim
        # dedup collapsing them.
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="fastapi crud endpoint scaffold",
                embedding_text="unique-fastapi-pattern-alpha-scaffold",
                framework="fastapi",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="fastapi auth dependency injection",
                embedding_text="unique-fastapi-pattern-beta-authx",
                framework="fastapi",
            )
        )
        soil.deposit(
            Nutrient(
                nutrient_type=NutrientType.PATTERN,
                content="click CLI entry",
                embedding_text="unique-cli-pattern-gamma-entry",
                tags=["cli"],
            )
        )
        # Invalidated — should NOT appear in clusters.
        ghost = Nutrient(
            nutrient_type=NutrientType.PATTERN,
            content="fastapi deprecated pattern",
            embedding_text="unique-fastapi-ghost-delta-deprecated",
            framework="fastapi",
        )
        soil.deposit(ghost)
        soil.invalidate_nutrient(ghost.nutrient_id, "deprecated-api")

        report = build_manifold(soil, gap_threshold=3)
        assert report.total_active == 3  # 4 deposited, 1 invalidated
        assert report.total_invalidated == 1
        fastapi = next(c for c in report.clusters if c.domain == "fastapi")
        assert fastapi.size == 2  # ghost excluded
        cli = next(c for c in report.clusters if c.domain == "cli")
        assert cli.size == 1
        assert "cli" in report.coverage_gaps  # 1 < 3
