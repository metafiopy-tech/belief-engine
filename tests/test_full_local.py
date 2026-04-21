"""Tests for Session 17: full-local primitives.

Covers:
  - ModelRouter.backend_for_call: probe-gated escalation when in
    local mode; no-op when escalation disabled, confidence missing,
    or already in cloud mode; escalation_count bookkeeping
  - NutrientProfile.compact limits per category and never mutates
    the source; format_context_block_compact matches expected caps
  - skeleton_cache: round-trip, stable fingerprint regardless of key
    order, corrupt file treated as miss, invalidate cleans up, hit
    count increments
  - ast_cache: parse_cached returns the same Module object for
    identical input, respects filename as part of the key, syntax
    errors cached without raising, clear_parse_cache resets state
  - robust_parse.try_parse_json: code-fence strip, trailing comma,
    single-quoted keys, bare keys, truncated JSON, wrapping prose,
    bytes input, non-JSON fallback to default
  - robust_parse.extract_code_block: language-typed + bare
    extraction, missing block returns None
  - local_benchmark.LocalBenchmarkReport round-trip through JSON,
    build_report from both dict shapes, compare_reports delta,
    format_comparison smoke
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import pytest

# ── ModelRouter probe escalation ──────────────────────────────────────────


from belief.config.models import (
    Backend,
    DEFAULT_ESCALATION_THRESHOLD,
    ModelRole,
    ModelRouter,
    RouteMode,
)


class TestProbeGatedEscalation:
    def _local_router(self) -> ModelRouter:
        r = ModelRouter()
        r.set_mode(RouteMode.LOCAL)
        return r

    def test_low_confidence_escalates(self):
        r = self._local_router()
        r.enable_probe_escalation()
        backend = r.backend_for_call(ModelRole.INTAKE, confidence=0.1)
        assert backend is Backend.CLOUD
        assert r.escalation_count == 1

    def test_high_confidence_stays_local(self):
        r = self._local_router()
        r.enable_probe_escalation()
        backend = r.backend_for_call(ModelRole.INTAKE, confidence=0.9)
        assert backend is Backend.LOCAL
        assert r.escalation_count == 0

    def test_threshold_boundary_strict_less_than(self):
        r = self._local_router()
        r.enable_probe_escalation(threshold=0.4)
        # Equal to threshold → stays local (spec: "< 0.4")
        assert r.backend_for_call(ModelRole.INTAKE, confidence=0.4) is Backend.LOCAL
        # Below → escalates
        assert r.backend_for_call(ModelRole.INTAKE, confidence=0.39) is Backend.CLOUD

    def test_no_confidence_no_escalation(self):
        r = self._local_router()
        r.enable_probe_escalation()
        assert r.backend_for_call(ModelRole.INTAKE) is Backend.LOCAL
        assert r.escalation_count == 0

    def test_escalation_disabled_by_default(self):
        """Routers that haven't opted in never escalate."""
        r = self._local_router()
        assert r.escalation_threshold is None
        assert r.backend_for_call(ModelRole.INTAKE, confidence=0.01) is Backend.LOCAL

    def test_cloud_mode_ignores_escalation(self):
        """Already-cloud calls aren't subject to further escalation."""
        r = ModelRouter()  # defaults to CLOUD
        r.enable_probe_escalation()
        assert r.backend_for_call(ModelRole.INTAKE, confidence=0.01) is Backend.CLOUD
        assert r.escalation_count == 0

    def test_disable_returns_to_legacy(self):
        r = self._local_router()
        r.enable_probe_escalation()
        r.disable_probe_escalation()
        assert r.escalation_threshold is None
        assert r.backend_for_call(ModelRole.INTAKE, confidence=0.0) is Backend.LOCAL

    def test_default_threshold_constant_matches_spec(self):
        assert DEFAULT_ESCALATION_THRESHOLD == 0.4


# ── NutrientProfile.compact (needs chromadb for memory package init) ─────


try:
    from belief.memory.nutrients import (
        Nutrient,
        NutrientProfile,
        NutrientType,
    )
    _HAS_MEMORY_PACKAGE = True
except ImportError:
    _HAS_MEMORY_PACKAGE = False


def _mk(content: str):
    return Nutrient(
        nutrient_type=NutrientType.PATTERN,
        content=content,
        embedding_text=content,
    )


@pytest.mark.skipif(not _HAS_MEMORY_PACKAGE,
                    reason="belief.memory requires chromadb")
class TestNutrientProfileCompact:
    def _populated(self) -> NutrientProfile:
        return NutrientProfile(
            covenants=[_mk(f"cov-{i}") for i in range(5)],
            antipatterns=[_mk(f"anti-{i}") for i in range(5)],
            patterns=[_mk(f"pat-{i}") for i in range(10)],
            skeletons=[_mk(f"skel-{i}") for i in range(3)],
        )

    def test_default_caps_match_spec(self):
        """Spec says "top-3 most relevant soil nutrients instead of top-10"."""
        p = self._populated()
        c = p.compact()
        assert len(c.patterns) == 3
        assert len(c.antipatterns) == 3
        assert len(c.skeletons) == 1

    def test_covenants_kept_by_default(self):
        """All covenants stay — dropping them loses hard-won invariants."""
        p = self._populated()
        c = p.compact()
        assert len(c.covenants) == 5

    def test_custom_covenant_cap(self):
        p = self._populated()
        c = p.compact(max_covenants=2)
        assert len(c.covenants) == 2

    def test_source_unchanged(self):
        p = self._populated()
        p.compact()
        # Source must not be mutated.
        assert len(p.patterns) == 10

    def test_empty_profile(self):
        assert NutrientProfile().compact().is_empty

    def test_format_context_block_compact_includes_expected_slugs(self):
        """The short-form context block has exactly top-N content."""
        p = self._populated()
        block = p.format_context_block_compact()
        # patterns top-3: pat-0, pat-1, pat-2; pat-3 should be absent
        assert "pat-0" in block
        assert "pat-2" in block
        assert "pat-3" not in block


# ── Skeleton cache ────────────────────────────────────────────────────────


from belief.cache.skeleton_cache import (
    CacheEntry,
    cache_size,
    cache_skeleton,
    fingerprint_spec,
    get_cached_skeleton,
    invalidate,
)


class TestSkeletonCache:
    def test_roundtrip(self, tmp_path):
        spec = {"goal": "x", "framework": "fastapi", "complexity": 2}
        skeleton = {"files": {"main.py": "print('x')"}}
        key = cache_skeleton(spec, skeleton, base_dir=tmp_path)
        got = get_cached_skeleton(spec, base_dir=tmp_path)
        assert got is not None
        assert got.key == key
        assert got.skeleton == skeleton
        assert got.spec == spec
        assert got.hit_count >= 1

    def test_fingerprint_stable_across_key_order(self):
        a = fingerprint_spec({"a": 1, "b": 2})
        b = fingerprint_spec({"b": 2, "a": 1})
        assert a == b

    def test_miss_when_not_cached(self, tmp_path):
        assert get_cached_skeleton({"goal": "nothing"},
                                    base_dir=tmp_path) is None

    def test_hit_count_increments(self, tmp_path):
        spec = {"goal": "y"}
        cache_skeleton(spec, {"files": {}}, base_dir=tmp_path)
        first = get_cached_skeleton(spec, base_dir=tmp_path)
        second = get_cached_skeleton(spec, base_dir=tmp_path)
        assert second.hit_count > first.hit_count

    def test_corrupt_file_treated_as_miss(self, tmp_path):
        spec = {"goal": "z"}
        key = cache_skeleton(spec, {"files": {}}, base_dir=tmp_path)
        # Corrupt the skeleton file.
        sk = tmp_path / key / "skeleton.json"
        sk.write_text("{ not json", encoding="utf-8")
        assert get_cached_skeleton(spec, base_dir=tmp_path) is None

    def test_invalidate_removes_entry(self, tmp_path):
        spec = {"goal": "to-remove"}
        cache_skeleton(spec, {"files": {}}, base_dir=tmp_path)
        assert invalidate(spec, base_dir=tmp_path)
        assert get_cached_skeleton(spec, base_dir=tmp_path) is None

    def test_invalidate_missing_returns_false(self, tmp_path):
        assert not invalidate({"goal": "never"}, base_dir=tmp_path)

    def test_cache_size_counts_entries(self, tmp_path):
        for i in range(3):
            cache_skeleton({"goal": f"g{i}"}, {"files": {}},
                            base_dir=tmp_path)
        assert cache_size(tmp_path) == 3

    def test_non_json_values_do_not_crash_fingerprint(self):
        """Fingerprint must not raise on unusual values."""
        class _Opaque:
            def __str__(self):
                return "opaque"
        fp = fingerprint_spec({"obj": _Opaque(), "n": 1})
        assert isinstance(fp, str)
        assert len(fp) == 16


# ── AST parse cache ───────────────────────────────────────────────────────


from belief.validators.ast_cache import (
    cache_stats,
    clear_parse_cache,
    parse_cached,
)


class TestASTCache:
    def setup_method(self):
        clear_parse_cache()

    def test_identical_source_returns_same_tree(self):
        src = "x = 1\ndef f():\n    return x\n"
        t1 = parse_cached(src)
        t2 = parse_cached(src)
        assert t1 is t2  # same object, not just equal

    def test_filename_affects_key(self):
        src = "x = 1"
        a = parse_cached(src, filename="a.py")
        b = parse_cached(src, filename="b.py")
        # Different filenames → different cache entries → distinct trees.
        assert a is not b

    def test_syntax_error_returns_none_by_default(self):
        result = parse_cached("def (( bad")
        assert result is None

    def test_syntax_error_can_raise(self):
        with pytest.raises(SyntaxError):
            parse_cached("def (( bad", raise_on_syntax_error=True)

    def test_stats_reflect_hits_and_misses(self):
        clear_parse_cache()
        parse_cached("a = 1")      # miss
        parse_cached("a = 1")      # hit
        parse_cached("b = 2")      # miss
        stats = cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2

    def test_clear_resets_state(self):
        parse_cached("x = 1")
        clear_parse_cache()
        stats = cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0


# ── robust_parse ──────────────────────────────────────────────────────────


from belief.utils.robust_parse import (
    extract_code_block,
    strip_code_fences,
    try_parse_json,
)


class TestStripCodeFences:
    def test_json_fence(self):
        assert strip_code_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'

    def test_bare_fence(self):
        assert strip_code_fences("```\n[1,2]\n```") == "[1,2]"

    def test_no_fence(self):
        assert strip_code_fences('{"a":1}') == '{"a":1}'


class TestTryParseJson:
    def test_plain_json_passthrough(self):
        assert try_parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert try_parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_wrapping_prose(self):
        raw = 'Here is the result:\n{"a": 1}\nThanks!'
        assert try_parse_json(raw) == {"a": 1}

    def test_trailing_comma(self):
        assert try_parse_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_single_quoted_keys(self):
        assert try_parse_json("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    def test_bare_keys(self):
        assert try_parse_json('{a: 1, b: 2}') == {"a": 1, "b": 2}

    def test_truncated_object(self):
        result = try_parse_json('{"a": 1, "b": {"c": 2')
        assert result == {"a": 1, "b": {"c": 2}}

    def test_list_root(self):
        assert try_parse_json("[1, 2, 3]") == [1, 2, 3]

    def test_already_parsed_dict(self):
        assert try_parse_json({"a": 1}) == {"a": 1}

    def test_already_parsed_list(self):
        assert try_parse_json([1, 2]) == [1, 2]

    def test_bytes_input(self):
        assert try_parse_json(b'{"a": 1}') == {"a": 1}

    def test_total_garbage_returns_default(self):
        assert try_parse_json("total garbage with no braces", default={}) == {}

    def test_none_returns_default(self):
        assert try_parse_json(None, default="EMPTY") == "EMPTY"


class TestExtractCodeBlock:
    def test_language_typed(self):
        raw = "before\n```python\nprint('hi')\n```\nafter"
        assert extract_code_block(raw, language="python") == "print('hi')"

    def test_bare(self):
        raw = "```\nx=1\n```"
        assert extract_code_block(raw) == "x=1"

    def test_missing_returns_none(self):
        assert extract_code_block("no block here") is None

    def test_language_mismatch_returns_none(self):
        raw = "```python\nx=1\n```"
        assert extract_code_block(raw, language="javascript") is None


# ── Local benchmark reporter ──────────────────────────────────────────────


from belief.metrics.local_benchmark import (
    LocalBenchmarkReport,
    ReportDelta,
    build_report,
    compare_reports,
    format_comparison,
    run_local_benchmark,
)


class TestLocalBenchmarkReport:
    def test_write_and_reload_json(self, tmp_path):
        r = LocalBenchmarkReport(
            mode="local", pass_rate=0.8, weighted_score=0.82,
            total_challenges=20, passed_challenges=16,
            build_time_s=600.0, soil_before=100, soil_after=120,
            soil_deposited=20, cost_usd=0.0, escalations=2,
        )
        path = r.write_json(tmp_path / "report.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["pass_rate"] == 0.8
        assert loaded["soil_deposited"] == 20

    def test_build_report_from_pass_rate_shape(self):
        summary = {"pass_rate": 0.9, "total": 10, "cost": 0.0,
                    "passing_ids": [f"c-{i}" for i in range(9)]}
        r = build_report(summary, soil_before=5, soil_after=8,
                         build_time_s=30.0)
        assert r.pass_rate == 0.9
        assert r.total_challenges == 10
        assert r.passed_challenges == 9
        assert r.soil_deposited == 3

    def test_build_report_from_challenges_shape(self):
        summary = {"challenges": [
            {"id": "c-1", "passed": True, "score": 1.0},
            {"id": "c-2", "passed": False, "score": 0.0},
        ]}
        r = build_report(summary, soil_before=0, soil_after=0)
        assert r.total_challenges == 2
        assert r.passed_challenges == 1
        assert r.pass_rate == pytest.approx(0.5)

    def test_run_local_benchmark(self, tmp_path):
        async def runner(tiers=None, ids=None, **kw):
            return {
                "pass_rate": 0.75, "total": 4,
                "passing_ids": ["a", "b", "c"],
                "cost": 0.0,
            }
        class _Soil:
            n = 10
            def count(self): return self.n
        soil = _Soil()

        async def _run():
            return await run_local_benchmark(
                runner, tiers=[1, 2], soil=soil, notes="smoke",
            )
        report = asyncio.run(_run())
        assert report.pass_rate == 0.75
        assert report.tiers == [1, 2]
        assert report.notes == "smoke"
        assert report.soil_before == 10
        assert report.soil_after == 10


class TestCompareReports:
    def _make(self, pass_rate: float, score: float,
              challenges: Optional[list[dict]] = None) -> LocalBenchmarkReport:
        return LocalBenchmarkReport(
            pass_rate=pass_rate,
            weighted_score=score,
            challenges=challenges or [],
        )

    def test_delta_fields(self):
        left = self._make(0.6, 0.58)
        right = self._make(0.8, 0.78)
        d = compare_reports(left, right)
        assert d.pass_rate_delta == pytest.approx(0.2)
        assert d.weighted_score_delta == pytest.approx(0.2)

    def test_challenge_flip_recorded(self):
        left = self._make(0.5, 0.5, challenges=[
            {"id": "c-1", "passed": True},
            {"id": "c-2", "passed": False, "error": "timeout"},
        ])
        right = self._make(0.5, 0.5, challenges=[
            {"id": "c-1", "passed": False, "error": "new-regression"},
            {"id": "c-2", "passed": True},
        ])
        d = compare_reports(left, right)
        assert "c-1" in d.challenges_diff
        assert "c-2" in d.challenges_diff

    def test_format_comparison_text(self):
        left = self._make(0.5, 0.5)
        right = self._make(0.8, 0.8)
        text = format_comparison(compare_reports(left, right))
        assert "Benchmark comparison" in text
        assert "pass rate" in text
