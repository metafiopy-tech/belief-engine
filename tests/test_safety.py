"""Tests for safety guardrails and metrics dashboard.

Covers:
  - initialize_probes hashes files correctly
  - check_evaluator_integrity catches modifications
  - check_test_harness_edits catches diffs touching forbidden paths
  - check_environment_tampering detects env changes
  - check_resource_consumption catches cost spikes
  - Goodhart canary detects divergence (synthetic score sequences)
  - Dashboard records and loads metrics
  - Growth analysis returns correct fit types
  - Dashboard prints formatted output
  - Overseer instantiation and violation tracking
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from belief.safety.overseer import AsyncOverseer, SafetyViolation


# ── SafetyViolation ────────────────────────────────────────────────────────


class TestSafetyViolation:
    def test_violation_attributes(self):
        v = SafetyViolation("test_probe", "something bad", "critical")
        assert v.probe_name == "test_probe"
        assert v.description == "something bad"
        assert v.severity == "critical"
        assert "critical" in str(v)

    def test_violation_is_exception(self):
        v = SafetyViolation("p", "d", "warning")
        assert isinstance(v, Exception)


# ── Overseer ───────────────────────────────────────────────────────────────


class TestOverseer:
    def test_init(self):
        o = AsyncOverseer()
        assert o.model == "claude-haiku-4-5-20251001"
        assert o.check_interval == 30.0
        assert not o.has_critical

    def test_violations_tracking(self):
        o = AsyncOverseer()
        assert o.violations == []
        o._violations.append(SafetyViolation("t", "d", "warning"))
        assert len(o.violations) == 1
        assert not o.has_critical

    def test_has_critical(self):
        o = AsyncOverseer()
        o._violations.append(SafetyViolation("t", "d", "critical"))
        assert o.has_critical

    def test_stop(self):
        o = AsyncOverseer()
        o._running = True
        o.stop()
        assert not o._running


# ── Probes: initialize ─────────────────────────────────────────────────────


class TestInitializeProbes:
    def test_hashes_files(self, tmp_path):
        """initialize_probes should hash existing critical files."""
        from belief.safety.probes import _CRITICAL_FILE_HASHES, initialize_probes

        # Create fake critical files
        (tmp_path / "belief").mkdir()
        (tmp_path / "belief" / "benchmark.py").write_text("# benchmark code")
        (tmp_path / "belief" / "hardening.py").write_text("# hardening code")
        (tmp_path / "belief" / "validators").mkdir()
        (tmp_path / "belief" / "validators" / "__init__.py").write_text("# validators")

        initialize_probes(str(tmp_path))
        assert len(_CRITICAL_FILE_HASHES) == 3
        # Hashes are stored with absolute paths
        assert any("benchmark.py" in k for k in _CRITICAL_FILE_HASHES)

    def test_hashes_missing_files_skipped(self, tmp_path):
        from belief.safety.probes import _CRITICAL_FILE_HASHES, initialize_probes

        initialize_probes(str(tmp_path))
        # No files exist, so no hashes
        assert len(_CRITICAL_FILE_HASHES) == 0


# ── Probes: evaluator integrity ────────────────────────────────────────────


class TestEvaluatorIntegrity:
    @pytest.mark.asyncio
    async def test_catches_modification(self, tmp_path):
        from belief.safety.probes import (
            check_evaluator_integrity,
            initialize_probes,
        )

        (tmp_path / "belief").mkdir()
        bench_path = tmp_path / "belief" / "benchmark.py"
        bench_path.write_text("original content")

        initialize_probes(str(tmp_path))

        # Modify the file
        bench_path.write_text("modified content")

        with pytest.raises(SafetyViolation, match="modified"):
            await check_evaluator_integrity()

    @pytest.mark.asyncio
    async def test_catches_deletion(self, tmp_path):
        from belief.safety.probes import (
            check_evaluator_integrity,
            initialize_probes,
        )

        (tmp_path / "belief").mkdir()
        bench_path = tmp_path / "belief" / "benchmark.py"
        bench_path.write_text("content")

        initialize_probes(str(tmp_path))

        # Delete the file
        bench_path.unlink()

        with pytest.raises(SafetyViolation, match="deleted"):
            await check_evaluator_integrity()

    @pytest.mark.asyncio
    async def test_passes_when_unchanged(self, tmp_path):
        from belief.safety.probes import (
            check_evaluator_integrity,
            initialize_probes,
        )

        (tmp_path / "belief").mkdir()
        (tmp_path / "belief" / "benchmark.py").write_text("content")

        initialize_probes(str(tmp_path))

        # Should not raise
        await check_evaluator_integrity()


# ── Probes: test harness edits ─────────────────────────────────────────────


class TestTestHarnessEdits:
    @pytest.mark.asyncio
    async def test_catches_benchmark_diff(self):
        from belief.safety.probes import check_test_harness_edits

        mock_version = MagicMock()
        mock_version.diff_from_parent = "Modified belief/benchmark.py to increase pass rate"

        with patch("belief.evolution.archive.Archive") as MockArchive:
            MockArchive.return_value.get_all_versions.return_value = [
                MagicMock(),  # seed
                mock_version,
            ]

            with pytest.raises(SafetyViolation, match="benchmark"):
                await check_test_harness_edits()

    @pytest.mark.asyncio
    async def test_catches_tests_dir_diff(self):
        from belief.safety.probes import check_test_harness_edits

        mock_version = MagicMock()
        mock_version.diff_from_parent = "Updated tests/ to make them pass"

        with patch("belief.evolution.archive.Archive") as MockArchive:
            MockArchive.return_value.get_all_versions.return_value = [
                MagicMock(),
                mock_version,
            ]

            with pytest.raises(SafetyViolation, match="tests/"):
                await check_test_harness_edits()

    @pytest.mark.asyncio
    async def test_passes_safe_diff(self):
        from belief.safety.probes import check_test_harness_edits

        mock_version = MagicMock()
        mock_version.diff_from_parent = "Updated belief/prompts/__init__.py to improve builder"

        with patch("belief.evolution.archive.Archive") as MockArchive:
            MockArchive.return_value.get_all_versions.return_value = [
                MagicMock(),
                mock_version,
            ]

            # Should not raise
            await check_test_harness_edits()


# ── Probes: environment tampering ──────────────────────────────────────────


class TestEnvironmentTampering:
    @pytest.mark.asyncio
    async def test_detects_key_change(self):
        from belief.safety.probes import _ENV_SNAPSHOTS, check_environment_tampering

        # Reset state
        _ENV_SNAPSHOTS.clear()

        # First call: snapshot
        await check_environment_tampering()

        # Simulate key change
        original = _ENV_SNAPSHOTS.get("ANTHROPIC_API_KEY", "")
        _ENV_SNAPSHOTS["ANTHROPIC_API_KEY"] = "original_key_value"

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "different_key"}):
            with pytest.raises(SafetyViolation, match="ANTHROPIC_API_KEY"):
                await check_environment_tampering()

        # Restore
        _ENV_SNAPSHOTS.clear()

    @pytest.mark.asyncio
    async def test_passes_when_unchanged(self):
        from belief.safety.probes import _ENV_SNAPSHOTS, check_environment_tampering

        _ENV_SNAPSHOTS.clear()
        await check_environment_tampering()  # Snapshot

        # Second call with same env — should not raise
        await check_environment_tampering()

        _ENV_SNAPSHOTS.clear()


# ── Probes: resource consumption ───────────────────────────────────────────


class TestResourceConsumption:
    @pytest.mark.asyncio
    async def test_catches_high_avg_cost(self):
        from belief.safety.probes import check_resource_consumption

        mock_results = [MagicMock(cost_usd=8.0) for _ in range(5)]

        with patch("belief.evolution.archive.Archive") as MockArchive:
            MockArchive.return_value.get_all_results_recent.return_value = mock_results

            with pytest.raises(SafetyViolation, match="Avg cost"):
                await check_resource_consumption()

    @pytest.mark.asyncio
    async def test_catches_doubling_trend(self):
        from belief.safety.probes import check_resource_consumption

        # First half: $1 avg, second half: $4 avg (>2x)
        mock_results = [
            MagicMock(cost_usd=1.0), MagicMock(cost_usd=1.0), MagicMock(cost_usd=1.0),
            MagicMock(cost_usd=4.0), MagicMock(cost_usd=4.0), MagicMock(cost_usd=4.0),
        ]

        with patch("belief.evolution.archive.Archive") as MockArchive:
            MockArchive.return_value.get_all_results_recent.return_value = mock_results

            with pytest.raises(SafetyViolation, match="doubling"):
                await check_resource_consumption()

    @pytest.mark.asyncio
    async def test_passes_normal_costs(self):
        from belief.safety.probes import check_resource_consumption

        mock_results = [MagicMock(cost_usd=0.5) for _ in range(5)]

        with patch("belief.evolution.archive.Archive") as MockArchive:
            MockArchive.return_value.get_all_results_recent.return_value = mock_results

            # Should not raise
            await check_resource_consumption()

    @pytest.mark.asyncio
    async def test_skips_insufficient_data(self):
        from belief.safety.probes import check_resource_consumption

        mock_results = [MagicMock(cost_usd=100.0)]  # Only 1 result

        with patch("belief.evolution.archive.Archive") as MockArchive:
            MockArchive.return_value.get_all_results_recent.return_value = mock_results

            # Should not raise (too few results)
            await check_resource_consumption()


# ── Goodhart Canary ────────────────────────────────────────────────────────


class TestGoodhartCanary:
    def test_canary_challenges_exist(self):
        from belief.safety.goodhart_canary import CANARY_CHALLENGES
        assert len(CANARY_CHALLENGES) == 3
        assert all("id" in c and "goal" in c for c in CANARY_CHALLENGES)

    def test_detects_divergence(self):
        from belief.safety.goodhart_canary import check_goodhart_divergence

        # Proxy improving, canary declining
        diverging = check_goodhart_divergence(
            proxy_scores=[0.5, 0.6, 0.7, 0.8, 0.85],
            canary_scores=[0.5, 0.5, 0.48, 0.45, 0.42],
        )
        assert diverging is True

    def test_no_divergence_when_both_improve(self):
        from belief.safety.goodhart_canary import check_goodhart_divergence

        # Both improving at similar rates — gap stays small
        not_diverging = check_goodhart_divergence(
            proxy_scores=[0.5, 0.55, 0.6, 0.65, 0.7],
            canary_scores=[0.5, 0.55, 0.6, 0.65, 0.7],
        )
        assert not_diverging is False

    def test_no_divergence_insufficient_data(self):
        from belief.safety.goodhart_canary import check_goodhart_divergence

        assert check_goodhart_divergence([0.5, 0.6], [0.5, 0.6]) is False

    def test_detects_gap_widening(self):
        from belief.safety.goodhart_canary import check_goodhart_divergence

        # Both improving but proxy faster — gap widens
        diverging = check_goodhart_divergence(
            proxy_scores=[0.5, 0.55, 0.6, 0.7, 0.8],
            canary_scores=[0.5, 0.5, 0.5, 0.5, 0.5],
        )
        assert diverging is True

    def test_canary_ids_unique(self):
        from belief.safety.goodhart_canary import CANARY_CHALLENGES
        ids = [c["id"] for c in CANARY_CHALLENGES]
        assert len(ids) == len(set(ids))

    def test_canary_not_in_benchmark(self):
        """Canary challenges must NOT appear in benchmark.py."""
        from belief.safety.goodhart_canary import CANARY_CHALLENGES

        try:
            from belief.benchmark import CHALLENGES
            benchmark_ids = {c.id for c in CHALLENGES}
            for canary in CANARY_CHALLENGES:
                assert canary["id"] not in benchmark_ids, (
                    f"Canary {canary['id']} found in benchmark — Goodhart violation!"
                )
        except ImportError:
            pass  # benchmark not available in test env


# ── Metrics Dashboard ──────────────────────────────────────────────────────


class TestDashboard:
    def test_record_and_load(self, tmp_path):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        db_path = str(tmp_path / "metrics.jsonl")
        dashboard = MetricsDashboard(db_path=db_path)

        m = IterationMetrics(
            iteration=1,
            timestamp="2025-01-01T00:00:00",
            benchmark_score=0.65,
            cost_per_solved=1.50,
            novel_capabilities=3,
        )
        dashboard.record(m)

        loaded = dashboard.load_all()
        assert len(loaded) == 1
        assert loaded[0].iteration == 1
        assert loaded[0].benchmark_score == 0.65

    def test_multiple_records(self, tmp_path):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        db_path = str(tmp_path / "metrics.jsonl")
        dashboard = MetricsDashboard(db_path=db_path)

        for i in range(10):
            dashboard.record(IterationMetrics(
                iteration=i,
                timestamp=f"2025-01-{i+1:02d}T00:00:00",
                benchmark_score=0.5 + i * 0.03,
            ))

        loaded = dashboard.load_all()
        assert len(loaded) == 10

    def test_load_empty(self, tmp_path):
        from belief.metrics.dashboard import MetricsDashboard

        dashboard = MetricsDashboard(db_path=str(tmp_path / "empty.jsonl"))
        assert dashboard.load_all() == []

    def test_canary_score_optional(self, tmp_path):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        db_path = str(tmp_path / "metrics.jsonl")
        dashboard = MetricsDashboard(db_path=db_path)

        # Record without canary
        dashboard.record(IterationMetrics(
            iteration=1, timestamp="t", benchmark_score=0.5,
        ))
        # Record with canary
        dashboard.record(IterationMetrics(
            iteration=2, timestamp="t", benchmark_score=0.6, canary_score=0.55,
        ))

        loaded = dashboard.load_all()
        assert loaded[0].canary_score is None
        assert loaded[1].canary_score == 0.55


# ── Growth Analysis ────────────────────────────────────────────────────────


class TestGrowthAnalysis:
    def test_insufficient_data(self, tmp_path):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        dashboard = MetricsDashboard(db_path=str(tmp_path / "m.jsonl"))
        for i in range(3):
            dashboard.record(IterationMetrics(
                iteration=i, timestamp="t", benchmark_score=0.5,
            ))

        result = dashboard.compute_growth_analysis()
        assert result["status"] == "insufficient_data"

    def test_linear_fit(self, tmp_path):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        dashboard = MetricsDashboard(db_path=str(tmp_path / "m.jsonl"))
        # Linear growth: 0.3, 0.35, 0.4, 0.45, 0.5, 0.55
        for i in range(6):
            dashboard.record(IterationMetrics(
                iteration=i, timestamp="t", benchmark_score=0.3 + i * 0.05,
            ))

        result = dashboard.compute_growth_analysis()
        assert result["status"] == "ok"
        assert "linear" in result
        assert result["linear"]["slope"] > 0

    def test_best_fit_selected(self, tmp_path):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        dashboard = MetricsDashboard(db_path=str(tmp_path / "m.jsonl"))
        for i in range(8):
            dashboard.record(IterationMetrics(
                iteration=i, timestamp="t", benchmark_score=0.3 + i * 0.05,
            ))

        result = dashboard.compute_growth_analysis()
        assert result["best_fit"] in ("linear", "exponential")


# ── Dashboard Printing ─────────────────────────────────────────────────────


class TestDashboardPrinting:
    def test_print_dashboard_empty(self, tmp_path, capsys):
        from belief.metrics.dashboard import MetricsDashboard

        dashboard = MetricsDashboard(db_path=str(tmp_path / "m.jsonl"))
        dashboard.print_dashboard()
        captured = capsys.readouterr()
        assert "No metrics" in captured.out

    def test_print_dashboard_with_data(self, tmp_path, capsys):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        dashboard = MetricsDashboard(db_path=str(tmp_path / "m.jsonl"))
        dashboard.record(IterationMetrics(
            iteration=5,
            timestamp="2025-01-01T00:00:00",
            benchmark_score=0.75,
            cost_per_solved=1.20,
            novel_capabilities=2,
            tool_library_size=8,
            covenant_count=15,
        ))
        dashboard.print_dashboard()
        captured = capsys.readouterr()
        assert "BELIEF ENGINE" in captured.out
        assert "75.0%" in captured.out
        assert "8 tools" in captured.out

    def test_print_json(self, tmp_path, capsys):
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        dashboard = MetricsDashboard(db_path=str(tmp_path / "m.jsonl"))
        dashboard.record(IterationMetrics(
            iteration=1, timestamp="t", benchmark_score=0.5,
        ))
        dashboard.print_json()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "metrics" in data
        assert len(data["metrics"]) == 1


# ── CLI ─────────────────────────────────────────────────────────────────────


class TestCLI:
    def test_dashboard_cmd_exists(self):
        from belief.cli import _run_dashboard_cmd
        assert callable(_run_dashboard_cmd)
