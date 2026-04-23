"""Tests for the evolutionary archive (belief/evolution/archive.py).

Covers:
  - Create archive and save versions with parent relationships
  - Parent selection biases toward high-utility, low-children versions
  - get_lineage returns correct ancestor chain
  - compute_utility formula is correct
  - Niche map works (best per niche)
  - Cascaded evaluation gates (mock, test accept/reject logic)
  - Seed version creation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from belief.evolution.archive import (
    AgentVersion,
    Archive,
    BenchmarkResult,
    create_seed_version,
)


@pytest.fixture
def archive(tmp_path):
    """Create an Archive backed by a temp SQLite file."""
    db_path = str(tmp_path / "test_archive.db")
    return Archive(db_path=db_path)


def _make_version(
    parent_id=None,
    utility=0.5,
    niche=(0, 0, 0),
    children_count=0,
    canary_passed=True,
    **kwargs,
) -> AgentVersion:
    """Helper to create a test AgentVersion."""
    return AgentVersion(
        id=kwargs.get("id", str(uuid.uuid4())),
        parent_id=parent_id,
        created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
        system_prompts=kwargs.get("system_prompts", {"builder": "abc123"}),
        tool_ids=kwargs.get("tool_ids", []),
        principle_ids=kwargs.get("principle_ids", []),
        covenant_ids=kwargs.get("covenant_ids", []),
        model_config=kwargs.get("model_config", {"builder": "claude-sonnet-4-6"}),
        diff_from_parent=kwargs.get("diff_from_parent", "test change"),
        proposal_rationale=kwargs.get("proposal_rationale", "test reason"),
        utility=utility,
        children_count=children_count,
        niche_descriptor=niche,
        canary_passed=canary_passed,
    )


def _make_result(
    version_id: str,
    challenge_id: str = "t1-fizzbuzz",
    passed: bool = True,
    score: float = 1.0,
    cost_usd: float = 0.5,
    time_seconds: float = 30.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        version_id=version_id,
        challenge_id=challenge_id,
        passed=passed,
        score=score,
        cost_usd=cost_usd,
        time_seconds=time_seconds,
    )


# ── Basic CRUD ──────────────────────────────────────────────────────────────


class TestArchiveCRUD:
    def test_save_and_get_version(self, archive):
        v = _make_version(utility=0.7)
        archive.save_version(v)
        got = archive.get_version(v.id)
        assert got.id == v.id
        assert got.utility == 0.7
        assert got.parent_id is None

    def test_get_all_versions(self, archive):
        for i in range(5):
            archive.save_version(_make_version())
        assert len(archive.get_all_versions()) == 5

    def test_save_and_get_results(self, archive):
        v = _make_version()
        archive.save_version(v)
        r1 = _make_result(v.id, "t1-fizzbuzz", True, 1.0)
        r2 = _make_result(v.id, "t2-todo-cli", False, 0.3)
        archive.save_result(r1)
        archive.save_result(r2)
        results = archive.get_results(v.id)
        assert len(results) == 2
        assert results[0].challenge_id == "t1-fizzbuzz"
        assert results[0].passed is True
        assert results[1].passed is False

    def test_get_version_not_found(self, archive):
        with pytest.raises(KeyError):
            archive.get_version("nonexistent")


# ── Parent-child relationships ──────────────────────────────────────────────


class TestParentChild:
    def test_five_versions_with_lineage(self, archive):
        """Create 5 versions with parent relationships: seed -> v1 -> v2 -> v3 -> v4."""
        ids = [str(uuid.uuid4()) for _ in range(5)]

        # Seed (no parent)
        archive.save_version(_make_version(id=ids[0], parent_id=None, utility=0.5))

        # Chain: each version's parent is the previous one
        for i in range(1, 5):
            archive.save_version(
                _make_version(id=ids[i], parent_id=ids[i - 1], utility=0.5 + i * 0.1)
            )

        assert len(archive.get_all_versions()) == 5

        # Children of seed
        children = archive.get_children(ids[0])
        assert len(children) == 1
        assert children[0].id == ids[1]

    def test_get_lineage(self, archive):
        """get_lineage should return root-to-leaf ancestor chain."""
        ids = [str(uuid.uuid4()) for _ in range(4)]

        archive.save_version(_make_version(id=ids[0], parent_id=None, utility=0.4))
        archive.save_version(_make_version(id=ids[1], parent_id=ids[0], utility=0.5))
        archive.save_version(_make_version(id=ids[2], parent_id=ids[1], utility=0.6))
        archive.save_version(_make_version(id=ids[3], parent_id=ids[2], utility=0.7))

        lineage = archive.get_lineage(ids[3])
        assert len(lineage) == 4
        # Root first
        assert lineage[0].id == ids[0]
        assert lineage[0].parent_id is None
        # Leaf last
        assert lineage[-1].id == ids[3]

    def test_get_lineage_single(self, archive):
        """Lineage of the root should be just the root."""
        v = _make_version(parent_id=None)
        archive.save_version(v)
        lineage = archive.get_lineage(v.id)
        assert len(lineage) == 1
        assert lineage[0].id == v.id

    def test_increment_children(self, archive):
        v = _make_version(children_count=0)
        archive.save_version(v)
        archive.increment_children(v.id)
        archive.increment_children(v.id)
        got = archive.get_version(v.id)
        assert got.children_count == 2


# ── Parent selection (DGM sampling) ─────────────────────────────────────────


class TestParentSelection:
    def test_empty_archive_raises(self, archive):
        with pytest.raises(ValueError, match="empty"):
            archive.select_parent()

    def test_single_version(self, archive):
        v = _make_version(utility=0.8)
        archive.save_version(v)
        selected = archive.select_parent()
        assert selected.id == v.id

    def test_biases_toward_high_utility(self, archive):
        """High-utility versions should be selected more often."""
        low = _make_version(utility=0.1, children_count=0)
        high = _make_version(utility=0.9, children_count=0)
        archive.save_version(low)
        archive.save_version(high)

        # Sample 200 times
        selections = {}
        for _ in range(200):
            p = archive.select_parent()
            selections[p.id] = selections.get(p.id, 0) + 1

        # High utility should be selected significantly more
        assert selections.get(high.id, 0) > selections.get(low.id, 0), (
            f"High-utility should dominate: high={selections.get(high.id, 0)}, "
            f"low={selections.get(low.id, 0)}"
        )

    def test_penalizes_over_explored(self, archive):
        """Versions with many children should be selected less often."""
        fresh = _make_version(utility=0.6, children_count=0)
        explored = _make_version(utility=0.6, children_count=20)
        archive.save_version(fresh)
        archive.save_version(explored)

        selections = {}
        for _ in range(200):
            p = archive.select_parent()
            selections[p.id] = selections.get(p.id, 0) + 1

        assert selections.get(fresh.id, 0) > selections.get(explored.id, 0), (
            f"Fresh should be preferred: fresh={selections.get(fresh.id, 0)}, "
            f"explored={selections.get(explored.id, 0)}"
        )


# ── Utility computation ────────────────────────────────────────────────────


class TestComputeUtility:
    def test_perfect_score(self):
        """Perfect results: score=1.0, low cost, low time."""
        results = [
            _make_result("v1", score=1.0, cost_usd=0.5, time_seconds=10.0),
            _make_result("v1", score=1.0, cost_usd=0.5, time_seconds=10.0),
        ]
        u = Archive.compute_utility(results)
        # 0.5*1.0 + 0.25*(1-1/10) + 0.25*(1-20/300)
        assert u > 0.8

    def test_zero_results(self):
        assert Archive.compute_utility([]) == 0.0

    def test_high_cost_penalized(self):
        cheap = [_make_result("v1", score=0.8, cost_usd=0.5, time_seconds=30.0)]
        expensive = [_make_result("v1", score=0.8, cost_usd=9.0, time_seconds=30.0)]
        assert Archive.compute_utility(cheap) > Archive.compute_utility(expensive)

    def test_high_time_penalized(self):
        fast = [_make_result("v1", score=0.8, cost_usd=1.0, time_seconds=10.0)]
        slow = [_make_result("v1", score=0.8, cost_usd=1.0, time_seconds=280.0)]
        assert Archive.compute_utility(fast) > Archive.compute_utility(slow)

    def test_clamped_to_01(self):
        """Utility should be in [0, 1]."""
        # Extreme values
        extreme = [_make_result("v1", score=100.0, cost_usd=0.0, time_seconds=0.0)]
        u = Archive.compute_utility(extreme)
        assert 0.0 <= u <= 1.0

    def test_formula_exact(self):
        """Verify the exact formula: U = 0.5*score + 0.25*(1-cost/10) + 0.25*(1-time/300)."""
        results = [_make_result("v1", score=0.6, cost_usd=2.0, time_seconds=60.0)]
        u = Archive.compute_utility(results)
        expected = 0.5 * 0.6 + 0.25 * (1.0 - 2.0 / 10.0) + 0.25 * (1.0 - 60.0 / 300.0)
        assert abs(u - expected) < 0.001


# ── Niche map ───────────────────────────────────────────────────────────────


class TestNicheMap:
    def test_best_per_niche(self, archive):
        """get_best_in_niche returns the highest-utility version in that niche."""
        archive.save_version(_make_version(utility=0.3, niche=(0, 0, 0)))
        archive.save_version(_make_version(utility=0.8, niche=(0, 0, 0)))
        archive.save_version(_make_version(utility=0.5, niche=(1, 0, 0)))

        best_00 = archive.get_best_in_niche((0, 0, 0))
        assert best_00 is not None
        assert best_00.utility == 0.8

        best_10 = archive.get_best_in_niche((1, 0, 0))
        assert best_10 is not None
        assert best_10.utility == 0.5

    def test_niche_map_all_niches(self, archive):
        archive.save_version(_make_version(utility=0.5, niche=(0, 0, 0)))
        archive.save_version(_make_version(utility=0.7, niche=(1, 0, 0)))
        archive.save_version(_make_version(utility=0.9, niche=(2, 0, 0)))

        niche_map = archive.get_niche_map()
        assert len(niche_map) == 3
        assert niche_map[(0, 0, 0)].utility == 0.5
        assert niche_map[(2, 0, 0)].utility == 0.9

    def test_empty_niche(self, archive):
        assert archive.get_best_in_niche((99, 99, 99)) is None


# ── Cascaded evaluation (mocked) ───────────────────────────────────────────


class TestCascadeGates:
    @pytest.mark.asyncio
    async def test_canary_failure_rejects(self):
        """If canary fails, cascade should reject with 'canary_failed'."""
        from belief.evolution.cascade import cascaded_evaluate

        version = _make_version()
        mock_graph = MagicMock()

        with patch("belief.evolution.cascade._run_canary") as mock_canary:
            mock_canary.return_value = BenchmarkResult(
                version_id=version.id,
                challenge_id="canary-hello-world",
                passed=False,
                score=0.0,
                cost_usd=0.01,
                time_seconds=2.0,
                error_summary="Script failed to run",
            )

            accepted, results, reason = await cascaded_evaluate(version, mock_graph)

            assert accepted is False
            assert reason == "canary_failed"
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_smoke_failure_rejects(self):
        """If smoke pass rate < 60%, cascade should reject with 'smoke_failed'."""
        from belief.evolution.cascade import cascaded_evaluate

        version = _make_version()
        mock_graph = MagicMock()

        with (
            patch("belief.evolution.cascade._run_canary") as mock_canary,
            patch("belief.evolution.cascade._run_smoke") as mock_smoke,
        ):
            mock_canary.return_value = BenchmarkResult(
                version_id=version.id,
                challenge_id="canary-hello-world",
                passed=True,
                score=1.0,
                cost_usd=0.01,
                time_seconds=2.0,
            )
            # 2/5 pass = 40% < 60% threshold
            mock_smoke.return_value = [
                _make_result(version.id, f"t{i}", passed=(i < 2), score=float(i < 2))
                for i in range(5)
            ]

            accepted, results, reason = await cascaded_evaluate(version, mock_graph)

            assert accepted is False
            assert reason == "smoke_failed"

    @pytest.mark.asyncio
    async def test_full_pass_accepts(self):
        """If all gates pass, cascade should accept."""
        from belief.evolution.cascade import cascaded_evaluate

        version = _make_version()
        mock_graph = MagicMock()

        with (
            patch("belief.evolution.cascade._run_canary") as mock_canary,
            patch("belief.evolution.cascade._run_smoke") as mock_smoke,
            patch("belief.evolution.cascade._run_full_benchmark") as mock_full,
        ):
            mock_canary.return_value = BenchmarkResult(
                version_id=version.id,
                challenge_id="canary-hello-world",
                passed=True,
                score=1.0,
                cost_usd=0.01,
                time_seconds=2.0,
            )
            mock_smoke.return_value = [
                _make_result(version.id, f"smoke-{i}", passed=True) for i in range(5)
            ]
            mock_full.return_value = [
                _make_result(version.id, f"full-{i}", passed=True) for i in range(10)
            ]

            accepted, results, reason = await cascaded_evaluate(version, mock_graph)

            assert accepted is True
            assert reason == ""
            assert len(results) == 16  # 1 canary + 5 smoke + 10 full

    @pytest.mark.asyncio
    async def test_regression_flagged_not_rejected(self):
        """Gate 4 should flag regressions but NOT auto-reject."""
        from belief.evolution.cascade import cascaded_evaluate

        version = _make_version()
        mock_graph = MagicMock()

        parent_results = [
            _make_result("parent", "t1-fizzbuzz", passed=True),
            _make_result("parent", "t2-todo-cli", passed=True),
        ]

        with (
            patch("belief.evolution.cascade._run_canary") as mock_canary,
            patch("belief.evolution.cascade._run_smoke") as mock_smoke,
            patch("belief.evolution.cascade._run_full_benchmark") as mock_full,
        ):
            mock_canary.return_value = BenchmarkResult(
                version_id=version.id,
                challenge_id="canary-hello-world",
                passed=True,
                score=1.0,
                cost_usd=0.01,
                time_seconds=2.0,
            )
            mock_smoke.return_value = [
                _make_result(version.id, "t1-fizzbuzz", passed=True),
                _make_result(version.id, "t2-todo-cli", passed=False),  # Regression!
                _make_result(version.id, "t2-health-api", passed=True),
                _make_result(version.id, "t3-url-shortener", passed=True),
                _make_result(version.id, "t3-bookmark-api", passed=True),
            ]
            mock_full.return_value = []

            accepted, results, reason = await cascaded_evaluate(
                version, mock_graph, parent_results=parent_results
            )

            # Should still be accepted (DGM: regressions may be stepping stones)
            assert accepted is True
            # But regression should be flagged in results
            regression_results = [r for r in results if r.challenge_id.startswith("regression:")]
            assert len(regression_results) == 1
            assert "t2-todo-cli" in regression_results[0].challenge_id


# ── Seed version ────────────────────────────────────────────────────────────


class TestSeedVersion:
    def test_create_seed(self, archive):
        """create_seed_version should create the root of the DAG."""
        seed = create_seed_version(archive)
        assert seed.parent_id is None
        assert seed.utility == 0.5
        assert seed.canary_passed is True
        assert seed.diff_from_parent.startswith("seed")

        # Should have system prompts
        assert "builder" in seed.system_prompts
        assert "architect" in seed.system_prompts

        # Should have model config
        assert "builder" in seed.model_config
        assert "sonnet" in seed.model_config["builder"]

    def test_seed_idempotent(self, archive):
        """Calling create_seed_version twice should return the same seed."""
        s1 = create_seed_version(archive)
        s2 = create_seed_version(archive)
        assert s1.id == s2.id
        assert len(archive.get_all_versions()) == 1

    def test_seed_is_in_archive(self, archive):
        seed = create_seed_version(archive)
        versions = archive.get_all_versions()
        assert len(versions) == 1
        assert versions[0].id == seed.id

    def test_seed_selectable_as_parent(self, archive):
        create_seed_version(archive)
        parent = archive.select_parent()
        assert parent.parent_id is None  # The seed


# ── Config persistence roundtrip ────────────────────────────────────────────


class TestConfigPersistence:
    def test_system_prompts_roundtrip(self, archive):
        """system_prompts dict should survive save/load."""
        v = _make_version(
            system_prompts={"builder": "hash1", "tester": "hash2", "architect": "hash3"}
        )
        archive.save_version(v)
        got = archive.get_version(v.id)
        assert got.system_prompts == {"builder": "hash1", "tester": "hash2", "architect": "hash3"}

    def test_tool_ids_roundtrip(self, archive):
        v = _make_version(tool_ids=["tool-1", "tool-2"])
        archive.save_version(v)
        got = archive.get_version(v.id)
        assert got.tool_ids == ["tool-1", "tool-2"]

    def test_niche_roundtrip(self, archive):
        v = _make_version(niche=(2, 3, 1))
        archive.save_version(v)
        got = archive.get_version(v.id)
        assert got.niche_descriptor == (2, 3, 1)

    def test_model_config_roundtrip(self, archive):
        v = _make_version(
            model_config={
                "builder": "claude-sonnet-4-6",
                "planner": "claude-opus-4-6",
            }
        )
        archive.save_version(v)
        got = archive.get_version(v.id)
        assert got.model_config["planner"] == "claude-opus-4-6"
