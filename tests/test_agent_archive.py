"""Hermetic tests for the Session 6 (v3.2) agent archive.

Uses ChromaDB's ephemeral client — no disk artefacts, fully
self-contained.  Run with::

    python3 -m pytest tests/test_agent_archive.py -v
"""

from __future__ import annotations

import random

import pytest

from belief.archive import (
    AgentArchive,
    AgentConfiguration,
    BuildOutcome,
    parent_sample,
    utility,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _HashEmbedder:
    """Deterministic hash-based embedding function for hermetic tests.

    Avoids the HuggingFace ONNX download that DefaultEmbeddingFunction
    triggers on first use — the sandbox blocks that network call, and
    we don't need semantic quality for tests that only check "query
    returns persisted docs in roughly-goal-similar order".

    Uses a simple bag-of-characters → 32-dim projection so that
    lexically-similar inputs get mathematically-close vectors.
    ChromaDB requires a ``name`` attribute on embedding functions
    (used for collection-level cache keys).
    """

    def name(self) -> str:
        return "belief-test-hash-embedder-v1"

    def _embed(self, texts):
        out = []
        for text in texts:
            v = [0.0] * 32
            for c in (text or "").lower():
                v[ord(c) % 32] += 1.0
            # Normalise so cosine distance means something.
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    def __call__(self, input):  # ChromaDB signature: list[str] → list[list[float]]
        return self._embed(input)

    def embed_query(self, input):
        # ChromaDB 1.x query-time path calls `embed_query` separately.
        return self._embed(input)

    def embed_documents(self, input):
        return self._embed(input)


@pytest.fixture()
def archive(tmp_path) -> AgentArchive:
    # Per-test tmp_path → each archive is fully isolated.  (ChromaDB's
    # EphemeralClient is a process-wide singleton, so ephemeral=True
    # leaks state across tests; tmp_path is the safe choice.)
    return AgentArchive(path=tmp_path / "archive", embedding_function=_HashEmbedder())


def _cfg(agent_name: str = "planner", prompt: str = "You plan things.") -> AgentConfiguration:
    return AgentConfiguration(
        agent_name=agent_name,
        system_prompt=prompt,
        model="qwen2.5-coder:14b",
    )


def _outcome(
    run_id: str,
    goal: str,
    *,
    verdict: str = "pass",
    score: float = 1.0,
    wall: float = 255.0,
    cost: float = 0.0,
) -> BuildOutcome:
    o = BuildOutcome(
        run_id=run_id,
        goal=goal,
        verdict=verdict,
        tests_passed=3,
        tests_total=3,
        weighted_score=score,
        wallclock_s=wall,
        estimated_cost_usd=cost,
        agent_configurations={"planner": _cfg()},
    )
    o.trajectory_signature = o.compute_trajectory_signature(["planner", "builder", "executor"])
    return o


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------


class TestDataclassRoundTrip:
    def test_agent_configuration_json_round_trip(self) -> None:
        c = _cfg(agent_name="architect", prompt="You design.")
        restored = AgentConfiguration.from_json(c.to_json())
        assert restored == c

    def test_build_outcome_json_round_trip(self) -> None:
        o = _outcome("r-1", "Build fizzbuzz")
        restored = BuildOutcome.from_json(o.to_json())
        assert restored.run_id == o.run_id
        assert restored.goal == o.goal
        assert restored.weighted_score == o.weighted_score
        assert restored.agent_configurations["planner"] == o.agent_configurations["planner"]

    def test_code_hash_stable_and_distinct(self) -> None:
        h1 = AgentConfiguration.compute_code_hash("def foo(): pass")
        h2 = AgentConfiguration.compute_code_hash("def foo(): pass")
        h3 = AgentConfiguration.compute_code_hash("def bar(): pass")
        assert h1 == h2
        assert h1 != h3


# ---------------------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------------------


class TestUtility:
    def test_perfect_cheap_fast_scores_high(self) -> None:
        o = _outcome("u-1", "perfect", score=1.0, wall=10.0, cost=0.0)
        u = utility(o, expected_covenants=7)
        assert u > 0.9

    def test_failed_slow_expensive_scores_low(self) -> None:
        o = _outcome("u-2", "bad", verdict="fail_hard", score=0.0, wall=600.0, cost=10.0)
        u = utility(o, expected_covenants=7)
        # Even a zero-quality, max-cost, max-time build retains some
        # utility from the covenant_rate fallback (no violations → 1.0
        # × 0.15 = 0.15).
        assert 0.0 <= u <= 0.2

    def test_utility_is_clipped_to_unit_interval(self) -> None:
        # Even with nonsense inputs the utility must stay in [0, 1].
        o = _outcome("u-3", "odd", score=2.0, wall=-100.0, cost=-5.0)
        u = utility(o, expected_covenants=7)
        assert 0.0 <= u <= 1.0


# ---------------------------------------------------------------------------
# Archive: persist + query
# ---------------------------------------------------------------------------


class TestArchivePersist:
    def test_persist_increments_size(self, archive: AgentArchive) -> None:
        assert archive.size() == 0
        archive.persist(_outcome("r-1", "Build a FizzBuzz"))
        archive.persist(_outcome("r-2", "Build a URL shortener"))
        assert archive.size() == 2

    def test_query_returns_similar_goals_first(self, archive: AgentArchive) -> None:
        archive.persist(_outcome("r-fizz", "Build a FizzBuzz Python script"))
        archive.persist(_outcome("r-url", "Build a URL shortener with FastAPI"))
        archive.persist(_outcome("r-todo", "Build a todo CLI with Click"))
        hits = archive.query_by_goal("Build a FizzBuzz clone", k=3)
        assert hits, "query returned nothing"
        # The FizzBuzz prior should be the closest (smallest distance).
        top = hits[0]
        assert top["id"] == "r-fizz"

    def test_query_excludes_fail_hard_by_default(self, archive: AgentArchive) -> None:
        archive.persist(_outcome("r-good", "Good build", verdict="pass"))
        archive.persist(_outcome("r-bad", "Bad build", verdict="fail_hard"))
        hits = archive.query_by_goal("Similar task", k=10)
        ids = {h["id"] for h in hits}
        assert "r-good" in ids
        assert "r-bad" not in ids, "fail_hard outcome must be filtered out"


# ---------------------------------------------------------------------------
# parent_sample — Boltzmann behaviour
# ---------------------------------------------------------------------------


class TestParentSample:
    def test_returns_up_to_k_results(self, archive: AgentArchive) -> None:
        for i in range(5):
            archive.persist(_outcome(f"r-{i}", f"Build feature {i}"))
        hits = parent_sample("Build feature X", archive=archive, k=3, rng=random.Random(0))
        assert len(hits) <= 3

    def test_boltzmann_is_not_purely_greedy(self, archive: AgentArchive) -> None:
        """Two seeds should occasionally pick different top-1 parents
        when utilities are close — that's the whole point of
        Boltzmann sampling over argmax."""
        # Three priors with slightly-different scores.
        archive.persist(_outcome("r-high", "Build a FizzBuzz script", score=0.95))
        archive.persist(_outcome("r-mid", "Build a FizzBuzz script", score=0.90))
        archive.persist(_outcome("r-low", "Build a FizzBuzz script", score=0.85))

        seen_tops: set[str] = set()
        for seed in range(0, 50):
            rng = random.Random(seed)
            picks = parent_sample(
                "Build a FizzBuzz variant",
                archive=archive,
                k=1,
                temperature=0.5,
                rng=rng,
            )
            if picks:
                seen_tops.add(picks[0]["id"])
        # With τ=0.5 and three priors, we expect ≥2 distinct top picks
        # across 50 seeds.  (τ=0.0 would collapse to argmax and only
        # yield one.)
        assert len(seen_tops) >= 2, (
            f"Boltzmann sampler is collapsing to greedy — seen tops={seen_tops}"
        )

    def test_empty_archive_returns_empty_list(self, archive: AgentArchive) -> None:
        assert parent_sample("anything", archive=archive, k=3) == []


# ---------------------------------------------------------------------------
# Priors block — planner prompt injection
# ---------------------------------------------------------------------------


class TestPriorsBlock:
    def test_format_priors_block_empty_archive(self, archive: AgentArchive) -> None:
        from belief.archive.priors import format_priors_block

        assert format_priors_block("anything", archive=archive) == ""

    def test_format_priors_block_includes_prior_entries(self, archive: AgentArchive) -> None:
        from belief.archive.priors import format_priors_block

        archive.persist(_outcome("r-fizz", "Build a FizzBuzz Python script"))
        archive.persist(_outcome("r-other", "Build a URL shortener"))
        block = format_priors_block("Build a FizzBuzz clone", archive=archive, k=2)
        assert block  # non-empty
        assert "PRIOR SUCCESSFUL CONFIGURATIONS" in block
        assert "### Prior 1" in block


# ---------------------------------------------------------------------------
# BuildOutcome from state — persist.py
# ---------------------------------------------------------------------------


class TestPersistFromState:
    def test_persist_build_outcome_from_minimal_state(
        self, monkeypatch: pytest.MonkeyPatch, archive: AgentArchive
    ) -> None:
        from belief.archive.persist import persist_build_outcome

        state = {
            "run_id": "belief-test-1",
            "user_goal": "Build a fizzbuzz",
            "validation_result": {"verdict": "pass", "weighted_score": 1.0},
            "execution_result": {"tests_passed": 3, "tests_total": 3},
            "agent_timings": {"planner": 60.0, "builder": 15.0, "executor": 0.1},
            "iteration": 0,
        }
        persist_build_outcome(state, archive=archive)
        assert archive.size() == 1
        hits = archive.query_by_goal("Build a fizzbuzz", k=1)
        assert len(hits) == 1
        assert hits[0]["id"] == "belief-test-1"

    def test_persist_silently_skips_on_missing_run_id(self, archive: AgentArchive) -> None:
        from belief.archive.persist import persist_build_outcome

        persist_build_outcome({"user_goal": "no run_id"}, archive=archive)
        assert archive.size() == 0
