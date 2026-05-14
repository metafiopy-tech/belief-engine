"""Tests for cycle-level rejection-reason visibility (SE Session 7.9).

The cross-domain phase used to silently swallow xd_result.reason
when xd_result.spec was None. From the operator's POV this looked
like `rejected=N errors=0` with no signal about WHICH pipeline
stage rejected. S7.9 adds a summary.errors entry +
logger.info line so the reason surfaces both to the CLI's print
loop and to logs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from belief.photosynthesis.config import PhotoConfig
from belief.photosynthesis.sources.word_set import emit
from belief.photosynthesis.state import PhotosynthesisState
from belief.photosynthesis.synthesis.cross_domain_generator import (
    CrossDomainResult,
)
from belief.photosynthesis.synthesis.cycle import (
    CycleSummary,
    _run_cross_domain_phase,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def state(tmp_path: Path) -> PhotosynthesisState:
    db = tmp_path / "signals.sqlite"
    return PhotosynthesisState(db_path=str(db))


@pytest.fixture
def config(tmp_path: Path) -> PhotoConfig:
    return PhotoConfig(
        state_dir=tmp_path,
        log_dir=tmp_path / "logs",
        config_dir=tmp_path / "cfg",
    )


def _install_fake_synthesize(
    monkeypatch,
    *,
    reason: str,
    critic_result=None,
) -> None:
    """Stub synthesize_cross_domain to return a None-spec result with
    the supplied reason. This is the path the cycle's rejection
    visibility branch handles. Optionally attach a CriticResult so
    the drill-down branch (S7.10) can be exercised."""

    async def fake(*, words, bundle_id, generator_client, critic_client=None, **kw):
        return CrossDomainResult(
            spec=None,
            mechanism=None,
            critic=critic_result,
            reason=reason,
            raw_passes={},
        )

    monkeypatch.setattr(
        "belief.photosynthesis.synthesis.cycle.synthesize_cross_domain",
        fake,
        raising=False,
    )
    # ALSO patch the import inside _run_cross_domain_phase. That function
    # does a local `from ... import synthesize_cross_domain` -- patch the
    # source module too so both lookups resolve to the fake.
    monkeypatch.setattr(
        "belief.photosynthesis.synthesis.cross_domain_generator.synthesize_cross_domain",
        fake,
        raising=False,
    )


async def _noop_generator(prompt, *, temperature, max_tokens):
    return ""


def test_rejection_reason_lands_in_summary_errors(monkeypatch, state, config) -> None:
    """When the synthesizer returns spec=None, cycle records the reason
    in summary.errors so the CLI's `err:` print loop surfaces it."""
    _install_fake_synthesize(monkeypatch, reason="critic_rejected")
    _run(emit(state, config, words=["mantis", "camera"], bundle_id="bundle_xyz"))

    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=None,
            critic_client=None,
        )
    )

    assert summary.surveyed == 1
    assert summary.rejected == 1
    assert summary.promoted == 0
    # The visibility addition: summary.errors carries a structured entry.
    assert len(summary.errors) == 1
    err = summary.errors[0]
    assert err.startswith("cross_domain_rejected:")
    assert "bundle=bundle_xyz" in err
    assert "reason=critic_rejected" in err


def test_rejection_reason_logged_at_info(monkeypatch, state, config, caplog) -> None:
    """Logger.info fires with bundle id + reason so log scrapers see it."""
    _install_fake_synthesize(monkeypatch, reason="schema_invalid")
    _run(emit(state, config, words=["a", "b"], bundle_id="bundle_log_test"))

    caplog.set_level("INFO", logger="belief.photosynthesis.synthesis.cycle")
    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=None,
            critic_client=None,
        )
    )

    info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
    matched = [m for m in info_messages if "cross_domain bundle rejected" in m]
    assert matched, f"expected info log; got: {info_messages}"
    assert "bundle=bundle_log_test" in matched[0]
    assert "reason=schema_invalid" in matched[0]


def test_unknown_reason_falls_back_to_literal_unknown(monkeypatch, state, config) -> None:
    """If the synthesizer returns spec=None with reason=None (defensive),
    the visibility branch still produces a non-empty error entry."""
    _install_fake_synthesize(monkeypatch, reason="")  # falsy -> "unknown"
    _run(emit(state, config, words=["a", "b"], bundle_id="bundle_unknown"))

    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=None,
            critic_client=None,
        )
    )

    assert len(summary.errors) == 1
    assert "reason=unknown" in summary.errors[0]


# ---------------------------------------------------------------------------
# S7.10: critic-check drill-down. When reason=critic_rejected, the
# cycle should pull failed check names out of xd_result.critic.checks
# so the CLI prints them alongside the bundle id.
# ---------------------------------------------------------------------------


def test_critic_failed_checks_surface_in_error_entry(monkeypatch, state, config) -> None:
    """When reason=critic_rejected and the CriticResult has failed
    checks, the error entry includes their names."""
    from belief.photosynthesis.synthesis.cross_domain_critic import (
        CheckResult,
        CriticResult,
    )

    critic = CriticResult(
        verdict="REJECT",
        checks=[
            CheckResult(id=1, name="predicate_not_attribute_style", passed=True, reason=""),
            CheckResult(id=2, name="roles_are_process_oriented", passed=True, reason=""),
            CheckResult(
                id=4, name="near_miss_plausibly_fits_domains", passed=False, reason="implausible"
            ),
            CheckResult(id=8, name="analogy_is_non_trivial", passed=False, reason="trivial"),
        ],
    )
    _install_fake_synthesize(monkeypatch, reason="critic_rejected", critic_result=critic)
    _run(emit(state, config, words=["x", "y"], bundle_id="bundle_drilldown"))

    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=None,
            critic_client=None,
        )
    )

    assert len(summary.errors) == 1
    err = summary.errors[0]
    assert "reason=critic_rejected" in err
    assert "failed=" in err
    # Both failed check names appear (passed ones do not).
    assert "near_miss_plausibly_fits_domains" in err
    assert "analogy_is_non_trivial" in err
    assert "predicate_not_attribute_style" not in err
    assert "roles_are_process_oriented" not in err


def test_critic_no_failed_checks_omits_failed_clause(monkeypatch, state, config) -> None:
    """If the critic exists but has no failed checks (defensive case),
    the error entry should not include an empty `failed=` clause."""
    from belief.photosynthesis.synthesis.cross_domain_critic import (
        CheckResult,
        CriticResult,
    )

    critic = CriticResult(
        verdict="REJECT",  # verdict says reject but all checks pass (defensive)
        checks=[
            CheckResult(id=1, name="predicate_not_attribute_style", passed=True, reason=""),
            CheckResult(id=2, name="roles_are_process_oriented", passed=True, reason=""),
        ],
    )
    _install_fake_synthesize(monkeypatch, reason="critic_rejected", critic_result=critic)
    _run(emit(state, config, words=["x", "y"], bundle_id="bundle_no_failed"))

    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=None,
            critic_client=None,
        )
    )

    err = summary.errors[0]
    assert "reason=critic_rejected" in err
    assert "failed=" not in err


def test_non_critic_rejection_has_no_failed_clause(monkeypatch, state, config) -> None:
    """Non-critic reasons (e.g. schema_invalid) carry no critic
    result; the error entry shouldn't have a `failed=` clause."""
    _install_fake_synthesize(monkeypatch, reason="schema_invalid", critic_result=None)
    _run(emit(state, config, words=["x", "y"], bundle_id="bundle_schema"))

    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=None,
            critic_client=None,
        )
    )

    err = summary.errors[0]
    assert "reason=schema_invalid" in err
    assert "failed=" not in err


# ---------------------------------------------------------------------------
# S7.13: novelty gate enable/disable + threshold knobs
# ---------------------------------------------------------------------------


def _install_synthesize_returning_accepted(monkeypatch):
    """Stub synthesize_cross_domain to return a fully-accepted spec
    (no critic rejection) -- exercises the gate path on a live mechanism."""
    from belief.photosynthesis.synthesis.cross_domain_critic import CriticResult
    from belief.photosynthesis.synthesis.cross_domain_generator import CrossDomainResult
    from belief.photosynthesis.synthesis.generator import GoalSpec
    from belief.photosynthesis.synthesis.structural_mechanism import (
        DomainEvidence,
        HigherOrderRelation,
        NearMiss,
        PredicateInstance,
        StructuralMechanism,
    )

    pred = PredicateInstance(
        name="downsamples_at_source",
        arity=2,
        roles=["source", "downstream"],
        marr_level="algorithmic",
    )
    mech = StructuralMechanism(
        mechanism_id="m1",
        source_domain="biology",
        target_domain="computing",
        predicate_in_source=pred,
        predicate_in_target=pred.model_copy(),
        higher_order_relations=[
            HigherOrderRelation(
                name="reduces_downstream_compute",
                relates=["downsamples_at_source", "compresses_at_sensor"],
            ),
        ],
        near_miss=NearMiss(
            description="A bee compound eye sums signals downstream",
            breaks_at_argument="predicate_in_source.argument[0]",
        ),
        considered_and_rejected_attributes=["color_count", "spectral_count"],
        domain_evidence=[
            DomainEvidence(
                domain="biology",
                citation="general",
                excerpt="mantis_shrimp pre-classifies",
            ),
        ],
    )
    from belief.photosynthesis.synthesis.generator import AcceptanceCriterion

    spec = GoalSpec(
        goal_id="m1-goal",
        title="Build a downsampling sensor",
        one_paragraph_description="A FastAPI mount of a downsampling sensor.",
        artifact_type="api",
        primary_libraries=["fastapi"],
        new_libraries_introduced=[],
        acceptance_criteria=[
            AcceptanceCriterion(kind="endpoint", spec="POST /sample handles raw frames"),
        ],
        estimated_build_time_min=60,
        estimated_difficulty=3,
        prerequisite_skills=["fastapi"],
        relevance_rationale="cross-domain demo",
        novelty_rationale="first run",
        source_citation="word_set",
        structural_mechanism=mech,
    )

    async def fake(*, words, bundle_id, generator_client, critic_client=None, **kw):
        return CrossDomainResult(
            spec=spec,
            mechanism=mech,
            critic=CriticResult(verdict="ACCEPT", checks=[]),
            reason="accepted",
            raw_passes={},
        )

    monkeypatch.setattr(
        "belief.photosynthesis.synthesis.cycle.synthesize_cross_domain",
        fake,
        raising=False,
    )
    monkeypatch.setattr(
        "belief.photosynthesis.synthesis.cross_domain_generator.synthesize_cross_domain",
        fake,
        raising=False,
    )
    return mech


class _FakeBioStore:
    """Bio store stub: novelty_score returns whatever you set."""

    def __init__(self, score: float):
        self._score = score

    def novelty_score(self, mech) -> float:  # noqa: ANN001
        return self._score

    def add(self, mech) -> None:  # noqa: ANN001
        pass


def test_novelty_gate_enabled_rejects_below_threshold(monkeypatch, state, config, tmp_path) -> None:
    """Default behavior preserved: low-novelty mechanism rejected."""
    _install_synthesize_returning_accepted(monkeypatch)
    _run(emit(state, config, words=["a", "b"], bundle_id="bundle_low_novel"))

    bio = _FakeBioStore(score=0.1)  # 0.1 < default 0.30 -> reject
    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=tmp_path,
            critic_client=None,
            bio_store=bio,
            novelty_gate_enabled=True,
        )
    )
    assert summary.rejected == 1
    assert summary.promoted == 0
    assert any("cross_domain_redundant" in e for e in summary.errors)


def test_novelty_gate_disabled_admits_low_novelty(monkeypatch, state, config, tmp_path) -> None:
    """`--no-novelty-gate` lets a duplicate-ish mechanism through."""
    _install_synthesize_returning_accepted(monkeypatch)
    _run(emit(state, config, words=["a", "b"], bundle_id="bundle_bypass"))

    bio = _FakeBioStore(score=0.0)  # would normally reject hard
    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=tmp_path,
            critic_client=None,
            bio_store=bio,
            novelty_gate_enabled=False,
        )
    )
    assert summary.promoted == 1
    assert summary.rejected == 0
    assert not any("cross_domain_redundant" in e for e in summary.errors)


def test_novelty_threshold_lowered_admits_borderline(monkeypatch, state, config, tmp_path) -> None:
    """Setting threshold=0.05 admits a 0.10-novel mechanism."""
    _install_synthesize_returning_accepted(monkeypatch)
    _run(emit(state, config, words=["a", "b"], bundle_id="bundle_lower"))

    bio = _FakeBioStore(score=0.10)  # < default 0.30 but > 0.05
    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=tmp_path,
            critic_client=None,
            bio_store=bio,
            novelty_gate_enabled=True,
            novelty_threshold=0.05,
        )
    )
    assert summary.promoted == 1
    assert summary.rejected == 0


def test_novelty_threshold_raised_rejects_borderline(monkeypatch, state, config, tmp_path) -> None:
    """Setting threshold=0.50 rejects a 0.40-novel mechanism."""
    _install_synthesize_returning_accepted(monkeypatch)
    _run(emit(state, config, words=["a", "b"], bundle_id="bundle_strict"))

    bio = _FakeBioStore(score=0.40)  # passes default 0.30 but not 0.50
    summary = CycleSummary()
    _run(
        _run_cross_domain_phase(
            state=state,
            config=config,
            summary=summary,
            generator_client=_noop_generator,
            embedder=None,
            archive=None,
            pending_dir=tmp_path,
            critic_client=None,
            bio_store=bio,
            novelty_gate_enabled=True,
            novelty_threshold=0.50,
        )
    )
    assert summary.rejected == 1
