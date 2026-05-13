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
