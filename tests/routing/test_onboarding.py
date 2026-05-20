"""Tests for the onboarding gate (mycorrhizal Stage 6, Area 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.memory.reciprocity import ReciprocityLedger
from belief.routing._store import RoutingStore
from belief.routing.onboarding import (
    OnboardingGate,
    OnboardingOutcome,
)


@pytest.fixture
def recip(tmp_path: Path) -> ReciprocityLedger:
    r = ReciprocityLedger(db_path=tmp_path / "reciprocity.db")
    yield r
    r.close()


@pytest.fixture
def store(tmp_path: Path) -> RoutingStore:
    s = RoutingStore(db_path=tmp_path / "routing.db")
    yield s
    s.close()


@pytest.fixture
def gate(store: RoutingStore, recip: ReciprocityLedger) -> OnboardingGate:
    return OnboardingGate(store, recip)


# ── Known agents skip onboarding ─────────────────────────────────────────────


def test_known_agent_already_onboarded(gate: OnboardingGate, recip: ReciprocityLedger) -> None:
    recip.record_request("belief_engine", cost=1.0)
    result = gate.submit("belief_engine")
    assert result.outcome is OnboardingOutcome.ALREADY_KNOWN


# ── New agent demo-task flow ─────────────────────────────────────────────────


def test_new_agent_assigned_task(gate: OnboardingGate) -> None:
    result = gate.submit("newcomer", self_description="a fresh agent")
    assert result.outcome is OnboardingOutcome.TASK_ASSIGNED
    assert result.task is not None
    assert "sum" in result.task.prompt.lower()


def test_successful_onboarding_admits_to_ledger(
    gate: OnboardingGate, recip: ReciprocityLedger
) -> None:
    submit = gate.submit("newcomer")
    task = submit.task
    assert task is not None
    # The default first task is "sum of 2 and 3" → 5.
    result = gate.complete("newcomer", output=5)
    assert result.outcome is OnboardingOutcome.APPROVED
    assert result.awarded_value > 0
    # Agent is now known.
    assert gate.is_known("newcomer") is True


def test_failed_onboarding_rejected(gate: OnboardingGate) -> None:
    gate.submit("clumsy")
    result = gate.complete("clumsy", output=999)  # wrong answer
    assert result.outcome is OnboardingOutcome.REJECTED
    assert gate.is_known("clumsy") is False


def test_complete_without_pending_raises(gate: OnboardingGate) -> None:
    with pytest.raises(ValueError):
        gate.complete("never-submitted", output=5)


def test_rate_limited_after_max_attempts(store: RoutingStore, recip: ReciprocityLedger) -> None:
    gate = OnboardingGate(store, recip, max_attempts=2)
    # Two failed attempts.
    gate.submit("persistent")
    gate.complete("persistent", output=0)
    gate.submit("persistent")
    gate.complete("persistent", output=0)
    # Third submit is rate-limited.
    result = gate.submit("persistent")
    assert result.outcome is OnboardingOutcome.RATE_LIMITED


# ── Graveyard re-entry ───────────────────────────────────────────────────────


def test_graveyard_reentry_requires_manual_review(
    gate: OnboardingGate, store: RoutingStore
) -> None:
    store.archive_agent("known-defector", reason="terminated for parasitism")
    result = gate.submit("known-defector")
    assert result.outcome is OnboardingOutcome.MANUAL_REVIEW_REQUIRED


def test_manual_approval_admits(
    gate: OnboardingGate, recip: ReciprocityLedger, store: RoutingStore
) -> None:
    store.archive_agent("rehabilitated", reason="prior termination")
    assert gate.submit("rehabilitated").outcome is OnboardingOutcome.MANUAL_REVIEW_REQUIRED
    result = gate.approve_manually("rehabilitated")
    assert result.outcome is OnboardingOutcome.APPROVED
    assert gate.is_known("rehabilitated") is True


# ── Validation ───────────────────────────────────────────────────────────────


def test_submit_requires_agent_id(gate: OnboardingGate) -> None:
    with pytest.raises(ValueError):
        gate.submit("")


def test_second_task_on_retry(gate: OnboardingGate) -> None:
    """A failed first attempt rotates to the next demo task on retry, so a
    leaked first-task answer doesn't trivially pass."""
    first = gate.submit("retrier")
    first_task_id = first.task.task_id
    gate.complete("retrier", output=-1)  # fail
    second = gate.submit("retrier")
    assert second.task is not None
    assert second.task.task_id != first_task_id
