"""Onboarding gate for new agents (mycorrhizal Stage 6, Area 5).

West, Griffin & Gardner 2007 frame partner choice as one of the four
mechanisms that maintain cooperation. The plant-fungus analogue is
pre-symbiosis signalling (strigolactones, Myc factors) by which both sides
pre-screen partners before infection. The Belief Engine analogue: a new
agent — one with no reciprocity-ledger history — must demonstrate a
validated contribution before it can send requests. This stops an arbitrary
new identity from consuming compute without first proving it can return
value.

Graveyard interaction (Stage 5): if a submitting ``agent_id`` matches a
previously-archived (sanction-terminated) agent, onboarding requires manual
operator approval rather than the automatic demo-task path — the system
should notice a known defector trying to re-enter under the same id.

**Build-path safety.** This gate is standalone. The recomposer's advisory
routing hook does NOT consult it — builds run as ``belief_engine`` which is
already a known ledger agent. Wiring the gate into a real request-admission
path waits for autonomous agents, like the rest of the routing enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from belief.routing._store import RoutingStore

logger = logging.getLogger("belief.routing.onboarding")


class OnboardingOutcome(str, Enum):
    ALREADY_KNOWN = "already_known"
    TASK_ASSIGNED = "task_assigned"
    APPROVED = "approved"  # demo task validated → admitted
    REJECTED = "rejected"  # demo task failed → watchlist
    MANUAL_REVIEW_REQUIRED = "manual_review_required"  # graveyard re-entry
    RATE_LIMITED = "rate_limited"


@dataclass
class DemoTask:
    """A pre-validated task with a known-good validator. The validator
    takes the agent's submitted output and returns (passed, value)."""

    task_id: str
    prompt: str
    validator: Callable[[object], tuple[bool, float]]


@dataclass
class OnboardingResult:
    outcome: OnboardingOutcome
    agent_id: str
    task: Optional[DemoTask] = None
    reason: str = ""
    awarded_value: float = 0.0


def _default_task_pool() -> list[DemoTask]:
    """A tiny pool of demo tasks with deterministic validators. Real
    deployments would pull from a curated set; for Stage 6 these prove the
    gate mechanics."""

    def _sum_validator(expected: int) -> Callable[[object], tuple[bool, float]]:
        def v(output: object) -> tuple[bool, float]:
            try:
                return (int(output) == expected, 1.0 if int(output) == expected else 0.0)
            except (TypeError, ValueError):
                return (False, 0.0)

        return v

    return [
        DemoTask("demo-sum-2-3", "Return the sum of 2 and 3.", _sum_validator(5)),
        DemoTask("demo-sum-10-7", "Return the sum of 10 and 7.", _sum_validator(17)),
    ]


class OnboardingGate:
    """Admits new agents via a demo-task gate.

    ``reciprocity_ledger`` is the Stage 1 ledger (to detect known agents +
    award initial credit). ``store`` is the shared routing store (to detect
    graveyard re-entry). ``task_pool`` defaults to the built-in demo tasks.
    """

    def __init__(
        self,
        store: RoutingStore,
        reciprocity_ledger,
        task_pool: Optional[list[DemoTask]] = None,
        max_attempts: int = 3,
    ) -> None:
        self._store = store
        self._ledger = reciprocity_ledger
        self._tasks = task_pool if task_pool is not None else _default_task_pool()
        self.max_attempts = int(max_attempts)
        self._pending: dict[str, DemoTask] = {}
        self._attempts: dict[str, int] = {}

    def is_known(self, agent_id: str) -> bool:
        """An agent is known iff it has any reciprocity-ledger history."""
        stats = self._ledger.stats(agent_id, window="all")
        return stats.request_count > 0 or stats.contribution_count > 0

    def submit(self, agent_id: str, self_description: str = "") -> OnboardingResult:
        """Begin onboarding. Returns a result describing the next step."""
        if not agent_id:
            raise ValueError("agent_id is required")

        if self.is_known(agent_id):
            return OnboardingResult(
                outcome=OnboardingOutcome.ALREADY_KNOWN,
                agent_id=agent_id,
                reason="agent already has ledger history",
            )

        # Graveyard re-entry → manual review.
        if self._store.is_archived(agent_id):
            return OnboardingResult(
                outcome=OnboardingOutcome.MANUAL_REVIEW_REQUIRED,
                agent_id=agent_id,
                reason="agent_id matches a previously-archived agent; "
                "manual operator approval required",
            )

        # Rate-limit repeated failed attempts.
        if self._attempts.get(agent_id, 0) >= self.max_attempts:
            return OnboardingResult(
                outcome=OnboardingOutcome.RATE_LIMITED,
                agent_id=agent_id,
                reason=f"exceeded {self.max_attempts} onboarding attempts",
            )

        # Assign the next demo task (round-robin by attempt count).
        task = self._tasks[self._attempts.get(agent_id, 0) % len(self._tasks)]
        self._pending[agent_id] = task
        return OnboardingResult(
            outcome=OnboardingOutcome.TASK_ASSIGNED,
            agent_id=agent_id,
            task=task,
            reason="complete the demo task to be admitted",
        )

    def complete(self, agent_id: str, output: object) -> OnboardingResult:
        """Submit the demo-task output. On pass, the agent is admitted to
        the ledger with an initial nutrient credit equal to the task value.
        On fail, the attempt is counted toward the rate limit."""
        task = self._pending.get(agent_id)
        if task is None:
            raise ValueError(f"no pending onboarding task for {agent_id!r}")

        passed, value = task.validator(output)
        self._attempts[agent_id] = self._attempts.get(agent_id, 0) + 1

        if not passed:
            self._pending.pop(agent_id, None)
            return OnboardingResult(
                outcome=OnboardingOutcome.REJECTED,
                agent_id=agent_id,
                task=task,
                reason="demo task output failed validation",
            )

        # Admit: record an initial contribution so the agent becomes known.
        self._pending.pop(agent_id, None)
        self._attempts.pop(agent_id, None)
        try:
            self._ledger.record_contribution(
                agent_id=agent_id,
                nutrient_value=value,
                nutrient_id=f"onboarding:{task.task_id}",
                idempotency_key=f"onboarding:{agent_id}:{task.task_id}",
            )
        except Exception as e:  # pragma: no cover — best-effort
            logger.debug("onboarding credit skipped: %s", e)
        return OnboardingResult(
            outcome=OnboardingOutcome.APPROVED,
            agent_id=agent_id,
            task=task,
            reason="demo task validated; agent admitted to the ledger",
            awarded_value=value,
        )

    def approve_manually(self, agent_id: str, initial_value: float = 1.0) -> OnboardingResult:
        """Operator override for a graveyard re-entry (or any manual case).
        Admits the agent with an initial credit."""
        try:
            self._ledger.record_contribution(
                agent_id=agent_id,
                nutrient_value=initial_value,
                nutrient_id="onboarding:manual",
                idempotency_key=f"onboarding:manual:{agent_id}",
            )
        except Exception as e:  # pragma: no cover — best-effort
            logger.debug("manual onboarding credit skipped: %s", e)
        return OnboardingResult(
            outcome=OnboardingOutcome.APPROVED,
            agent_id=agent_id,
            reason="manually approved by operator",
            awarded_value=initial_value,
        )
