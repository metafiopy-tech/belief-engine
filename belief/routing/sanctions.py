"""Soft sanctions (mycorrhizal Stage 5, Area 5).

Kiers & Denison 2008 + Kiers et al. 2011: sanctions are how a biological
market enforces its lower bound without manual intervention. Crucially the
biological data show sanctions are *soft* — less-effective symbionts persist,
and that tolerance is adaptive (it preserves diversity and avoids the cost of
perfect enforcement). So this engine throttles, it doesn't ban; and a
throttled agent that re-establishes reciprocity is restored automatically
because the decision reads the live exchange rate every time.

**Advisory at this stage.** ``evaluate`` returns what *would* happen
(ALLOW / THROTTLE / TERMINATE). It does not refuse anything. The recomposer
hook reads the decision for the ``belief topology`` diagnostic only.
Termination archives the agent to the graveyard so future onboarding
(Stage 6) can detect re-entry, but it does not delete or block — actual
enforcement waits for ``BELIEF_ROUTING_ENFORCE`` and real agents.

**Bypass property.** The singular ``belief_engine`` agent that currently
receives all build credit sits well above both thresholds the moment any
build deposits nutrients, and below the ``grace_period_n`` request count for
termination. Either way the decision is ALLOW in practice — and even a
THROTTLE verdict is advisory, so builds are unaffected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from belief.routing._store import RoutingStore

logger = logging.getLogger("belief.routing.sanctions")

# 7-day exchange rate below this → throttle (advisory).
DEFAULT_THROTTLE_THRESHOLD = 0.1
# 30-day exchange rate below this (with enough history) → terminate.
DEFAULT_TERMINATE_THRESHOLD = 0.01
# Minimum lifetime request count before termination can fire — the grace
# period that lets a new agent establish itself before being judged.
DEFAULT_GRACE_PERIOD_N = 50


class SanctionAction(str, Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class SanctionDecision:
    """Advisory verdict for one agent. ``backoff_hint_s`` is populated for
    THROTTLE so a future enforcement path can return a Retry-After-style
    hint without recomputing."""

    action: SanctionAction
    agent_id: str
    reason: str
    exchange_rate_7d: float
    exchange_rate_30d: float
    lifetime_requests: int
    backoff_hint_s: Optional[int] = None

    @property
    def is_allow(self) -> bool:
        return self.action is SanctionAction.ALLOW


class SanctionsEngine:
    """Computes soft-sanction verdicts from the reciprocity ledger.

    ``reciprocity_ledger`` is the Stage 1 ``ReciprocityLedger``. ``store``
    is the shared routing store (for graveyard archival).
    """

    def __init__(
        self,
        store: RoutingStore,
        reciprocity_ledger,
        throttle_threshold: float = DEFAULT_THROTTLE_THRESHOLD,
        terminate_threshold: float = DEFAULT_TERMINATE_THRESHOLD,
        grace_period_n: int = DEFAULT_GRACE_PERIOD_N,
        throttle_backoff_s: int = 30,
    ) -> None:
        self._store = store
        self._ledger = reciprocity_ledger
        self.throttle_threshold = float(throttle_threshold)
        self.terminate_threshold = float(terminate_threshold)
        self.grace_period_n = int(grace_period_n)
        self.throttle_backoff_s = int(throttle_backoff_s)

    def evaluate(self, agent_id: str, archive_on_terminate: bool = True) -> SanctionDecision:
        """Return the advisory sanction verdict for ``agent_id``.

        Order: TERMINATE (worst) is checked before THROTTLE so an agent
        that crosses both only gets the more severe verdict. An unknown
        agent (no ledger history) is ALLOW — never sanction someone we've
        never charged.
        """
        stats_7d = self._ledger.stats(agent_id, window="7d")
        stats_30d = self._ledger.stats(agent_id, window="30d")
        lifetime = self._ledger.stats(agent_id, window="all")
        lifetime_requests = lifetime.request_count
        rate_7d = stats_7d.exchange_rate
        rate_30d = stats_30d.exchange_rate

        # Unknown / never-charged agents: ALLOW.
        if lifetime_requests == 0 and lifetime.contribution_count == 0:
            return SanctionDecision(
                action=SanctionAction.ALLOW,
                agent_id=agent_id,
                reason="no ledger history — allow",
                exchange_rate_7d=rate_7d,
                exchange_rate_30d=rate_30d,
                lifetime_requests=lifetime_requests,
            )

        # TERMINATE: persistently parasitic AND past the grace period.
        if rate_30d < self.terminate_threshold and lifetime_requests > self.grace_period_n:
            if archive_on_terminate:
                try:
                    self._store.archive_agent(
                        agent_id,
                        reason=(
                            f"30d exchange {rate_30d:.4f} < "
                            f"{self.terminate_threshold} after "
                            f"{lifetime_requests} requests"
                        ),
                    )
                except Exception as e:  # pragma: no cover — best-effort
                    logger.debug("graveyard archival skipped: %s", e)
            return SanctionDecision(
                action=SanctionAction.TERMINATE,
                agent_id=agent_id,
                reason=(
                    f"30d exchange {rate_30d:.4f} below terminate threshold "
                    f"{self.terminate_threshold} after {lifetime_requests} requests"
                ),
                exchange_rate_7d=rate_7d,
                exchange_rate_30d=rate_30d,
                lifetime_requests=lifetime_requests,
            )

        # THROTTLE: recent free-riding.
        if rate_7d < self.throttle_threshold:
            return SanctionDecision(
                action=SanctionAction.THROTTLE,
                agent_id=agent_id,
                reason=(
                    f"7d exchange {rate_7d:.4f} below throttle threshold {self.throttle_threshold}"
                ),
                exchange_rate_7d=rate_7d,
                exchange_rate_30d=rate_30d,
                lifetime_requests=lifetime_requests,
                backoff_hint_s=self.throttle_backoff_s,
            )

        return SanctionDecision(
            action=SanctionAction.ALLOW,
            agent_id=agent_id,
            reason="exchange rate above thresholds — allow",
            exchange_rate_7d=rate_7d,
            exchange_rate_30d=rate_30d,
            lifetime_requests=lifetime_requests,
        )

    def is_archived(self, agent_id: str) -> bool:
        return self._store.is_archived(agent_id)
