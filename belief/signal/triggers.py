"""Trigger registration + evaluation (mycorrhizal Stage 4, Area 1).

Triggers are predicates over the signal-concentration profile of an agent.
They fire actions when conditions are met — e.g. "agent is sustainedly
stressed and requesting help, escalate to the engine immediately."

The biological pattern this mirrors is induced-defense priming
(Heil & Karban 2009): receivers commit to costly behavior only when
sustained signal integration crosses a threshold, not on single events.
Our triggers query ``SignalStore.concentration`` (which is itself an
exponentially-decayed integral) so single-event spikes don't fire.

This module is intentionally thin — Stage 5's router will wire concrete
actions to the engine's request flow. For now we ship the registry plus
two example predicates that demonstrate the patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from belief.signal.store import (
    DEFAULT_HALF_LIFE,
    DEFAULT_WINDOW,
    SignalStore,
    get_default_store,
)

logger = logging.getLogger("belief.signal.triggers")


# ── Trigger primitive ──────────────────────────────────────────────────────


# A trigger's predicate is a function from (store, agent_id, now) -> bool.
# Keeping the signature wide gives concrete predicates access to whatever
# read APIs they need (concentration, joint_concentration, recent_signals).
TriggerPredicate = Callable[[SignalStore, str, Optional[datetime]], bool]

# The action receives the agent_id of the agent whose state triggered the
# fire. Actions are best-effort — exceptions are logged and swallowed.
TriggerAction = Callable[[str], None]


@dataclass
class Trigger:
    """A named (predicate, action) pair. Stage 5+ will populate the
    registry with real actions; for now actions are no-op stubs that
    just log."""

    name: str
    predicate: TriggerPredicate
    action: TriggerAction = field(default=lambda agent_id: None)
    description: str = ""


@dataclass
class TriggerFireEvent:
    """Returned from ``TriggerRegistry.evaluate`` so callers can inspect
    which triggers fired in a given pass."""

    trigger_name: str
    agent_id: str
    when: datetime
    error: Optional[str] = None


class TriggerRegistry:
    """Registers triggers, evaluates them against every active agent on
    demand, and dispatches actions for ones that fire.

    The registry doesn't own the SignalStore — pass one in (tests inject
    a tmp_path store; production uses ``get_default_store``).
    """

    def __init__(self, store: Optional[SignalStore] = None) -> None:
        self._store = store
        self._triggers: dict[str, Trigger] = {}

    @property
    def store(self) -> SignalStore:
        if self._store is None:
            self._store = get_default_store()
        return self._store

    def register(self, trigger: Trigger) -> None:
        """Register or replace a trigger by name."""
        if not trigger.name:
            raise ValueError("trigger name must be non-empty")
        self._triggers[trigger.name] = trigger

    def unregister(self, name: str) -> None:
        self._triggers.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self._triggers.keys())

    def evaluate(
        self,
        agent_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> list[TriggerFireEvent]:
        """Evaluate all registered triggers.

        If ``agent_id`` is None, evaluate against every agent that has
        emitted at least one signal. Returns a list of
        ``TriggerFireEvent`` entries for the predicates that fired.
        """
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        targets = [agent_id] if agent_id else self.store.known_agents()
        now = now or _dt.now(_tz.utc)
        events: list[TriggerFireEvent] = []
        for agent in targets:
            for trig in self._triggers.values():
                try:
                    fired = bool(trig.predicate(self.store, agent, now))
                except Exception as e:  # pragma: no cover — predicates may bug
                    logger.warning(
                        "trigger %r predicate failed for agent %r: %s",
                        trig.name,
                        agent,
                        e,
                    )
                    events.append(
                        TriggerFireEvent(
                            trigger_name=trig.name,
                            agent_id=agent,
                            when=now,
                            error=str(e),
                        )
                    )
                    continue
                if not fired:
                    continue
                err: Optional[str] = None
                try:
                    trig.action(agent)
                except Exception as e:  # pragma: no cover — action best-effort
                    err = str(e)
                    logger.warning(
                        "trigger %r action raised for agent %r: %s",
                        trig.name,
                        agent,
                        e,
                    )
                events.append(
                    TriggerFireEvent(
                        trigger_name=trig.name,
                        agent_id=agent,
                        when=now,
                        error=err,
                    )
                )
        return events


# ── Example predicates ─────────────────────────────────────────────────────
#
# These are the building blocks Stage 5 will hook real actions onto. They
# also serve as worked examples for how to write concentration- vs
# joint-concentration-based predicates.


def stress_request_conjunction_predicate(
    stress_threshold: float = 0.6,
    request_threshold: float = 0.3,
    window=DEFAULT_WINDOW,
    half_life=DEFAULT_HALF_LIFE,
) -> TriggerPredicate:
    """Fires when an agent is sustainedly stressed AND asking for help.

    The biological analogue: a damaged plant that's also emitting
    distress volatiles — the conjunction is the high-confidence signal
    that the receiver should respond to (vs. either alone, which could
    be a false positive)."""

    def predicate(store: SignalStore, agent_id: str, now: Optional[datetime]) -> bool:
        s = store.concentration(agent_id, "STRESS", window, half_life, now)
        r = store.concentration(agent_id, "REQUEST", window, half_life, now)
        return s > stress_threshold and r > request_threshold

    predicate.__name__ = "stress_request_conjunction"
    return predicate


def sustained_offer_predicate(
    threshold: float = 0.8,
    window: timedelta = timedelta(hours=1),
    half_life: timedelta = timedelta(minutes=15),
) -> TriggerPredicate:
    """Fires when an agent has been sustainedly contributing.

    Wider window + longer half-life than the stress conjunction — we
    want hours of consistent offers, not a single burst. Stage 5's hub
    promotion uses this as one of its inputs."""

    def predicate(store: SignalStore, agent_id: str, now: Optional[datetime]) -> bool:
        return store.concentration(agent_id, "OFFER", window, half_life, now) > threshold

    predicate.__name__ = "sustained_offer"
    return predicate


def covenant_warn_predicate(
    threshold: float = 0.5,
    window=DEFAULT_WINDOW,
    half_life=DEFAULT_HALF_LIFE,
) -> TriggerPredicate:
    """Fires when an agent has emitted persistent WARN signals.

    Stage 6's defense-priming propagation will wire its broadcast
    action onto this predicate so covenant-class warnings escalate
    eagerly when an agent keeps seeing the same failure mode."""

    def predicate(store: SignalStore, agent_id: str, now: Optional[datetime]) -> bool:
        return store.concentration(agent_id, "WARN", window, half_life, now) > threshold

    predicate.__name__ = "covenant_warn"
    return predicate
