"""Hub-mediated request routing (mycorrhizal Stage 5, Area 3).

Receives every incoming request and decides whether it goes directly to the
engine or through a hub. Hubs maintain a small LRU cache of recent
(query, response) pairs and can serve cache hits without hitting the engine.

**Bypass invariant.** When no hubs exist — the current state, since no
autonomous agents have run — every decision is ``ROUTE_DIRECT``. This is not
a special case bolted on; it falls out of ``HubRegistry.current_hubs()``
being empty. The router is therefore a transparent pass-through until the
reciprocity ledger has earned hubs, which is exactly the property that keeps
the existing test gate green.

**Enforcement flag.** ``BELIEF_ROUTING_ENFORCE=1`` switches the router from
observability-only (compute + log the decision, caller ignores it) to
active (caller honours VIA_HUB / CACHE_HIT). Default OFF. At this stage the
recomposer hook always treats the decision as advisory regardless of the
flag — actually rerouting builds waits for a future session when there are
real agents to route. The flag exists so that future session can flip it
without a schema change.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from belief.routing._store import RoutingStore
from belief.routing.hubs import HubRegistry

logger = logging.getLogger("belief.routing.router")

_ENFORCE_ENV = "BELIEF_ROUTING_ENFORCE"


def enforcement_enabled() -> bool:
    """True iff ``BELIEF_ROUTING_ENFORCE`` is set to a truthy value.

    Read at call time (not import time) so tests + operators can toggle it
    per-process without reimporting.
    """
    return os.environ.get(_ENFORCE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class RoutingKind(str, Enum):
    DIRECT = "direct"  # straight to the engine
    VIA_HUB = "via_hub"  # forwarded through a hub
    CACHE_HIT = "cache_hit"  # served from a hub's LRU cache


@dataclass(frozen=True)
class RoutingDecision:
    """The router's verdict for one request.

    ``enforced`` records whether the decision is being treated as binding
    (enforcement flag on) or advisory (default). Callers in observability
    mode log the decision and proceed exactly as before.
    """

    kind: RoutingKind
    agent_id: str
    hub_id: Optional[str] = None
    reason: str = ""
    enforced: bool = False
    cached_response: Optional[object] = None

    @property
    def is_bypass(self) -> bool:
        return self.kind is RoutingKind.DIRECT


@dataclass
class _HubCache:
    """Per-hub LRU cache of (query_key -> response). Only exercised when
    hubs exist and a query key is supplied; inert for the build pipeline
    which doesn't pass query keys."""

    capacity: int = 128
    _data: "OrderedDict[str, object]" = field(default_factory=OrderedDict)

    def get(self, key: str) -> Optional[object]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: object) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)


class Router:
    """Decides + records the route for each request.

    Construct with a shared ``RoutingStore`` and a ``HubRegistry``. The
    router never raises on the decision path — a routing failure must not
    take down a build, so callers can treat ``route`` as best-effort.
    """

    def __init__(
        self,
        store: RoutingStore,
        hub_registry: HubRegistry,
        cache_capacity: int = 128,
    ) -> None:
        self._store = store
        self._hubs = hub_registry
        self._caches: dict[str, _HubCache] = {}
        self._cache_capacity = cache_capacity

    def _cache_for(self, hub_id: str) -> _HubCache:
        if hub_id not in self._caches:
            self._caches[hub_id] = _HubCache(capacity=self._cache_capacity)
        return self._caches[hub_id]

    def route(
        self,
        agent_id: str,
        query_key: Optional[str] = None,
        record: bool = True,
        now: Optional[datetime] = None,
    ) -> RoutingDecision:
        """Compute (and optionally record) the route for ``agent_id``.

        Decision order:
          1. No hubs exist → DIRECT (the bypass invariant).
          2. Sender is itself a hub → DIRECT.
          3. Sender is peripheral → VIA_HUB(nearest). If a ``query_key``
             is supplied and the hub has a cached response → CACHE_HIT.
        """
        enforced = enforcement_enabled()
        decision = self._decide(agent_id, query_key, enforced)
        if record:
            try:
                self._store.record_event(
                    agent_id=agent_id,
                    decision_kind=decision.kind.value,
                    hub_id=decision.hub_id,
                    ts=now,
                )
            except Exception as e:  # pragma: no cover — recording is best-effort
                logger.debug("routing event record skipped: %s", e)
        return decision

    def _decide(self, agent_id: str, query_key: Optional[str], enforced: bool) -> RoutingDecision:
        hubs = self._hubs.current_hubs()
        if not hubs:
            return RoutingDecision(
                kind=RoutingKind.DIRECT,
                agent_id=agent_id,
                reason="no hubs exist — bypass to engine",
                enforced=enforced,
            )
        if self._hubs.is_hub(agent_id):
            return RoutingDecision(
                kind=RoutingKind.DIRECT,
                agent_id=agent_id,
                reason="sender is a hub — direct to engine",
                enforced=enforced,
            )
        hub = self._hubs.nearest_hub(agent_id)
        if hub is None:
            return RoutingDecision(
                kind=RoutingKind.DIRECT,
                agent_id=agent_id,
                reason="no eligible hub for sender — bypass to engine",
                enforced=enforced,
            )
        if query_key is not None:
            cached = self._cache_for(hub).get(query_key)
            if cached is not None:
                return RoutingDecision(
                    kind=RoutingKind.CACHE_HIT,
                    agent_id=agent_id,
                    hub_id=hub,
                    reason=f"served from hub {hub} cache",
                    enforced=enforced,
                    cached_response=cached,
                )
        return RoutingDecision(
            kind=RoutingKind.VIA_HUB,
            agent_id=agent_id,
            hub_id=hub,
            reason=f"forwarded through hub {hub}",
            enforced=enforced,
        )

    def cache_response(self, hub_id: str, query_key: str, response: object) -> None:
        """Populate a hub's LRU cache. Called by the (future) hub-serving
        path once it has an engine response to memoise."""
        self._cache_for(hub_id).put(query_key, response)
