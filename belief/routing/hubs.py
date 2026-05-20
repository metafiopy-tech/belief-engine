"""Hub designation by reciprocity (mycorrhizal Stage 5, Area 3).

Hubs are *derived*, never assigned. An agent becomes a hub when its recent
contribution profile puts it in the top tier of the reciprocity ledger
(Stage 1) — specifically, top-decile 30-day exchange rate AND a lifetime
``nutrients_returned`` above a floor. This is the load-bearing half of the
Beiler 2010 finding (scale-free topology with reciprocity-earned hubs); the
"mother tree" altruism story attached to biological hubs is deliberately NOT
imported (Karst et al. 2023 — see module docstring in __init__).

Demotion: a hub that falls below the threshold for N consecutive recomputes
is demoted. This rotates the hub set over time so connectivity doesn't
ossify around early winners — the mycorrhizal brief's explicit guard against
the fragility-to-hub-loss failure mode.

Bypass property: with an empty or sparse reciprocity ledger (the current
state — no autonomous agents have run), the lifetime-floor gate ensures
*no* agent qualifies, so ``current_hubs()`` is empty and the router falls
through to direct routing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from belief.routing._store import RoutingStore

logger = logging.getLogger("belief.routing.hubs")

# Lifetime nutrients_returned an agent must exceed to be hub-eligible. This
# is the real gate that keeps the hub set empty until agents have
# demonstrated sustained contribution — top-decile alone would promote the
# best of a tiny, low-activity population, which isn't what we want.
DEFAULT_LIFETIME_FLOOR = 100.0

# Fraction of agents (by exchange rate) eligible for hub status. 0.1 = top
# decile, per the brief.
DEFAULT_TOP_FRACTION = 0.1

# Consecutive sub-threshold recomputes before a hub is demoted.
DEFAULT_DEMOTE_AFTER = 3


@dataclass(frozen=True)
class HubCandidate:
    """An agent's standing relative to the hub threshold at recompute time."""

    agent_id: str
    exchange_rate_30d: float
    lifetime_nutrients: float
    qualifies: bool


class HubRegistry:
    """Derives + persists hub membership from the reciprocity ledger.

    ``reciprocity_ledger`` must expose ``rank_agents(window)`` and
    ``stats(agent_id, window)`` (the Stage 1 ``ReciprocityLedger`` API).
    Tests inject both the ledger and the routing store against tmp paths.
    """

    def __init__(
        self,
        store: RoutingStore,
        reciprocity_ledger,
        lifetime_floor: float = DEFAULT_LIFETIME_FLOOR,
        top_fraction: float = DEFAULT_TOP_FRACTION,
        demote_after: int = DEFAULT_DEMOTE_AFTER,
    ) -> None:
        self._store = store
        self._ledger = reciprocity_ledger
        self.lifetime_floor = float(lifetime_floor)
        if not (0.0 < top_fraction <= 1.0):
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        self.top_fraction = float(top_fraction)
        self.demote_after = int(demote_after)

    # ── Recompute ───────────────────────────────────────────────────────

    def _eligible_candidates(self) -> list[HubCandidate]:
        """Compute which agents currently qualify for hub status.

        Qualification: 30-day exchange rate in the top ``top_fraction`` of
        all agents AND lifetime nutrients_returned > floor. With fewer
        agents than ``ceil(1/top_fraction)``, the top-fraction cut still
        admits at least the single best agent — but the lifetime floor is
        what actually gates promotion in a small population.
        """
        ranked = self._ledger.rank_agents(window="30d")  # sorted desc by rate
        if not ranked:
            return []
        n = len(ranked)
        top_k = max(1, math.ceil(n * self.top_fraction))
        top_set = {s.agent_id for s in ranked[:top_k]}
        candidates: list[HubCandidate] = []
        for s in ranked:
            lifetime = self._ledger.stats(s.agent_id, window="all").nutrients_returned
            qualifies = (
                s.agent_id in top_set and s.exchange_rate > 0.0 and lifetime > self.lifetime_floor
            )
            candidates.append(
                HubCandidate(
                    agent_id=s.agent_id,
                    exchange_rate_30d=s.exchange_rate,
                    lifetime_nutrients=lifetime,
                    qualifies=qualifies,
                )
            )
        return candidates

    def recompute(self) -> list[str]:
        """Recompute hub membership and persist it. Returns the current
        hub-id list after the recompute.

        Promotion is immediate on qualification. Demotion is lagged:
        a hub that fails to qualify increments its ``below_count`` and is
        only demoted once that reaches ``demote_after``. A re-qualifying
        hub resets its counter.
        """
        candidates = self._eligible_candidates()
        cand_by_id = {c.agent_id: c for c in candidates}
        # Existing hub status, so we can apply demotion hysteresis.
        existing = {r["agent_id"]: r for r in self._store.all_hub_status()}
        now_iso = datetime.now(timezone.utc).isoformat()

        # Union of agents we know about (candidates + already-tracked).
        all_ids = set(cand_by_id) | set(existing)
        for agent_id in all_ids:
            cand = cand_by_id.get(agent_id)
            row = existing.get(agent_id)
            was_hub = bool(row["is_hub"]) if row else False
            below_count = int(row["below_count"]) if row else 0
            promoted_at = row["promoted_at"] if row else None

            qualifies = cand.qualifies if cand else False

            if qualifies:
                # Promote (or keep) — reset the demotion counter.
                self._store.upsert_hub_status(
                    agent_id=agent_id,
                    is_hub=True,
                    below_count=0,
                    promoted_at=promoted_at or now_iso,
                )
            elif was_hub:
                # Currently a hub but didn't qualify — apply hysteresis.
                new_below = below_count + 1
                if new_below >= self.demote_after:
                    self._store.upsert_hub_status(
                        agent_id=agent_id,
                        is_hub=False,
                        below_count=0,
                        promoted_at=None,
                    )
                    logger.info(
                        "Hub demoted after %d sub-threshold windows: %s",
                        self.demote_after,
                        agent_id,
                    )
                else:
                    self._store.upsert_hub_status(
                        agent_id=agent_id,
                        is_hub=True,
                        below_count=new_below,
                        promoted_at=promoted_at,
                    )
            else:
                # Not a hub and doesn't qualify — record the standing so
                # the row exists, but stay non-hub.
                if row is not None:
                    self._store.upsert_hub_status(
                        agent_id=agent_id,
                        is_hub=False,
                        below_count=0,
                        promoted_at=None,
                    )
        return self.current_hubs()

    # ── Reads ───────────────────────────────────────────────────────────

    def current_hubs(self) -> list[str]:
        return self._store.current_hub_ids()

    def is_hub(self, agent_id: str) -> bool:
        row = self._store.get_hub_status(agent_id)
        return bool(row["is_hub"]) if row else False

    def nearest_hub(self, agent_id: str) -> Optional[str]:
        """Return the hub a peripheral agent should route through.

        Stage 5 simplification: the brief proposes semantic proximity via
        a ChromaDB nearest-neighbour query on a summary vector of the
        agent's recent activity. Agents don't have summary vectors yet
        (no autonomous agents exist), so this returns the highest-
        exchange-rate hub that isn't the agent itself. Semantic-proximity
        routing is deferred until agents carry activity vectors — recorded
        as a known simplification, not an oversight.
        """
        hubs = [h for h in self.current_hubs() if h != agent_id]
        if not hubs:
            return None
        # Rank hubs by 30d exchange rate, pick the top.
        best: Optional[str] = None
        best_rate = -1.0
        for h in hubs:
            rate = self._ledger.exchange_rate(h, window="30d")
            if rate > best_rate:
                best_rate = rate
                best = h
        return best
