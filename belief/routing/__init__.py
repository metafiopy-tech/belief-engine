"""Scale-free hub topology, routing, and sanctions (mycorrhizal Stage 5).

Areas 3 + 5 + 6 (partial) of the mycorrhizal brief. Beiler et al. 2010 mapped
real mycorrhizal networks and found scale-free / small-world topology: a few
hub trees carry most connectivity, peripheral nodes route through them, and
the graph is robust to random node loss but fragile to targeted hub loss.
Kiers & Denison 2008 + Kiers et al. 2011 established sanctions as the
enforcement mechanism that keeps the market honest without manual
intervention.

The Belief Engine analogue: high-reciprocity agents (Stage 1 ledger) are
*derived* as hubs (never assigned by tenure — that's the Area 6 "mother
tree" altruism story the 2023 literature debunked, deliberately excluded
here). Peripheral agents route through hubs; sanctions throttle persistent
free-riders.

**Critical safety property — bypass by construction.** The Belief Engine is
currently a one-shot LangGraph with no autonomous agents, so no hubs exist.
Every router decision in that state is ``ROUTE_DIRECT`` and every sanction
decision is ``ALLOW``. Enforcement (actually throttling or rerouting real
builds) only happens when ``BELIEF_ROUTING_ENFORCE=1`` AND hubs exist.
Default behaviour is observability-only: decisions are computed and logged
for the ``belief topology`` diagnostic, but the build pipeline is never
altered. This keeps the existing test gate green by construction.
"""

from belief.routing.hubs import (
    DEFAULT_LIFETIME_FLOOR,
    HubRegistry,
)
from belief.routing.router import (
    RoutingDecision,
    RoutingKind,
    Router,
    enforcement_enabled,
)
from belief.routing.sanctions import (
    SanctionAction,
    SanctionDecision,
    SanctionsEngine,
)
from belief.routing.diagnostics import (
    TopologyDiagnostics,
    TopologyReport,
)

__all__ = [
    "DEFAULT_LIFETIME_FLOOR",
    "HubRegistry",
    "RoutingDecision",
    "RoutingKind",
    "Router",
    "enforcement_enabled",
    "SanctionAction",
    "SanctionDecision",
    "SanctionsEngine",
    "TopologyDiagnostics",
    "TopologyReport",
]
