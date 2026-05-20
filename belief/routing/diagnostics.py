"""Topology diagnostics (mycorrhizal Stage 5, Area 3).

Makes the routing graph *visible* — the software equivalent of Beiler's
microsatellite mapping. Reads the routing-events log, reconstructs the
agent → hub → engine graph, and reports the metrics that tell the operator
whether the topology is healthy (scale-free, robust) or drifting toward
fragility (over-centralised on a few hubs).

Metrics:
  * degree distribution      — edges per node
  * mean path length         — average hops from a request origin to engine
  * clustering coefficient   — networkx average clustering (≈0 for the
                               bipartite agent/hub/engine graph; reported
                               for completeness + future multi-tier hubs)
  * over-centralisation flag — top-3 hubs carry > threshold of all hub edges

``belief topology`` renders this. With no routing events (the current
state) the report is empty and flags nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from belief.routing._store import RoutingStore

logger = logging.getLogger("belief.routing.diagnostics")

ENGINE_NODE = "__engine__"
DEFAULT_OVERCENTRALIZATION_THRESHOLD = 0.70


@dataclass
class TopologyReport:
    """Computed topology metrics over a window of routing events."""

    event_count: int
    node_count: int
    edge_count: int
    hub_ids: list[str]
    degree_distribution: dict[str, int]
    mean_path_length: float
    clustering_coefficient: float
    top3_hub_edge_share: float
    overcentralized: bool
    direct_count: int = 0
    via_hub_count: int = 0
    cache_hit_count: int = 0
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_count": self.event_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "hub_ids": list(self.hub_ids),
            "degree_distribution": dict(self.degree_distribution),
            "mean_path_length": round(self.mean_path_length, 4),
            "clustering_coefficient": round(self.clustering_coefficient, 4),
            "top3_hub_edge_share": round(self.top3_hub_edge_share, 4),
            "overcentralized": self.overcentralized,
            "direct_count": self.direct_count,
            "via_hub_count": self.via_hub_count,
            "cache_hit_count": self.cache_hit_count,
            "notes": self.notes,
        }


class TopologyDiagnostics:
    """Reconstructs + measures the routing graph from the events log."""

    def __init__(
        self,
        store: RoutingStore,
        overcentralization_threshold: float = DEFAULT_OVERCENTRALIZATION_THRESHOLD,
    ) -> None:
        self._store = store
        self.overcentralization_threshold = float(overcentralization_threshold)

    def report(self, window: Optional[str] = None) -> TopologyReport:
        """Build a ``TopologyReport`` over an optional window (e.g. ``"7d"``).

        ``None`` window = all recorded events.
        """
        cutoff_iso: Optional[str] = None
        if window is not None:
            from belief.memory.reciprocity import _parse_window

            delta = _parse_window(window)
            if delta is not None:
                cutoff_iso = (datetime.now(timezone.utc) - delta).isoformat()
        events = self._store.events_since(cutoff_iso)
        return self._compute(events)

    def _compute(self, events) -> TopologyReport:
        if not events:
            return TopologyReport(
                event_count=0,
                node_count=0,
                edge_count=0,
                hub_ids=[],
                degree_distribution={},
                mean_path_length=0.0,
                clustering_coefficient=0.0,
                top3_hub_edge_share=0.0,
                overcentralized=False,
                notes={"status": "no routing events recorded yet"},
            )

        # Tally decisions + reconstruct edges.
        direct = via_hub = cache_hit = 0
        # edges as (src, dst) with multiplicity → we keep counts.
        edge_counts: dict[tuple[str, str], int] = {}
        hub_edge_counts: dict[str, int] = {}
        path_lengths: list[int] = []

        for e in events:
            agent = e["agent_id"]
            kind = e["decision_kind"]
            hub = e["hub_id"]
            if kind == "direct":
                direct += 1
                edge_counts[(agent, ENGINE_NODE)] = edge_counts.get((agent, ENGINE_NODE), 0) + 1
                path_lengths.append(1)
            elif kind in ("via_hub", "cache_hit"):
                if kind == "via_hub":
                    via_hub += 1
                else:
                    cache_hit += 1
                if hub:
                    edge_counts[(agent, hub)] = edge_counts.get((agent, hub), 0) + 1
                    hub_edge_counts[hub] = hub_edge_counts.get(hub, 0) + 1
                    # cache hits stop at the hub (1 hop); via_hub continues
                    # to the engine (2 hops). Record the hub→engine edge
                    # only for via_hub.
                    if kind == "via_hub":
                        edge_counts[(hub, ENGINE_NODE)] = edge_counts.get((hub, ENGINE_NODE), 0) + 1
                        path_lengths.append(2)
                    else:
                        path_lengths.append(1)
                else:
                    # malformed event — treat as direct
                    path_lengths.append(1)

        nodes = set()
        for src, dst in edge_counts:
            nodes.add(src)
            nodes.add(dst)

        # Degree distribution (undirected degree = in + out multiplicity).
        degree: dict[str, int] = {}
        for (src, dst), c in edge_counts.items():
            degree[src] = degree.get(src, 0) + c
            degree[dst] = degree.get(dst, 0) + c

        # Over-centralisation: share of hub-incident edges held by top 3 hubs.
        total_hub_edges = sum(hub_edge_counts.values())
        top3_share = 0.0
        if total_hub_edges > 0:
            top3 = sorted(hub_edge_counts.values(), reverse=True)[:3]
            top3_share = sum(top3) / total_hub_edges
        overcentralized = total_hub_edges > 0 and top3_share > self.overcentralization_threshold

        mean_path = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
        clustering = self._clustering(edge_counts)

        return TopologyReport(
            event_count=len(events),
            node_count=len(nodes),
            edge_count=len(edge_counts),
            hub_ids=sorted(hub_edge_counts.keys()),
            degree_distribution=degree,
            mean_path_length=mean_path,
            clustering_coefficient=clustering,
            top3_hub_edge_share=top3_share,
            overcentralized=overcentralized,
            direct_count=direct,
            via_hub_count=via_hub,
            cache_hit_count=cache_hit,
        )

    def _clustering(self, edge_counts: dict[tuple[str, str], int]) -> float:
        """Average clustering coefficient via networkx if available.

        The agent/hub/engine graph is bipartite-ish so clustering is
        typically 0; we still report it honestly because future multi-tier
        hub layers (hub-to-hub links) will make it non-trivial, and a
        non-zero value on the current graph would itself be a signal worth
        seeing.
        """
        try:
            import networkx as nx  # noqa: PLC0415

            g = nx.Graph()
            for src, dst in edge_counts:
                g.add_edge(src, dst)
            if g.number_of_nodes() == 0:
                return 0.0
            return float(nx.average_clustering(g))
        except Exception as e:  # pragma: no cover — networkx optional
            logger.debug("clustering computation skipped: %s", e)
            return 0.0


# ── CLI rendering ──────────────────────────────────────────────────────────


def cli_format_report(report: TopologyReport, hub_ids: Optional[list[str]] = None) -> str:
    lines = [
        "Routing topology",
        f"  events analysed:      {report.event_count}",
        f"  decisions:            direct={report.direct_count} "
        f"via_hub={report.via_hub_count} cache_hit={report.cache_hit_count}",
        f"  nodes / edges:        {report.node_count} / {report.edge_count}",
        f"  mean path length:     {report.mean_path_length:.2f} hops",
        f"  clustering coeff:     {report.clustering_coefficient:.4f}",
    ]
    current_hubs = hub_ids if hub_ids is not None else report.hub_ids
    lines.append(f"  active hubs:          {len(current_hubs)}")
    if current_hubs:
        for h in current_hubs:
            deg = report.degree_distribution.get(h, 0)
            lines.append(f"    {h:<28} degree={deg}")
    if report.event_count == 0:
        lines.append(
            "  (no routing events yet — the topology is empty; "
            "every request bypasses straight to the engine)"
        )
    if report.overcentralized:
        lines.append(
            f"  ⚠ OVER-CENTRALISED: top-3 hubs carry "
            f"{report.top3_hub_edge_share:.0%} of hub edges "
            f"(threshold {DEFAULT_OVERCENTRALIZATION_THRESHOLD:.0%}); "
            f"consider forcing hub rotation"
        )
    return "\n".join(lines)
