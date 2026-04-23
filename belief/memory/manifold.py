"""
Domain Manifold — knowledge-topology analysis over the soil.

Session 14, Task 2.  Summarises how the soil is distributed across
knowledge domains (fastapi, cli, mcp, data, async, library, script,
general) so the Photosynthesis goal generator can direct attention to
under-explored areas.

The module answers three questions:

1. **Cluster density** — how many nutrients does each domain hold,
   and what's their health (stability / lapse rate)?
2. **Inter-domain connections** — which domain pairs share nutrients?
   A nutrient that belongs to two domains ("fastapi" tag + "async"
   content) forms a cross-edge.  Domains with many connections are
   well-stitched; isolated domains are candidates for bridge goals.
3. **Coverage gaps** — domains whose active-nutrient count sits below
   a configurable threshold.  These are the Photosynthesis daemon's
   priority targets.

Pure stdlib — no numpy / networkx / sklearn.  Consumes a
:class:`~belief.memory.soil.Soil` by iterating
``iter_all_nutrients(include_invalidated=False)`` so invalidated
nutrients (Session 14 Task 1) are correctly excluded from the active
topology.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Optional

from belief.evolution.progression import (
    DOMAINS,
    DOMAIN_DISPLAY_ORDER,
    GENERAL_DOMAIN,
    detect_domain,
)


# ── Configuration ──────────────────────────────────────────────────────────


# Default "sparse soil" threshold.  Domains with fewer active nutrients
# than this are flagged as coverage gaps and surfaced to the
# Photosynthesis daemon as priority targets for new goal generation.
DEFAULT_GAP_THRESHOLD = 5


# ── Data shapes ────────────────────────────────────────────────────────────


@dataclass
class DomainCluster:
    """Summary of a single domain's slice of the soil.

    Fields:
        domain:          Domain name (``"fastapi"``, ``"cli"``, ...).
        size:            Number of active nutrients in this domain.
        mean_stability:  Mean FSRS stability — higher = better-retained
                         knowledge.
        lapse_rate:      Fraction of nutrients with lapse_count > 0.
        sample_content:  Up to 3 representative nutrient contents,
                         ordered by reinforcement count (most-reused
                         first).  Used by the text renderer.
    """

    domain: str
    size: int = 0
    mean_stability: float = 0.0
    lapse_rate: float = 0.0
    sample_content: list[str] = field(default_factory=list)

    def is_sparse(self, threshold: int = DEFAULT_GAP_THRESHOLD) -> bool:
        """Whether this domain falls below the coverage-gap threshold."""
        return self.size < threshold


@dataclass
class CrossEdge:
    """A shared-nutrient connection between two domains."""

    domain_a: str
    domain_b: str
    count: int = 0

    @property
    def key(self) -> tuple[str, str]:
        """Canonical unordered pair, for dict keys."""
        return tuple(sorted((self.domain_a, self.domain_b)))  # type: ignore[return-value]


@dataclass
class ManifoldReport:
    """Complete manifold summary.

    Bundled so CLI renderers and tests can consume a single value
    instead of juggling three separate collections.
    """

    clusters: list[DomainCluster] = field(default_factory=list)
    cross_edges: list[CrossEdge] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    total_active: int = 0
    total_invalidated: int = 0

    def to_json(self) -> str:
        """Serialise as JSON for ``belief manifold --json``."""
        payload = {
            "total_active": self.total_active,
            "total_invalidated": self.total_invalidated,
            "clusters": [
                {
                    "domain": c.domain,
                    "size": c.size,
                    "mean_stability": round(c.mean_stability, 3),
                    "lapse_rate": round(c.lapse_rate, 3),
                    "sample_content": list(c.sample_content),
                }
                for c in self.clusters
            ],
            "cross_edges": [
                {
                    "domain_a": e.domain_a,
                    "domain_b": e.domain_b,
                    "count": e.count,
                }
                for e in self.cross_edges
            ],
            "coverage_gaps": list(self.coverage_gaps),
        }
        return json.dumps(payload, indent=2, sort_keys=False)


# ── Domain classification ──────────────────────────────────────────────────


def nutrient_domains(nutrient) -> set[str]:
    """Return every domain a nutrient belongs to.

    A nutrient can belong to multiple domains because its
    ``framework``, ``tags``, and content often span verticals (e.g. a
    "FastAPI async queue" straddles ``fastapi`` and ``async``).
    Cross-edges are derived from the resulting sets.

    Priority:
      1. ``framework`` field matches a known DOMAINS key → add that
      2. ``tags`` overlap with any domain keyword → add that domain
      3. fallback: ``detect_domain(content, tags)`` primary bucket

    The primary bucket (used for cluster membership) is the first
    element of :func:`primary_domain`.  This function returns the
    full set, so cross-edges between the primary and secondary
    buckets are counted once per nutrient.
    """
    domains: set[str] = set()

    framework = (getattr(nutrient, "framework", None) or "").lower()
    if framework in DOMAINS:
        domains.add(framework)

    tags = [str(t).lower() for t in (getattr(nutrient, "tags", None) or [])]
    tag_set = set(tags)
    for dom, keywords in DOMAINS.items():
        # Normalise keyword variants: strip whitespace (DOMAINS uses
        # "async " with a trailing space to avoid substring collisions
        # in goal text, but tag lookup should match the bare "async")
        # and also tolerate hyphenated forms of multi-word keywords.
        kw_variants = {kw.strip() for kw in keywords} | {
            kw.strip().replace(" ", "-") for kw in keywords
        }
        if tag_set & kw_variants:
            domains.add(dom)

    # Always include the detect_domain fallback so nutrients without
    # explicit tags/framework still land somewhere.
    content = getattr(nutrient, "content", "") or ""
    primary = detect_domain(content, tags=tags)
    if primary != GENERAL_DOMAIN or not domains:
        domains.add(primary)

    return domains


def primary_domain(nutrient) -> str:
    """Return the single "canonical" domain used for cluster membership.

    Prefers the framework field, then the first tag that maps to a
    domain, then ``detect_domain(content)``, finally ``general``.  The
    "first" rule mirrors :func:`detect_domain`'s ordering so results
    are reproducible.
    """
    framework = (getattr(nutrient, "framework", None) or "").lower()
    if framework in DOMAINS:
        return framework

    tags = [str(t).lower() for t in (getattr(nutrient, "tags", None) or [])]
    for dom, keywords in DOMAINS.items():
        # Normalise keyword variants: strip whitespace (DOMAINS uses
        # "async " with a trailing space to avoid substring collisions
        # in goal text, but tag lookup should match the bare "async")
        # and also tolerate hyphenated forms of multi-word keywords.
        kw_variants = {kw.strip() for kw in keywords} | {
            kw.strip().replace(" ", "-") for kw in keywords
        }
        if set(tags) & kw_variants:
            return dom

    content = getattr(nutrient, "content", "") or ""
    return detect_domain(content, tags=tags)


# ── Analysis ───────────────────────────────────────────────────────────────


def _top_samples(nutrients: Iterable, k: int = 3) -> list[str]:
    """Pick up to *k* representative contents, preferring most-reused."""
    ranked = sorted(
        nutrients,
        key=lambda n: (
            -int(getattr(n, "reinforcement_count", 0) or 0),
            getattr(n, "nutrient_id", ""),
        ),
    )
    samples: list[str] = []
    for n in ranked:
        content = (getattr(n, "content", "") or "").strip().splitlines()[0:1]
        if content:
            samples.append(content[0][:120])
            if len(samples) >= k:
                break
    return samples


def build_manifold(
    soil,
    gap_threshold: int = DEFAULT_GAP_THRESHOLD,
    as_of: Optional[float] = None,
) -> ManifoldReport:
    """Compute the full manifold report for *soil*.

    Walks the active view (``include_invalidated=False``) exactly once
    and bucketises nutrients into domains.  Inter-domain edges are
    derived from nutrients whose :func:`nutrient_domains` returns more
    than one bucket.  Coverage gaps are the domains whose size falls
    below *gap_threshold*.

    Args:
        soil:          Any object implementing ``iter_all_nutrients``
                       with the Session 14 signature.
        gap_threshold: Minimum active-nutrient count for a domain to
                       be considered "covered" (default: 5).
        as_of:         Optional timestamp to rewind the view; passed
                       straight to ``iter_all_nutrients``.  Useful for
                       historical comparison (e.g. "what did the
                       topology look like last week?").

    Returns:
        Populated :class:`ManifoldReport`.
    """
    # Bucket: domain -> list of nutrients (for samples / stats).
    buckets: dict[str, list] = {d: [] for d in DOMAIN_DISPLAY_ORDER}
    # Edge counter keyed by canonical unordered pair.
    edge_counts: dict[tuple[str, str], int] = {}
    total_active = 0

    for n in soil.iter_all_nutrients(include_invalidated=False, as_of=as_of):
        total_active += 1
        bucket = primary_domain(n)
        buckets.setdefault(bucket, []).append(n)

        # Cross-edges: every pair of distinct domains this nutrient
        # touches counts as one connection.
        doms = list(nutrient_domains(n))
        if len(doms) > 1:
            for i in range(len(doms)):
                for j in range(i + 1, len(doms)):
                    key = tuple(sorted((doms[i], doms[j])))
                    edge_counts[key] = edge_counts.get(key, 0) + 1  # type: ignore[index]

    clusters: list[DomainCluster] = []
    for domain in DOMAIN_DISPLAY_ORDER:
        members = buckets.get(domain, [])
        if not members:
            clusters.append(DomainCluster(domain=domain, size=0))
            continue
        stabilities = [float(getattr(n, "stability", 1.0)) for n in members]
        lapse_flags = [1 if int(getattr(n, "lapse_count", 0) or 0) > 0 else 0 for n in members]
        clusters.append(
            DomainCluster(
                domain=domain,
                size=len(members),
                mean_stability=sum(stabilities) / len(stabilities),
                lapse_rate=sum(lapse_flags) / len(lapse_flags),
                sample_content=_top_samples(members, k=3),
            )
        )

    cross_edges = [
        CrossEdge(domain_a=a, domain_b=b, count=c)
        for (a, b), c in sorted(
            edge_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]

    gaps = [c.domain for c in clusters if c.is_sparse(gap_threshold)]

    # Count invalidated via a separate scan (cheap in-memory pass).
    total_invalidated = 0
    for n in soil.iter_all_nutrients(include_invalidated=True):
        valid_until = float(getattr(n, "valid_until", 0.0) or 0.0)
        if valid_until > 0.0:
            total_invalidated += 1

    return ManifoldReport(
        clusters=clusters,
        cross_edges=cross_edges,
        coverage_gaps=gaps,
        total_active=total_active,
        total_invalidated=total_invalidated,
    )


# ── Rendering ──────────────────────────────────────────────────────────────


def _bar(count: int, scale: int, width: int = 20) -> str:
    """Tiny ASCII bar for the cluster histogram."""
    if scale <= 0:
        return ""
    fill = min(width, int(round(width * count / scale)))
    return "▇" * fill if fill > 0 else ""


def format_report(
    report: ManifoldReport,
    gap_threshold: int = DEFAULT_GAP_THRESHOLD,
) -> str:
    """Pretty-print a :class:`ManifoldReport` as plain text.

    Three sections:

        Domain clusters   bar chart, stability, lapse rate, samples
        Inter-domain links  top N shared-nutrient edges
        Coverage gaps     domains below the threshold

    The CLI calls this when ``--json`` is not passed.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("═" * 62)
    lines.append("  Domain Manifold")
    lines.append("═" * 62)
    lines.append(
        f"  active nutrients: {report.total_active}   invalidated: {report.total_invalidated}"
    )
    lines.append("")

    # Cluster bar chart
    scale = max((c.size for c in report.clusters), default=1)
    lines.append("  ─── Clusters ─────────────────────────────────────────────")
    lines.append(f"  {'domain':<9} {'size':>4}  {'bar':<22}  stab   lapse")
    for c in report.clusters:
        lines.append(
            f"  {c.domain:<9} {c.size:>4}  {_bar(c.size, scale):<22}  "
            f"{c.mean_stability:>5.2f}  {c.lapse_rate:>5.1%}"
        )
    lines.append("")

    # Top samples per non-empty cluster (two-line format)
    non_empty = [c for c in report.clusters if c.size > 0]
    if non_empty:
        lines.append("  ─── Samples (top-3 by reuse) ────────────────────────────")
        for c in non_empty:
            if not c.sample_content:
                continue
            lines.append(f"  [{c.domain}]")
            for s in c.sample_content:
                lines.append(f"    • {s}")
        lines.append("")

    # Cross-edges
    lines.append("  ─── Inter-domain links ──────────────────────────────────")
    if not report.cross_edges:
        lines.append("  (no nutrients span multiple domains)")
    else:
        for e in report.cross_edges[:10]:
            lines.append(f"  {e.domain_a:<9} ── {e.count:>4} nutrients ──> {e.domain_b}")
        if len(report.cross_edges) > 10:
            lines.append(f"  … and {len(report.cross_edges) - 10} more edges")
    lines.append("")

    # Coverage gaps
    lines.append(f"  ─── Coverage gaps (< {gap_threshold} active nutrients) ─────────────")
    if not report.coverage_gaps:
        lines.append("  (every domain is adequately covered)")
    else:
        lines.append("  photosynthesis should target: " + ", ".join(report.coverage_gaps))
    lines.append("═" * 62)
    return "\n".join(lines)
