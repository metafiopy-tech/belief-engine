"""
Progression Tracker — generative chain stage detection.

Tracks which stage the engine has reached in building its own toolchain:

  Stage 0: Seed       — hand-authored tools only
  Stage 1: Cluster    — tools form coherent clusters (HDBSCAN or threshold)
  Stage 2: Tessellation — tools cover most of the benchmark space
  Stage 3: Basis      — tools are diverse (high SVD rank ratio)
  Stage 4: Connectivity — tools co-occur in builds (shared context)
  Stage 5: Archetypes — recurring build patterns with high reuse

sklearn is optional.  If unavailable, uses threshold-based grouping.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("belief.evolution.progression")


# ---------------------------------------------------------------------------
# Session 7: per-domain (vertical) progression tracking
# ---------------------------------------------------------------------------


# Keyword -> domain dictionary. Matched as a substring against the goal
# text (case-insensitive). First domain with a hit wins; falls back to
# "general". Deliberately simple — no LLM in the hot path (spec).
DOMAINS: dict[str, list[str]] = {
    "fastapi": ["fastapi", "api", "rest", "crud", "endpoint", "route"],
    "cli":     ["click", "cli", "command-line", "command line", "typer"],
    "mcp":     ["mcp", "fastmcp", "tool-server", "tool server"],
    "data":    ["csv", "pipeline", "etl", "data pipeline"],
    "async":   ["asyncio", "websocket", "queue", "async "],
    "library": ["sdk", "wrapper", "library", "package"],
    "script":  ["script", "fizzbuzz", "fibonacci"],
}

GENERAL_DOMAIN = "general"

# Stable ordering for display (matches the spec's example).
DOMAIN_DISPLAY_ORDER: tuple[str, ...] = (
    "fastapi", "cli", "mcp", "data", "async", "library", "script", GENERAL_DOMAIN,
)


def detect_domain(goal: str, tags: Optional[list[str]] = None) -> str:
    """Classify a goal into one of the DOMAINS buckets.

    Keyword-match only. Checks the goal text first, then falls back to
    any supplied tags. Returns "general" when nothing matches.
    """
    goal_lower = (goal or "").lower()
    for domain, keywords in DOMAINS.items():
        if any(kw in goal_lower for kw in keywords):
            return domain
    if tags:
        tag_set = {str(t).lower() for t in tags}
        for domain, keywords in DOMAINS.items():
            if tag_set.intersection({kw.replace(" ", "-") for kw in keywords}):
                return domain
            if tag_set.intersection(keywords):
                return domain
    return GENERAL_DOMAIN


@dataclass
class ProgressionMetrics:
    """Current position in the generative chain."""

    seed_tool_count: int = 0            # Stage 0: hand-authored tools
    cluster_count: int = 0              # Stage 1: knowledge clusters
    cluster_silhouette: float = 0.0     # Quality of clusters
    coverage_fraction: float = 0.0      # Stage 2: benchmark space coverage
    basis_rank_ratio: float = 0.0       # Stage 3: tool diversity
    connectivity_fraction: float = 0.0  # Stage 4: tool co-occurrence
    archetype_count: int = 0            # Stage 5: recurring patterns
    archetype_reuse: float = 0.0        # Stage 5: reuse on novel tasks
    current_stage: int = 0              # 0-5
    total_tool_count: int = 0
    # Session 7: which domain this snapshot represents. "general" means
    # the cross-domain view; otherwise one of DOMAINS.
    domain: str = GENERAL_DOMAIN


def compute_progression(
    soil,
    tool_registry,
    build_traces: list[dict],
    domain: str = GENERAL_DOMAIN,
) -> ProgressionMetrics:
    """Compute current generative chain stage.

    Args:
        soil:           Soil instance for accessing embeddings.
        tool_registry:  ToolRegistry instance.
        build_traces:   Recent build traces for connectivity/archetype analysis.

    Returns:
        ProgressionMetrics with current_stage set to 0-5.
    """
    all_tools = tool_registry.get_active_tools()

    # Session 7: if a specific domain was requested, filter tools and
    # build_traces whose tags/goal overlap that domain. "general"
    # keeps the cross-domain behavior (no filter).
    if domain != GENERAL_DOMAIN:
        all_tools = [t for t in all_tools if _tool_in_domain(t, domain)]
        build_traces = [tr for tr in build_traces if _trace_in_domain(tr, domain)]

    # Stage 0: Seed — count hand-authored vs self-authored
    seed_count = len([t for t in all_tools if t.created_by == "human"])
    total_count = len(all_tools)

    # Get embeddings from ChromaDB
    embeddings = _get_tool_embeddings(soil, all_tools)

    # Stage 1: Cluster
    cluster_count, silhouette = _compute_clusters(embeddings)

    # Stage 2: Tessellation — coverage of benchmark space
    coverage = _compute_coverage(soil, all_tools)

    # Stage 3: Basis — tool diversity
    rank_ratio = _compute_rank_ratio(embeddings)

    # Stage 4: Connectivity — tool co-occurrence
    connectivity = _compute_connectivity(all_tools, build_traces)

    # Stage 5: Archetypes
    archetype_count, reuse = _detect_archetypes(build_traces)

    # Determine current stage (highest stage with all prerequisites met)
    stage = 0
    if cluster_count >= 3 and silhouette > 0.2:
        stage = 1
    if stage >= 1 and coverage > 0.85:
        stage = 2
    if stage >= 2 and (rank_ratio > 0.7):
        stage = 3
    if stage >= 3 and connectivity > 0.4:
        stage = 4
    if stage >= 4 and archetype_count >= 3 and reuse > 0.5:
        stage = 5

    return ProgressionMetrics(
        seed_tool_count=seed_count,
        cluster_count=cluster_count,
        cluster_silhouette=silhouette,
        coverage_fraction=coverage,
        basis_rank_ratio=rank_ratio,
        connectivity_fraction=connectivity,
        archetype_count=archetype_count,
        archetype_reuse=reuse,
        current_stage=stage,
        total_tool_count=total_count,
        domain=domain,
    )


# ---------------------------------------------------------------------------
# Session 7: per-domain helpers + reporting
# ---------------------------------------------------------------------------


def _tool_in_domain(tool, domain: str) -> bool:
    """Is this tool relevant to the domain?

    Matches tag overlap and name/description substrings. Defensive to
    tools with minimal metadata (accepts missing fields).
    """
    if domain == GENERAL_DOMAIN:
        return True
    keywords = DOMAINS.get(domain, [])
    tags = [str(t).lower() for t in (getattr(tool, "tags", []) or [])]
    name = (getattr(tool, "name", "") or "").lower()
    desc = (getattr(tool, "description", "") or "").lower()
    for kw in keywords:
        if kw in tags or kw in name or kw in desc:
            return True
        if kw.replace(" ", "-") in tags:
            return True
    return False


def _trace_in_domain(trace: dict, domain: str) -> bool:
    """Is this build-trace dict relevant to the domain?"""
    if domain == GENERAL_DOMAIN:
        return True
    goal = str(trace.get("goal", "") or trace.get("user_goal", "")).lower()
    tags = [str(t).lower() for t in (trace.get("tags") or [])]
    if detect_domain(goal, tags) == domain:
        return True
    # Trace may store domain explicitly
    if str(trace.get("domain", "")).lower() == domain:
        return True
    return False


def compute_all_domains(
    soil,
    tool_registry,
    build_traces: list[dict],
    *,
    include_general: bool = True,
) -> dict[str, ProgressionMetrics]:
    """Compute per-domain progression for every entry in DOMAINS.

    Returns {domain_name: ProgressionMetrics}. When include_general is
    True (default) the cross-domain view is also included under the key
    "general".
    """
    out: dict[str, ProgressionMetrics] = {}
    for domain in DOMAINS:
        out[domain] = compute_progression(
            soil, tool_registry, build_traces, domain=domain
        )
    if include_general:
        out[GENERAL_DOMAIN] = compute_progression(
            soil, tool_registry, build_traces, domain=GENERAL_DOMAIN
        )
    return out


def format_all_domains_report(
    by_domain: dict[str, ProgressionMetrics],
    *,
    bar_width: int = 10,
) -> str:
    """Render the spec's per-domain stage bar chart.

    Each row: "  fastapi:  Stage 2 (Tessellation) ████████░░ 85% coverage"
    """
    stage_names = {
        0: "Seed",
        1: "Cluster",
        2: "Tessellation",
        3: "Basis",
        4: "Connectivity",
        5: "Archetypes",
    }

    lines = ["Generative Chain Stages:", ""]
    # Stable ordering
    ordered = [d for d in DOMAIN_DISPLAY_ORDER if d in by_domain]
    # Any unknown domains tacked on alphabetically
    for extra in sorted(by_domain.keys() - set(ordered)):
        ordered.append(extra)

    for domain in ordered:
        m = by_domain[domain]
        stage_label = stage_names.get(m.current_stage, "?")
        fraction = _progress_fraction(m)
        filled = round(bar_width * max(0.0, min(1.0, fraction)))
        bar = "█" * filled + "░" * (bar_width - filled)
        note = _progress_note(m)
        lines.append(
            f"  {domain:<9} Stage {m.current_stage} ({stage_label:<12}) {bar} {note}"
        )
    return "\n".join(lines)


def _progress_fraction(metrics: ProgressionMetrics) -> float:
    """A 0..1 "how much progress" signal for the bar.

    Uses coverage when we're past stage 1, else fraction-of-cluster-target
    (3 clusters), else fraction-of-seed-target (10 seeds). Degenerate
    empty domains produce 0.0 — no builds yet.
    """
    if metrics.total_tool_count == 0:
        return 0.0
    if metrics.current_stage >= 2:
        return max(metrics.coverage_fraction, 0.5)
    if metrics.current_stage == 1:
        return min(1.0, metrics.cluster_count / 5.0)
    return min(1.0, metrics.total_tool_count / 10.0)


def _progress_note(metrics: ProgressionMetrics) -> str:
    """Short right-side annotation: most informative stat for this stage."""
    if metrics.total_tool_count == 0:
        return "no builds"
    if metrics.current_stage >= 2:
        return f"{metrics.coverage_fraction:.0%} coverage"
    if metrics.current_stage == 1:
        return f"{metrics.cluster_count} clusters"
    return f"{metrics.total_tool_count} tool{'s' if metrics.total_tool_count != 1 else ''}"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_tool_embeddings(soil, tools: list) -> list[list[float]]:
    """Get embedding vectors for tools from ChromaDB."""
    if not tools:
        return []

    col = soil._collections.get("belief_tools")
    if col is None or col.count() == 0:
        return []

    embeddings: list[list[float]] = []
    tool_ids = [t.id for t in tools]

    try:
        result = col.get(
            ids=tool_ids,
            include=["embeddings"],
        )
        if result.get("embeddings"):
            embeddings = [e for e in result["embeddings"] if e is not None]
    except Exception as e:
        logger.debug(f"Tool embedding retrieval skipped: {e}")

    return embeddings


def _compute_clusters(embeddings: list[list[float]]) -> tuple[int, float]:
    """Cluster tool embeddings.  Returns (cluster_count, silhouette_score)."""
    if len(embeddings) < 5:
        return 0, 0.0

    # Try sklearn HDBSCAN first
    try:
        from sklearn.cluster import HDBSCAN
        from sklearn.metrics import silhouette_score
        import numpy as np

        X = np.array(embeddings)
        clusterer = HDBSCAN(min_cluster_size=3)
        labels = clusterer.fit_predict(X)
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)

        if n_clusters > 1:
            sil = float(silhouette_score(X, labels))
        else:
            sil = 0.0

        return n_clusters, sil

    except ImportError:
        pass

    # Fallback: threshold-based grouping on cosine distance
    return _threshold_cluster(embeddings)


def _threshold_cluster(embeddings: list[list[float]]) -> tuple[int, float]:
    """Simple threshold-based clustering when sklearn is unavailable."""
    n = len(embeddings)
    if n < 3:
        return 0, 0.0

    # Compute pairwise cosine distances
    assigned = [-1] * n
    cluster_id = 0
    threshold = 0.3  # Cosine distance threshold

    for i in range(n):
        if assigned[i] >= 0:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            dist = _cosine_distance(embeddings[i], embeddings[j])
            if dist < threshold:
                assigned[j] = cluster_id
        cluster_id += 1

    n_clusters = len(set(assigned))

    # Simple silhouette approximation
    if n_clusters > 1 and n_clusters < n:
        sil = max(0.0, 1.0 - n_clusters / n)
    else:
        sil = 0.0

    return n_clusters, sil


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity


def _compute_coverage(soil, tools: list) -> float:
    """Fraction of benchmark challenges with a relevant tool.

    A challenge is "covered" if at least one tool has cosine similarity > 0.7
    to the challenge description.
    """
    if not tools:
        return 0.0

    try:
        from belief.benchmark import CHALLENGES
        challenge_goals = [c.goal for c in CHALLENGES]
    except Exception:
        return 0.0

    if not challenge_goals:
        return 0.0

    col = soil._collections.get("belief_tools")
    if col is None or col.count() == 0:
        return 0.0

    covered = 0
    for goal in challenge_goals:
        try:
            result = col.query(
                query_texts=[goal],
                n_results=1,
                include=["distances"],
            )
            if (result["distances"] and result["distances"][0] and
                    result["distances"][0][0] < 0.3):  # cosine distance < 0.3 means sim > 0.7
                covered += 1
        except Exception as e:
            logger.debug(f"Coverage query skipped: {e}")

    return covered / len(challenge_goals)


def _compute_rank_ratio(embeddings: list[list[float]]) -> float:
    """SVD rank ratio of tool embeddings — measures diversity.

    Higher ratio = more diverse tools (good).
    """
    if len(embeddings) < 3:
        return 0.0

    try:
        import numpy as np
        X = np.array(embeddings)
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        # Rank ratio: number of significant singular values / total
        threshold = s[0] * 0.1
        rank = int(np.sum(s > threshold))
        return rank / len(s)
    except (ImportError, Exception):
        return 0.0


def _compute_connectivity(tools: list, traces: list[dict]) -> float:
    """Fraction of tool pairs that co-occur in the same build trace."""
    if len(tools) < 2 or not traces:
        return 0.0

    tool_names = {t.name for t in tools}
    cooccurrence_pairs = set()
    total_pairs = 0

    for name1 in tool_names:
        for name2 in tool_names:
            if name1 >= name2:
                continue
            total_pairs += 1

            # Check if both appear in any trace
            for trace in traces:
                trace_text = str(trace.get("code_files", {})) + str(trace.get("user_goal", ""))
                if name1 in trace_text and name2 in trace_text:
                    cooccurrence_pairs.add((name1, name2))
                    break

    return len(cooccurrence_pairs) / max(total_pairs, 1)


def _detect_archetypes(traces: list[dict]) -> tuple[int, float]:
    """Detect recurring build patterns.

    Returns (archetype_count, reuse_fraction).
    """
    if len(traces) < 5:
        return 0, 0.0

    # Group traces by rough pattern (file structure + framework)
    patterns: dict[str, int] = {}
    for trace in traces:
        files = sorted(trace.get("code_files", {}).keys())
        key = ",".join(f.split("/")[-1] for f in files[:5])
        if key:
            patterns[key] = patterns.get(key, 0) + 1

    # Archetypes are patterns that appear 2+ times
    archetypes = {k: v for k, v in patterns.items() if v >= 2}
    archetype_count = len(archetypes)

    # Reuse: fraction of traces that match an archetype
    archetype_traces = sum(archetypes.values())
    reuse = archetype_traces / len(traces) if traces else 0.0

    return archetype_count, reuse


def format_progression_report(metrics: ProgressionMetrics) -> str:
    """Format progression metrics for CLI display."""
    stage_names = {
        0: "Seed",
        1: "Cluster",
        2: "Tessellation",
        3: "Basis",
        4: "Connectivity",
        5: "Archetypes",
    }

    lines = [
        f"Generative Chain Stage: {metrics.current_stage} ({stage_names.get(metrics.current_stage, '?')})",
        "",
        f"  Tools: {metrics.total_tool_count} total ({metrics.seed_tool_count} hand-authored)",
        f"  Clusters: {metrics.cluster_count} (silhouette={metrics.cluster_silhouette:.2f})",
        f"  Coverage: {metrics.coverage_fraction:.0%} of benchmark space",
        f"  Diversity: rank_ratio={metrics.basis_rank_ratio:.2f}",
        f"  Connectivity: {metrics.connectivity_fraction:.0%} tool co-occurrence",
        f"  Archetypes: {metrics.archetype_count} (reuse={metrics.archetype_reuse:.0%})",
        "",
    ]

    # Stage progress indicators
    thresholds = [
        ("Stage 1 (Cluster)", metrics.cluster_count >= 3 and metrics.cluster_silhouette > 0.2),
        ("Stage 2 (Tessellation)", metrics.coverage_fraction > 0.85),
        ("Stage 3 (Basis)", metrics.basis_rank_ratio > 0.7),
        ("Stage 4 (Connectivity)", metrics.connectivity_fraction > 0.4),
        ("Stage 5 (Archetypes)", metrics.archetype_count >= 3 and metrics.archetype_reuse > 0.5),
    ]
    for name, reached in thresholds:
        marker = "+" if reached else "-"
        lines.append(f"  [{marker}] {name}")

    return "\n".join(lines)
