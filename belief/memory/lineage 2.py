"""
Lineage tracking and decay maintenance for the Metabolization Architecture.

The food chain property: nutrients from Tier 1 builds feed Tier 2 builds,
which feed Tier 3. Each tier carries the accumulated essence of every tier
below it. This module provides:

  1. Cross-build correlation: detect when 3+ Tier N nutrients describe the
     same pattern, promoting to a Tier N+1 nutrient with lineage links.

  2. Covenant promotion: detect when 3+ antipatterns share the same root
     cause, and promote to a covenant (immutable rule).

  3. Lineage queries: given a nutrient, trace its ancestry back through
     the tiers to see how knowledge evolved.

  4. Decay maintenance: scheduled soil.decay_all() with logging.

Source: METABOLIZATION_BUILD_PLAN.md Phase 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from belief.memory.nutrients import Nutrient, NutrientTier, NutrientType

logger = logging.getLogger("belief.memory.lineage")

# Similarity threshold for cross-build correlation
_CORRELATION_THRESHOLD = 0.85

# Minimum antipatterns with same signature to promote to covenant
_COVENANT_PROMOTION_COUNT = 3


def trace_lineage(nutrient: Nutrient, soil) -> list[Nutrient]:
    """Trace a nutrient's ancestry through the food chain.

    Given a Tier 3 nutrient, returns [Tier 1 parent, Tier 2 parent, self]
    showing how knowledge evolved across build tiers.

    Returns the chain from oldest ancestor to the nutrient itself.
    """
    chain: list[Nutrient] = []
    visited: set[str] = set()

    def _walk(n: Nutrient) -> None:
        if n.nutrient_id in visited:
            return
        visited.add(n.nutrient_id)

        # Walk parents first (depth-first, oldest ancestor first)
        for parent_id in n.lineage_parent_ids:
            parent = soil.get(parent_id)
            if parent:
                _walk(parent)

        chain.append(n)

    _walk(nutrient)
    return chain


def correlate_and_promote(soil) -> list[Nutrient]:
    """Scan for cross-build patterns and promote to higher tiers.

    Finds clusters of 3+ nutrients at the same tier with >0.85 similarity
    and creates a higher-tier nutrient that subsumes them.

    Also promotes antipattern clusters to covenants.

    Returns list of newly created nutrients.
    """
    promoted: list[Nutrient] = []

    # Process each tier from lowest to highest
    for tier_val in [1, 2, 3, 4]:
        tier = NutrientTier(tier_val)
        next_tier = NutrientTier(tier_val + 1)

        # Get all patterns at this tier
        patterns = soil.retrieve(
            query="",  # Empty query returns all
            n=50,
            nutrient_type=NutrientType.PATTERN,
            min_retrievability=0.2,
        )
        tier_patterns = [p for p in patterns if p.tier == tier]

        if len(tier_patterns) < 3:
            continue

        # Find clusters of similar patterns
        clusters = _find_clusters(tier_patterns, soil)

        for cluster in clusters:
            if len(cluster) < 3:
                continue

            # Create a promoted nutrient that subsumes the cluster
            promoted_nutrient = _promote_cluster(cluster, next_tier)
            if promoted_nutrient:
                nutrient_id = soil.deposit(promoted_nutrient)
                stored = soil.get(nutrient_id)
                if stored:
                    promoted.append(stored)
                    logger.info(
                        f"Lineage: promoted {len(cluster)} Tier {tier_val} patterns → "
                        f"Tier {next_tier.value} ({stored.content[:60]}...)"
                    )

    # Covenant promotion: 3+ antipatterns with same root cause
    covenant_promotions = _promote_antipatterns_to_covenants(soil)
    promoted.extend(covenant_promotions)

    return promoted


def _find_clusters(
    nutrients: list[Nutrient],
    soil,
) -> list[list[Nutrient]]:
    """Find clusters of semantically similar nutrients.

    Uses a simple greedy clustering: pick a seed, find all neighbors
    above threshold, form a cluster, remove from pool, repeat.
    """
    clusters: list[list[Nutrient]] = []
    remaining = list(nutrients)

    while len(remaining) >= 3:
        seed = remaining[0]
        cluster = [seed]
        rest = []

        for candidate in remaining[1:]:
            # Use soil's embedding to check similarity
            results = soil._collection.query(
                query_texts=[seed.embedding_text],
                n_results=len(remaining),
                include=["distances"],
            )

            # Find this candidate's distance from seed
            sim = _get_similarity(candidate.nutrient_id, results)
            if sim is not None and sim >= _CORRELATION_THRESHOLD:
                cluster.append(candidate)
            else:
                rest.append(candidate)

        if len(cluster) >= 3:
            clusters.append(cluster)
        remaining = rest

    return clusters


def _get_similarity(
    nutrient_id: str,
    query_results: dict,
) -> Optional[float]:
    """Extract similarity score for a specific nutrient from query results."""
    if not query_results["ids"] or not query_results["ids"][0]:
        return None

    for i, doc_id in enumerate(query_results["ids"][0]):
        if doc_id == nutrient_id:
            distance = query_results["distances"][0][i]
            return 1.0 - distance  # cosine distance → similarity

    return None


def _promote_cluster(
    cluster: list[Nutrient],
    target_tier: NutrientTier,
) -> Optional[Nutrient]:
    """Create a promoted nutrient from a cluster of similar lower-tier nutrients.

    The promoted nutrient:
    - Combines the essence of all cluster members
    - Has lineage_parent_ids pointing to all members
    - Starts at the target tier
    - Inherits the highest stability from the cluster
    """
    if not cluster:
        return None

    # Synthesize content from the cluster
    contents = [n.content for n in cluster]
    # Use the most-reinforced member as the base content
    best = max(cluster, key=lambda n: n.reinforcement_count)

    # Collect all unique tags
    all_tags = list(set(t for n in cluster for t in n.tags))

    # Framework: use the most common one
    frameworks = [n.framework for n in cluster if n.framework]
    framework = max(set(frameworks), key=frameworks.count) if frameworks else None

    return Nutrient(
        nutrient_type=NutrientType.PATTERN,
        tier=target_tier,
        content=best.content,
        embedding_text=best.embedding_text,
        code_sample=best.code_sample,
        stability=max(n.stability for n in cluster),
        difficulty=sum(n.difficulty for n in cluster) / len(cluster),
        source_build_id=best.source_build_id,
        lineage_parent_ids=[n.nutrient_id for n in cluster],
        tags=all_tags[:10],
        framework=framework,
    )


def _promote_antipatterns_to_covenants(soil) -> list[Nutrient]:
    """Find antipattern clusters and promote to covenants.

    When 3+ antipatterns describe the same root cause, create a covenant
    (immutable rule) that prevents the failure from recurring.
    """
    promoted: list[Nutrient] = []

    antipatterns = soil.retrieve(
        query="",
        n=50,
        nutrient_type=NutrientType.ANTIPATTERN,
        min_retrievability=0.2,
    )

    if len(antipatterns) < _COVENANT_PROMOTION_COUNT:
        return promoted

    clusters = _find_clusters(antipatterns, soil)

    for cluster in clusters:
        if len(cluster) < _COVENANT_PROMOTION_COUNT:
            continue

        # Check if a covenant for this already exists
        existing = soil.retrieve(
            query=cluster[0].embedding_text,
            n=1,
            nutrient_type=NutrientType.COVENANT,
            min_retrievability=0.1,
        )
        if existing:
            # Reinforce existing covenant instead
            soil.reinforce(existing[0].nutrient_id)
            continue

        # Promote to covenant
        best = max(cluster, key=lambda n: n.reinforcement_count)
        covenant = Nutrient(
            nutrient_type=NutrientType.COVENANT,
            tier=max(n.tier for n in cluster),
            content=f"NEVER: {best.content}",
            embedding_text=best.embedding_text,
            stability=10.0,  # Covenants start with high stability
            difficulty=8.0,  # Covenants are complex rules
            source_build_id=best.source_build_id,
            lineage_parent_ids=[n.nutrient_id for n in cluster],
            tags=list(set(t for n in cluster for t in n.tags))[:10],
            framework=best.framework,
        )

        nutrient_id = soil.deposit(covenant)
        stored = soil.get(nutrient_id)
        if stored:
            promoted.append(stored)
            logger.info(
                f"Lineage: promoted {len(cluster)} antipatterns → COVENANT: "
                f"{stored.content[:80]}"
            )

    return promoted


def run_maintenance(soil) -> dict[str, Any]:
    """Run full soil maintenance: decay + correlation + promotion.

    Intended to run on CLI startup or as a periodic job.

    Returns summary dict with maintenance results.
    """
    from typing import Any

    summary: dict[str, Any] = {}

    # Phase 1: Decay stale nutrients
    decay_result = soil.decay_all()
    summary["decay"] = decay_result

    # Phase 2: Cross-build correlation and promotion
    promoted = correlate_and_promote(soil)
    summary["promoted"] = len(promoted)
    summary["promoted_details"] = [
        {"id": n.nutrient_id, "type": n.nutrient_type.value, "tier": n.tier.value}
        for n in promoted
    ]

    # Phase 3: Stats
    summary["active_count"] = soil.count()
    summary["archived_count"] = soil.count_archived()
    summary["by_type"] = soil.count_by_type()

    logger.info(
        f"Soil maintenance complete: "
        f"{decay_result['archived']} archived, {len(promoted)} promoted, "
        f"{soil.count()} active nutrients"
    )

    return summary
