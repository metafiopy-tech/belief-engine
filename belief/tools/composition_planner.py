"""
Composition Pattern — Milestone 5

Instead of generating everything from scratch, the Research agent
searches for existing libraries that solve each requirement.

Two components:
1. PackageEvaluator: Scores packages on quality metrics
2. CompositionPlanner: Decides "use library X" vs "generate" per component

Based on:
- Libraries.io API (9.96M packages, SourceRank scoring)
- IndieStack MCP (curated tool discovery)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Package evaluation
# ---------------------------------------------------------------------------

class PackageSource(str, Enum):
    PYPI = "pypi"
    NPM = "npm"
    GITHUB = "github"


@dataclass
class PackageCandidate:
    """A candidate package found during research."""
    name: str
    source: PackageSource = PackageSource.PYPI
    version: Optional[str] = None
    description: str = ""
    downloads_monthly: int = 0
    stars: int = 0
    last_updated: Optional[str] = None  # ISO date
    license: str = ""
    source_rank: float = 0.0  # Libraries.io SourceRank (0-30+)
    maintained: bool = True
    homepage: Optional[str] = None

    @property
    def quality_score(self) -> float:
        """
        Composite quality score (0-100).

        Factors:
        - Downloads (logarithmic, max 30 points)
        - Source rank (max 30 points)
        - Stars (logarithmic, max 20 points)
        - Maintained (20 points)
        """
        import math
        dl_score = min(30, math.log10(max(self.downloads_monthly, 1)) * 5)
        rank_score = min(30, self.source_rank)
        star_score = min(20, math.log10(max(self.stars, 1)) * 5)
        maint_score = 20 if self.maintained else 0
        return dl_score + rank_score + star_score + maint_score


# ---------------------------------------------------------------------------
# Well-known package registry (offline fallback)
# ---------------------------------------------------------------------------

WELL_KNOWN_PACKAGES: dict[str, PackageCandidate] = {
    "fastapi": PackageCandidate(
        name="fastapi", description="Modern web framework for APIs",
        downloads_monthly=20_000_000, stars=70000, source_rank=28,
        license="MIT", maintained=True,
    ),
    "httpx": PackageCandidate(
        name="httpx", description="Async HTTP client",
        downloads_monthly=15_000_000, stars=12000, source_rank=25,
        license="BSD-3", maintained=True,
    ),
    "pydantic": PackageCandidate(
        name="pydantic", description="Data validation using Python type annotations",
        downloads_monthly=50_000_000, stars=18000, source_rank=30,
        license="MIT", maintained=True,
    ),
    "fastmcp": PackageCandidate(
        name="fastmcp", description="Fast MCP server framework",
        downloads_monthly=100_000, stars=2000, source_rank=15,
        license="MIT", maintained=True,
    ),
    "uvicorn": PackageCandidate(
        name="uvicorn", description="ASGI server",
        downloads_monthly=20_000_000, stars=7000, source_rank=26,
        license="BSD-3", maintained=True,
    ),
    "sqlalchemy": PackageCandidate(
        name="sqlalchemy", description="SQL toolkit and ORM",
        downloads_monthly=30_000_000, stars=8000, source_rank=29,
        license="MIT", maintained=True,
    ),
    "celery": PackageCandidate(
        name="celery", description="Distributed task queue",
        downloads_monthly=10_000_000, stars=23000, source_rank=27,
        license="BSD-3", maintained=True,
    ),
    "beautifulsoup4": PackageCandidate(
        name="beautifulsoup4", description="HTML/XML parser",
        downloads_monthly=25_000_000, stars=0, source_rank=24,
        license="MIT", maintained=True,
    ),
    "scrapy": PackageCandidate(
        name="scrapy", description="Web scraping framework",
        downloads_monthly=5_000_000, stars=50000, source_rank=26,
        license="BSD-3", maintained=True,
    ),
    "pytest": PackageCandidate(
        name="pytest", description="Testing framework",
        downloads_monthly=50_000_000, stars=11000, source_rank=30,
        license="MIT", maintained=True,
    ),
}


def evaluate_package(name: str, api_result: Optional[dict] = None) -> Optional[PackageCandidate]:
    """
    Evaluate a package by name.

    First checks the well-known registry (offline).
    If api_result is provided (from Libraries.io), uses that data.
    """
    # Check well-known packages first
    if name.lower() in WELL_KNOWN_PACKAGES:
        return WELL_KNOWN_PACKAGES[name.lower()]

    # Parse API result if provided
    if api_result:
        return PackageCandidate(
            name=api_result.get("name", name),
            description=api_result.get("description", ""),
            downloads_monthly=api_result.get("downloads", 0),
            stars=api_result.get("stars", 0),
            source_rank=api_result.get("rank", 0),
            license=api_result.get("licenses", ""),
            last_updated=api_result.get("latest_release_published_at"),
            maintained=api_result.get("status") != "Deprecated",
        )

    return None


# ---------------------------------------------------------------------------
# Composition decision
# ---------------------------------------------------------------------------

class ComponentStrategy(str, Enum):
    USE_LIBRARY = "use_library"      # Use existing package
    GENERATE = "generate"            # Generate from scratch
    WRAP_LIBRARY = "wrap_library"    # Use library + generate wrapper


@dataclass
class ComponentDecision:
    """Decision for a single component: use library or generate."""
    component_name: str
    strategy: ComponentStrategy
    package: Optional[PackageCandidate] = None
    reason: str = ""
    wrapper_notes: str = ""  # How to wrap the library if WRAP_LIBRARY


# Quality threshold for "use library" decision
QUALITY_THRESHOLD = 40.0  # Score out of 100


def decide_component_strategy(
    component_name: str,
    component_description: str,
    candidate_packages: list[PackageCandidate],
    threshold: float = QUALITY_THRESHOLD,
) -> ComponentDecision:
    """
    Decide whether to use a library or generate from scratch.

    Rules:
    1. If a candidate scores above threshold → USE_LIBRARY
    2. If a candidate scores above threshold/2 → WRAP_LIBRARY (use + generate adapter)
    3. Otherwise → GENERATE from scratch
    """
    if not candidate_packages:
        return ComponentDecision(
            component_name=component_name,
            strategy=ComponentStrategy.GENERATE,
            reason="No candidate packages found",
        )

    # Sort by quality score
    best = max(candidate_packages, key=lambda p: p.quality_score)

    if best.quality_score >= threshold:
        return ComponentDecision(
            component_name=component_name,
            strategy=ComponentStrategy.USE_LIBRARY,
            package=best,
            reason=f"{best.name} scores {best.quality_score:.0f}/100 (above threshold {threshold})",
        )
    elif best.quality_score >= threshold / 2:
        return ComponentDecision(
            component_name=component_name,
            strategy=ComponentStrategy.WRAP_LIBRARY,
            package=best,
            reason=f"{best.name} scores {best.quality_score:.0f}/100 — usable with adapter",
            wrapper_notes=f"Import {best.name} and wrap with project-specific interface",
        )
    else:
        return ComponentDecision(
            component_name=component_name,
            strategy=ComponentStrategy.GENERATE,
            reason=f"Best candidate {best.name} scores {best.quality_score:.0f}/100 (below threshold)",
        )


# ---------------------------------------------------------------------------
# Composition Planner
# ---------------------------------------------------------------------------

@dataclass
class CompositionPlan:
    """Complete plan of what to build vs what to reuse."""
    decisions: list[ComponentDecision] = field(default_factory=list)

    @property
    def libraries_to_install(self) -> list[str]:
        return [
            d.package.name for d in self.decisions
            if d.strategy in (ComponentStrategy.USE_LIBRARY, ComponentStrategy.WRAP_LIBRARY)
            and d.package
        ]

    @property
    def components_to_generate(self) -> list[str]:
        return [
            d.component_name for d in self.decisions
            if d.strategy in (ComponentStrategy.GENERATE, ComponentStrategy.WRAP_LIBRARY)
        ]

    def summary(self) -> str:
        use = [d for d in self.decisions if d.strategy == ComponentStrategy.USE_LIBRARY]
        wrap = [d for d in self.decisions if d.strategy == ComponentStrategy.WRAP_LIBRARY]
        gen = [d for d in self.decisions if d.strategy == ComponentStrategy.GENERATE]
        return (
            f"Composition Plan: {len(use)} use-library, {len(wrap)} wrap-library, {len(gen)} generate\n"
            + "\n".join(f"  {d.component_name}: {d.strategy.value} — {d.reason}" for d in self.decisions)
        )


def plan_composition(
    requirements: list[tuple[str, str]],  # [(component_name, description)]
) -> CompositionPlan:
    """
    Create a composition plan for a set of requirements.

    For each requirement, searches well-known packages and decides strategy.

    Args:
        requirements: List of (component_name, description) tuples.

    Returns:
        CompositionPlan with decisions for each component.
    """
    plan = CompositionPlan()

    for name, description in requirements:
        # Search for candidates
        candidates = _search_candidates(name, description)

        decision = decide_component_strategy(
            component_name=name,
            component_description=description,
            candidate_packages=candidates,
        )
        plan.decisions.append(decision)

    logger.info(plan.summary())
    return plan


def _search_candidates(component_name: str, description: str) -> list[PackageCandidate]:
    """
    Search for candidate packages matching a component.

    Uses keyword matching against well-known packages.
    In production, this would call Libraries.io API.
    """
    keywords = set(component_name.lower().split("_") + description.lower().split())
    # Remove noise words
    keywords -= {"a", "an", "the", "for", "with", "and", "or", "to", "from", "in", "of"}

    candidates = []
    for pkg_name, pkg in WELL_KNOWN_PACKAGES.items():
        pkg_keywords = set(pkg_name.lower().split() + pkg.description.lower().split())
        overlap = keywords & pkg_keywords
        if overlap:
            candidates.append(pkg)

    return candidates
