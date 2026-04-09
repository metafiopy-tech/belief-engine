"""Procedural Benchmark Generator — BeTaL-Inspired Challenge Calibration.

Generates parameterized code generation challenges at a target difficulty
level (~50% for maximum information content per evaluation). Each challenge
is deterministically reproducible from its seed but unique.

Research basis:
- BeTaL (arXiv 2510.25039): LLM-driven benchmark calibration, 5.3-13.2% deviation
- Procgen (OpenAI, arXiv 1912.01588): procedural environment generation
- Item Response Theory: 50% difficulty = maximum information

Controllable dimensions:
- Number of models (1-20)
- API endpoints (2-50)
- Business rules (0-20)
- Authentication level (none → OAuth2)
- Data relationship depth (flat → polymorphic)
- Error handling requirements

Usage:
    from belief.benchmark_generator import generate_challenges
    challenges = generate_challenges(n=10, target_difficulty=0.5, seed=42)
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field

logger = logging.getLogger("belief.benchmark_generator")


@dataclass
class GeneratedChallenge:
    """A procedurally generated benchmark challenge."""
    id: str
    tier: int
    goal: str
    acceptance_criteria: list[str]
    seed: int
    difficulty_estimate: float
    parameters: dict = field(default_factory=dict)


# ── Parameter Templates ──────────────────────────────────────────────────────

_DOMAINS = [
    "library", "restaurant", "hospital", "university", "warehouse",
    "theater", "gym", "hotel", "airline", "bank",
    "pharmacy", "garage", "farm", "museum", "studio",
]

_ENTITIES = {
    "library": ["books", "members", "loans", "authors", "categories"],
    "restaurant": ["menus", "orders", "tables", "reservations", "reviews"],
    "hospital": ["patients", "doctors", "appointments", "departments", "prescriptions"],
    "university": ["students", "courses", "enrollments", "professors", "grades"],
    "warehouse": ["products", "shipments", "suppliers", "locations", "inventory"],
    "theater": ["shows", "tickets", "seats", "venues", "performers"],
    "gym": ["members", "classes", "trainers", "equipment", "bookings"],
    "hotel": ["rooms", "guests", "bookings", "services", "invoices"],
    "airline": ["flights", "passengers", "bookings", "airports", "crew"],
    "bank": ["accounts", "transactions", "customers", "loans", "cards"],
    "pharmacy": ["medicines", "prescriptions", "customers", "suppliers", "orders"],
    "garage": ["vehicles", "services", "mechanics", "appointments", "parts"],
    "farm": ["crops", "animals", "fields", "harvests", "workers"],
    "museum": ["exhibits", "visitors", "tickets", "galleries", "curators"],
    "studio": ["projects", "artists", "clients", "assets", "schedules"],
}

_FEATURES = [
    "full-text search",
    "pagination with cursor",
    "filtering by multiple fields",
    "sorting by any column",
    "bulk import from CSV",
    "export to JSON",
    "soft delete with restore",
    "audit log of all changes",
    "role-based access control",
    "rate limiting per user",
    "webhook notifications on changes",
    "scheduled reports",
    "data validation with custom rules",
    "file attachments",
    "tagging and categorization",
]

_AUTH_LEVELS = [
    ("none", "No authentication required"),
    ("api_key", "API key authentication via header"),
    ("jwt", "JWT token authentication with login/register"),
]


# ── Generator ────────────────────────────────────────────────────────────────

def generate_challenges(
    n: int = 10,
    target_difficulty: float = 0.5,
    seed: int = 42,
    min_tier: int = 2,
    max_tier: int = 5,
) -> list[GeneratedChallenge]:
    """Generate N unique challenges at approximately target difficulty.

    Difficulty is estimated from parameter complexity:
    - More entities = harder
    - More features = harder
    - Auth = harder
    - Cross-entity relationships = harder

    Args:
        n: Number of challenges to generate
        target_difficulty: Target pass probability (0.5 = maximum information)
        seed: Random seed for reproducibility
        min_tier: Minimum tier
        max_tier: Maximum tier

    Returns:
        List of GeneratedChallenge objects
    """
    rng = random.Random(seed)
    challenges = []

    for i in range(n):
        challenge_seed = seed * 1000 + i
        challenge_rng = random.Random(challenge_seed)

        # Adjust parameters to target difficulty
        params = _sample_parameters(challenge_rng, target_difficulty, min_tier, max_tier)
        challenge = _build_challenge(params, challenge_seed)
        challenges.append(challenge)

    logger.info(
        f"Generated {len(challenges)} challenges "
        f"(target difficulty={target_difficulty:.1%}, "
        f"tiers {min_tier}-{max_tier})"
    )
    return challenges


def _sample_parameters(
    rng: random.Random,
    target_difficulty: float,
    min_tier: int,
    max_tier: int,
) -> dict:
    """Sample challenge parameters targeting a specific difficulty."""
    # Map difficulty to complexity budget
    # 0.3 (easy) → budget 3-5
    # 0.5 (medium) → budget 6-10
    # 0.7 (hard) → budget 11-16
    budget = int(3 + target_difficulty * 20)

    # Distribute budget across dimensions
    domain = rng.choice(_DOMAINS)
    available_entities = list(_ENTITIES[domain])

    # Number of entities (1-5, each costs ~2 budget)
    max_entities = min(len(available_entities), budget // 2)
    n_entities = rng.randint(1, max(1, max_entities))
    entities = rng.sample(available_entities, n_entities)
    budget -= n_entities * 2

    # Number of extra features (0-5, each costs ~2 budget)
    n_features = min(budget // 2, rng.randint(0, 3))
    features = rng.sample(_FEATURES, min(n_features, len(_FEATURES)))
    budget -= n_features * 2

    # Auth level (costs 0-3 budget)
    auth_idx = 0
    if budget >= 3 and rng.random() < target_difficulty:
        auth_idx = rng.randint(1, len(_AUTH_LEVELS) - 1)
        budget -= auth_idx * 1.5

    # Cross-entity relationships
    relationships = []
    if n_entities >= 2 and budget >= 2:
        n_rels = rng.randint(1, min(n_entities - 1, 3))
        for _ in range(n_rels):
            e1, e2 = rng.sample(entities, 2)
            relationships.append((e1, e2))

    # Estimate tier from total complexity
    complexity = n_entities * 2 + n_features * 2 + auth_idx * 3 + len(relationships) * 2
    if complexity <= 4:
        tier = max(min_tier, 2)
    elif complexity <= 8:
        tier = max(min_tier, 3)
    elif complexity <= 14:
        tier = min(max_tier, 4)
    else:
        tier = min(max_tier, 5)

    return {
        "domain": domain,
        "entities": entities,
        "features": features,
        "auth": _AUTH_LEVELS[auth_idx],
        "relationships": relationships,
        "tier": tier,
        "complexity": complexity,
    }


def _build_challenge(params: dict, seed: int) -> GeneratedChallenge:
    """Build a Challenge from sampled parameters."""
    domain = params["domain"]
    entities = params["entities"]
    features = params["features"]
    auth_name, auth_desc = params["auth"]
    relationships = params["relationships"]
    tier = params["tier"]
    complexity = params["complexity"]

    # Build goal
    entity_desc = ", ".join(entities)
    goal_parts = [
        f"Build a {domain} management API with FastAPI",
        f"— CRUD for {entity_desc}",
    ]

    if relationships:
        rel_desc = ", ".join(f"{e1} linked to {e2}" for e1, e2 in relationships)
        goal_parts.append(f"with relationships: {rel_desc}")

    if features:
        feat_desc = ", ".join(features[:3])
        goal_parts.append(f"Additional features: {feat_desc}")

    if auth_name != "none":
        goal_parts.append(f"{auth_desc}")

    goal_parts.append("SQLite storage.")
    goal = ". ".join(goal_parts)

    # Build acceptance criteria
    criteria = []
    for entity in entities:
        criteria.append(f"CRUD for {entity} works")
    for rel in relationships:
        criteria.append(f"{rel[0]} and {rel[1]} are properly linked")
    for feat in features[:3]:
        criteria.append(f"{feat} is implemented")
    if auth_name != "none":
        criteria.append(f"{auth_desc} works")

    # Generate deterministic ID
    id_hash = hashlib.md5(f"{seed}:{goal[:50]}".encode()).hexdigest()[:8]
    challenge_id = f"t{tier}-gen-{domain}-{id_hash}"

    # Estimate difficulty (0-1 scale)
    difficulty_estimate = min(1.0, complexity / 20.0)

    return GeneratedChallenge(
        id=challenge_id,
        tier=tier,
        goal=goal,
        acceptance_criteria=criteria,
        seed=seed,
        difficulty_estimate=difficulty_estimate,
        parameters=params,
    )


def challenges_to_benchmark_format(
    challenges: list[GeneratedChallenge],
) -> list[dict]:
    """Convert GeneratedChallenges to the benchmark.py Challenge format."""
    from belief.benchmark import Challenge

    return [
        Challenge(
            id=c.id,
            tier=c.tier,
            goal=c.goal,
            acceptance_criteria=c.acceptance_criteria,
            verify_commands=["curl localhost:8000/docs"],
            timeout_seconds=600 + c.tier * 200,
            tags=["generated", c.parameters.get("domain", "")],
        )
        for c in challenges
    ]
