"""
Nutrient models for the Metabolization Architecture.

A nutrient is an atomic, verified, reusable unit of knowledge extracted
from a build result. The food chain: every build decomposes its results
into nutrients, stores them in ChromaDB soil, and future builds
recompose relevant nutrients into enriched architect context.

Four nutrient types:
  PATTERN     — what worked (verified success)
  ANTIPATTERN — what failed and why (verified failure)
  SKELETON    — a SkeletonArtifact that produced clean imports
  COVENANT    — an immutable rule learned from repeated failures

Confidence uses FSRS (Free Spaced Repetition Scheduler) — no stored
confidence field. retrievability() is computed from stability, difficulty,
and time since last reinforcement. Single source of truth.

Source: METABOLIZATION_BUILD_PLAN.md Phase 1
Research: Voyager skill library, Reflexion verbal RL, FSRS algorithm
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class NutrientType(str, Enum):
    PATTERN = "pattern"
    ANTIPATTERN = "antipattern"
    SKELETON = "skeleton"
    COVENANT = "covenant"


class NutrientTier(int, Enum):
    """Which build complexity tier produced this nutrient."""
    TIER_1 = 1  # Single file scripts
    TIER_2 = 2  # MCP servers, simple APIs
    TIER_3 = 3  # Multi-file packages
    TIER_4 = 4  # Multi-service architectures
    TIER_5 = 5  # Distributed systems


# FSRS constants (from open-spaced-repetition/fsrs4anki)
_FSRS_DECAY = -0.5
_FSRS_FACTOR = 0.9 ** (-1.0 / _FSRS_DECAY) - 1  # ≈ 19/81

# Tunable FSRS weight parameters — reasonable defaults, can be tuned
# after accumulating build data
_W8 = 1.5    # Base stability growth rate
_W9 = 0.2    # Diminishing returns on high stability
_W10 = 0.5   # Spacing effect strength (bigger = more reward for spaced reuse)
_W11 = 0.5   # Lapse stability floor multiplier
_W12 = 0.2   # Lapse difficulty scaling
_W13 = 0.2   # Lapse stability recovery


def _now_ts() -> float:
    """Current UTC timestamp in seconds."""
    return datetime.now(timezone.utc).timestamp()


def _make_id() -> str:
    """Generate a nutrient ID."""
    return f"n-{uuid.uuid4().hex[:12]}"


class Nutrient(BaseModel):
    """A single atomic unit of knowledge in the soil.

    Confidence is NOT stored — it's computed via retrievability() from
    FSRS parameters (stability, difficulty, last_reinforced). This
    ensures a single source of truth for nutrient relevance scoring.
    """

    # Identity
    nutrient_id: str = Field(default_factory=_make_id)
    nutrient_type: NutrientType
    tier: NutrientTier = NutrientTier.TIER_1

    # Content
    content: str                               # The knowledge itself
    embedding_text: str                        # What gets embedded (NL description)
    code_sample: Optional[str] = None          # Representative code if applicable

    # FSRS parameters (no stored confidence — use retrievability())
    stability: float = 1.0                     # Days until 90% retrievability
    difficulty: float = 5.0                    # 1-10 scale (simple idioms=2, arch patterns=8)
    reinforcement_count: int = 0
    lapse_count: int = 0                       # Times found incorrect

    # Timestamps (stored as UTC floats for FSRS math)
    created_at: float = Field(default_factory=_now_ts)
    last_reinforced: float = Field(default_factory=_now_ts)

    @field_validator("created_at", "last_reinforced", mode="before")
    @classmethod
    def _parse_ts(cls, v):
        """Accept float | int | ISO-8601 string for stored timestamps.

        Legacy nutrients in ChromaDB may have been written with ISO-8601
        strings instead of UTC floats; coerce those on read so FSRS math
        keeps working. Unparseable strings fall back to 0.0 rather than
        raising (a bad timestamp shouldn't poison the whole recomposer).
        """
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v).timestamp()
            except ValueError:
                return 0.0
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    # Provenance
    source_build_id: str = ""
    lineage_parent_ids: list[str] = Field(default_factory=list)

    # Categorization
    tags: list[str] = Field(default_factory=list)
    framework: Optional[str] = None            # e.g., "fastapi", "fastmcp"

    def retrievability(self) -> float:
        """Calculate current retrievability using FSRS power-law decay.

        Returns a float 0.0-1.0 representing the probability that this
        nutrient is still valid/useful. Equals 0.9 when elapsed time
        equals stability (the defining property of FSRS stability).

        R(t, S) = (1 + factor * t/S) ^ DECAY
        """
        elapsed_days = max(0.0, (_now_ts() - self.last_reinforced) / 86400.0)
        if self.stability <= 0:
            return 0.0
        inner = 1.0 + _FSRS_FACTOR * elapsed_days / self.stability
        if inner <= 0:
            # Extreme elapsed time — nutrient is effectively forgotten
            return 0.0
        return inner ** _FSRS_DECAY

    def reinforce(self) -> None:
        """Called when this nutrient is successfully reused in a build.

        FSRS stability growth formula with three key properties:
        1. Higher existing stability → smaller proportional gain (S^-w9)
        2. Lower retrievability (longer gap) → bigger boost (spacing effect)
        3. Lower difficulty → faster stability growth (11 - D)
        """
        self.reinforcement_count += 1
        r = self.retrievability()

        # Cap R at 0.9 for growth calculation. In FSRS, R=1.0 means
        # "just reviewed" and gives zero growth (no spacing benefit).
        # For code patterns, even immediate reuse should grow stability
        # since it confirms the pattern works across builds.
        r = min(r, 0.9)

        # S_new = S * (e^w8 * (11-D) * S^(-w9) * (e^(w10*(1-R)) - 1) + 1)
        growth = (
            math.exp(_W8)
            * (11.0 - self.difficulty)
            * (self.stability ** -_W9)
            * (math.exp(_W10 * (1.0 - r)) - 1.0)
            + 1.0
        )
        self.stability = self.stability * max(growth, 1.0)  # Never shrink on reinforce
        self.last_reinforced = _now_ts()

    def lapse(self) -> None:
        """Called when this nutrient led to a build failure.

        Stability drops significantly but doesn't reset to zero — a
        pattern that worked 20 times and failed once retains some
        credibility. Uses FSRS lapse formula:
        S_new = w11 * D^(-w12) * ((S+1)^w13 - 1)
        """
        self.lapse_count += 1
        new_s = (
            _W11
            * (self.difficulty ** -_W12)
            * ((self.stability + 1.0) ** _W13 - 1.0)
        )
        self.stability = max(0.5, new_s)  # Floor at 0.5 days

    def to_chromadb_metadata(self) -> dict:
        """Convert to flat dict for ChromaDB metadata storage.

        ChromaDB metadata supports: str, int, float, bool, list[str].
        No nested objects. Timestamps as floats for range queries.
        The document field stores embedding_text (used for similarity search).
        The original content is stored in metadata for retrieval.
        """
        meta = {
            "nutrient_type": self.nutrient_type.value,
            "tier": self.tier.value,
            "content": self.content,  # Original content (distinct from embedding_text)
            "stability": self.stability,
            "difficulty": self.difficulty,
            "reinforcement_count": self.reinforcement_count,
            "lapse_count": self.lapse_count,
            "created_at": self.created_at,
            "last_reinforced": self.last_reinforced,
            "source_build_id": self.source_build_id,
            "framework": self.framework or "",
            # ChromaDB rejects empty lists — use sentinel for empty
            "tags": self.tags if self.tags else ["_none"],
            "lineage_parent_ids": self.lineage_parent_ids if self.lineage_parent_ids else ["_none"],
        }
        if self.code_sample:
            meta["code_sample"] = self.code_sample
        return meta

    @classmethod
    def from_chromadb(cls, doc_id: str, document: str, metadata: dict) -> Nutrient:
        """Reconstruct a Nutrient from ChromaDB query result.

        document = embedding_text (what was embedded for similarity search)
        metadata["content"] = original content (what the nutrient says)
        """
        return cls(
            nutrient_id=doc_id,
            nutrient_type=NutrientType(metadata.get("nutrient_type", "pattern")),
            tier=NutrientTier(metadata.get("tier", 1)),
            content=metadata.get("content", document),  # Prefer stored content
            embedding_text=document,
            code_sample=metadata.get("code_sample"),
            stability=metadata.get("stability", 1.0),
            difficulty=metadata.get("difficulty", 5.0),
            reinforcement_count=metadata.get("reinforcement_count", 0),
            lapse_count=metadata.get("lapse_count", 0),
            created_at=metadata.get("created_at", _now_ts()),
            last_reinforced=metadata.get("last_reinforced", _now_ts()),
            source_build_id=metadata.get("source_build_id", ""),
            framework=metadata.get("framework", "") or None,
            tags=[t for t in metadata.get("tags", []) if t != "_none"],
            lineage_parent_ids=[p for p in metadata.get("lineage_parent_ids", []) if p != "_none"],
        )


# Token budget per complexity tier (review correction #3)
_TOKEN_BUDGET_BY_TIER = {
    1: 2000,
    2: 2000,
    3: 4000,
    4: 8000,
    5: 12000,
}


class NutrientProfile(BaseModel):
    """Retrieved nutrients organized for injection into the architect.

    Priority order for context block:
    1. Covenants (MUST follow — always injected)
    2. Antipatterns (mistakes to avoid)
    3. Patterns (proven approaches)
    4. Skeletons (reusable templates)

    When soil is empty, format_context_block() returns "" — no empty
    INSTITUTIONAL MEMORY header (review correction #5).
    """
    covenants: list[Nutrient] = Field(default_factory=list)
    antipatterns: list[Nutrient] = Field(default_factory=list)
    patterns: list[Nutrient] = Field(default_factory=list)
    skeletons: list[Nutrient] = Field(default_factory=list)

    @property
    def total_nutrients(self) -> int:
        return len(self.covenants) + len(self.antipatterns) + len(self.patterns) + len(self.skeletons)

    @property
    def is_empty(self) -> bool:
        return self.total_nutrients == 0

    def token_budget(self, complexity: int = 3) -> int:
        """Return token budget for nutrient context based on project complexity."""
        return _TOKEN_BUDGET_BY_TIER.get(complexity, 4000)

    def format_context_block(self, complexity: int = 3) -> str:
        """Format nutrients into a context block for the architect's prompt.

        Returns empty string if no nutrients — never injects an empty
        INSTITUTIONAL MEMORY section (review correction #5).
        """
        if self.is_empty:
            return ""

        sections: list[str] = []
        sections.append(f"## INSTITUTIONAL MEMORY (from prior builds)\n")

        if self.covenants:
            sections.append("### Covenants (MUST follow these rules):")
            for n in self.covenants:
                sections.append(f"- {n.content}")
            sections.append("")

        if self.antipatterns:
            sections.append("### Avoid These Mistakes:")
            for n in self.antipatterns:
                sections.append(f"- {n.content}")
            sections.append("")

        if self.patterns:
            sections.append("### Proven Patterns:")
            for n in self.patterns:
                sections.append(f"- {n.content}")
            sections.append("")

        if self.skeletons:
            sections.append("### Suggested Skeleton:")
            for n in self.skeletons:
                # Skeletons may have code_sample — include abbreviated version
                if n.code_sample:
                    abbreviated = n.code_sample[:1500]
                    if len(n.code_sample) > 1500:
                        abbreviated += "\n# ... (truncated)"
                    sections.append(f"```\n{abbreviated}\n```")
                else:
                    sections.append(f"- {n.content}")
            sections.append("")

        sections.append("Adapt these insights to the current build. Do not blindly copy — use as guidance.")

        block = "\n".join(sections)

        # Rough token estimate (~4 chars per token) and truncate if over budget
        budget = self.token_budget(complexity)
        estimated_tokens = len(block) // 4
        if estimated_tokens > budget:
            # Truncate from the bottom (skeletons first, then patterns)
            # Covenants and antipatterns are never truncated
            block = self._truncate_to_budget(budget)

        return block

    def _truncate_to_budget(self, budget: int) -> str:
        """Rebuild context block within token budget, dropping lowest-priority content."""
        sections: list[str] = []
        sections.append(f"## INSTITUTIONAL MEMORY (from prior builds)\n")

        if self.covenants:
            sections.append("### Covenants (MUST follow these rules):")
            for n in self.covenants:
                sections.append(f"- {n.content}")
            sections.append("")

        if self.antipatterns:
            sections.append("### Avoid These Mistakes:")
            for n in self.antipatterns:
                sections.append(f"- {n.content}")
            sections.append("")

        block = "\n".join(sections)
        remaining = (budget * 4) - len(block)  # Remaining chars

        if remaining > 200 and self.patterns:
            sections.append("### Proven Patterns:")
            for n in self.patterns:
                line = f"- {n.content}"
                if len(line) < remaining:
                    sections.append(line)
                    remaining -= len(line)
            sections.append("")

        if remaining > 500 and self.skeletons:
            sections.append("### Suggested Skeleton:")
            sections.append("(truncated for token budget)")
            sections.append("")

        sections.append("Adapt these insights to the current build. Do not blindly copy — use as guidance.")
        return "\n".join(sections)
