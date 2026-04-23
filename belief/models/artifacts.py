"""Typed artifacts passed between agents via LangGraph state.

Every agent reads structured input and writes structured output.
No conversational dialogue — agents communicate through these models.

Source: forge/models/artifacts.py (475 lines, production-tested)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# File Manifest (output of architect agent)
# ---------------------------------------------------------------------------


class FileManifest(BaseModel):
    filename: str = Field(description="Relative path, e.g. 'src/api/router.py'")
    purpose: str = Field(description="One-line description: what this file does")
    public_interface: str = Field(default="", description="What this file exports")
    depends_on: list[str] = Field(default_factory=list, description="Files this imports from")
    is_entry_point: bool = Field(default=False)


class FileManifestPlan(BaseModel):
    files: list[FileManifest] = Field(default_factory=list)
    architecture_notes: str = Field(default="")
    entry_point: str = Field(default="main.py")
    is_package_build: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GapSeverity(str, Enum):
    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    COSMETIC = "cosmetic"


class ValidationVerdict(str, Enum):
    PASS = "pass"
    FAIL_FIXABLE = "fail_fixable"
    FAIL_RETHINK = "fail_rethink"
    FAIL_UNFIXABLE = "fail_unfixable"


# ---------------------------------------------------------------------------
# Requirement Spec (output of intake)
# ---------------------------------------------------------------------------


class CredentialRequirement(BaseModel):
    name: str = Field(description="Human-readable name")
    env_var: str = Field(description="Environment variable name")
    instructions: str = Field(default="")
    provided: bool = Field(default=False)


class RequirementSpec(BaseModel):
    goal: str
    goal_refined: str = Field(default="")
    target_type: str = Field(default="python")
    complexity_score: int = Field(default=1, ge=1, le=5)
    acceptance_criteria: list[str] = Field(default_factory=list)
    credentials: list[CredentialRequirement] = Field(default_factory=list)
    tools_needed: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Build Memory
# ---------------------------------------------------------------------------


class BuildReference(BaseModel):
    goal: str
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    file_summaries: dict[str, str] = Field(default_factory=dict)
    quality_scores: dict[str, float] = Field(default_factory=dict)
    output_path: str = Field(default="")
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Research Report (output of research agent)
# ---------------------------------------------------------------------------


class SourceReference(BaseModel):
    url: str
    title: str = ""
    source_type: str = Field(default="web")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="")


class RepoCandidate(BaseModel):
    url: str = Field(default="")
    name: str = Field(default="")
    stars: int = 0
    language: str = ""
    description: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    clone_recommended: bool = False
    reasoning: str = Field(default="")


class ResearchReport(BaseModel):
    query_terms: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    repo_candidates: list[RepoCandidate] = Field(default_factory=list)
    patterns_found: list[str] = Field(default_factory=list)
    recommended_approach: str = Field(default="")
    clone_target: str | None = Field(default=None)
    similar_builds: list[BuildReference] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Implementation Plan (output of planner)
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    order: int
    description: str
    agent_responsible: str = Field(default="builder")
    estimated_complexity: str = Field(default="medium")
    dependencies: list[int] = Field(default_factory=list)


class ImplementationPlan(BaseModel):
    strategy: str = Field(default="generate_fresh")
    steps: list[PlanStep] = Field(default_factory=list)
    estimated_iterations: int = Field(default=1)
    risk_factors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Gap Report (output of gap analyst)
# ---------------------------------------------------------------------------


class Gap(BaseModel):
    description: str
    severity: GapSeverity = GapSeverity.MINOR
    category: str = Field(default="functionality")
    suggested_fix: str = Field(default="")
    requires_new_research: bool = Field(default=False)


class GapReport(BaseModel):
    gaps: list[Gap] = Field(default_factory=list)
    total_blockers: int = Field(default=0)
    total_major: int = Field(default=0)
    requires_research: bool = Field(default=False)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Execution Result (output of executor)
# ---------------------------------------------------------------------------


class PytestTestItem(BaseModel):
    node_id: str
    outcome: str
    longrepr: str = Field(default="")


class PytestResult(BaseModel):
    ran: bool = Field(default=False)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    errors: int = Field(default=0)
    total: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    items: list[PytestTestItem] = Field(default_factory=list)
    raw_output: str = Field(default="")

    @property
    def summary_line(self) -> str:
        if not self.ran:
            return "pytest not run"
        parts = []
        if self.passed:
            parts.append(f"{self.passed} passed")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} error(s)")
        return ", ".join(parts) or "no tests collected"

    @property
    def all_passed(self) -> bool:
        return self.ran and self.failed == 0 and self.errors == 0 and self.total > 0


class ExecutionResult(BaseModel):
    exit_code: int = Field(default=-1)
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    duration_seconds: float = Field(default=0.0)
    success: bool = Field(default=False)
    error_summary: str = Field(default="")
    install_success: bool = Field(default=True)
    install_stdout: str = Field(default="")
    install_stderr: str = Field(default="")
    install_duration_seconds: float = Field(default=0.0)
    pytest_result: PytestResult | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Token / Cost Tracking
# ---------------------------------------------------------------------------

ANTHROPIC_COST_PER_1K: dict[str, dict[str, float]] = {
    # Prices per 1,000 tokens (as of 2026)
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    #
    # Model           | Input/1K  | Output/1K | Cache Write/1K | Cache Read/1K
    # Sonnet 4        | $0.003    | $0.015    | $0.00375       | $0.0003
    # Haiku 3.5/4.5   | $0.0008   | $0.004    | $0.001         | $0.00008
    # Opus 4          | $0.015    | $0.075    | $0.01875       | $0.0015
    "claude-sonnet-4-6": {
        "input": 0.003,
        "output": 0.015,
        "cache_write": 0.00375,
        "cache_read": 0.0003,
    },
    "claude-sonnet-4-20250514": {
        "input": 0.003,
        "output": 0.015,
        "cache_write": 0.00375,
        "cache_read": 0.0003,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.0008,
        "output": 0.004,
        "cache_write": 0.001,
        "cache_read": 0.00008,
    },
    "claude-haiku-4-5": {
        "input": 0.0008,
        "output": 0.004,
        "cache_write": 0.001,
        "cache_read": 0.00008,
    },
    "claude-opus-4-6": {
        "input": 0.015,
        "output": 0.075,
        "cache_write": 0.01875,
        "cache_read": 0.0015,
    },
}

_DEFAULT_COST = {
    "input": 0.003,
    "output": 0.015,
    "cache_write": 0.00375,
    "cache_read": 0.0003,
}


def _cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_create_tokens: int = 0,
) -> float:
    """Calculate cost from token counts and model-specific pricing.

    The Anthropic API does NOT return cost — only token counts.
    We calculate cost client-side using the pricing table above.

    Cache pricing varies by model tier:
    - Cache read: 10% of input price (90% savings)
    - Cache write: 125% of input price (25% premium on first write)
    """
    rates = ANTHROPIC_COST_PER_1K.get(model, _DEFAULT_COST)
    input_rate = rates["input"]
    output_rate = rates["output"]
    cache_read_rate = rates.get("cache_read", input_rate * 0.1)
    cache_write_rate = rates.get("cache_write", input_rate * 1.25)

    # Standard input = total input minus cached tokens
    standard_input = max(0, prompt_tokens - cache_read_tokens - cache_create_tokens)

    cost = (
        standard_input * input_rate
        + cache_read_tokens * cache_read_rate
        + cache_create_tokens * cache_write_rate
        + completion_tokens * output_rate
    ) / 1000.0

    return round(cost, 6)


class RoleUsage(BaseModel):
    role: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class TokenUsage(BaseModel):
    backend: str = "unknown"
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    by_role: dict[str, RoleUsage] = Field(default_factory=dict)

    def add_call(
        self, role: str, prompt_tokens: int, completion_tokens: int, cost_usd: float = 0.0
    ) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost_usd
        if role not in self.by_role:
            self.by_role[role] = RoleUsage(role=role)
        r = self.by_role[role]
        r.calls += 1
        r.prompt_tokens += prompt_tokens
        r.completion_tokens += completion_tokens
        r.estimated_cost_usd += cost_usd

    def merge(self, other: TokenUsage) -> TokenUsage:
        merged = TokenUsage(
            backend=self.backend or other.backend,
            total_prompt_tokens=self.total_prompt_tokens + other.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens + other.total_completion_tokens,
            total_cost_usd=self.total_cost_usd + other.total_cost_usd,
        )
        for role, usage in self.by_role.items():
            merged.by_role[role] = RoleUsage(**usage.model_dump())
        for role, usage in other.by_role.items():
            if role in merged.by_role:
                existing = merged.by_role[role]
                existing.calls += usage.calls
                existing.prompt_tokens += usage.prompt_tokens
                existing.completion_tokens += usage.completion_tokens
                existing.estimated_cost_usd += usage.estimated_cost_usd
            else:
                merged.by_role[role] = RoleUsage(**usage.model_dump())
        return merged


# ---------------------------------------------------------------------------
# Validation Result (output of validator)
# ---------------------------------------------------------------------------


class TestTier(str, Enum):
    SMOKE = "smoke"  # P0 — must all pass
    FUNCTIONAL = "functional"  # P1 — business logic
    EDGE_CASE = "edge_case"  # P2 — boundary conditions
    ENVIRONMENT = "environment"  # import/dep failures — weight 0


# Tier weights for verdict scoring
TIER_WEIGHTS = {
    TestTier.SMOKE: 3.0,
    TestTier.FUNCTIONAL: 2.0,
    TestTier.EDGE_CASE: 1.0,
    TestTier.ENVIRONMENT: 0.0,
}


class TestCase(BaseModel):
    name: str
    description: str = ""
    passed: bool = False
    error: str = ""
    tier: TestTier = TestTier.FUNCTIONAL


class ValidationResult(BaseModel):
    verdict: ValidationVerdict = ValidationVerdict.FAIL_FIXABLE
    tests: list[TestCase] = Field(default_factory=list)
    tests_passed: int = Field(default=0)
    tests_total: int = Field(default=0)
    weighted_score: float = Field(default=0.0, ge=0.0, le=1.0)
    correctness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    code_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    security_score: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)
