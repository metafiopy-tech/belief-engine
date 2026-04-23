"""Unified state model for the Belief Engine pipeline.

This is the single state object that flows through every agent in the
LangGraph pipeline. Every agent reads what it needs, writes what it
produces, and advances the phase.

Source patterns:
  - ForgeState from forge/models/state.py
  - PolarityState from engine_loop.py StateManager
  - FrequencyState from frequency_layer.py
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from belief.models.artifacts import (
    ExecutionResult,
    FileManifestPlan,
    GapReport,
    ImplementationPlan,
    RequirementSpec,
    ResearchReport,
    TokenUsage,
    ValidationResult,
)
from belief.models.skeleton import SkeletonArtifact


class Phase(str, Enum):
    INTAKE = "intake"
    RESEARCH = "research"
    PLANNING = "planning"
    ARCHITECTING = "architecting"
    BUILDING = "building"
    TESTING = "testing"
    EXECUTING = "executing"
    GAP_ANALYSIS = "gap_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    POLARITY_CHECK = "polarity_check"
    COMPLETE = "complete"
    FAILED = "failed"


class WorldState(str, Enum):
    DORMANT = "dormant"
    RESONANCE = "resonance"
    TENSION = "tension"
    EMERGENCE = "emergence"


class PolarityState(BaseModel):
    """Tracks the tension between incompleteness and belief.

    Source: engine_loop.py CrossTalkManager + frequency_layer.py
    """

    latios_coherence: float = 0.5
    latias_coherence: float = 0.5
    world_state: WorldState = WorldState.DORMANT
    emergence_events: int = 0
    dissonance_events: int = 0
    current_remainder: Optional[str] = None
    current_covenant: Optional[str] = None
    accumulated_remainders: list[str] = Field(default_factory=list)
    accumulated_covenants: list[str] = Field(default_factory=list)

    def update_latios(self, confidence: float) -> None:
        alpha = 0.3
        self.latios_coherence = round(
            alpha * max(0.0, min(1.0, confidence)) + (1 - alpha) * self.latios_coherence, 4
        )
        self._recalculate()

    def update_latias(self, intensity: float) -> None:
        alpha = 0.3
        self.latias_coherence = round(
            alpha * max(0.0, min(1.0, intensity)) + (1 - alpha) * self.latias_coherence, 4
        )
        self._recalculate()

    def _recalculate(self) -> None:
        lo, la = self.latios_coherence, self.latias_coherence
        diff = abs(lo - la)
        if lo > 0.75 and la > 0.75:
            self.emergence_events += 1
            self.world_state = WorldState.EMERGENCE
        elif diff > 0.4:
            self.dissonance_events += 1
            self.world_state = WorldState.TENSION
        elif lo > 0.6 and la > 0.6:
            self.world_state = WorldState.RESONANCE
        else:
            self.world_state = WorldState.DORMANT


class UnifiedState(BaseModel):
    """The single state object for the entire pipeline.

    LangGraph passes this as a dict between nodes. BaseAgent.__call__
    hydrates it into this model, the agent's run() method modifies it,
    and __call__ serializes it back to a dict.
    """

    # ── Identity ─────────────────────────────────────────────
    run_id: str = ""
    user_goal: str = ""
    phase: Phase = Phase.INTAKE
    iteration: int = 0
    max_iterations: int = 3
    complexity_score: int = 3

    # ── Agent outputs (None until that agent runs) ───────────
    requirement_spec: Optional[RequirementSpec] = None
    research_report: Optional[ResearchReport] = None
    implementation_plan: Optional[ImplementationPlan] = None
    file_manifest: Optional[FileManifestPlan] = None
    code_files: dict[str, str] = Field(default_factory=dict)
    test_files: dict[str, str] = Field(default_factory=dict)
    execution_result: Optional[ExecutionResult] = None
    gap_report: Optional[GapReport] = None
    validation_result: Optional[ValidationResult] = None

    # ── Skeleton Architecture (Milestone 1) ───────────────────
    skeleton_artifact: Optional[SkeletonArtifact] = None
    skeleton_files: dict[str, str] = Field(default_factory=dict)
    skeleton_registry_context: str = ""  # Compressed symbol context for builder

    # ── Convergence tracking ─────────────────────────────────
    previous_gap_summaries: list[str] = Field(default_factory=list)

    # ── Build memory context ─────────────────────────────────
    similar_builds_context: str = ""  # Injected from CLI before pipeline runs

    # ── Metabolization (nutrient memory) ─────────────────────
    nutrient_profile: Optional[dict] = None  # Retrieved nutrients (from recomposer)
    nutrient_context: str = ""  # Formatted context block for architect
    extracted_nutrients: list[dict] = Field(default_factory=list)  # Decomposer output

    # ── Polarity (Latios/Latias) ─────────────────────────────
    polarity: PolarityState = Field(default_factory=PolarityState)

    # ── Cost tracking ────────────────────────────────────────
    token_usage: Optional[TokenUsage] = None
    max_cost_usd: float = 10.0

    # ── Diagnostics ──────────────────────────────────────────
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    agent_timings: dict[str, float] = Field(default_factory=dict)

    # ── Helper methods ───────────────────────────────────────

    def over_budget(self) -> bool:
        if self.token_usage is None:
            return False
        return self.token_usage.total_cost_usd >= self.max_cost_usd

    def over_iterations(self) -> bool:
        return self.iteration >= self.max_iterations

    def gap_is_oscillating(self) -> bool:
        if len(self.previous_gap_summaries) < 2:
            return False
        last = set(self.previous_gap_summaries[-1].lower().split())
        prev = set(self.previous_gap_summaries[-2].lower().split())
        union = last | prev
        if not union:
            return False
        return len(last & prev) / len(union) > 0.85
