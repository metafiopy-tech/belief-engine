"""Model routing configuration.

Maps agent roles to LLM models. Supports complexity-based upgrades
(e.g., use Opus for planning when complexity >= 4).

Session 6 adds a `mode` dimension on top of the per-role model map:

    mode="cloud"   All agents call Anthropic (v3.0 default; unchanged).
    mode="hybrid"  Mechanical agents route to local (Ollama); reasoning
                   agents stay on Anthropic. HYBRID_ROUTING owns the
                   per-role mapping.
    mode="local"   All agents route to the local backend. Anything
                   marked 'none' in HYBRID_ROUTING stays deterministic.

Source: forge/config/models.py + brain.py classify_complexity()
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelRole(str, Enum):
    INTAKE = "intake"
    RESEARCH = "research"
    PLANNER = "planner"
    ARCHITECT = "architect"
    BUILDER = "builder"
    TESTER = "tester"
    DEBUGGER = "debugger"
    GAP_ANALYST = "gap_analyst"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    LATIOS = "latios"
    EXECUTOR = "executor"


# Default model assignments per role
# Move 3: Route 6 of 11 agents to Haiku for 3x cost savings
# Sonnet: research, planner, architect, builder, debugger (need deep reasoning)
# Haiku: intake, tester, gap_analyst, synthesizer, latios, executor (mechanical tasks)
_DEFAULTS: dict[str, str] = {
    ModelRole.INTAKE: "claude-haiku-4-5-20251001",
    ModelRole.RESEARCH: "claude-sonnet-4-6",
    ModelRole.PLANNER: "claude-sonnet-4-6",
    ModelRole.ARCHITECT: "claude-sonnet-4-6",
    ModelRole.BUILDER: "claude-sonnet-4-6",
    ModelRole.TESTER: "claude-haiku-4-5-20251001",
    ModelRole.DEBUGGER: "claude-sonnet-4-6",
    ModelRole.GAP_ANALYST: "claude-haiku-4-5-20251001",
    ModelRole.SYNTHESIZER: "claude-haiku-4-5-20251001",
    ModelRole.VALIDATOR: "claude-haiku-4-5-20251001",  # Now deterministic (Move 1), only used as fallback
    ModelRole.LATIOS: "claude-haiku-4-5-20251001",
    ModelRole.EXECUTOR: "claude-haiku-4-5-20251001",
}

# Roles that upgrade to Opus at high complexity
_UPGRADE_AT_COMPLEXITY_4: set[str] = {
    ModelRole.PLANNER,
    ModelRole.ARCHITECT,
    ModelRole.SYNTHESIZER,
}

OPUS_MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Session 6: hybrid routing
# ---------------------------------------------------------------------------


class RouteMode(str, Enum):
    CLOUD = "cloud"
    HYBRID = "hybrid"
    LOCAL = "local"


class Backend(str, Enum):
    CLOUD = "cloud"       # Anthropic
    LOCAL = "local"       # Ollama
    NONE = "none"         # deterministic path (no LLM)


# Per-role intent under mode="hybrid". Mechanical tasks go local,
# reasoning tasks stay on Anthropic, deterministic paths declare 'none'
# so callers don't spin up either client.
HYBRID_ROUTING: dict[str, Backend] = {
    # Mechanical — local is fine
    ModelRole.INTAKE.value: Backend.LOCAL,
    ModelRole.GAP_ANALYST.value: Backend.LOCAL,
    ModelRole.SYNTHESIZER.value: Backend.LOCAL,
    ModelRole.TESTER.value: Backend.LOCAL,
    ModelRole.EXECUTOR.value: Backend.LOCAL,
    ModelRole.LATIOS.value: Backend.LOCAL,
    ModelRole.VALIDATOR.value: Backend.LOCAL,
    # Reasoning — Claude stays
    ModelRole.RESEARCH.value: Backend.CLOUD,
    ModelRole.PLANNER.value: Backend.CLOUD,
    ModelRole.ARCHITECT.value: Backend.CLOUD,
    ModelRole.BUILDER.value: Backend.CLOUD,
    ModelRole.DEBUGGER.value: Backend.CLOUD,
}

DEFAULT_LOCAL_MODEL = "qwen2.5-coder:14b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class ModelRouter(BaseModel):
    """Routes agent roles to specific LLM models.

    Reads overrides from environment variables:
      BELIEF_MODEL_INTAKE=claude-haiku-4-5-20251001
      BELIEF_MODEL_BUILDER=claude-sonnet-4-6
      etc.

    Session 6 additions:
      BELIEF_MODEL_MODE=cloud|hybrid|local   (default: cloud)
      BELIEF_LOCAL_MODEL=qwen2.5-coder:14b   (default: see DEFAULT_LOCAL_MODEL)
      BELIEF_OLLAMA_URL=http://host:11434    (default: localhost)
    """
    backend: str = Field(default="anthropic")
    overrides: dict[str, str] = Field(default_factory=dict)
    mode: RouteMode = Field(default=RouteMode.CLOUD)
    local_model: str = Field(default=DEFAULT_LOCAL_MODEL)
    ollama_base_url: str = Field(default=DEFAULT_OLLAMA_BASE_URL)
    # Per-router counter: how many times a local call fell back to cloud.
    fallback_count: int = Field(default=0)

    def model_post_init(self, __context) -> None:
        # Load overrides from environment
        for role in ModelRole:
            env_key = f"BELIEF_MODEL_{role.value.upper()}"
            env_val = os.environ.get(env_key)
            if env_val:
                self.overrides[role.value] = env_val

        # Session 6 env overrides
        env_mode = os.environ.get("BELIEF_MODEL_MODE", "").strip().lower()
        if env_mode in {m.value for m in RouteMode}:
            self.mode = RouteMode(env_mode)
        env_local = os.environ.get("BELIEF_LOCAL_MODEL", "").strip()
        if env_local:
            self.local_model = env_local
        env_url = os.environ.get("BELIEF_OLLAMA_URL", "").strip()
        if env_url:
            self.ollama_base_url = env_url

    # ---------------------------------------------------------- mode API
    def set_mode(self, mode: str | RouteMode) -> None:
        """Switch mode at runtime; resets the fallback counter."""
        m = RouteMode(mode.value) if isinstance(mode, RouteMode) else RouteMode(str(mode))
        self.mode = m
        self.fallback_count = 0

    def backend_for(self, role: ModelRole | str) -> Backend:
        """Return which backend serves `role` under the current mode.

        - cloud mode  : always Backend.CLOUD (unchanged from v3.0)
        - hybrid mode : HYBRID_ROUTING table
        - local mode  : Backend.LOCAL (unless HYBRID_ROUTING says 'none')
        """
        role_str = role.value if isinstance(role, ModelRole) else role
        if self.mode is RouteMode.CLOUD:
            return Backend.CLOUD
        table_entry = HYBRID_ROUTING.get(role_str, Backend.CLOUD)
        if self.mode is RouteMode.HYBRID:
            return table_entry
        # LOCAL
        if table_entry is Backend.NONE:
            return Backend.NONE
        return Backend.LOCAL

    def record_fallback(self) -> None:
        """Increment the counter when a local call degrades to cloud."""
        self.fallback_count += 1

    # ---------------------------------------------------------- model API
    def get_model(self, role: ModelRole | str, complexity: int = 1) -> str:
        """Get the model name for a given role and complexity.

        Cloud-side model name (Haiku / Sonnet / Opus) — unchanged from
        v3.0. The mode/backend decision is a separate axis; callers
        that dispatch to Ollama use `router.local_model` directly.
        """
        role_str = role.value if isinstance(role, ModelRole) else role

        # Check explicit override first
        if role_str in self.overrides:
            return self.overrides[role_str]

        # Complexity-based upgrade
        if complexity >= 4 and role_str in _UPGRADE_AT_COMPLEXITY_4:
            return OPUS_MODEL

        return _DEFAULTS.get(role_str, "claude-sonnet-4-6")

    def routing_table(self) -> list[tuple[str, Backend, str]]:
        """Snapshot the effective routing per role — used by `belief models`.

        Each row: (role_name, backend, model_name_or_note).
        """
        out: list[tuple[str, Backend, str]] = []
        for role in ModelRole:
            backend = self.backend_for(role)
            if backend is Backend.LOCAL:
                model = self.local_model
            elif backend is Backend.CLOUD:
                model = self.get_model(role)
            else:
                model = "(deterministic)"
            out.append((role.value, backend, model))
        return out


__all__ = [
    "Backend",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "HYBRID_ROUTING",
    "ModelRole",
    "ModelRouter",
    "OPUS_MODEL",
    "RouteMode",
]
