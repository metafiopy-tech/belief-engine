"""Model routing configuration.

Maps agent roles to LLM models. Supports complexity-based upgrades
(e.g., use Opus for planning when complexity >= 4).

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
_DEFAULTS: dict[str, str] = {
    ModelRole.INTAKE: "claude-haiku-4-5-20251001",
    ModelRole.RESEARCH: "claude-sonnet-4-6",
    ModelRole.PLANNER: "claude-sonnet-4-6",
    ModelRole.ARCHITECT: "claude-sonnet-4-6",
    ModelRole.BUILDER: "claude-sonnet-4-6",
    ModelRole.TESTER: "claude-sonnet-4-6",
    ModelRole.DEBUGGER: "claude-sonnet-4-6",
    ModelRole.GAP_ANALYST: "claude-sonnet-4-6",
    ModelRole.SYNTHESIZER: "claude-sonnet-4-6",
    ModelRole.VALIDATOR: "claude-sonnet-4-6",
    ModelRole.LATIOS: "claude-sonnet-4-6",
    ModelRole.EXECUTOR: "claude-sonnet-4-6",
}

# Roles that upgrade to Opus at high complexity
_UPGRADE_AT_COMPLEXITY_4: set[str] = {
    ModelRole.PLANNER,
    ModelRole.ARCHITECT,
    ModelRole.SYNTHESIZER,
}

OPUS_MODEL = "claude-opus-4-6"


class ModelRouter(BaseModel):
    """Routes agent roles to specific LLM models.

    Reads overrides from environment variables:
      BELIEF_MODEL_INTAKE=claude-haiku-4-5-20251001
      BELIEF_MODEL_BUILDER=claude-sonnet-4-6
      etc.
    """
    backend: str = Field(default="anthropic")
    overrides: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        # Load overrides from environment
        for role in ModelRole:
            env_key = f"BELIEF_MODEL_{role.value.upper()}"
            env_val = os.environ.get(env_key)
            if env_val:
                self.overrides[role.value] = env_val

    def get_model(self, role: ModelRole | str, complexity: int = 1) -> str:
        """Get the model name for a given role and complexity."""
        role_str = role.value if isinstance(role, ModelRole) else role

        # Check explicit override first
        if role_str in self.overrides:
            return self.overrides[role_str]

        # Complexity-based upgrade
        if complexity >= 4 and role_str in _UPGRADE_AT_COMPLEXITY_4:
            return OPUS_MODEL

        return _DEFAULTS.get(role_str, "claude-sonnet-4-6")
