"""Intake Agent — parse goal into RequirementSpec.

Source: forge/agents/intake.py
"""

from __future__ import annotations

import logging

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.artifacts import RequirementSpec
from belief.models.state import Phase, UnifiedState
from belief.prompts import INTAKE_SYSTEM, INTAKE_PROMPT

logger = logging.getLogger("belief.agents.intake")


class IntakeAgent(BaseAgent):
    role = ModelRole.INTAKE
    name = "Intake"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.INTAKE

        if state.requirement_spec is not None:
            logger.info("Intake: spec already present — skipping")
            state.phase = Phase.RESEARCH
            return state

        goal = state.user_goal.strip()
        if not goal:
            state.errors.append("Intake: empty goal")
            state.phase = Phase.FAILED
            return state

        if len(goal) > 2000:
            goal = goal[:2000]
            state.warnings.append("Intake: goal truncated to 2000 chars")
        state.user_goal = goal

        llm = LLMClient(self.router)
        try:
            spec = await llm.generate_structured(
                role=self.role,
                system=INTAKE_SYSTEM,
                prompt=INTAKE_PROMPT.format(goal=goal),
                response_schema=RequirementSpec,
                temperature=0.3,
            )
            spec.goal = state.user_goal
            if not spec.goal_refined:
                spec.goal_refined = state.user_goal
            if not spec.acceptance_criteria:
                spec.acceptance_criteria = [
                    f"The automation for '{goal}' executes without errors",
                    "The automation produces the expected output",
                ]
            spec.complexity_score = max(1, min(5, spec.complexity_score))
            state.requirement_spec = spec
            state.complexity_score = spec.complexity_score
            logger.info(
                f"Intake: {len(spec.acceptance_criteria)} criteria, complexity={spec.complexity_score}"
            )

        except (ConnectionError, ValueError) as e:
            logger.warning(f"Intake fallback: {e}")
            state.requirement_spec = RequirementSpec(
                goal=goal,
                goal_refined=goal,
                target_type="python",
                acceptance_criteria=[
                    f"The automation for '{goal}' executes without errors",
                    "The automation produces the expected output",
                ],
            )
            state.warnings.append(f"Intake used fallback: {e}")
        finally:
            await llm.close()

        state.phase = Phase.RESEARCH
        return state
