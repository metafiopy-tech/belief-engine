"""Planner Agent — create implementation plan from research + requirements."""

from __future__ import annotations
import logging
from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.artifacts import ImplementationPlan, PlanStep
from belief.models.state import Phase, UnifiedState
from belief.prompts import PLANNER_SYSTEM, PLANNER_PROMPT

logger = logging.getLogger("belief.agents.planner")

class PlannerAgent(BaseAgent):
    role = ModelRole.PLANNER
    name = "Planner"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.PLANNING
        spec = state.requirement_spec
        research = state.research_report

        if not spec:
            state.errors.append("Planner: no requirement_spec")
            state.implementation_plan = _fallback()
            state.phase = Phase.ARCHITECTING
            return state

        llm = LLMClient(self.router)
        try:
            prompt = PLANNER_PROMPT.format(
                goal=spec.goal, goal_refined=spec.goal_refined or spec.goal,
                target_type=spec.target_type,
                acceptance_criteria="\n".join(f"  - {c}" for c in spec.acceptance_criteria),
                constraints=", ".join(spec.constraints) if spec.constraints else "none",
                credentials=", ".join(c.env_var for c in spec.credentials) if spec.credentials else "none",
                tools=", ".join(spec.tools_needed) if spec.tools_needed else "none",
                recommended_approach=research.recommended_approach if research else "Build from scratch",
                clone_target=research.clone_target if research else "none",
                patterns=", ".join(research.patterns_found[:5]) if research and research.patterns_found else "none",
                repo_candidates="\n".join(
                    f"  - {r.name} ({r.stars}★): {r.description}" for r in (research.repo_candidates[:5] if research else [])
                ) or "  none found",
            )
            plan = await llm.generate_structured(
                role=self.role, system=PLANNER_SYSTEM, prompt=prompt,
                response_schema=ImplementationPlan, temperature=0.3,
                complexity=state.complexity_score,
            )
            if not plan.steps:
                plan = _fallback()
            state.implementation_plan = plan
            logger.info(f"Planner: {len(plan.steps)} steps, strategy={plan.strategy}")
        except (ConnectionError, ValueError) as e:
            logger.warning(f"Planner fallback: {e}")
            state.implementation_plan = _fallback()
            state.warnings.append(f"Planner used fallback: {e}")
        finally:
            await llm.close()

        state.phase = Phase.ARCHITECTING
        return state

def _fallback() -> ImplementationPlan:
    return ImplementationPlan(
        strategy="generate_fresh",
        steps=[
            PlanStep(order=1, description="Generate automation code", agent_responsible="builder"),
            PlanStep(order=2, description="Execute and test", agent_responsible="executor", dependencies=[1]),
            PlanStep(order=3, description="Finalize and document", agent_responsible="synthesizer", dependencies=[1, 2]),
        ],
        estimated_iterations=2,
        risk_factors=["Planning done without LLM — plan is generic"],
    )
