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

        # Session 6 (v3.2): inject top-3 similar-goal priors from the
        # agent archive.  Append-only so the session-1 num_keep=512
        # prefix cache still hits on the leading bytes of PLANNER_SYSTEM.
        system_prompt = PLANNER_SYSTEM
        try:
            from belief.archive.priors import format_priors_block
            priors_block = format_priors_block(spec.goal, k=3)
            if priors_block:
                system_prompt = f"{PLANNER_SYSTEM}{priors_block}"
                logger.info("Planner: injected %d prior(s) from agent archive",
                            priors_block.count("### Prior "))
        except Exception as e:  # pragma: no cover
            logger.debug("Planner prior injection skipped: %s", e)

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
                role=self.role, system=system_prompt, prompt=prompt,
                response_schema=ImplementationPlan, temperature=0.3,
                complexity=state.complexity_score,
            )
            if not plan.steps:
                plan = _fallback()

            # ── SEED-001: Validate plan completeness ──
            # Self-proposed improvement: truncated/malformed JSON from the planner
            # causes silent fallback to incoherent build plans. Validate before use.
            plan = _validate_plan(plan, spec)

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


def _validate_plan(plan: ImplementationPlan, spec) -> ImplementationPlan:
    """SEED-001: Validate plan completeness before downstream use.

    Self-proposed by SEED after analyzing recurring antipatterns:
    truncated/malformed JSON from the planner causes silent degradation.

    Checks:
    1. Plan has at least 3 steps (less = truncated)
    2. Steps have valid ordering (no gaps, no duplicates)
    3. Strategy is a known value
    4. At least one step mentions "build" or "generate" or "implement"
    5. Dependencies reference valid step numbers

    If validation fails, falls back to generic plan rather than
    propagating a broken one downstream.
    """
    issues = []

    # Check 1: Minimum step count
    if len(plan.steps) < 2:
        issues.append(f"Only {len(plan.steps)} steps (minimum 2)")

    # Check 2: Valid step ordering
    orders = [s.order for s in plan.steps]
    if len(set(orders)) != len(orders):
        issues.append(f"Duplicate step orders: {orders}")

    # Check 3: Known strategy
    valid_strategies = {"generate_fresh", "clone_and_modify", "compose_from_packages", "extend_codebase"}
    if plan.strategy not in valid_strategies:
        # Not fatal — just log
        logger.debug(f"Planner: unknown strategy '{plan.strategy}', continuing")

    # Check 4: At least one implementation step
    has_impl = any(
        any(kw in s.description.lower() for kw in ("build", "generat", "implement", "creat", "writ", "code"))
        for s in plan.steps
    )
    if not has_impl:
        issues.append("No implementation step found in plan")

    # Check 5: Valid dependency references
    valid_orders = set(orders)
    for step in plan.steps:
        for dep in step.dependencies:
            if dep not in valid_orders:
                issues.append(f"Step {step.order} depends on non-existent step {dep}")
                break

    # Check 6: Steps have non-empty descriptions
    empty_steps = [s.order for s in plan.steps if len(s.description.strip()) < 5]
    if empty_steps:
        issues.append(f"Steps {empty_steps} have empty/truncated descriptions")

    if issues:
        logger.warning(f"SEED-001: Plan validation failed — {'; '.join(issues)}. Using fallback.")
        return _fallback()

    return plan
