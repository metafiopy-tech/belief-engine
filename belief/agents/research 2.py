"""Research Agent — find existing solutions + composition planning (M5).

Before calling the LLM for research, runs the composition planner to
identify well-known packages that can be reused. This information is
injected into the LLM prompt so the research agent can make informed
"use library" vs "generate from scratch" recommendations.
"""

from __future__ import annotations

import logging

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.artifacts import ResearchReport
from belief.models.state import Phase, UnifiedState
from belief.prompts import RESEARCH_SYSTEM, RESEARCH_PROMPT

logger = logging.getLogger("belief.agents.research")


class ResearchAgent(BaseAgent):
    role = ModelRole.RESEARCH
    name = "Research"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.RESEARCH
        spec = state.requirement_spec
        if not spec:
            state.errors.append("Research: no requirement_spec")
            state.phase = Phase.PLANNING
            return state

        # --- M5: Composition planning (zero-cost, runs before LLM) ---
        composition_context = _run_composition_planner(spec)

        llm = LLMClient(self.router)
        try:
            memory_context = ""
            if state.similar_builds_context:
                memory_context = (
                    f"\n\nSIMILAR PAST BUILDS (from build memory):\n"
                    f"{state.similar_builds_context}\n"
                    f"Use these as reference for architecture decisions."
                )

            prompt = RESEARCH_PROMPT.format(
                goal=spec.goal,
                goal_refined=spec.goal_refined or spec.goal,
                target_type=spec.target_type,
                complexity=spec.complexity_score,
                acceptance_criteria="\n".join(f"  - {c}" for c in spec.acceptance_criteria),
                tools=", ".join(spec.tools_needed) if spec.tools_needed else "none",
            ) + memory_context + composition_context

            report = await llm.generate_structured(
                role=self.role, system=RESEARCH_SYSTEM, prompt=prompt,
                response_schema=ResearchReport, temperature=0.3,
                complexity=state.complexity_score,
            )
            state.research_report = report
            logger.info(f"Research: {len(report.repo_candidates)} repos, approach={report.recommended_approach[:60]}")

        except (ConnectionError, ValueError) as e:
            logger.warning(f"Research fallback: {e}")
            state.research_report = ResearchReport(recommended_approach="Build from scratch — research unavailable")
            state.warnings.append(f"Research used fallback: {e}")
        finally:
            await llm.close()

        state.phase = Phase.PLANNING
        return state


def _run_composition_planner(spec) -> str:
    """Run M5 composition planner to find reusable packages.

    Returns a context string to inject into the research prompt.
    Zero LLM cost — uses local package registry.
    """
    try:
        from belief.tools.composition_planner import plan_composition, ComponentStrategy

        # Build requirements from acceptance criteria and tools
        requirements = []
        for criterion in spec.acceptance_criteria:
            # Extract component hints from criteria
            lower = criterion.lower()
            if "api" in lower or "server" in lower or "endpoint" in lower:
                requirements.append(("api_framework", "web API framework"))
            if "http" in lower or "fetch" in lower or "request" in lower:
                requirements.append(("http_client", "HTTP client for API calls"))
            if "scrape" in lower or "html" in lower or "parse" in lower:
                requirements.append(("web_scraper", "web scraping and HTML parsing"))
            if "mcp" in lower:
                requirements.append(("mcp_server", "MCP server framework"))
            if "database" in lower or "sql" in lower:
                requirements.append(("database", "database toolkit"))
            if "queue" in lower or "worker" in lower or "celery" in lower:
                requirements.append(("task_queue", "distributed task queue"))
            if "test" in lower:
                requirements.append(("testing", "testing framework"))

        for tool in (spec.tools_needed or []):
            requirements.append((tool.lower().replace(" ", "_"), tool))

        if not requirements:
            return ""

        # Deduplicate
        seen = set()
        unique_reqs = []
        for name, desc in requirements:
            if name not in seen:
                seen.add(name)
                unique_reqs.append((name, desc))

        plan = plan_composition(unique_reqs)

        # Format as context for the LLM
        lines = ["\n\nCOMPOSITION ANALYSIS (pre-evaluated packages):"]
        for d in plan.decisions:
            if d.strategy == ComponentStrategy.USE_LIBRARY and d.package:
                lines.append(
                    f"  ✓ {d.component_name}: USE {d.package.name} "
                    f"(score {d.package.quality_score:.0f}/100, {d.reason})"
                )
            elif d.strategy == ComponentStrategy.WRAP_LIBRARY and d.package:
                lines.append(
                    f"  ~ {d.component_name}: WRAP {d.package.name} "
                    f"(score {d.package.quality_score:.0f}/100, needs adapter)"
                )
            else:
                lines.append(f"  ✗ {d.component_name}: GENERATE from scratch ({d.reason})")

        if plan.libraries_to_install:
            lines.append(f"  Libraries to install: {', '.join(plan.libraries_to_install)}")

        logger.info(f"Composition: {len(plan.decisions)} components evaluated")
        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"Composition planner skipped: {e}")
        return ""
