"""Gap Analyst Agent — compare build state against requirements.

Source: forge/agents/gap_analyst.py (deterministic checks + LLM analysis)
"""

from __future__ import annotations

import logging

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.artifacts import Gap, GapReport, GapSeverity
from belief.models.state import Phase, UnifiedState
from belief.prompts import GAP_ANALYST_SYSTEM, GAP_ANALYST_PROMPT

logger = logging.getLogger("belief.agents.gap_analyst")


class GapAnalystAgent(BaseAgent):
    role = ModelRole.GAP_ANALYST
    name = "Gap Analyst"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.GAP_ANALYSIS

        # Always run deterministic checks
        basic_gaps = _check_basic_gaps(state)

        # Skip LLM if no spec
        if not state.requirement_spec:
            state.gap_report = _build_report(basic_gaps, confidence=0.4)
            state.previous_gap_summaries.append(state.gap_report.summary)
            state.phase = Phase.SYNTHESIS
            return state

        llm = LLMClient(self.router)
        try:
            spec = state.requirement_spec
            exec_result = state.execution_result
            prev = state.previous_gap_summaries[-3:]
            prev_fmt = (
                "\n".join(f"  [{i + 1}] {s}" for i, s in enumerate(reversed(prev)))
                if prev
                else "  (first iteration)"
            )

            pytest_summary = "  (not run)"
            if exec_result and exec_result.pytest_result and exec_result.pytest_result.ran:
                pytest_summary = f"  {exec_result.pytest_result.summary_line}"

            code_preview = (
                "\n\n".join(
                    f"  --- {f} ---\n{c[:1500]}" for f, c in sorted(state.code_files.items())
                )[:6000]
                or "  (no files)"
            )

            prompt = GAP_ANALYST_PROMPT.format(
                goal=spec.goal,
                goal_refined=spec.goal_refined,
                target_type=spec.target_type,
                acceptance_criteria="\n".join(
                    f"  {i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)
                ),
                constraints=", ".join(spec.constraints) if spec.constraints else "none",
                code_files=code_preview,
                exec_success=exec_result.success if exec_result else "not executed",
                exec_exit_code=exec_result.exit_code if exec_result else -1,
                exec_stdout=(
                    exec_result.stdout[-1000:] if exec_result and exec_result.stdout else "(empty)"
                ),
                exec_stderr=(
                    exec_result.stderr[-1000:] if exec_result and exec_result.stderr else "(empty)"
                ),
                exec_error_summary=(
                    exec_result.error_summary if exec_result else "no execution data"
                ),
                pytest_summary=pytest_summary,
                iteration=state.iteration + 1,
                max_iterations=state.max_iterations,
                previous_summaries=prev_fmt,
            )

            llm_report = await llm.generate_structured(
                role=self.role,
                system=GAP_ANALYST_SYSTEM,
                prompt=prompt,
                response_schema=GapReport,
                temperature=0.2,
                complexity=state.complexity_score,
            )

            # Merge deterministic gaps
            llm_descs = {g.description.lower() for g in llm_report.gaps}
            for bg in basic_gaps:
                if bg.description.lower() not in llm_descs:
                    llm_report.gaps.append(bg)

            llm_report.total_blockers = sum(
                1 for g in llm_report.gaps if g.severity == GapSeverity.BLOCKER
            )
            llm_report.total_major = sum(
                1 for g in llm_report.gaps if g.severity == GapSeverity.MAJOR
            )
            llm_report.requires_research = any(g.requires_new_research for g in llm_report.gaps)

            state.gap_report = llm_report
            logger.info(
                f"Gap analysis: {llm_report.total_blockers} blockers, {llm_report.total_major} major"
            )

        except (ConnectionError, ValueError) as e:
            logger.warning(f"Gap analyst fallback: {e}")
            state.gap_report = _build_report(basic_gaps, confidence=0.4)
            state.warnings.append(f"Gap analyst used fallback: {e}")
        finally:
            await llm.close()

        state.previous_gap_summaries.append(state.gap_report.summary)
        state.phase = Phase.SYNTHESIS
        return state


def _check_basic_gaps(state: UnifiedState) -> list[Gap]:
    gaps: list[Gap] = []

    if not state.code_files:
        gaps.append(
            Gap(
                description="No code files produced",
                severity=GapSeverity.BLOCKER,
                category="functionality",
            )
        )
        return gaps

    for fname, content in state.code_files.items():
        if "# TODO" in content or "pass  # placeholder" in content:
            gaps.append(
                Gap(
                    description=f"'{fname}' contains TODO/placeholder",
                    severity=GapSeverity.MAJOR,
                    category="functionality",
                )
            )

    exec_result = state.execution_result
    if exec_result and not exec_result.success:
        if not exec_result.install_success:
            gaps.append(
                Gap(
                    description=f"Dependency install failed: {exec_result.install_stderr[-200:]}",
                    severity=GapSeverity.BLOCKER,
                    category="dependency",
                )
            )
        elif "SyntaxError" in (exec_result.stderr or ""):
            gaps.append(
                Gap(
                    description="Syntax error in code",
                    severity=GapSeverity.BLOCKER,
                    category="functionality",
                )
            )
        elif "ModuleNotFoundError" in (exec_result.stderr or ""):
            gaps.append(
                Gap(
                    description=f"Missing dependency: {exec_result.stderr.split('ModuleNotFoundError')[-1][:100]}",
                    severity=GapSeverity.BLOCKER,
                    category="dependency",
                )
            )
        else:
            gaps.append(
                Gap(
                    description=f"Execution failed (exit {exec_result.exit_code})",
                    severity=GapSeverity.BLOCKER,
                    category="functionality",
                    suggested_fix=exec_result.error_summary[:200],
                )
            )

    return gaps


def _build_report(gaps: list[Gap], confidence: float) -> GapReport:
    return GapReport(
        gaps=gaps,
        total_blockers=sum(1 for g in gaps if g.severity == GapSeverity.BLOCKER),
        total_major=sum(1 for g in gaps if g.severity == GapSeverity.MAJOR),
        confidence=confidence,
        summary=f"Deterministic analysis: {len(gaps)} issue(s) found"
        if gaps
        else "No issues detected",
    )
