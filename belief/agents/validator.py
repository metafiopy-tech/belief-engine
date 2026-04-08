"""Validator Agent — adversarial review of the final build."""

from __future__ import annotations
import logging
from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.artifacts import TestCase, ValidationResult, ValidationVerdict
from belief.models.state import Phase, UnifiedState
from belief.prompts import VALIDATOR_SYSTEM, VALIDATOR_PROMPT

logger = logging.getLogger("belief.agents.validator")

class ValidatorAgent(BaseAgent):
    role = ModelRole.VALIDATOR
    name = "Validator"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.VALIDATION
        if not state.code_files:
            state.validation_result = ValidationResult(
                verdict=ValidationVerdict.FAIL_FIXABLE, summary="No code files",
                issues=["No code files produced"],
            )
            state.phase = Phase.COMPLETE
            return state

        llm = LLMClient(self.router)
        try:
            spec = state.requirement_spec
            exec_result = state.execution_result
            pytest_summary = "  (not run)"
            if exec_result and exec_result.pytest_result and exec_result.pytest_result.ran:
                pytest_summary = f"  {exec_result.pytest_result.summary_line}"

            code_preview = "\n\n".join(
                f"--- {f} ---\n{c[:4000]}" for f, c in sorted(state.code_files.items())
            )

            prompt = VALIDATOR_PROMPT.format(
                goal=spec.goal if spec else state.user_goal,
                goal_refined=spec.goal_refined if spec else state.user_goal,
                target_type=spec.target_type if spec else "python",
                acceptance_criteria="\n".join(f"  {i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)) if spec else "  (none)",
                constraints=", ".join(spec.constraints) if spec and spec.constraints else "none",
                code_files=code_preview,
                exec_success=exec_result.success if exec_result else "not executed",
                exec_exit_code=exec_result.exit_code if exec_result else -1,
                exec_stdout=(exec_result.stdout[-500:] if exec_result and exec_result.stdout else "(empty)"),
                exec_stderr=(exec_result.stderr[-500:] if exec_result and exec_result.stderr else "(empty)"),
                pytest_summary=pytest_summary,
            )
            result = await llm.generate_structured(
                role=self.role, system=VALIDATOR_SYSTEM, prompt=prompt,
                response_schema=ValidationResult, temperature=0.2,
                complexity=state.complexity_score,
            )
            result.tests_passed = sum(1 for t in result.tests if t.passed)
            result.tests_total = len(result.tests)
            for attr in ("correctness_score", "completeness_score", "code_quality_score", "security_score"):
                setattr(result, attr, max(0.0, min(1.0, getattr(result, attr))))

            state.validation_result = result
            logger.info(f"Validator: {result.verdict.value}, {result.tests_passed}/{result.tests_total} tests")

        except (ConnectionError, ValueError) as e:
            logger.warning(f"Validator fallback: {e}")
            state.validation_result = _deterministic(state)
            state.warnings.append(f"Validator fallback: {e}")
        finally:
            await llm.close()

        state.phase = Phase.COMPLETE
        return state


def _deterministic(state: UnifiedState) -> ValidationResult:
    tests: list[TestCase] = []
    issues: list[str] = []

    has_code = bool(state.code_files)
    tests.append(TestCase(name="code_exists", description="Code files produced", passed=has_code))
    if not has_code:
        issues.append("No code files")

    exec_ok = state.execution_result and state.execution_result.success
    tests.append(TestCase(name="execution", description="Code executes", passed=bool(exec_ok)))
    if not exec_ok:
        issues.append("Execution failed")

    passed = sum(1 for t in tests if t.passed)
    return ValidationResult(
        verdict=ValidationVerdict.PASS if passed == len(tests) else ValidationVerdict.FAIL_FIXABLE,
        tests=tests, tests_passed=passed, tests_total=len(tests),
        correctness_score=1.0 if exec_ok else 0.2,
        completeness_score=passed / max(len(tests), 1),
        code_quality_score=0.5, security_score=0.5,
        issues=issues,
        summary=f"Deterministic: {passed}/{len(tests)} checks passed",
    )
