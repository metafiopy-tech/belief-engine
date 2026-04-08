"""Validator Agent — adversarial review of the final build."""

from __future__ import annotations
import logging
from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.artifacts import (
    TestCase, TestTier, TIER_WEIGHTS,
    ValidationResult, ValidationVerdict,
)
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

            # Classify tests into tiers and compute weighted score
            _classify_and_score(result)

            for attr in ("correctness_score", "completeness_score", "code_quality_score", "security_score"):
                setattr(result, attr, max(0.0, min(1.0, getattr(result, attr))))

            state.validation_result = result
            logger.info(
                f"Validator: {result.verdict.value}, "
                f"{result.tests_passed}/{result.tests_total} tests, "
                f"weighted={result.weighted_score:.2f}"
            )

        except (ConnectionError, ValueError) as e:
            logger.warning(f"Validator fallback: {e}")
            state.validation_result = _deterministic(state)
            state.warnings.append(f"Validator fallback: {e}")
        finally:
            await llm.close()

        state.phase = Phase.COMPLETE
        return state


def _classify_and_score(result: ValidationResult) -> None:
    """Classify tests into tiers and compute weighted verdict score.

    Tier classification:
    - Tests with "import", "instantiat", "exist" → P0 SMOKE
    - Tests with "error", "invalid", "edge", "empty", "boundary" → P2 EDGE_CASE
    - Tests with ImportError/ModuleNotFoundError in error → ENVIRONMENT (weight 0)
    - Everything else → P1 FUNCTIONAL

    Weighted score:
    - Score = sum(weight × pass) / sum(weight) for non-environment tests
    - All smoke tests passing + score ≥ 0.85 → PASS verdict
    """
    for test in result.tests:
        name_lower = (test.name + " " + test.description).lower()
        error_lower = test.error.lower()

        # Environment failures — weight 0
        if any(e in error_lower for e in ("importerror", "modulenotfounderror", "no module named")):
            test.tier = TestTier.ENVIRONMENT
        # Smoke tests
        elif any(k in name_lower for k in ("import", "instantiat", "exist", "smoke", "health", "startup", "p0")):
            test.tier = TestTier.SMOKE
        # Edge cases
        elif any(k in name_lower for k in ("error", "invalid", "edge", "empty", "boundary", "negative", "p2")):
            test.tier = TestTier.EDGE_CASE
        # Everything else is functional
        else:
            test.tier = TestTier.FUNCTIONAL

    # Compute weighted score (excluding environment tests)
    weighted_sum = 0.0
    weight_total = 0.0
    smoke_pass = True

    for test in result.tests:
        w = TIER_WEIGHTS[test.tier]
        if w == 0:
            continue  # Skip environment tests
        weight_total += w
        if test.passed:
            weighted_sum += w
        elif test.tier == TestTier.SMOKE:
            smoke_pass = False

    result.weighted_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    result.tests_passed = sum(1 for t in result.tests if t.passed)
    result.tests_total = len(result.tests)

    # Determine verdict based on weighted score
    if smoke_pass and result.weighted_score >= 0.85:
        result.verdict = ValidationVerdict.PASS
    elif result.weighted_score >= 0.5:
        result.verdict = ValidationVerdict.FAIL_FIXABLE
    else:
        result.verdict = ValidationVerdict.FAIL_FIXABLE


def _deterministic(state: UnifiedState) -> ValidationResult:
    tests: list[TestCase] = []
    issues: list[str] = []

    has_code = bool(state.code_files)
    tests.append(TestCase(name="code_exists", description="Code files produced", passed=has_code, tier=TestTier.SMOKE))
    if not has_code:
        issues.append("No code files")

    exec_ok = state.execution_result and state.execution_result.success
    tests.append(TestCase(name="execution", description="Code executes", passed=bool(exec_ok), tier=TestTier.SMOKE))
    if not exec_ok:
        issues.append("Execution failed")

    passed = sum(1 for t in tests if t.passed)
    weighted = passed / max(len(tests), 1)
    return ValidationResult(
        verdict=ValidationVerdict.PASS if passed == len(tests) else ValidationVerdict.FAIL_FIXABLE,
        tests=tests, tests_passed=passed, tests_total=len(tests),
        weighted_score=weighted,
        correctness_score=1.0 if exec_ok else 0.2,
        completeness_score=passed / max(len(tests), 1),
        code_quality_score=0.5, security_score=0.5,
        issues=issues,
        summary=f"Deterministic: {passed}/{len(tests)} checks passed",
    )
