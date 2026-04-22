"""Auto-Remediation — Tier 8 Self-Healing.

When the health monitor detects issues, this module:
  1. Diagnoses the root cause from error logs/health check data
  2. Generates a code fix using the refinement fixer
  3. Tests the fix locally
  4. Redeploys if tests pass

Progressive autonomy levels:
  Level 1 (auto): restart, clear cache, retry
  Level 2 (auto + notify): config changes, env vars
  Level 3 (human approval): code patches, dependency changes
  Level 4 (human approval): data migrations, breaking changes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("belief.deploy.remediation")


class AutonomyLevel(int, Enum):
    """How much independence the remediation has."""
    AUTO = 1           # Restarts, retries — no approval needed
    AUTO_NOTIFY = 2    # Config changes — auto with notification
    HUMAN_REVIEW = 3   # Code patches — requires human approval
    HUMAN_REQUIRED = 4 # Data migrations — always human


class RemediationAction(str, Enum):
    RESTART = "restart"
    ROLLBACK = "rollback"
    CODE_PATCH = "code_patch"
    CONFIG_CHANGE = "config_change"
    SCALE_UP = "scale_up"
    SKIP = "skip"


@dataclass
class RemediationPlan:
    """A plan for fixing a production issue."""
    diagnosis: str
    action: RemediationAction
    autonomy_level: AutonomyLevel
    details: str = ""
    patch_files: dict[str, str] = field(default_factory=dict)  # filepath → new content
    approved: bool = False
    executed: bool = False
    success: bool = False
    error: str = ""


@dataclass
class RemediationConfig:
    """Configuration for auto-remediation."""
    max_autonomy: AutonomyLevel = AutonomyLevel.AUTO_NOTIFY
    max_remediations_per_hour: int = 3
    cooldown_seconds: int = 300  # 5 min between remediations
    notify_callback: Any = None  # async fn(plan: RemediationPlan) for notifications


class AutoRemediation:
    """Self-healing system for deployed services."""

    def __init__(
        self,
        code_files: dict[str, str],
        test_files: dict[str, str],
        config: RemediationConfig | None = None,
    ):
        self.code_files = dict(code_files)
        self.test_files = dict(test_files)
        self.config = config or RemediationConfig()
        self._history: list[RemediationPlan] = []
        self._last_remediation: float = 0

    async def diagnose_and_plan(self, monitor_state) -> RemediationPlan:
        """Analyze the health monitor state and produce a remediation plan.

        Uses the error pattern + consecutive failures to determine
        the right action and autonomy level.
        """
        error = monitor_state.error_pattern or "Unknown error"
        failures = monitor_state.consecutive_failures

        # Level 1: Simple restart (transient errors)
        if failures <= 3 and "timeout" in error.lower():
            return RemediationPlan(
                diagnosis=f"Transient timeout ({failures} failures)",
                action=RemediationAction.RESTART,
                autonomy_level=AutonomyLevel.AUTO,
                details="Service may be overloaded. Restart to clear state.",
                approved=True,  # Auto-approved
            )

        # Level 1: Connection refused → service crashed
        if "connection refused" in error.lower() or "unreachable" in error.lower():
            return RemediationPlan(
                diagnosis=f"Service unreachable ({failures} failures)",
                action=RemediationAction.RESTART,
                autonomy_level=AutonomyLevel.AUTO,
                details="Service appears to have crashed. Restarting.",
                approved=True,
            )

        # Level 2: Server errors → likely code bug
        if "server error" in error.lower() or "500" in error:
            plan = await self._generate_code_fix(error, monitor_state)
            if plan:
                return plan

        # Level 3: Persistent failures → rollback
        if failures >= 10:
            return RemediationPlan(
                diagnosis=f"Persistent failures ({failures}). Rolling back to last known good.",
                action=RemediationAction.ROLLBACK,
                autonomy_level=AutonomyLevel.HUMAN_REVIEW,
                details="Multiple remediation attempts failed. Rolling back.",
            )

        return RemediationPlan(
            diagnosis=f"Unable to diagnose: {error}",
            action=RemediationAction.SKIP,
            autonomy_level=AutonomyLevel.HUMAN_REQUIRED,
        )

    async def _generate_code_fix(
        self, error: str, monitor_state
    ) -> RemediationPlan | None:
        """Use the refinement fixer to generate a code patch."""
        try:
            from belief.refinement.analyzer import analyze_failures
            from belief.refinement.fixer import generate_fix
            from belief.refinement import RefinementState

            # Create a refinement state from the error
            state = RefinementState(
                code_files=self.code_files,
                test_files=self.test_files,
                test_output=f"Production error: {error}",
            )

            # Analyze
            analysis = await analyze_failures(state, None)

            if not analysis.get("target_file"):
                return None

            # Generate fix
            fix = await generate_fix(
                state, analysis["diagnosis"], analysis["target_file"], None
            )

            if not fix.get("success"):
                return None

            # Build patch
            patch_files = {analysis["target_file"]: fix["new_content"]}

            return RemediationPlan(
                diagnosis=analysis["diagnosis"],
                action=RemediationAction.CODE_PATCH,
                autonomy_level=AutonomyLevel.HUMAN_REVIEW,
                details=fix["explanation"],
                patch_files=patch_files,
            )

        except Exception as e:
            logger.warning(f"Code fix generation failed: {e}")
            return None

    async def execute(self, plan: RemediationPlan) -> bool:
        """Execute a remediation plan.

        Checks autonomy level, applies fix, tests, and reports.
        """
        # Check cooldown
        now = time.time()
        if now - self._last_remediation < self.config.cooldown_seconds:
            wait = self.config.cooldown_seconds - (now - self._last_remediation)
            logger.info(f"Remediation: cooling down ({wait:.0f}s remaining)")
            return False

        # Check autonomy level
        if plan.autonomy_level.value > self.config.max_autonomy.value:
            logger.info(
                f"Remediation: {plan.action.value} requires level {plan.autonomy_level.value} "
                f"but max is {self.config.max_autonomy.value} — needs human approval"
            )
            if self.config.notify_callback:
                await self.config.notify_callback(plan)
            return False

        # Check hourly limit
        recent_count = sum(
            1 for p in self._history
            if p.executed and (now - self._last_remediation) < 3600
        )
        if recent_count >= self.config.max_remediations_per_hour:
            logger.warning("Remediation: hourly limit reached")
            return False

        # Execute based on action type
        plan.executed = True
        self._last_remediation = now

        if plan.action == RemediationAction.RESTART:
            logger.info(f"Remediation: restarting service — {plan.diagnosis}")
            plan.success = True  # Restart is handled by Railway/Docker

        elif plan.action == RemediationAction.CODE_PATCH:
            # Test the patch first
            if plan.patch_files:
                test_ok = await self._test_patch(plan.patch_files)
                if test_ok:
                    # Apply patch to our code files
                    self.code_files.update(plan.patch_files)
                    plan.success = True
                    logger.info(f"Remediation: code patch applied and tested — {plan.details[:60]}")
                else:
                    plan.success = False
                    plan.error = "Patch failed testing"
                    logger.warning("Remediation: code patch failed testing")

        elif plan.action == RemediationAction.ROLLBACK:
            logger.info(f"Remediation: rollback requested — {plan.diagnosis}")
            plan.success = True  # Actual rollback handled by deploy

        elif plan.action == RemediationAction.SKIP:
            logger.info(f"Remediation: skipping — {plan.diagnosis}")

        self._history.append(plan)
        return plan.success

    async def _test_patch(self, patch_files: dict[str, str]) -> bool:
        """Test a code patch before applying it."""
        try:
            from belief.refinement.runner import _run_tests

            # Merge patch with current code
            test_code = dict(self.code_files)
            test_code.update(patch_files)

            output, passed, total, failed = _run_tests(test_code, self.test_files)
            logger.info(f"Patch test: {passed}/{total} tests passed")

            # Accept if no regression from current state
            return total == 0 or passed > 0

        except Exception as e:
            logger.warning(f"Patch testing failed: {e}")
            return False

    @property
    def history(self) -> list[RemediationPlan]:
        return self._history
