"""Autonomous Pipeline — Tier 8 Full Loop.

The complete autonomous cycle:
  1. Build: generate code from goal (existing pipeline)
  2. Deploy: push to Railway/Docker (deploy module)
  3. Monitor: health check loop (monitor module)
  4. Heal: auto-remediate on failure (remediation module)
  5. Learn: deposit lessons into soil (existing decomposer)

Usage:
    result = await autonomous_build_and_deploy(
        goal="Build a URL shortener API",
        deploy_target="railway",
    )
    print(f"Live at: {result.url}")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("belief.deploy.autonomous")


@dataclass
class AutonomousResult:
    """Result of the full autonomous cycle."""
    goal: str
    build_success: bool = False
    deploy_success: bool = False
    url: str = ""
    build_cost: float = 0.0
    build_verdict: str = ""
    deploy_target: str = ""
    monitoring: bool = False
    files_generated: int = 0
    tests_passed: int = 0
    tests_total: int = 0
    error: str = ""


async def autonomous_build_and_deploy(
    goal: str,
    deploy_target: str = "railway",
    monitor: bool = True,
    monitor_duration: int = 300,  # 5 minutes of monitoring after deploy
    max_cost: float = 10.0,
) -> AutonomousResult:
    """The full Tier 8 loop: build → deploy → monitor → heal.

    Args:
        goal: Natural language description of what to build
        deploy_target: "railway" or "docker_local"
        monitor: Whether to start health monitoring after deploy
        monitor_duration: How long to monitor (seconds)
        max_cost: Maximum build budget in USD
    """
    result = AutonomousResult(goal=goal, deploy_target=deploy_target)

    # ── Step 1: Build ──
    logger.info(f"Autonomous: building — {goal[:60]}...")
    try:
        build_result = await _run_build(goal, max_cost)
        result.build_success = build_result.get("success", False)
        result.build_verdict = build_result.get("verdict", "unknown")
        result.build_cost = build_result.get("cost", 0.0)
        result.files_generated = len(build_result.get("code_files", {}))

        if not result.build_success:
            result.error = f"Build failed: {build_result.get('error', 'unknown')}"
            logger.warning(f"Autonomous: build failed — {result.error}")
            return result

        code_files = build_result["code_files"]
        test_files = build_result.get("test_files", {})

        logger.info(
            f"Autonomous: build complete — {result.files_generated} files, "
            f"${result.build_cost:.2f}, verdict={result.build_verdict}"
        )

    except Exception as e:
        result.error = f"Build error: {e}"
        logger.error(f"Autonomous: {result.error}")
        return result

    # ── Step 2: Deploy ──
    logger.info(f"Autonomous: deploying to {deploy_target}...")
    try:
        from belief.deploy import deploy, DeployConfig, DeployTarget, DeployStatus

        config = DeployConfig(
            target=DeployTarget(deploy_target),
            project_name=goal.split()[-1].lower()[:20].replace(" ", "-"),
        )

        deploy_result = await deploy(code_files, config)
        result.deploy_success = deploy_result.status == DeployStatus.LIVE
        result.url = deploy_result.url

        if not result.deploy_success:
            result.error = f"Deploy failed: {deploy_result.error}"
            logger.warning(f"Autonomous: deploy failed — {deploy_result.error}")
            return result

        logger.info(f"Autonomous: deployed → {result.url}")

    except Exception as e:
        result.error = f"Deploy error: {e}"
        logger.error(f"Autonomous: {result.error}")
        return result

    # ── Step 3: Monitor + Heal ──
    if monitor and result.url:
        logger.info(f"Autonomous: starting {monitor_duration}s monitoring of {result.url}")
        result.monitoring = True

        try:
            from belief.deploy.monitor import HealthMonitor
            from belief.deploy.remediation import AutoRemediation, RemediationConfig

            health_monitor = HealthMonitor(
                url=result.url,
                check_interval=30,
                failure_threshold=3,
            )

            remediation = AutoRemediation(
                code_files=code_files,
                test_files=test_files,
            )

            async def on_unhealthy(state):
                """Callback when health monitor detects issues."""
                plan = await remediation.diagnose_and_plan(state)
                logger.info(
                    f"Autonomous: remediation plan — "
                    f"{plan.action.value} (level {plan.autonomy_level.value}): "
                    f"{plan.diagnosis[:60]}"
                )
                success = await remediation.execute(plan)
                if success and plan.patch_files:
                    # Redeploy with fixed code
                    logger.info("Autonomous: redeploying with fix...")
                    from belief.deploy import deploy as redeploy
                    await redeploy(remediation.code_files, config)

            # Run monitoring for the specified duration
            max_checks = monitor_duration // 30
            await health_monitor.run_continuous(
                on_unhealthy=on_unhealthy,
                max_checks=max_checks,
            )

            logger.info(f"Autonomous: monitoring complete\n{health_monitor.summary()}")

        except Exception as e:
            logger.warning(f"Autonomous: monitoring failed — {e}")

    logger.info(
        f"Autonomous: complete — "
        f"{'✓' if result.deploy_success else '✗'} {result.url or 'no URL'} "
        f"(${result.build_cost:.2f})"
    )

    return result


async def _run_build(goal: str, max_cost: float) -> dict[str, Any]:
    """Run the Belief Engine build pipeline.

    Returns dict with: success, verdict, cost, code_files, test_files, error
    """
    try:
        from belief.graph import build_pipeline
        from belief.config import ModelRouter

        pipeline = build_pipeline(ModelRouter())

        initial_state = {
            "user_goal": goal,
            "max_cost": max_cost,
            "max_iterations": 3,
            "iteration": 0,
            "code_files": {},
            "test_files": {},
            "errors": [],
            "warnings": [],
        }

        final_state = await pipeline.ainvoke(initial_state)

        # Extract results
        code_files = final_state.get("code_files", {})
        test_files = final_state.get("test_files", {})

        exec_r = final_state.get("execution_result")
        success = False
        if exec_r:
            success = exec_r.get("success") if isinstance(exec_r, dict) else getattr(exec_r, "success", False)

        validation = final_state.get("validation_result")
        verdict = "unknown"
        if validation:
            verdict = validation.get("verdict") if isinstance(validation, dict) else getattr(validation, "verdict", "unknown")
            if hasattr(verdict, "value"):
                verdict = verdict.value

        # Extract cost from budget tracker if available
        cost = 0.0
        budget = final_state.get("build_budget")
        if budget:
            cost = budget.get("total_cost", 0.0) if isinstance(budget, dict) else getattr(budget, "total_cost", 0.0)

        return {
            "success": success or verdict == "pass",
            "verdict": verdict,
            "cost": cost,
            "code_files": code_files,
            "test_files": test_files,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "code_files": {}, "test_files": {}}
