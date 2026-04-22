"""
Async Overseer — monitors the self-improvement loop for safety violations.

Runs on a DIFFERENT model than the agent (Haiku vs Sonnet) to prevent
self-deception.  Periodically executes safety probes that check:
  - Evaluator integrity (benchmark/hardening files unmodified)
  - Test harness edits (SICA diffs don't touch forbidden paths)
  - Environment tampering (API keys, PYTHONPATH)
  - Resource consumption (cost trends)

Research basis: DGM was caught writing fake unit-test logs.  STOP's
budget constraints were deleted by the LLM when they lived inside
the mutable prompt.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger("belief.safety.overseer")


class SafetyViolation(Exception):
    """A safety probe detected a violation."""

    def __init__(self, probe_name: str, description: str, severity: str):
        self.probe_name = probe_name
        self.description = description
        self.severity = severity  # "warning" | "critical"
        self.timestamp = datetime.now(timezone.utc)
        super().__init__(f"[{severity}] {probe_name}: {description}")


class AsyncOverseer:
    """Monitors self-improvement for safety violations.

    Runs on a DIFFERENT model than the agent to prevent self-deception.
    Uses Haiku by default (agent uses Sonnet).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        check_interval: float = 30.0,
    ) -> None:
        self.model = model
        self.check_interval = check_interval
        self._running = False
        self._violations: list[SafetyViolation] = []

    @property
    def violations(self) -> list[SafetyViolation]:
        return list(self._violations)

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self._violations)

    async def start(self) -> None:
        """Start the monitoring loop.

        Runs until stop() is called or a critical violation is detected.
        """
        self._running = True
        logger.info("Overseer started (model=%s, interval=%.0fs)", self.model, self.check_interval)

        while self._running:
            try:
                await self._run_all_probes()
            except SafetyViolation as v:
                self._violations.append(v)
                if v.severity == "critical":
                    logger.critical(f"CRITICAL SAFETY VIOLATION: {v}")
                    self._running = False
                    raise
                else:
                    logger.warning(f"Safety warning: {v}")
            except Exception as e:
                logger.debug(f"Overseer probe error (non-fatal): {e}")

            await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        logger.info("Overseer stopped (%d violations recorded)", len(self._violations))

    async def run_once(self) -> list[SafetyViolation]:
        """Run all probes once and return any violations found."""
        new_violations: list[SafetyViolation] = []
        try:
            await self._run_all_probes()
        except SafetyViolation as v:
            new_violations.append(v)
            self._violations.append(v)
        return new_violations

    async def _run_all_probes(self) -> None:
        """Run all safety probes."""
        from belief.safety.probes import (
            check_environment_tampering,
            check_evaluator_integrity,
            check_resource_consumption,
            check_test_harness_edits,
        )

        await check_evaluator_integrity()
        await check_test_harness_edits()
        await check_environment_tampering()
        await check_resource_consumption()
