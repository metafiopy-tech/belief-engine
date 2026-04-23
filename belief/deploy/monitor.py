"""Health Monitor — Tier 8 Autonomous Operation.

Monitors deployed services and triggers remediation when issues arise:
  1. Periodic health checks (HTTP GET /health)
  2. Error log parsing
  3. Degradation detection (response time, error rate)
  4. Triggers the remediation pipeline when issues are found

Usage:
    monitor = HealthMonitor(url="https://myapp.up.railway.app")
    status = await monitor.check()
    if not status.healthy:
        print(f"Issue: {status.diagnosis}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("belief.deploy.monitor")


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"


@dataclass
class HealthCheck:
    """Result of a single health check."""

    timestamp: str
    status: HealthStatus
    response_time_ms: float = 0.0
    status_code: int = 0
    body: str = ""
    error: str = ""
    diagnosis: str = ""


@dataclass
class MonitorState:
    """Accumulated state from multiple health checks."""

    url: str
    checks: list[HealthCheck] = field(default_factory=list)
    consecutive_failures: int = 0
    last_healthy: str = ""
    error_pattern: str = ""  # Most common error
    needs_remediation: bool = False
    remediation_triggered: int = 0  # How many times remediation was triggered

    @property
    def is_healthy(self) -> bool:
        if not self.checks:
            return True
        return self.checks[-1].status == HealthStatus.HEALTHY

    @property
    def failure_rate(self) -> float:
        if not self.checks:
            return 0.0
        recent = self.checks[-10:]
        failures = sum(1 for c in recent if c.status != HealthStatus.HEALTHY)
        return failures / len(recent)


class HealthMonitor:
    """Monitors a deployed service's health."""

    def __init__(
        self,
        url: str,
        health_path: str = "/health",
        check_interval: int = 30,
        failure_threshold: int = 3,
        timeout: float = 10.0,
    ):
        self.url = url.rstrip("/")
        self.health_path = health_path
        self.health_url = f"{self.url}{health_path}"
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = MonitorState(url=url)
        self._running = False

    async def check_once(self) -> HealthCheck:
        """Run a single health check."""
        import httpx

        t0 = time.time()
        check = HealthCheck(
            timestamp=datetime.now().isoformat(),
            status=HealthStatus.UNREACHABLE,
        )

        try:
            from belief.core.http import get_async_client

            async with get_async_client(timeout=self.timeout) as client:
                resp = await client.get(self.health_url)
                elapsed = (time.time() - t0) * 1000
                check.response_time_ms = elapsed
                check.status_code = resp.status_code
                check.body = resp.text[:500]

                if resp.status_code == 200:
                    check.status = HealthStatus.HEALTHY
                    self.state.consecutive_failures = 0
                    self.state.last_healthy = check.timestamp
                elif resp.status_code < 500:
                    check.status = HealthStatus.DEGRADED
                    check.diagnosis = f"Non-OK response: {resp.status_code}"
                else:
                    check.status = HealthStatus.UNHEALTHY
                    check.diagnosis = f"Server error: {resp.status_code}"
                    check.error = resp.text[:200]

                # Check response time degradation
                if elapsed > 5000 and check.status == HealthStatus.HEALTHY:
                    check.status = HealthStatus.DEGRADED
                    check.diagnosis = f"Slow response: {elapsed:.0f}ms"

        except httpx.ConnectError:
            check.status = HealthStatus.UNREACHABLE
            check.error = "Connection refused"
            check.diagnosis = "Service is not reachable — may be down or not deployed"
        except httpx.TimeoutException:
            check.status = HealthStatus.UNREACHABLE
            check.error = f"Timeout after {self.timeout}s"
            check.diagnosis = "Service did not respond in time — may be overloaded"
        except Exception as e:
            check.status = HealthStatus.UNREACHABLE
            check.error = str(e)
            check.diagnosis = f"Unexpected error: {e}"

        # Update state
        if check.status != HealthStatus.HEALTHY:
            self.state.consecutive_failures += 1
        self.state.checks.append(check)

        # Keep only last 100 checks
        if len(self.state.checks) > 100:
            self.state.checks = self.state.checks[-100:]

        # Determine if remediation is needed
        if self.state.consecutive_failures >= self.failure_threshold:
            self.state.needs_remediation = True
            self.state.error_pattern = check.diagnosis or check.error
            logger.warning(
                f"Monitor: {self.state.consecutive_failures} consecutive failures "
                f"— remediation needed: {self.state.error_pattern[:80]}"
            )

        return check

    async def run_continuous(
        self,
        on_unhealthy=None,
        max_checks: int = 0,
    ):
        """Run continuous health monitoring.

        Args:
            on_unhealthy: async callback(MonitorState) when remediation is needed
            max_checks: stop after N checks (0 = run forever)
        """
        self._running = True
        checks_done = 0

        logger.info(f"Monitor: starting continuous monitoring of {self.health_url}")

        while self._running:
            check = await self.check_once()
            checks_done += 1

            if check.status == HealthStatus.HEALTHY:
                logger.debug(f"Monitor: ✓ {check.response_time_ms:.0f}ms")
            else:
                logger.warning(
                    f"Monitor: ✗ {check.status.value} — {check.diagnosis or check.error}"
                )

            if self.state.needs_remediation and on_unhealthy:
                self.state.needs_remediation = False
                self.state.remediation_triggered += 1
                await on_unhealthy(self.state)

            if max_checks > 0 and checks_done >= max_checks:
                break

            await asyncio.sleep(self.check_interval)

        logger.info(f"Monitor: stopped after {checks_done} checks")

    def stop(self):
        """Stop continuous monitoring."""
        self._running = False

    def summary(self) -> str:
        """Human-readable summary of monitoring state."""
        total = len(self.state.checks)
        healthy = sum(1 for c in self.state.checks if c.status == HealthStatus.HEALTHY)
        rate = (healthy / total * 100) if total > 0 else 0

        avg_ms = 0
        healthy_checks = [c for c in self.state.checks if c.response_time_ms > 0]
        if healthy_checks:
            avg_ms = sum(c.response_time_ms for c in healthy_checks) / len(healthy_checks)

        return (
            f"URL: {self.url}\n"
            f"Status: {'✓ healthy' if self.state.is_healthy else '✗ ' + (self.state.checks[-1].status.value if self.state.checks else 'unknown')}\n"
            f"Uptime: {rate:.0f}% ({healthy}/{total} checks)\n"
            f"Avg response: {avg_ms:.0f}ms\n"
            f"Remediations triggered: {self.state.remediation_triggered}"
        )
