"""Daemons — background processes that run alongside the build pipeline.

Health: Zero-LLM system health monitoring.
Autonomous: Scheduled tasks (research, self-scan, nightly synthesis).

Source: health.py, autonomous_loop.py
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("belief.daemons")


# ── Health Daemon ─────────────────────────────────────────────────────────────


class HealthDaemon:
    """Zero-LLM system health monitoring.

    Source: health.py + autonomous_loop.py _health_check()

    Runs in a background thread. Checks system health every N seconds.
    Reports issues but never blocks the pipeline.
    """

    def __init__(
        self,
        project_root: Path,
        check_interval: int = 300,  # 5 minutes
        notify_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.project_root = project_root
        self.check_interval = check_interval
        self.notify = notify_fn
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_status: dict = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Health daemon started")

    def stop(self) -> None:
        self._running = False

    def check_now(self) -> dict:
        """Run all health checks immediately. Returns status dict."""
        import os

        issues = []
        checks = {}

        # Check 1: API key present
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        checks["api_key"] = has_key
        if not has_key:
            issues.append("ANTHROPIC_API_KEY not set")

        # Check 2: Core files exist
        core_files = [
            "belief/__init__.py",
            "belief/graph.py",
            "belief/cli.py",
            "belief/llm.py",
        ]
        for f in core_files:
            if not (self.project_root / f).exists():
                issues.append(f"Missing core file: {f}")
                checks[f"file_{f}"] = False
            else:
                checks[f"file_{f}"] = True

        # Check 3: Output directory writable
        try:
            test_file = self.project_root / "output" / ".health_check"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("ok")
            test_file.unlink()
            checks["output_writable"] = True
        except Exception:
            checks["output_writable"] = False
            issues.append("Output directory not writable")

        # Check 4: ChromaDB available (optional)
        try:
            import chromadb  # noqa: F401 — presence check, not used here

            checks["chromadb"] = True
        except ImportError:
            checks["chromadb"] = False
            # Not an issue — it's optional

        # Check 5: Disk space
        import shutil

        usage = shutil.disk_usage(str(self.project_root))
        free_gb = usage.free / (1024**3)
        checks["disk_free_gb"] = round(free_gb, 1)
        if free_gb < 1.0:
            issues.append(f"Low disk space: {free_gb:.1f}GB free")

        self._last_status = {
            "timestamp": datetime.now().isoformat(),
            "healthy": len(issues) == 0,
            "checks": checks,
            "issues": issues,
        }

        if issues and self.notify:
            self.notify("⚠️ Health issues:\n" + "\n".join(f"• {i}" for i in issues))

        return self._last_status

    def _loop(self) -> None:
        while self._running:
            try:
                self.check_now()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            time.sleep(self.check_interval)

    def status(self) -> str:
        if not self._last_status:
            return "Health daemon: no checks run yet"
        s = self._last_status
        status = "✓ healthy" if s["healthy"] else f"✗ {len(s['issues'])} issue(s)"
        return f"Health: {status} (last check: {s['timestamp'][:19]})"


# ── Autonomous Loop ───────────────────────────────────────────────────────────


class AutonomousLoop:
    """Background scheduler for periodic tasks.

    Source: autonomous_loop.py

    Tasks and intervals:
      - health_check: every 2 hours
      - self_scan: every 4 hours (proposes improvements via SEED)
      - nightly_synthesis: every 24 hours (summarize what happened)
    """

    INTERVALS = {
        "health_check": 2 * 3600,
        "self_scan": 4 * 3600,
        "nightly_synthesis": 24 * 3600,
    }

    def __init__(
        self,
        health_daemon: Optional[HealthDaemon] = None,
        seed=None,  # SEED instance
        notify_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.health = health_daemon
        self.seed = seed
        self.notify = notify_fn
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_run: dict[str, float] = {t: 0 for t in self.INTERVALS}
        self._activity_log: list[str] = []

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Initialize — don't fire on first startup
        now = time.time()
        for task in self.INTERVALS:
            if self._last_run[task] == 0:
                self._last_run[task] = now
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Autonomous loop started")

    def stop(self) -> None:
        self._running = False

    def _due(self, task: str) -> bool:
        return (time.time() - self._last_run[task]) >= self.INTERVALS[task]

    def _mark_done(self, task: str) -> None:
        self._last_run[task] = time.time()

    def _loop(self) -> None:
        while self._running:
            try:
                if self._due("health_check") and self.health:
                    self.health.check_now()
                    self._mark_done("health_check")
                    self._activity_log.append(f"{datetime.now():%H:%M} health check")

                if self._due("self_scan") and self.seed:
                    # SEED tick is handled in the build pipeline
                    self._mark_done("self_scan")
                    self._activity_log.append(f"{datetime.now():%H:%M} self-scan")

                if self._due("nightly_synthesis"):
                    self._nightly_synthesis()
                    self._mark_done("nightly_synthesis")
            except Exception as e:
                logger.error(f"Autonomous loop error: {e}")

            time.sleep(300)  # Check every 5 minutes

    def _nightly_synthesis(self) -> None:
        """Summarize the day's activity."""
        if not self._activity_log:
            return

        summary = f"🌙 Nightly — {len(self._activity_log)} activities today"
        if self.notify:
            self.notify(summary)

        self._activity_log.clear()

    def status(self) -> str:
        now = time.time()
        lines = ["Autonomous Loop:"]
        for task, interval in self.INTERVALS.items():
            last = self._last_run.get(task, 0)
            remaining = interval - (now - last)
            if remaining <= 0:
                next_run = "due now"
            else:
                h = int(remaining // 3600)
                m = int((remaining % 3600) // 60)
                next_run = f"in {h}h {m}m" if h else f"in {m}m"
            lines.append(f"  {task}: {next_run}")
        return "\n".join(lines)
