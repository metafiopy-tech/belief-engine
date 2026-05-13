"""GrinderDaemon — the continuous build loop.

Main loop shape:

    while not stopping:
        pick_next_goal()         -> GoalEnvelope | None
        if None:
            sleep(idle_seconds)
            continue
        result = await build(goal)
        mark_completed/failed(goal)
        if count % 10 == 0:  run_sica()
        if count % 20 == 0:  run_jitterbug()
        if count % 15 == 0:  run_crystallization()

Design choices:
  - Pause/resume uses the kill-switch control table ('grinder' tag)
    that Session 5 already ships. `belief grinder pause` flips to
    PAUSED; the daemon sees it before picking the next goal.
  - Graceful stop: a SIGTERM sets `self._stop_requested`. The loop
    finishes the current build (if any) then returns cleanly.
  - Budget enforcement: per-build cap via BuildBudget (existing
    hardening.py, unchanged). Daily cap via CostTracker from Session 5.
    Local-mode calls are $0 and bypass both caps.
  - Injectable collaborators make the daemon testable without a real
    Anthropic/Ollama/benchmark loop:
        build_runner(envelope) -> BuildResult
        sica_runner() / jitterbug_runner() / crystallize_runner()
        ollama_probe() -> bool
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from belief.grinder.goal_queue import GoalEnvelope, GoalQueue
from belief.grinder.status import (
    DEFAULT_STATUS_PATH,
    GrinderStatus,
    write_status,
)


logger = logging.getLogger("belief.grinder.daemon")


DEFAULT_PENDING_DIR = Path("pending_sessions")
DEFAULT_IDLE_SECONDS = 60
DEFAULT_INTER_BUILD_SECONDS = 5


@dataclass
class BuildResult:
    goal_id: str
    success: bool
    cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str = ""


BuildRunner = Callable[[GoalEnvelope], Awaitable[BuildResult]]
ImprovementRunner = Callable[[], Awaitable[None]]


@dataclass
class DaemonStats:
    builds_completed: int = 0
    builds_failed: int = 0
    sica_triggered: int = 0
    jitterbug_triggered: int = 0
    crystallize_triggered: int = 0
    budget_paused: bool = False


class GrinderDaemon:
    """See module docstring."""

    def __init__(
        self,
        *,
        pending_dir: Path = DEFAULT_PENDING_DIR,
        status_path: Path = DEFAULT_STATUS_PATH,
        model_mode: str = "hybrid",
        max_cost_per_build: float = 1.0,
        daily_cost_cap: float = 10.0,
        builds_between_sica: int = 10,
        builds_between_jitterbug: int = 20,
        builds_between_crystallize: int = 15,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        inter_build_seconds: float = DEFAULT_INTER_BUILD_SECONDS,
        build_runner: Optional[BuildRunner] = None,
        sica_runner: Optional[ImprovementRunner] = None,
        jitterbug_runner: Optional[ImprovementRunner] = None,
        crystallize_runner: Optional[ImprovementRunner] = None,
    ) -> None:
        self.pending_dir = Path(pending_dir)
        self.status_path = Path(status_path)
        self.model_mode = model_mode
        self.max_cost_per_build = float(max_cost_per_build)
        self.daily_cost_cap = float(daily_cost_cap)
        self.builds_between_sica = int(builds_between_sica)
        self.builds_between_jitterbug = int(builds_between_jitterbug)
        self.builds_between_crystallize = int(builds_between_crystallize)
        self.idle_seconds = float(idle_seconds)
        self.inter_build_seconds = float(inter_build_seconds)

        self.queue = GoalQueue(pending_dir=self.pending_dir)

        self.build_runner = build_runner
        self.sica_runner = sica_runner
        self.jitterbug_runner = jitterbug_runner
        self.crystallize_runner = crystallize_runner

        self.stats = DaemonStats()
        self._stop_requested = False
        self._current_goal: Optional[GoalEnvelope] = None
        self._started_at = 0.0
        # Last completed build's result carried across iterations so the
        # status file doesn't lose it when persisted at loop top/bottom.
        self._last_result: str = ""
        self._last_cost_usd: float = 0.0
        self._last_duration_s: float = 0.0

    # ---------------------------------------------------------------- control
    def request_stop(self) -> None:
        """Ask the loop to exit at the next safe boundary."""
        self._stop_requested = True

    def install_signal_handlers(self) -> None:
        def _stop(_signum: int, _frame: Any) -> None:
            logger.info("signal received, requesting stop")
            self.request_stop()

        try:
            signal.signal(signal.SIGTERM, _stop)
            signal.signal(signal.SIGINT, _stop)
        except (AttributeError, ValueError):  # pragma: no cover - Windows
            pass

    def _is_paused(self) -> bool:
        """Consult the kill-switch control table for 'grinder' tag.

        Returns True if the daemon should hold off on picking a new goal.
        Kept local so tests can override without touching the global
        KillSwitchState singleton.
        """
        try:
            from belief.photosynthesis.safety.kill_switch import (
                ControlStatus,
                get_default_state,
            )

            status = get_default_state().current_status()
        except Exception:
            return False
        return status is ControlStatus.PAUSED

    def _daily_spend_exceeds_cap(self) -> bool:
        """Consult CostTracker daily total vs configured cap.

        Cloud/hybrid mode only — local mode has $0 cost and we don't
        gate on the daily cap there. If CostTracker isn't available
        (tests) we return False.
        """
        if self.model_mode == "local" or self.daily_cost_cap <= 0:
            return False
        try:
            from belief.photosynthesis.safety.cost_tracker import CostTracker

            spent = CostTracker().spent("1 day")
        except Exception:
            return False
        return spent >= self.daily_cost_cap

    # ---------------------------------------------------------------- status
    def _current_status(self) -> GrinderStatus:
        s = GrinderStatus(
            state="stopping"
            if self._stop_requested
            else (
                "paused" if self._is_paused() else ("building" if self._current_goal else "idle")
            ),
            builds_completed=self.stats.builds_completed,
            builds_failed=self.stats.builds_failed,
            current_goal_id=self._current_goal.goal_id if self._current_goal else "",
            current_goal_text=(self._current_goal.goal_text if self._current_goal else ""),
            queue_depth=self.queue.queue_depth(),
            started_at=self._started_at,
            last_result=self._last_result,
            last_cost_usd=self._last_cost_usd,
            last_duration_s=self._last_duration_s,
        )
        return s

    def _persist_status(self, **overrides: Any) -> None:
        s = self._current_status()
        for k, v in overrides.items():
            if hasattr(s, k):
                setattr(s, k, v)
        write_status(s, path=self.status_path)

    # ---------------------------------------------------------------- loop
    async def run_forever(self, *, max_builds: Optional[int] = None) -> DaemonStats:
        """Main loop. Returns the final stats snapshot.

        `max_builds` caps the number of builds in one invocation (used
        by tests and `belief grinder start --max-builds N`). None = run
        until stopped by a signal or pause.
        """
        self._started_at = time.time()
        self._persist_status()
        logger.info(
            "grinder started: mode=%s pending_dir=%s max_builds=%s",
            self.model_mode,
            self.pending_dir,
            max_builds,
        )

        while not self._stop_requested:
            # max_builds caps TOTAL processed so a run of only-failures
            # still exits. Equivalently: stop after N dispatch slots.
            total_processed = self.stats.builds_completed + self.stats.builds_failed
            if max_builds is not None and total_processed >= max_builds:
                break

            if self._is_paused():
                self._persist_status()
                await asyncio.sleep(self.idle_seconds)
                continue

            if self._daily_spend_exceeds_cap():
                self.stats.budget_paused = True
                logger.warning("daily cost cap hit; pausing")
                self._persist_status()
                await asyncio.sleep(self.idle_seconds)
                continue
            else:
                self.stats.budget_paused = False

            envelope = self.queue.pick_next()
            if envelope is None:
                self._persist_status()
                await asyncio.sleep(self.idle_seconds)
                continue

            self._current_goal = envelope
            self._persist_status()
            result = await self._build(envelope)
            self._current_goal = None

            if result.success:
                self.queue.mark_completed(envelope)
                self.stats.builds_completed += 1
            else:
                self.queue.mark_failed(envelope)
                self.stats.builds_failed += 1

            # Snapshot the result onto the daemon so subsequent
            # status persists (including the one right after the
            # max_builds break) carry these fields.
            self._last_result = "pass" if result.success else "fail"
            self._last_cost_usd = float(result.cost_usd)
            self._last_duration_s = float(result.duration_s)

            total = self.stats.builds_completed + self.stats.builds_failed
            self._persist_status()

            # Periodic improvement cycles — fire on every-Nth completion.
            # Base on total (successes + failures) so a run of failures
            # still triggers SICA's self-improvement cycle.
            if self.builds_between_sica > 0 and total > 0 and total % self.builds_between_sica == 0:
                await self._run_improvement("sica", self.sica_runner)
                self.stats.sica_triggered += 1
            if (
                self.builds_between_jitterbug > 0
                and total > 0
                and total % self.builds_between_jitterbug == 0
            ):
                await self._run_improvement("jitterbug", self.jitterbug_runner)
                self.stats.jitterbug_triggered += 1
            if (
                self.builds_between_crystallize > 0
                and total > 0
                and total % self.builds_between_crystallize == 0
            ):
                await self._run_improvement("crystallize", self.crystallize_runner)
                self.stats.crystallize_triggered += 1

            await asyncio.sleep(self.inter_build_seconds)

        # Final status
        self._persist_status()
        logger.info(
            "grinder stopped: %d completed, %d failed",
            self.stats.builds_completed,
            self.stats.builds_failed,
        )
        return self.stats

    # ---------------------------------------------------------------- build
    async def _build(self, envelope: GoalEnvelope) -> BuildResult:
        """Dispatch one build. Catches exceptions so the loop survives."""
        start = time.monotonic()
        runner = self.build_runner or _default_build_runner
        try:
            result = await runner(envelope)
            return BuildResult(
                goal_id=envelope.goal_id,
                success=bool(result.success),
                cost_usd=float(result.cost_usd),
                duration_s=float(result.duration_s or (time.monotonic() - start)),
                error=result.error,
            )
        except Exception as exc:
            logger.exception("build crashed for %s", envelope.goal_id)
            return BuildResult(
                goal_id=envelope.goal_id,
                success=False,
                cost_usd=0.0,
                duration_s=time.monotonic() - start,
                error=str(exc),
            )

    async def _run_improvement(self, name: str, runner: Optional[ImprovementRunner]) -> None:
        if runner is None:
            logger.info("%s hook not wired; skipping", name)
            return
        try:
            await runner()
        except Exception:
            logger.exception("%s hook raised", name)


# ---------------------------------------------------------------------------
# Default build runner — wraps the real build graph
# ---------------------------------------------------------------------------


async def _default_build_runner(envelope: GoalEnvelope) -> BuildResult:
    """Run the real pipeline. Keep thin — tests always inject a fake."""
    from belief.graph import build_pipeline
    from belief.photosynthesis.synthesis.sidecar_loader import (
        extract_structural_mechanism,
    )

    t0 = time.monotonic()
    # SE Session 7.5: hydrate structural_mechanism from the sidecar so
    # the IntakeAgent can route the build through the cross-domain
    # adapter and surface predicate / relations / probes as constraints.
    initial_state: dict[str, object] = {"user_goal": envelope.goal_text}
    mechanism = extract_structural_mechanism(envelope.sidecar)
    if mechanism is not None:
        initial_state["structural_mechanism"] = mechanism
        logger.info(
            "build %s: hydrated structural_mechanism from sidecar (%d open probes)",
            envelope.goal_id,
            len(mechanism.incompleteness_probes_open),
        )
    try:
        graph = build_pipeline()
        out = await graph.ainvoke(initial_state)
    except Exception as exc:
        return BuildResult(
            goal_id=envelope.goal_id,
            success=False,
            cost_usd=0.0,
            duration_s=time.monotonic() - t0,
            error=str(exc),
        )

    phase = str(out.get("phase", "") or "")
    success = phase == "complete"
    cost_usd = 0.0
    # BuildBudget (hardening.py) attaches to state — pull the total if present.
    budget = out.get("budget")
    if budget is not None and hasattr(budget, "spent_usd"):
        cost_usd = float(budget.spent_usd)
    return BuildResult(
        goal_id=envelope.goal_id,
        success=success,
        cost_usd=cost_usd,
        duration_s=time.monotonic() - t0,
        error="" if success else f"phase={phase}",
    )


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - invoked by systemd
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    import os

    daemon = GrinderDaemon(
        model_mode=os.environ.get("BELIEF_MODEL_MODE", "hybrid"),
    )
    daemon.install_signal_handlers()
    asyncio.run(daemon.run_forever())


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "BuildResult",
    "BuildRunner",
    "DaemonStats",
    "DEFAULT_IDLE_SECONDS",
    "DEFAULT_INTER_BUILD_SECONDS",
    "DEFAULT_PENDING_DIR",
    "GrinderDaemon",
    "ImprovementRunner",
    "main",
]
