"""GrinderDaemon: main loop, pause, intervals, budget gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from belief.grinder.daemon import (
    BuildResult,
    GrinderDaemon,
)
from belief.grinder.goal_queue import GoalEnvelope


@pytest.fixture()
def workspace(tmp_path: Path) -> dict[str, Path]:
    pending = tmp_path / "pending"
    pending.mkdir()
    status = tmp_path / "status.json"
    return {"pending": pending, "status": status, "tmp": tmp_path}


def _write_goal(dir: Path, goal_id: str, priority: float = 0.5) -> None:
    (dir / f"{goal_id}.json").write_text(
        json.dumps({"goal_id": goal_id, "title": goal_id, "value": priority})
    )
    (dir / f"{goal_id}.md").write_text(f"# {goal_id}\n")


# ---------------------------------------------------------------------------
# Basic loop
# ---------------------------------------------------------------------------


class _RunRecorder:
    def __init__(self, always_succeed: bool = True) -> None:
        self.calls: list[str] = []
        self.always_succeed = always_succeed

    async def __call__(self, env: GoalEnvelope) -> BuildResult:
        self.calls.append(env.goal_id)
        return BuildResult(
            goal_id=env.goal_id,
            success=self.always_succeed,
            cost_usd=0.01,
            duration_s=0.1,
        )


@pytest.mark.asyncio
async def test_processes_all_goals_up_to_max_builds(workspace) -> None:
    _write_goal(workspace["pending"], "a", priority=0.9)
    _write_goal(workspace["pending"], "b", priority=0.5)
    _write_goal(workspace["pending"], "c", priority=0.2)

    runner = _RunRecorder()
    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=0,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=runner,
    )
    stats = await daemon.run_forever(max_builds=3)
    assert stats.builds_completed == 3
    # Highest priority runs first
    assert runner.calls[0] == "a"


@pytest.mark.asyncio
async def test_failed_build_lands_in_failed_dir(workspace) -> None:
    _write_goal(workspace["pending"], "f", priority=0.5)

    runner = _RunRecorder(always_succeed=False)
    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=0,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=runner,
    )
    await daemon.run_forever(max_builds=1)
    assert daemon.stats.builds_failed == 1
    failed_dir = workspace["pending"].parent / "failed_sessions"
    assert (failed_dir / "f.json").exists()


@pytest.mark.asyncio
async def test_exception_in_runner_marks_build_failed(workspace) -> None:
    _write_goal(workspace["pending"], "boom", priority=0.5)

    async def exploding(env: GoalEnvelope) -> BuildResult:
        raise RuntimeError("pipeline crashed")

    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=0,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=exploding,
    )
    await daemon.run_forever(max_builds=1)
    assert daemon.stats.builds_failed == 1


# ---------------------------------------------------------------------------
# Improvement hooks fire on the right cadence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sica_hook_fires_every_n_builds(workspace) -> None:
    for i in range(6):
        _write_goal(workspace["pending"], f"g{i}", priority=0.5)

    sica_calls: list[int] = []

    async def sica_hook() -> None:
        sica_calls.append(1)

    runner = _RunRecorder()
    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=2,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=runner,
        sica_runner=sica_hook,
    )
    await daemon.run_forever(max_builds=6)
    # Every 2nd build => 3 triggers (builds 2, 4, 6)
    assert len(sica_calls) == 3


@pytest.mark.asyncio
async def test_all_three_hooks_independent(workspace) -> None:
    for i in range(15):
        _write_goal(workspace["pending"], f"h{i}", priority=0.5)

    sica: list[int] = []
    jitter: list[int] = []
    crystal: list[int] = []

    async def mk(bucket: list[int]) -> None:
        bucket.append(1)

    runner = _RunRecorder()
    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=5,
        builds_between_jitterbug=3,
        builds_between_crystallize=15,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=runner,
        sica_runner=lambda: mk(sica),
        jitterbug_runner=lambda: mk(jitter),
        crystallize_runner=lambda: mk(crystal),
    )
    await daemon.run_forever(max_builds=15)
    assert len(sica) == 3  # builds 5, 10, 15
    assert len(jitter) == 5  # builds 3, 6, 9, 12, 15
    assert len(crystal) == 1  # build 15


# ---------------------------------------------------------------------------
# Pause gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_halts_main_loop_without_picking_new_goal(workspace) -> None:
    """While paused the daemon must not invoke the build runner at all.

    After unpause it processes the pending goal. The test uses max_builds=1
    so we don't race with fallback templates once the queue drains.
    """
    _write_goal(workspace["pending"], "p1", priority=0.5)

    paused = [True]
    picks: list[str] = []

    class Recorder:
        async def __call__(self, env: GoalEnvelope) -> BuildResult:
            picks.append(env.goal_id)
            return BuildResult(goal_id=env.goal_id, success=True)

    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=0,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.02,
        inter_build_seconds=0.0,
        build_runner=Recorder(),
    )
    daemon._is_paused = lambda: paused[0]  # type: ignore[method-assign]

    async def unpause_after_a_few_ticks() -> None:
        # Give the loop several paused iterations so we can verify
        # no build runner invocations happened during that window.
        await asyncio.sleep(0.1)
        assert picks == [], "build runner must not fire while paused"
        paused[0] = False

    task = asyncio.create_task(unpause_after_a_few_ticks())
    await daemon.run_forever(max_builds=1)
    await task

    assert daemon.stats.builds_completed == 1
    assert picks == ["p1"]


# ---------------------------------------------------------------------------
# Budget cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_cap_exceeded_pauses_loop(workspace) -> None:
    _write_goal(workspace["pending"], "c1", priority=0.5)

    runner = _RunRecorder()
    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=0,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=runner,
    )
    # Force daily spend above cap
    daemon._daily_spend_exceeds_cap = lambda: True  # type: ignore[method-assign]

    async def stop_soon() -> None:
        await asyncio.sleep(0.1)
        daemon.request_stop()

    task = asyncio.create_task(stop_soon())
    await daemon.run_forever()
    await task
    # No builds should have happened; budget_paused recorded
    assert daemon.stats.builds_completed == 0
    assert daemon.stats.budget_paused is True
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Graceful stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_stop_ends_loop(workspace) -> None:
    for i in range(5):
        _write_goal(workspace["pending"], f"s{i}", priority=0.5)

    runner = _RunRecorder()
    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=0,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=runner,
    )

    async def stop_after(n: int) -> None:
        while daemon.stats.builds_completed < n:
            await asyncio.sleep(0.01)
        daemon.request_stop()

    task = asyncio.create_task(stop_after(2))
    await daemon.run_forever()
    await task
    # Graceful stop — at least 2 builds completed, less than 5 total
    assert daemon.stats.builds_completed >= 2


# ---------------------------------------------------------------------------
# Status file gets persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_file_written_after_build(workspace) -> None:
    _write_goal(workspace["pending"], "s1", priority=0.5)

    runner = _RunRecorder()
    daemon = GrinderDaemon(
        pending_dir=workspace["pending"],
        status_path=workspace["status"],
        builds_between_sica=0,
        builds_between_jitterbug=0,
        builds_between_crystallize=0,
        idle_seconds=0.01,
        inter_build_seconds=0.0,
        build_runner=runner,
    )
    await daemon.run_forever(max_builds=1)

    data = json.loads(workspace["status"].read_text())
    assert data["builds_completed"] == 1
    assert data["last_result"] == "pass"
