"""GoalQueue: priority ordering, file move on dispatch, fallback templates."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from belief.grinder.goal_queue import (
    GoalQueue,
    render_fallback_goal,
)


@pytest.fixture()
def queue_dirs(tmp_path: Path) -> GoalQueue:
    return GoalQueue(
        pending_dir=tmp_path / "pending",
        completed_dir=tmp_path / "completed",
        failed_dir=tmp_path / "failed",
        rng=random.Random(42),
    )


def _write_sidecar(
    dir: Path, goal_id: str, *, priority: float | None = None,
    build_time: int | None = None, difficulty: int | None = None,
    mtime_offset: float = 0.0,
) -> None:
    dir.mkdir(parents=True, exist_ok=True)
    side: dict = {
        "goal_id": goal_id,
        "title": goal_id,
        "one_paragraph_description": f"goal {goal_id}",
    }
    if priority is not None:
        side["value"] = priority
    if build_time is not None:
        side["estimated_build_time_min"] = build_time
    if difficulty is not None:
        side["estimated_difficulty"] = difficulty
    json_path = dir / f"{goal_id}.json"
    json_path.write_text(json.dumps(side))
    md_path = dir / f"{goal_id}.md"
    md_path.write_text(f"# {goal_id}\n")
    if mtime_offset:
        new_ts = time.time() + mtime_offset
        import os

        os.utime(json_path, (new_ts, new_ts))


def test_pick_next_honors_explicit_value(queue_dirs: GoalQueue) -> None:
    _write_sidecar(queue_dirs.pending_dir, "a", priority=0.40)
    _write_sidecar(queue_dirs.pending_dir, "b", priority=0.80)
    _write_sidecar(queue_dirs.pending_dir, "c", priority=0.60)

    top = queue_dirs.pick_next()
    assert top is not None
    assert top.goal_id == "b"


def test_pick_next_derived_priority_prefers_easy_short(
    queue_dirs: GoalQueue,
) -> None:
    _write_sidecar(queue_dirs.pending_dir, "hard_long", build_time=200, difficulty=5)
    _write_sidecar(queue_dirs.pending_dir, "easy_short", build_time=10, difficulty=1)
    top = queue_dirs.pick_next()
    assert top is not None and top.goal_id == "easy_short"


def test_queue_depth(queue_dirs: GoalQueue) -> None:
    assert queue_dirs.queue_depth() == 0
    for gid in ("a", "b", "c"):
        _write_sidecar(queue_dirs.pending_dir, gid, priority=0.5)
    assert queue_dirs.queue_depth() == 3


def test_mark_completed_moves_both_files(queue_dirs: GoalQueue) -> None:
    _write_sidecar(queue_dirs.pending_dir, "x", priority=0.5)
    env = queue_dirs.pick_next()
    assert env is not None
    queue_dirs.mark_completed(env)
    assert not (queue_dirs.pending_dir / "x.json").exists()
    assert (queue_dirs.completed_dir / "x.json").exists()
    assert (queue_dirs.completed_dir / "x.md").exists()


def test_mark_failed_moves_files(queue_dirs: GoalQueue) -> None:
    _write_sidecar(queue_dirs.pending_dir, "y", priority=0.5)
    env = queue_dirs.pick_next()
    assert env is not None
    queue_dirs.mark_failed(env)
    assert (queue_dirs.failed_dir / "y.json").exists()


def test_fallback_fires_when_queue_empty(queue_dirs: GoalQueue) -> None:
    env = queue_dirs.pick_next()
    assert env is not None
    assert env.source == "fallback"
    assert env.goal_text
    assert env.goal_id.startswith("fallback-")


def test_fallback_move_is_noop_no_crash(queue_dirs: GoalQueue) -> None:
    env = queue_dirs.pick_next()
    assert env is not None and env.source == "fallback"
    # Fallback envelopes aren't file-backed — marking should be safe
    queue_dirs.mark_completed(env)
    queue_dirs.mark_failed(env)


def test_render_fallback_goal_produces_filled_text() -> None:
    text = render_fallback_goal(rng=random.Random(0))
    assert text
    # Template placeholders must be resolved
    assert "{" not in text and "}" not in text


def test_render_fallback_uses_known_templates() -> None:
    seen: set[str] = set()
    for i in range(30):
        t = render_fallback_goal(rng=random.Random(i))
        base = t.split()[1] if len(t.split()) > 1 else t
        # Just confirm each rendered goal starts with "Build"
        assert t.startswith("Build"), t
        seen.add(t)
    assert len(seen) >= 4  # some variety


def test_malformed_json_is_skipped(queue_dirs: GoalQueue) -> None:
    (queue_dirs.pending_dir / "bad.json").write_text("{not valid")
    _write_sidecar(queue_dirs.pending_dir, "good", priority=0.5)
    candidates = queue_dirs.list_pending()
    assert len(candidates) == 1
    assert candidates[0].goal_id == "good"
