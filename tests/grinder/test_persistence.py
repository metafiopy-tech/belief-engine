"""Tests for Grinder restart/persistence — Session 8.5c.

The audit called out a gap: the file-backed GoalQueue must survive a
process boundary.  If the daemon crashes mid-build, the next process
that starts should see the queue exactly as it was — no in-memory
state to lose, no "ghost" completed entries, no missing goals.

These tests exercise that contract by creating one queue, mutating
it, discarding the instance, creating a fresh instance against the
same directories, and asserting the state is consistent.
"""

from __future__ import annotations

import json
from pathlib import Path


from belief.grinder.goal_queue import GoalQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_goal(pending_dir: Path, goal_id: str, text: str, priority: float = 0.5) -> None:
    """Write a minimal pending goal pair (.json + .md) into ``pending_dir``.

    Field names match what :func:`belief.grinder.goal_queue._load_envelope`
    actually reads:

    * ``title`` → populates ``GoalEnvelope.goal_text`` (falls back to goal_id).
    * ``value`` → populates ``GoalEnvelope.priority`` via ``_derive_priority``
      (falls back to 0.5 neutral if absent).
    """
    pending_dir.mkdir(parents=True, exist_ok=True)
    (pending_dir / f"{goal_id}.md").write_text(f"# {text}\n")
    (pending_dir / f"{goal_id}.json").write_text(
        json.dumps({"goal_id": goal_id, "title": text, "value": priority})
    )


# ---------------------------------------------------------------------------
# Restart persistence
# ---------------------------------------------------------------------------


class TestRestartPersistence:
    def test_pending_goals_survive_fresh_queue_instance(self, tmp_path: Path) -> None:
        """A fresh GoalQueue on the same pending_dir sees everything
        the previous instance wrote.  File-backed queue state is the
        source of truth; nothing in RAM."""
        pending = tmp_path / "pending"
        _write_goal(pending, "g-a", "Build a JSON schema validator", priority=2.0)
        _write_goal(pending, "g-b", "Build a CLI progress bar", priority=1.0)

        # First queue instance
        q1 = GoalQueue(pending_dir=pending)
        assert q1.queue_depth() == 2

        # Simulate process restart — discard q1, create q2
        del q1
        q2 = GoalQueue(pending_dir=pending)
        assert q2.queue_depth() == 2
        ids = {e.goal_id for e in q2.list_pending()}
        assert ids == {"g-a", "g-b"}

    def test_priority_order_stable_across_restart(self, tmp_path: Path) -> None:
        pending = tmp_path / "pending"
        _write_goal(pending, "low", "low-priority goal", priority=0.1)
        _write_goal(pending, "high", "high-priority goal", priority=0.9)

        q1 = GoalQueue(pending_dir=pending)
        pick1 = q1.pick_next()
        assert pick1 is not None and pick1.goal_id == "high"

        # Fresh process shouldn't see a different order
        del q1
        q2 = GoalQueue(pending_dir=pending)
        pick2 = q2.pick_next()
        assert pick2 is not None and pick2.goal_id == "high"

    def test_completion_survives_restart(self, tmp_path: Path) -> None:
        """Mark completed in one instance, confirm in a fresh instance
        that the goal is gone from pending and present in completed."""
        pending = tmp_path / "pending"
        completed = tmp_path / "completed"
        _write_goal(pending, "g-1", "a goal")

        q1 = GoalQueue(pending_dir=pending, completed_dir=completed)
        env = q1.list_pending()[0]
        q1.mark_completed(env)

        del q1
        q2 = GoalQueue(pending_dir=pending, completed_dir=completed)
        assert q2.queue_depth() == 0
        assert (completed / "g-1.json").exists()
        assert (completed / "g-1.md").exists()

    def test_failure_survives_restart(self, tmp_path: Path) -> None:
        pending = tmp_path / "pending"
        failed = tmp_path / "failed"
        _write_goal(pending, "g-bad", "a goal that fails")

        q1 = GoalQueue(pending_dir=pending, failed_dir=failed)
        env = q1.list_pending()[0]
        q1.mark_failed(env)

        del q1
        q2 = GoalQueue(pending_dir=pending, failed_dir=failed)
        assert q2.queue_depth() == 0
        assert (failed / "g-bad.json").exists()


# ---------------------------------------------------------------------------
# Resume-after-crash simulation
# ---------------------------------------------------------------------------


class TestResumeAfterCrash:
    def test_partially_processed_goal_stays_pending(self, tmp_path: Path) -> None:
        """If a process picks a goal but crashes before mark_completed
        or mark_failed, the goal must remain in pending for the next
        restart to pick up.  (`pick_next` is pure-read.)"""
        pending = tmp_path / "pending"
        _write_goal(pending, "g-pick", "picked but not moved")

        q1 = GoalQueue(pending_dir=pending)
        env = q1.pick_next()
        assert env is not None

        # Simulate crash — no mark_completed / mark_failed call
        del q1
        q2 = GoalQueue(pending_dir=pending)
        ids = {e.goal_id for e in q2.list_pending()}
        assert "g-pick" in ids
        assert q2.queue_depth() == 1

    def test_fallback_only_fires_when_pending_empty(self, tmp_path: Path) -> None:
        """The fallback-goal generator must never fire when there's
        real work queued — otherwise crash-recovery would leak real
        goals into fallback territory."""
        pending = tmp_path / "pending"
        _write_goal(pending, "real", "a real queued goal")

        q = GoalQueue(pending_dir=pending)
        env = q.pick_next()
        assert env is not None
        assert env.source == "queue"
        assert env.goal_id == "real"

    def test_fallback_fires_on_empty_queue(self, tmp_path: Path) -> None:
        """With no pending goals, pick_next produces a fallback so the
        daemon always has something to do on resume (keeps the build
        cadence going even if the last crash drained the queue)."""
        pending = tmp_path / "pending"
        pending.mkdir()

        q = GoalQueue(pending_dir=pending)
        env = q.pick_next()
        assert env is not None
        assert env.source == "fallback"
        assert env.goal_text.startswith("Build")


# ---------------------------------------------------------------------------
# Malformed envelope tolerance (resilience, not persistence per se)
# ---------------------------------------------------------------------------


class TestMalformedEnvelope:
    def test_unreadable_json_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """A corrupted JSON sidecar must not prevent the queue from
        loading the other envelopes — otherwise a single bad file
        could lock the daemon out of all its work."""
        pending = tmp_path / "pending"
        _write_goal(pending, "g-good", "good goal")
        pending.mkdir(parents=True, exist_ok=True)
        (pending / "g-bad.json").write_text("{not json")
        (pending / "g-bad.md").write_text("# corrupt\n")

        q = GoalQueue(pending_dir=pending)
        ids = {e.goal_id for e in q.list_pending()}
        assert "g-good" in ids
        assert "g-bad" not in ids  # skipped, not fatal
