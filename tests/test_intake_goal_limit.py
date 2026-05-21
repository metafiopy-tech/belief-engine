"""Hermetic tests for the intake goal-length guard.

The old hard cap (``goal[:2000]``) silently amputated legitimate
multi-deliverable goals before the planner saw them. The guard is now a
generous, named limit that leaves real specs untouched and only trims (loudly)
pathologically long input.
"""

from __future__ import annotations

from belief.agents.intake import MAX_GOAL_CHARS, enforce_goal_limit


def test_normal_goal_passes_through_untouched() -> None:
    # The capsule goal that triggered this fix was ~3K chars — well under the
    # limit. It must come back byte-identical with no warning.
    goal = "Build a handheld idea-capsule voice recorder. " * 60  # ~2.8K chars
    assert len(goal) < MAX_GOAL_CHARS
    out, warning = enforce_goal_limit(goal)
    assert out == goal
    assert warning is None


def test_goal_at_limit_is_untouched() -> None:
    goal = "x" * MAX_GOAL_CHARS
    out, warning = enforce_goal_limit(goal)
    assert out == goal
    assert warning is None


def test_oversized_goal_is_trimmed_and_warns() -> None:
    goal = "y" * (MAX_GOAL_CHARS + 5000)
    out, warning = enforce_goal_limit(goal)
    assert len(out) == MAX_GOAL_CHARS
    assert warning is not None
    # Warning names both the original and clamped sizes so it isn't a silent cut.
    assert str(MAX_GOAL_CHARS) in warning
    assert str(MAX_GOAL_CHARS + 5000) in warning


def test_limit_is_generous_enough_for_multi_deliverable_specs() -> None:
    # Regression guard: the limit must comfortably exceed the 3061-char goal
    # that was being amputated at 2000.
    assert MAX_GOAL_CHARS >= 10000
