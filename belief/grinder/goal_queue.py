"""Goal queue for the Grinder.

Inputs: JSON sidecars written by Photosynthesis's renderer into
`pending_sessions/` (one file per goal). Output: an ordered stream of
GoalEnvelope records, highest-priority first. Built-in fallback
templates keep the daemon fed when Photosynthesis isn't running.

Priority (highest wins):

    1. Explicit 'value' field on the sidecar, if present.
    2. Otherwise a derived priority: a shorter estimated build time and
       lower difficulty rank higher (easy, fast wins build first).
    3. Ties broken by file mtime (oldest first, FIFO).

Files move after dispatch:
    pending_sessions/{id}.{md,json}
      -> completed_sessions/ on success
      -> failed_sessions/   on failure

Idempotent: if the daemon crashes between build and move, the goal
stays in pending and gets retried on next boot.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("belief.grinder.goal_queue")


# ---------------------------------------------------------------------------
# Fallback templates (spec verbatim)
# ---------------------------------------------------------------------------


FALLBACK_GOAL_TEMPLATES: tuple[str, ...] = (
    "Build a FastAPI REST API with SQLite for managing {resource}",
    "Build a Click CLI that {action}",
    "Build a Python module that {function}",
    "Build an MCP server that wraps {api}",
    "Build a WebSocket {app_type} with FastAPI",
    "Build a data pipeline that {pipeline_action}",
)

TEMPLATE_FILLS: dict[str, tuple[str, ...]] = {
    "resource": (
        "books", "recipes", "tasks", "users", "products", "events",
        "invoices", "projects", "tickets", "comments",
    ),
    "action": (
        "converts CSV to JSON",
        "counts words in files",
        "finds duplicate files",
        "generates passwords",
        "monitors a URL and alerts on downtime",
    ),
    "function": (
        "implements a LRU cache",
        "validates email addresses",
        "parses cron expressions",
        "generates UUIDs",
        "implements a rate limiter",
    ),
    "api": (
        "a weather API", "a dictionary API", "a random quote API",
        "a GitHub user lookup", "a currency converter",
    ),
    "app_type": (
        "chat server with rooms", "live dashboard",
        "notification system", "collaborative editor",
    ),
    "pipeline_action": (
        "reads JSON logs and computes error rates",
        "normalizes CSV files from different formats",
        "validates and deduplicates contact records",
    ),
}


def render_fallback_goal(*, rng: random.Random | None = None) -> str:
    """Pick a template and realize it with random fills.

    Pure function — deterministic when a seeded rng is supplied.
    """
    r = rng or random.Random()
    template = r.choice(FALLBACK_GOAL_TEMPLATES)
    fills: dict[str, str] = {}
    for key, choices in TEMPLATE_FILLS.items():
        if "{" + key + "}" in template:
            fills[key] = r.choice(choices)
    return template.format(**fills)


# ---------------------------------------------------------------------------
# GoalEnvelope — the Grinder's view of a pending goal
# ---------------------------------------------------------------------------


@dataclass
class GoalEnvelope:
    goal_id: str
    goal_text: str
    priority: float
    md_path: Optional[Path] = None
    json_path: Optional[Path] = None
    sidecar: dict[str, Any] | None = None
    source: str = "queue"  # 'queue' | 'fallback'

    def is_file_backed(self) -> bool:
        return self.json_path is not None


# ---------------------------------------------------------------------------
# GoalQueue
# ---------------------------------------------------------------------------


class GoalQueue:
    """File-backed queue with fallback-template generation."""

    def __init__(
        self,
        *,
        pending_dir: Path,
        completed_dir: Optional[Path] = None,
        failed_dir: Optional[Path] = None,
        rng: random.Random | None = None,
    ) -> None:
        self.pending_dir = Path(pending_dir)
        self.completed_dir = (
            Path(completed_dir) if completed_dir else self.pending_dir.parent / "completed_sessions"
        )
        self.failed_dir = (
            Path(failed_dir) if failed_dir else self.pending_dir.parent / "failed_sessions"
        )
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self._rng = rng or random.Random()

    # -------------------------------------------------------------- picking
    def list_pending(self) -> list[GoalEnvelope]:
        """All pending goals in priority order (highest first)."""
        out: list[GoalEnvelope] = []
        for json_path in sorted(self.pending_dir.glob("*.json")):
            env = _load_envelope(json_path)
            if env is None:
                continue
            out.append(env)
        out.sort(
            key=lambda e: (
                -e.priority,
                (e.json_path.stat().st_mtime if e.json_path else 0),
            )
        )
        return out

    def queue_depth(self) -> int:
        return len(list(self.pending_dir.glob("*.json")))

    def pick_next(self) -> Optional[GoalEnvelope]:
        """Return the highest-priority pending goal, or a fallback."""
        candidates = self.list_pending()
        if candidates:
            return candidates[0]
        text = render_fallback_goal(rng=self._rng)
        return GoalEnvelope(
            goal_id=_fallback_id(text),
            goal_text=text,
            priority=0.0,
            source="fallback",
        )

    # ------------------------------------------------------------- file move
    def mark_completed(self, env: GoalEnvelope) -> None:
        self._move_pair(env, self.completed_dir)

    def mark_failed(self, env: GoalEnvelope) -> None:
        self._move_pair(env, self.failed_dir)

    def _move_pair(self, env: GoalEnvelope, dest_dir: Path) -> None:
        if not env.is_file_backed():
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in (env.md_path, env.json_path):
            if path is None or not path.exists():
                continue
            target = dest_dir / path.name
            try:
                shutil.move(str(path), str(target))
            except Exception as exc:
                logger.warning("failed to move %s -> %s: %s", path, target, exc)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_envelope(json_path: Path) -> Optional[GoalEnvelope]:
    try:
        raw = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to load %s: %s", json_path, exc)
        return None
    if not isinstance(raw, dict):
        return None

    goal_id = str(raw.get("goal_id") or json_path.stem)
    goal_text = str(
        raw.get("title") or raw.get("one_paragraph_description") or goal_id
    )
    priority = _derive_priority(raw)

    md_path = json_path.with_suffix(".md")
    return GoalEnvelope(
        goal_id=goal_id,
        goal_text=goal_text,
        priority=priority,
        json_path=json_path,
        md_path=md_path if md_path.exists() else None,
        sidecar=raw,
        source="queue",
    )


def _derive_priority(sidecar: dict[str, Any]) -> float:
    """Priority score in [0, 1]: higher means run-first.

    Preference order:
      1. Explicit 'value' float (what Photosynthesis's ranker emits).
      2. Derived from estimated_build_time_min + estimated_difficulty:
         smaller = easier = ships faster = higher priority.
    """
    explicit = sidecar.get("value")
    if isinstance(explicit, (int, float)):
        return max(0.0, min(1.0, float(explicit)))

    build_time = sidecar.get("estimated_build_time_min")
    difficulty = sidecar.get("estimated_difficulty")
    if isinstance(build_time, (int, float)) and isinstance(
        difficulty, (int, float)
    ):
        # Map 5..240 minutes -> 1.0..0.0; 1..5 difficulty -> 1.0..0.0.
        t = 1.0 - max(0.0, min(1.0, (float(build_time) - 5.0) / 235.0))
        d = 1.0 - max(0.0, min(1.0, (float(difficulty) - 1.0) / 4.0))
        return 0.5 * t + 0.5 * d
    return 0.5  # no signal, neutral


def _fallback_id(goal_text: str) -> str:
    """Deterministic-looking id for fallback goals: slug of first 30 chars."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", goal_text.lower()).strip("-")
    return f"fallback-{slug[:30]}"


__all__ = [
    "FALLBACK_GOAL_TEMPLATES",
    "GoalEnvelope",
    "GoalQueue",
    "TEMPLATE_FILLS",
    "render_fallback_goal",
]
