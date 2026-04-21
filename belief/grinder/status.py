"""Grinder status file: atomic writer + reader.

The daemon writes a snapshot after every state transition (start,
goal picked, build completed, pause, shutdown). The CLI's
`belief grinder status` command reads it and formats for human output.

Atomic write pattern: write to `.status.tmp`, then `os.replace` onto
the final path. Crash-safe on all modern filesystems.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


DEFAULT_STATUS_PATH = Path("/var/lib/grinder/status.json")


@dataclass
class GrinderStatus:
    """Snapshot of the Grinder's current state."""

    state: str = "idle"  # idle | building | paused | stopping | stopped
    builds_completed: int = 0
    builds_failed: int = 0
    current_goal_id: str = ""
    current_goal_text: str = ""
    queue_depth: int = 0
    last_result: str = ""            # 'pass' | 'fail' | ''
    last_cost_usd: float = 0.0
    last_duration_s: float = 0.0
    started_at: float = 0.0          # unix seconds; 0 == never started
    updated_at: float = field(default_factory=lambda: float(time.time()))

    @property
    def uptime_seconds(self) -> float:
        if self.started_at == 0:
            return 0.0
        return max(0.0, self.updated_at - self.started_at)


# ---------------------------------------------------------------------------
# Atomic IO
# ---------------------------------------------------------------------------


def write_status(status: GrinderStatus, *, path: Path = DEFAULT_STATUS_PATH) -> None:
    """Atomically persist the status snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status.updated_at = time.time()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(status), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_status(*, path: Path = DEFAULT_STATUS_PATH) -> Optional[GrinderStatus]:
    """Load the last-persisted status, or None if the file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Tolerate unknown keys (forward compat) and missing ones (defaults).
    known = set(GrinderStatus.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in known}
    try:
        return GrinderStatus(**filtered)
    except (TypeError, ValueError):
        return None


def format_status(status: Optional[GrinderStatus]) -> str:
    """Human-readable status line. Returns a note when None."""
    if status is None:
        return "grinder: no status file found (daemon hasn't started?)"
    lines = [
        f"grinder: {status.state}",
        f"  builds: {status.builds_completed} completed, "
        f"{status.builds_failed} failed, "
        f"queue depth {status.queue_depth}",
    ]
    if status.current_goal_id:
        lines.append(f"  current: {status.current_goal_id}")
        if status.current_goal_text:
            short = status.current_goal_text[:80]
            lines.append(f"           {short}")
    if status.last_result:
        lines.append(
            f"  last:   {status.last_result} "
            f"(${status.last_cost_usd:.4f} / {status.last_duration_s:.1f}s)"
        )
    if status.started_at:
        lines.append(f"  uptime: {status.uptime_seconds:.0f}s")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_STATUS_PATH",
    "GrinderStatus",
    "format_status",
    "read_status",
    "write_status",
]
