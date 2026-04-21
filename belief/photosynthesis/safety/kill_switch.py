"""Kill-switch with three independent trip mechanisms.

1. KILL file on disk (spec default: /run/photosynthesis/KILL). Triggers
   SystemExit — intended for unrecoverable situations where the daemon
   must stop entirely.

2. Control-table status in SQLite: one row keyed by id=1. Status is one
   of 'running' | 'paused' | 'draining'. Anything other than 'running'
   raises KillSwitchTripped and gated callers abort. 'draining' further
   restricts to the 'finalize' and 'log' tags so in-flight work can
   complete without starting new.

3. SIGUSR1/SIGUSR2 in-memory flags. SIGUSR1 -> pause, SIGUSR2 -> resume.
   Useful for ops without touching the DB.

Public surface:

    kill_switch(tag) -> decorator (sync or async)
    KillSwitchState  -- the underlying state machine (injectable for tests)

Session 4 stubbed kill_switch as a pass-through. Session 5 replaces
that with the real state machine. Both decorations still use
@kill_switch(tag="...") so no Session-4 call site needs to change.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import signal
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Optional, TypeVar


logger = logging.getLogger("belief.photosynthesis.safety.kill_switch")


DEFAULT_CONTROL_DB = Path("/var/lib/photosynthesis/control.db")
DEFAULT_KILL_FILE = Path("/run/photosynthesis/KILL")


class ControlStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"


class KillSwitchTripped(RuntimeError):
    """Raised when a tripped switch gates a call."""


DRAINING_ALLOWED_TAGS = frozenset({"finalize", "log"})


SCHEMA = """
CREATE TABLE IF NOT EXISTS control (
    id         INTEGER PRIMARY KEY CHECK(id = 1),
    status     TEXT    NOT NULL CHECK(status IN ('running','paused','draining')),
    reason     TEXT    NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL
);

INSERT OR IGNORE INTO control(id, status, reason, updated_at)
VALUES(1, 'running', 'initial', strftime('%s','now'));
"""


# ---------------------------------------------------------------------------
# KillSwitchState — the concrete state machine
# ---------------------------------------------------------------------------


@dataclass
class KillSwitchState:
    """Holds all three trip mechanisms. Singleton in production.

    Inject a fresh instance in tests to avoid colliding with the
    process-wide default.
    """

    control_db: Path = field(default_factory=lambda: DEFAULT_CONTROL_DB)
    kill_file: Path = field(default_factory=lambda: DEFAULT_KILL_FILE)

    # SIGUSR-driven soft pause, in-memory only
    _paused_in_memory: bool = False
    _signal_handlers_installed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        # Normalize string paths to Path so callers can pass either.
        self.control_db = Path(self.control_db)
        self.kill_file = Path(self.kill_file)
        self.control_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------ SQLite plumbing
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(str(self.control_db), timeout=5.0, isolation_level=None)
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("PRAGMA busy_timeout = 5000;")
            c.row_factory = sqlite3.Row
            yield c
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    def current_status(self) -> ControlStatus:
        with self._conn() as c:
            row = c.execute(
                "SELECT status FROM control WHERE id = 1;"
            ).fetchone()
            if row is None:
                return ControlStatus.RUNNING
            try:
                return ControlStatus(row["status"])
            except ValueError:
                return ControlStatus.RUNNING

    def set_status(self, status: ControlStatus, reason: str = "") -> None:
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "UPDATE control SET status = ?, reason = ?, updated_at = ? "
                "WHERE id = 1;",
                (status.value, reason, int(time.time())),
            )
            c.execute("COMMIT;")

    # ------------------------------------------------------ SIGUSR handlers
    def install_signal_handlers(self) -> None:
        """Wire SIGUSR1 (pause) / SIGUSR2 (resume) to in-memory flag.

        Safe to call once at daemon startup. A second call is a no-op.
        """
        with self._lock:
            if self._signal_handlers_installed:
                return
            try:
                signal.signal(signal.SIGUSR1, self._sigusr1)
                signal.signal(signal.SIGUSR2, self._sigusr2)
                self._signal_handlers_installed = True
            except (AttributeError, ValueError):  # pragma: no cover - Windows
                logger.info("SIGUSR not available on this platform")

    def _sigusr1(self, _signum: int, _frame: Any) -> None:
        logger.info("SIGUSR1 received — pausing kill switch")
        with self._lock:
            self._paused_in_memory = True

    def _sigusr2(self, _signum: int, _frame: Any) -> None:
        logger.info("SIGUSR2 received — resuming kill switch")
        with self._lock:
            self._paused_in_memory = False

    # ------------------------------------------------------ the actual gate
    def check(self, tag: str) -> None:
        """Raise SystemExit on KILL file; KillSwitchTripped on pause/drain."""
        if self.kill_file.exists():
            raise SystemExit(
                f"kill file present at {self.kill_file}; aborting {tag}"
            )

        if self._paused_in_memory:
            raise KillSwitchTripped(
                f"paused by SIGUSR1; tag={tag} blocked"
            )

        status = self.current_status()
        if status is ControlStatus.RUNNING:
            return
        if status is ControlStatus.PAUSED:
            raise KillSwitchTripped(
                f"control table paused; tag={tag} blocked"
            )
        # DRAINING — finalize / log allowed, everything else blocked
        if tag not in DRAINING_ALLOWED_TAGS:
            raise KillSwitchTripped(
                f"draining; tag={tag} blocked (only {sorted(DRAINING_ALLOWED_TAGS)} allowed)"
            )


# ---------------------------------------------------------------------------
# Global singleton + override hook
# ---------------------------------------------------------------------------


_DEFAULT_STATE: Optional[KillSwitchState] = None
_DEFAULT_STATE_LOCK = threading.Lock()


def get_default_state() -> KillSwitchState:
    """Return the process-wide singleton, creating it lazily.

    Tests call `use_state(fresh_instance)` to override; production code
    just imports `kill_switch` and lets it find the singleton.
    """
    global _DEFAULT_STATE
    with _DEFAULT_STATE_LOCK:
        if _DEFAULT_STATE is None:
            _DEFAULT_STATE = KillSwitchState()
        return _DEFAULT_STATE


def use_state(state: KillSwitchState | None) -> None:
    """Override the default state — tests only."""
    global _DEFAULT_STATE
    with _DEFAULT_STATE_LOCK:
        _DEFAULT_STATE = state


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


F = TypeVar("F", bound=Callable[..., Any])


def kill_switch(tag: str) -> Callable[[F], F]:
    """Decorator gating a function on the global kill-switch state.

    Every call consults:
      1. KILL file -> SystemExit
      2. SIGUSR1 soft pause -> KillSwitchTripped
      3. control table status -> KillSwitchTripped (unless draining+allowed)
    """

    def decorator(fn: F) -> F:
        fn.__kill_switch_tag__ = tag  # type: ignore[attr-defined]

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                get_default_state().check(tag)
                return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            get_default_state().check(tag)
            return fn(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "ControlStatus",
    "DEFAULT_CONTROL_DB",
    "DEFAULT_KILL_FILE",
    "DRAINING_ALLOWED_TAGS",
    "KillSwitchState",
    "KillSwitchTripped",
    "get_default_state",
    "kill_switch",
    "use_state",
]
