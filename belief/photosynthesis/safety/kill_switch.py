"""Kill-switch stub — Session 5 replaces with control-table-backed real impl.

The decorator wraps a callable; at call time it consults a per-tag
switch. If the switch is tripped, it raises KillSwitchTripped and the
caller aborts.

Session 4 stub behavior: always allow. Session 5 will:
  - Read from a SQLite control_table(tag PK, state, reason, updated_at),
  - Support both hard-trip (raise) and soft-trip (log + continue) modes,
  - Accept a WebHook for admin overrides.

The public surface (decorator signature + exception class) is stable.
"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, TypeVar


class KillSwitchTripped(RuntimeError):
    """Raised when a tripped kill switch gates a call."""


F = TypeVar("F", bound=Callable[..., Any])


def kill_switch(tag: str) -> Callable[[F], F]:
    """Decorator that gates a function on a named kill switch.

    Session 5 reads switch state from a control table. Session 4 stub:
    every call passes through unchanged (tag is recorded but ignored).
    """

    def decorator(fn: F) -> F:
        # Attach the tag for introspection; useful in tests.
        fn.__kill_switch_tag__ = tag  # type: ignore[attr-defined]

        if _is_coroutine_function(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Session 5: consult control table here.
                return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Session 5: consult control table here.
            return fn(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _is_coroutine_function(fn: Callable[..., Any]) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)


__all__ = ["KillSwitchTripped", "kill_switch"]
