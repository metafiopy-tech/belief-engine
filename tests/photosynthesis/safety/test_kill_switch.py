"""Kill-switch: KILL file, control table, SIGUSR flag all trip independently."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from belief.photosynthesis.safety.kill_switch import (
    ControlStatus,
    KillSwitchState,
    KillSwitchTripped,
    kill_switch,
    use_state,
)


@pytest.fixture()
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KillSwitchState:
    """Fresh KillSwitchState per test with tmp_path paths."""
    s = KillSwitchState(
        control_db=tmp_path / "control.db",
        kill_file=tmp_path / "KILL",
    )
    use_state(s)
    yield s
    use_state(None)


def test_default_state_is_running(state: KillSwitchState) -> None:
    assert state.current_status() is ControlStatus.RUNNING


def test_kill_file_raises_systemexit(state: KillSwitchState) -> None:
    state.kill_file.write_text("stop")

    @kill_switch(tag="work")
    def do_work() -> str:
        return "done"

    with pytest.raises(SystemExit):
        do_work()


def test_paused_blocks_all_tags(state: KillSwitchState) -> None:
    state.set_status(ControlStatus.PAUSED, reason="test")

    @kill_switch(tag="work")
    def do_work() -> str:
        return "done"

    with pytest.raises(KillSwitchTripped):
        do_work()


def test_draining_blocks_except_finalize(state: KillSwitchState) -> None:
    state.set_status(ControlStatus.DRAINING, reason="test")

    @kill_switch(tag="work")
    def do_work() -> None: ...

    @kill_switch(tag="finalize")
    def do_final() -> str:
        return "ok"

    @kill_switch(tag="log")
    def do_log() -> str:
        return "ok"

    with pytest.raises(KillSwitchTripped):
        do_work()
    assert do_final() == "ok"
    assert do_log() == "ok"


def test_sigusr1_in_memory_pause(state: KillSwitchState) -> None:
    # Simulate a SIGUSR1 without actually installing a handler
    state._paused_in_memory = True

    @kill_switch(tag="work")
    def do_work() -> None: ...

    with pytest.raises(KillSwitchTripped):
        do_work()

    state._paused_in_memory = False
    # Now it passes
    do_work()


def test_async_decorator_is_gated(state: KillSwitchState) -> None:
    import asyncio

    state.set_status(ControlStatus.PAUSED, reason="test")

    @kill_switch(tag="work")
    async def do_work() -> str:
        return "done"

    with pytest.raises(KillSwitchTripped):
        asyncio.run(do_work())


def test_tag_is_attached_to_function(state: KillSwitchState) -> None:
    @kill_switch(tag="synthesis")
    def f() -> None: ...

    assert getattr(f, "__kill_switch_tag__") == "synthesis"


def test_set_status_via_enum_and_string_roundtrip(state: KillSwitchState) -> None:
    state.set_status(ControlStatus.PAUSED, reason="test")
    assert state.current_status() is ControlStatus.PAUSED
    state.set_status(ControlStatus.RUNNING, reason="resumed")
    assert state.current_status() is ControlStatus.RUNNING
