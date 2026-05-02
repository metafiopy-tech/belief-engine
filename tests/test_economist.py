"""Hermetic tests for belief.ecology.economist (v3.3 Session 1).

All tests use tmp_path for state + audit paths so nothing touches the real
~/.belief-engine/ tree. Run with:

    python3 -m pytest tests/test_economist.py -q --timeout=60
"""

from __future__ import annotations

import json
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from belief.ecology.economist import (
    DEFAULT_DAILY_BUDGET_USD,
    Economist,
    QuoteRejected,
    cli_reset,
    cli_show,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def econ(tmp_path: Path) -> Economist:
    """Fresh Economist with $5/day budget against tmp_path state + audit."""
    return Economist(
        daily_budget_usd=5.0,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
    )


def _read_audit(econ: Economist) -> list[dict]:
    """Parse every JSONL line from the audit file."""
    if not econ.audit_path.exists():
        return []
    out = []
    with open(econ.audit_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ── 1. Basic quote behavior ────────────────────────────────────────────────


def test_quote_within_budget_is_approved(econ: Economist) -> None:
    q = econ.quote("test_action", 1.50)
    assert q.approved is True
    assert q.action == "test_action"
    assert q.estimated_usd == 1.50
    assert q.remaining_after == pytest.approx(3.50)
    assert "headroom" in q.reason


def test_quote_over_budget_rejected_with_reason(econ: Economist) -> None:
    q = econ.quote("expensive", 10.0)
    assert q.approved is False
    assert "exceed" in q.reason.lower()
    assert q.remaining_after < 0


def test_quote_at_exact_budget_edge_is_approved(econ: Economist) -> None:
    q = econ.quote("edge", 5.0)
    assert q.approved is True
    assert q.remaining_after == pytest.approx(0.0)


def test_quote_zero_dollars_always_approved(econ: Economist) -> None:
    # Zero-cost organs (Predator, GC) should always sail through.
    q = econ.quote("predator.run", 0.0)
    assert q.approved is True
    assert q.remaining_after == pytest.approx(5.0)


def test_quote_negative_estimate_raises(econ: Economist) -> None:
    with pytest.raises(ValueError):
        econ.quote("bad", -1.0)


# ── 2. Commit + remaining ──────────────────────────────────────────────────


def test_commit_reduces_remaining(econ: Economist) -> None:
    assert econ.remaining() == pytest.approx(5.0)
    econ.commit("a", 2.0)
    assert econ.remaining() == pytest.approx(3.0)


def test_multiple_commits_accumulate(econ: Economist) -> None:
    econ.commit("a", 1.0)
    econ.commit("b", 0.5)
    econ.commit("c", 0.25)
    assert econ.remaining() == pytest.approx(3.25)


def test_commit_negative_raises(econ: Economist) -> None:
    with pytest.raises(ValueError):
        econ.commit("a", -0.5)


def test_commit_smaller_than_estimate_reflects_actual(econ: Economist) -> None:
    # Caller estimates $2 but only spent $0.50 — remaining tracks actual.
    q = econ.quote("a", 2.0)
    assert q.approved
    econ.commit("a", 0.5)
    assert econ.remaining() == pytest.approx(4.5)


def test_commit_or_raise_blocks_overspend(econ: Economist) -> None:
    econ.commit("warmup", 4.0)  # $1 left
    with pytest.raises(QuoteRejected) as exc:
        econ.commit_or_raise("over", 2.0)
    assert exc.value.quote.approved is False
    # State unchanged because commit never happened.
    assert econ.remaining() == pytest.approx(1.0)


# ── 3. Daily reset ─────────────────────────────────────────────────────────


def test_daily_reset_zeroes_spend_at_utc_rollover(econ: Economist) -> None:
    # Spend $3 today, then advance the clock 24h.
    econ.commit("today", 3.0)
    assert econ.remaining() == pytest.approx(2.0)

    # Manually rewrite the state's date to yesterday.
    state = json.loads(econ.state_path.read_text())
    yesterday = (
        datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1)
    ).strftime("%Y-%m-%d")
    state["date_utc"] = yesterday
    econ.state_path.write_text(json.dumps(state))

    # Next call should detect rollover and zero spend.
    assert econ.remaining() == pytest.approx(5.0)


def test_daily_reset_preserves_audit_history(econ: Economist) -> None:
    econ.commit("a", 1.0)
    pre_lines = len(_read_audit(econ))
    # Force a rollover.
    state = json.loads(econ.state_path.read_text())
    state["date_utc"] = "1999-01-01"
    econ.state_path.write_text(json.dumps(state))
    econ.remaining()  # triggers reset
    post_lines = _read_audit(econ)
    # Audit appended a daily_reset event, did not lose anything.
    assert len(post_lines) >= pre_lines + 1
    assert any(e.get("event") == "daily_reset" for e in post_lines)


# ── 4. State persistence + crash safety ───────────────────────────────────


def test_missing_state_file_means_fresh_state(tmp_path: Path) -> None:
    econ = Economist(
        daily_budget_usd=5.0,
        state_path=tmp_path / "nonexistent.json",
        audit_path=tmp_path / "audit.jsonl",
    )
    assert econ.remaining() == pytest.approx(5.0)


def test_malformed_state_file_falls_back_to_defaults(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{not valid json at all")
    econ = Economist(
        daily_budget_usd=5.0,
        state_path=state_path,
        audit_path=tmp_path / "audit.jsonl",
    )
    # No exception, behaves as fresh.
    assert econ.remaining() == pytest.approx(5.0)


def test_state_file_missing_required_keys_falls_back(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"unrelated": "shape"}))
    econ = Economist(
        daily_budget_usd=5.0,
        state_path=state_path,
        audit_path=tmp_path / "audit.jsonl",
    )
    assert econ.remaining() == pytest.approx(5.0)


def test_state_writes_are_atomic(econ: Economist) -> None:
    # After a commit, the .tmp file should not exist (atomic rename completed).
    econ.commit("a", 1.0)
    tmp_artifacts = list(econ.state_path.parent.glob("state.json.tmp"))
    assert tmp_artifacts == []


def test_crash_midwrite_does_not_corrupt_state(econ: Economist, monkeypatch) -> None:
    """If os.replace fails mid-commit, state remains the prior valid value.

    We simulate by patching os.replace to raise on the second call. The
    first commit writes a valid state; the second commit's atomic rename
    fails, but the prior file is still readable and the Economist
    recovers without double-charging or corrupting.
    """
    econ.commit("first", 1.0)
    pre_state = json.loads(econ.state_path.read_text())

    real_replace = os.replace
    call_count = {"n": 0}

    def flaky_replace(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated crash mid-rename")
        return real_replace(src, dst)

    monkeypatch.setattr("belief.ecology.economist.os.replace", flaky_replace)
    with pytest.raises(OSError):
        econ.commit("second", 1.0)

    # Restore real replace; state file should still be the post-first-commit
    # value (no corruption, no double-charge).
    monkeypatch.setattr("belief.ecology.economist.os.replace", real_replace)
    post_state = json.loads(econ.state_path.read_text())
    assert post_state == pre_state


# ── 5. Audit log ───────────────────────────────────────────────────────────


def test_audit_log_contains_quote_and_commit_events(econ: Economist) -> None:
    econ.quote("a", 1.0)
    econ.commit("a", 1.0)
    events = _read_audit(econ)
    types = [e["event"] for e in events]
    assert "quote" in types
    assert "commit" in types


def test_audit_entries_have_required_shape(econ: Economist) -> None:
    econ.commit("test", 0.5)
    events = _read_audit(econ)
    commit_events = [e for e in events if e["event"] == "commit"]
    assert len(commit_events) == 1
    e = commit_events[0]
    for key in (
        "ts",
        "event",
        "action",
        "actual_usd",
        "spent_before",
        "spent_after",
        "remaining",
        "over_budget",
    ):
        assert key in e, f"missing audit key {key!r}"
    # Timestamp parses as ISO 8601.
    datetime.fromisoformat(e["ts"])


def test_audit_log_rotates_when_oversize(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    econ = Economist(
        daily_budget_usd=5.0,
        state_path=tmp_path / "state.json",
        audit_path=audit_path,
        audit_rotate_bytes=200,  # tiny so we trip rotation fast
        audit_keep=2,
    )
    # Commit enough times to push the file past 200 bytes.
    for i in range(20):
        econ.commit("x", 0.001)
    rotated = audit_path.with_suffix(audit_path.suffix + ".1")
    assert rotated.exists(), "expected .1 rotation file to exist"


def test_audit_failure_does_not_crash_commit(econ: Economist, monkeypatch) -> None:
    """If audit write fails, commit must still succeed (audit is best-effort)."""

    def boom(*a, **kw):
        raise OSError("disk full")

    # Patch the open() inside _audit by patching the whole method's writer path.
    # Simpler: make audit_path point to a path whose parent can't be created.
    monkeypatch.setattr(
        "belief.ecology.economist.json.dumps",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("encoding broke")),
    )
    # commit still updates state even though audit blows up.
    # Need to reload state_path manually since json.dumps is also used by atomic write.
    # Actually our atomic write uses json.dump, not json.dumps — so monkeypatch only hits audit.
    econ.commit("a", 0.5)
    assert econ.remaining() == pytest.approx(4.5)


# ── 6. Concurrency ─────────────────────────────────────────────────────────


def _concurrent_commit_worker(state_path: str, audit_path: str, action: str, amount: float) -> None:
    """Top-level so multiprocessing can pickle it."""
    from belief.ecology.economist import Economist as E

    e = E(
        daily_budget_usd=5.0,
        state_path=Path(state_path),
        audit_path=Path(audit_path),
    )
    e.commit(action, amount)


def test_concurrent_commits_serialize_through_lock(tmp_path: Path) -> None:
    """Two processes committing $1 each must result in $2 total spent.

    Without the file lock, the read-modify-write race would let one commit
    overwrite the other and we'd see only $1.
    """
    state_path = tmp_path / "state.json"
    audit_path = tmp_path / "audit.jsonl"
    procs = []
    for i in range(4):
        p = multiprocessing.Process(
            target=_concurrent_commit_worker,
            args=(str(state_path), str(audit_path), f"a{i}", 0.5),
        )
        procs.append(p)
        p.start()
    for p in procs:
        p.join(timeout=20)
        assert not p.is_alive(), "concurrent commit worker hung"

    state = json.loads(state_path.read_text())
    assert state["spent_usd"] == pytest.approx(2.0), (
        f"expected $2.00 after 4×$0.50 commits, got ${state['spent_usd']:.4f}"
    )
    assert state["commits"] == 4


# ── 7. CLI helpers ─────────────────────────────────────────────────────────


def test_cli_show_contains_key_fields(tmp_path: Path, monkeypatch) -> None:
    # Point Economist at tmp paths via the CLI helpers' default constructor.
    monkeypatch.setattr(
        "belief.ecology.economist._DEFAULT_STATE_PATH",
        tmp_path / "state.json",
    )
    monkeypatch.setattr(
        "belief.ecology.economist._DEFAULT_AUDIT_PATH",
        tmp_path / "audit.jsonl",
    )
    out = cli_show(daily_budget_usd=3.50)
    assert "Economist" in out
    assert "3.50" in out
    assert "spent today" in out
    assert "remaining" in out


def test_cli_reset_clears_today_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "belief.ecology.economist._DEFAULT_STATE_PATH",
        tmp_path / "state.json",
    )
    monkeypatch.setattr(
        "belief.ecology.economist._DEFAULT_AUDIT_PATH",
        tmp_path / "audit.jsonl",
    )
    # Spend something via the default-paths Economist.
    Economist(
        daily_budget_usd=5.0,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
    ).commit("seed", 2.0)
    out = cli_reset(daily_budget_usd=5.0)
    assert "reset" in out.lower()
    # Verify state is zero but audit retains both the commit and the reset.
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["spent_usd"] == pytest.approx(0.0)
    audit = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    events = [json.loads(line)["event"] for line in audit]
    assert "commit" in events
    assert "manual_reset" in events


# ── 8. Misc ────────────────────────────────────────────────────────────────


def test_status_round_trips_through_json(econ: Economist) -> None:
    econ.commit("a", 1.0)
    s = econ.status()
    encoded = json.dumps(s)
    decoded = json.loads(encoded)
    assert decoded["spent_usd"] == pytest.approx(1.0)
    assert decoded["daily_budget_usd"] == pytest.approx(5.0)


def test_negative_daily_budget_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Economist(
            daily_budget_usd=-1.0,
            state_path=tmp_path / "state.json",
            audit_path=tmp_path / "audit.jsonl",
        )


def test_default_budget_constant_is_five_dollars() -> None:
    # Pin the contract — Joe accepted $5/day in §6.
    assert DEFAULT_DAILY_BUDGET_USD == 5.0
