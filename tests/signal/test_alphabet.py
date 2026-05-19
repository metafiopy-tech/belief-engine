"""Tests for the signal alphabet (mycorrhizal Stage 4, Area 1)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from belief.signal.alphabet import (
    MAX_PAYLOAD_BYTES,
    SIGNAL_TOKENS,
    Signal,
)


def test_alphabet_is_closed_five_tokens() -> None:
    assert set(SIGNAL_TOKENS) == {
        "STRESS",
        "DISCOVER",
        "REQUEST",
        "OFFER",
        "WARN",
    }


def test_signal_round_trip() -> None:
    sig = Signal(agent_id="alice", token="STRESS", magnitude=0.5)
    assert sig.agent_id == "alice"
    assert sig.token == "STRESS"
    assert sig.magnitude == 0.5
    assert sig.timestamp.tzinfo is not None  # auto-set to tz-aware


def test_signal_magnitude_clamping_via_validation() -> None:
    with pytest.raises(ValidationError):
        Signal(agent_id="a", token="STRESS", magnitude=-0.01)
    with pytest.raises(ValidationError):
        Signal(agent_id="a", token="STRESS", magnitude=1.01)


def test_signal_rejects_naive_timestamp() -> None:
    """Ambiguous TZ semantics across systems → reject naive datetimes."""
    naive = datetime(2026, 1, 1, 12, 0, 0)
    with pytest.raises(ValidationError):
        Signal(agent_id="a", token="STRESS", magnitude=0.5, timestamp=naive)


def test_signal_rejects_invalid_token() -> None:
    with pytest.raises(ValidationError):
        Signal(agent_id="a", token="FOO", magnitude=0.5)  # type: ignore[arg-type]


def test_signal_rejects_whitespace_agent_id() -> None:
    with pytest.raises(ValidationError):
        Signal(agent_id="   ", token="STRESS", magnitude=0.5)


def test_payload_size_cap() -> None:
    """Payload exceeding the 200B cap is rejected."""
    big = {"x": "a" * (MAX_PAYLOAD_BYTES * 2)}
    with pytest.raises(ValidationError):
        Signal(agent_id="a", token="STRESS", magnitude=0.1, payload=big)


def test_payload_must_be_json_serializable() -> None:
    class NotJsonable:
        pass

    with pytest.raises(ValidationError):
        Signal(
            agent_id="a",
            token="STRESS",
            magnitude=0.1,
            payload={"x": NotJsonable()},
        )


def test_derived_idempotency_key_is_stable() -> None:
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    a = Signal(agent_id="alice", token="STRESS", magnitude=0.5, timestamp=ts)
    b = Signal(agent_id="alice", token="STRESS", magnitude=0.5, timestamp=ts)
    assert a.derived_idempotency_key() == b.derived_idempotency_key()


def test_explicit_idempotency_key_wins() -> None:
    sig = Signal(
        agent_id="a",
        token="STRESS",
        magnitude=0.1,
        idempotency_key="my-explicit-key",
    )
    assert sig.effective_idempotency_key() == "my-explicit-key"


def test_derived_keys_differ_on_content() -> None:
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    a = Signal(agent_id="alice", token="STRESS", magnitude=0.5, timestamp=ts)
    b = Signal(agent_id="alice", token="REQUEST", magnitude=0.5, timestamp=ts)
    assert a.derived_idempotency_key() != b.derived_idempotency_key()
