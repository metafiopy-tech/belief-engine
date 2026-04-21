"""Audit log: canonical JSON, hash chain, tamper detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.photosynthesis.safety.audit import (
    AuditLog,
    GENESIS_PREV_HASH,
    canonical_json,
    compute_hash,
)


@pytest.fixture()
def log(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "audit.db")


def test_canonical_json_sorts_keys_and_strips_whitespace() -> None:
    s = canonical_json({"b": 2, "a": 1})
    assert s == '{"a":1,"b":2}'


def test_compute_hash_is_deterministic() -> None:
    h1 = compute_hash('{"a":1}', GENESIS_PREV_HASH)
    h2 = compute_hash('{"a":1}', GENESIS_PREV_HASH)
    assert h1 == h2
    assert len(h1) == 64


def test_append_produces_linked_chain(log: AuditLog) -> None:
    h1 = log.append({"kind": "test", "n": 1})
    h2 = log.append({"kind": "test", "n": 2})
    h3 = log.append({"kind": "test", "n": 3})
    assert h1 != h2 != h3
    assert log.head_hash() == h3
    assert log.count() == 3


def test_verify_clean_chain(log: AuditLog) -> None:
    for i in range(20):
        log.append({"i": i})
    result = log.verify()
    assert result.ok is True
    assert result.break_seq is None
    assert result.reason == "ok"


def test_verify_detects_payload_tamper(log: AuditLog) -> None:
    log.append({"i": 1})
    log.append({"i": 2})
    log.append({"i": 3})

    # Flip a byte in the middle event's payload
    import sqlite3

    c = sqlite3.connect(log.db_path)
    c.execute("UPDATE events SET payload = '{\"i\":20}' WHERE seq = 2;")
    c.commit()
    c.close()

    result = log.verify()
    assert result.ok is False
    assert result.break_seq == 2
    assert "hash mismatch" in result.reason


def test_verify_detects_hash_tamper(log: AuditLog) -> None:
    log.append({"i": 1})
    log.append({"i": 2})

    import sqlite3

    c = sqlite3.connect(log.db_path)
    c.execute("UPDATE events SET hash = 'f' * 64 WHERE seq = 1;")
    c.commit()
    c.close()

    result = log.verify()
    assert result.ok is False
    assert result.break_seq == 1


def test_head_hash_on_empty_log(log: AuditLog) -> None:
    assert log.head_hash() == GENESIS_PREV_HASH
