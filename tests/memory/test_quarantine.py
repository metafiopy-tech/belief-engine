"""Tests for the quarantine collection (mycorrhizal Stage 7, Area 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.memory.quarantine import (
    QuarantineCollection,
    QuarantineStatus,
    is_toxic,
)


@pytest.fixture
def quar(tmp_path: Path) -> QuarantineCollection:
    q = QuarantineCollection(db_path=tmp_path / "quarantine.db")
    yield q
    q.close()


# ── Toxicity detection ───────────────────────────────────────────────────────


def test_is_toxic_detects_security() -> None:
    assert is_toxic(["SQL injection vulnerability found"]) is not None


def test_is_toxic_detects_rm_rf() -> None:
    assert is_toxic([], exec_error="ran rm -rf / by mistake") == "rm -rf"


def test_is_toxic_clean_build_none() -> None:
    assert is_toxic(["ImportError: no module named foo"]) is None


# ── Collection ───────────────────────────────────────────────────────────────


def test_quarantine_and_pending(quar: QuarantineCollection) -> None:
    quar.quarantine("b1", reason="security", evidence={"x": 1})
    pending = quar.pending()
    assert len(pending) == 1
    assert pending[0].build_id == "b1"
    assert pending[0].status is QuarantineStatus.PENDING
    assert quar.is_quarantined("b1") is True


def test_approve_removes_from_pending(quar: QuarantineCollection) -> None:
    quar.quarantine("b1", reason="security")
    assert quar.approve("b1") is True
    assert quar.pending() == []
    # Still in all_items, now approved.
    items = quar.all_items()
    assert len(items) == 1
    assert items[0].status is QuarantineStatus.APPROVED


def test_reject(quar: QuarantineCollection) -> None:
    quar.quarantine("b1", reason="loop")
    assert quar.reject("b1") is True
    assert quar.pending() == []


def test_delete(quar: QuarantineCollection) -> None:
    quar.quarantine("b1", reason="x")
    assert quar.delete("b1") is True
    assert quar.is_quarantined("b1") is False


def test_idempotent_quarantine(quar: QuarantineCollection) -> None:
    quar.quarantine("b1", reason="first")
    quar.quarantine("b1", reason="updated")  # same build_id
    items = quar.all_items()
    assert len(items) == 1
    assert items[0].reason == "updated"


def test_approve_missing_returns_false(quar: QuarantineCollection) -> None:
    assert quar.approve("ghost") is False
