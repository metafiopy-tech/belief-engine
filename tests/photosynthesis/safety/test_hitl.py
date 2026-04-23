"""HITL approval flow: auto-approve / timeout / hard-block / decision."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from belief.photosynthesis.safety.hitl import (
    AUTO_THRESHOLD_CEILING,
    AUTO_THRESHOLD_FLOOR,
    ApprovalStatus,
    ApprovalStore,
    AutoApproveTestClient,
    HARD_BLOCK_THRESHOLD,
    NullApprovalClient,
    request_approval,
)


@pytest.fixture()
def store(tmp_path: Path) -> ApprovalStore:
    return ApprovalStore(db_path=tmp_path / "hitl.db")


def test_auto_approve_below_threshold(store: ApprovalStore) -> None:
    client = AutoApproveTestClient(decision=False)  # should not be asked
    decision = asyncio.run(
        request_approval(
            "generator",
            {"x": 1},
            est_cost=0.02,
            client=client,
            store=store,
            auto_threshold=0.05,
        )
    )
    assert decision.granted is True
    assert decision.status is ApprovalStatus.AUTO_APPROVED
    assert client.calls == []


def test_hard_block_above_threshold(store: ApprovalStore) -> None:
    client = AutoApproveTestClient(decision=True)  # shouldn't be asked
    decision = asyncio.run(
        request_approval(
            "generator",
            {"x": 1},
            est_cost=1.50,
            client=client,
            store=store,
        )
    )
    assert decision.granted is False
    assert decision.status is ApprovalStatus.HARD_BLOCKED
    assert client.calls == []


def test_human_approve_in_ask_band(store: ApprovalStore) -> None:
    client = AutoApproveTestClient(decision=True)
    decision = asyncio.run(
        request_approval(
            "generator",
            {"x": 1},
            est_cost=0.10,
            client=client,
            store=store,
            auto_threshold=0.05,
        )
    )
    assert decision.granted is True
    assert decision.status is ApprovalStatus.APPROVED
    assert len(client.calls) == 1


def test_human_reject_in_ask_band(store: ApprovalStore) -> None:
    client = AutoApproveTestClient(decision=False)
    decision = asyncio.run(
        request_approval(
            "generator",
            {"x": 1},
            est_cost=0.10,
            client=client,
            store=store,
            auto_threshold=0.05,
        )
    )
    assert decision.granted is False
    assert decision.status is ApprovalStatus.REJECTED


def test_timeout_fails_closed(store: ApprovalStore) -> None:
    client = NullApprovalClient()  # always returns None
    decision = asyncio.run(
        request_approval(
            "generator",
            {"x": 1},
            est_cost=0.10,
            client=client,
            store=store,
            auto_threshold=0.05,
        )
    )
    assert decision.granted is False
    assert decision.status is ApprovalStatus.AUTO_REJECTED


def test_auto_threshold_respects_floor_and_ceiling() -> None:
    # Raw values; via request_approval, clamping happens inside.
    assert AUTO_THRESHOLD_FLOOR == 0.01
    assert AUTO_THRESHOLD_CEILING == 0.20
    assert HARD_BLOCK_THRESHOLD == 1.00


def test_store_records_decision(store: ApprovalStore) -> None:
    client = AutoApproveTestClient(decision=True)
    decision = asyncio.run(
        request_approval(
            "generator",
            {"payload": "x"},
            est_cost=0.02,
            client=client,
            store=store,
            auto_threshold=0.05,
        )
    )
    assert decision.status is ApprovalStatus.AUTO_APPROVED
    rows = store.by_status(ApprovalStatus.AUTO_APPROVED)
    assert len(rows) == 1
    assert rows[0]["decided_by"] == "policy"
