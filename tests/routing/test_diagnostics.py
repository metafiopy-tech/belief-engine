"""Tests for TopologyDiagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.routing._store import RoutingStore
from belief.routing.diagnostics import (
    DEFAULT_OVERCENTRALIZATION_THRESHOLD,
    TopologyDiagnostics,
    cli_format_report,
)


@pytest.fixture
def store(tmp_path: Path) -> RoutingStore:
    s = RoutingStore(db_path=tmp_path / "routing.db")
    yield s
    s.close()


def test_empty_report(store: RoutingStore) -> None:
    diag = TopologyDiagnostics(store)
    report = diag.report()
    assert report.event_count == 0
    assert report.node_count == 0
    assert report.overcentralized is False
    out = cli_format_report(report)
    assert "no routing events" in out


def test_direct_events_one_hop(store: RoutingStore) -> None:
    for i in range(5):
        store.record_event(agent_id=f"a{i}", decision_kind="direct", hub_id=None)
    report = TopologyDiagnostics(store).report()
    assert report.direct_count == 5
    assert report.mean_path_length == pytest.approx(1.0)
    assert report.via_hub_count == 0


def test_via_hub_two_hops(store: RoutingStore) -> None:
    # 3 peripheral agents all route via hub 'H'.
    for i in range(3):
        store.record_event(agent_id=f"a{i}", decision_kind="via_hub", hub_id="H")
    report = TopologyDiagnostics(store).report()
    assert report.via_hub_count == 3
    assert report.mean_path_length == pytest.approx(2.0)
    assert "H" in report.hub_ids


def test_cache_hit_one_hop(store: RoutingStore) -> None:
    store.record_event(agent_id="a", decision_kind="cache_hit", hub_id="H")
    report = TopologyDiagnostics(store).report()
    assert report.cache_hit_count == 1
    assert report.mean_path_length == pytest.approx(1.0)


def test_overcentralization_flag(store: RoutingStore) -> None:
    """Top-3 hubs carrying > threshold of hub edges raises the flag.

    Here a single hub carries 100% of hub edges → over-centralised."""
    for i in range(10):
        store.record_event(agent_id=f"a{i}", decision_kind="via_hub", hub_id="DOMINANT")
    report = TopologyDiagnostics(store).report()
    assert report.top3_hub_edge_share == pytest.approx(1.0)
    assert report.overcentralized is True


def test_balanced_hubs_not_overcentralized(store: RoutingStore) -> None:
    """Edges spread across many hubs → top-3 share below threshold."""
    # 10 hubs, one edge each → top-3 share = 3/10 = 0.3 < 0.7.
    for i in range(10):
        store.record_event(agent_id=f"a{i}", decision_kind="via_hub", hub_id=f"hub{i}")
    report = TopologyDiagnostics(store).report()
    assert report.top3_hub_edge_share == pytest.approx(0.3)
    assert report.overcentralized is False


def test_mixed_decisions_path_length(store: RoutingStore) -> None:
    # 2 direct (1 hop) + 2 via_hub (2 hops) → mean = (1+1+2+2)/4 = 1.5
    store.record_event(agent_id="a", decision_kind="direct", hub_id=None)
    store.record_event(agent_id="b", decision_kind="direct", hub_id=None)
    store.record_event(agent_id="c", decision_kind="via_hub", hub_id="H")
    store.record_event(agent_id="d", decision_kind="via_hub", hub_id="H")
    report = TopologyDiagnostics(store).report()
    assert report.mean_path_length == pytest.approx(1.5)


def test_window_filters_old_events(store: RoutingStore) -> None:
    from datetime import datetime, timedelta, timezone

    old = datetime.now(timezone.utc) - timedelta(days=30)
    store.record_event(agent_id="old", decision_kind="direct", hub_id=None, ts=old)
    store.record_event(agent_id="recent", decision_kind="direct", hub_id=None)
    # 7-day window excludes the 30-day-old event.
    report = TopologyDiagnostics(store).report(window="7d")
    assert report.event_count == 1


def test_threshold_is_sane() -> None:
    assert 0.0 < DEFAULT_OVERCENTRALIZATION_THRESHOLD <= 1.0


def test_to_dict_round_trip(store: RoutingStore) -> None:
    store.record_event(agent_id="a", decision_kind="via_hub", hub_id="H")
    report = TopologyDiagnostics(store).report()
    d = report.to_dict()
    assert d["via_hub_count"] == 1
    assert "H" in d["hub_ids"]
