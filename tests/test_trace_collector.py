"""Tests for belief.metrics.trace_collector."""

from __future__ import annotations

import csv
import sqlite3
import threading
from pathlib import Path

import pytest

from belief.metrics.trace_collector import (
    OUTPUT_SUMMARY_LIMIT,
    StepTrace,
    TraceCollector,
    is_tracing_enabled,
    record_step_from_state,
    set_default_collector,
)


@pytest.fixture()
def collector(tmp_path: Path) -> TraceCollector:
    c = TraceCollector(tmp_path / "traces.db")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# StepTrace normalization
# ---------------------------------------------------------------------------


def test_step_trace_truncates_long_output() -> None:
    trace = StepTrace(
        build_id="b1",
        step_index=0,
        agent_name="builder",
        output_summary="x" * 2000,
    )
    assert len(trace.output_summary) == OUTPUT_SUMMARY_LIMIT


def test_step_trace_preserves_short_output() -> None:
    trace = StepTrace(
        build_id="b1", step_index=0, agent_name="builder",
        output_summary="hello",
    )
    assert trace.output_summary == "hello"


# ---------------------------------------------------------------------------
# record_step + async write
# ---------------------------------------------------------------------------


def test_record_step_persists_after_close(collector: TraceCollector) -> None:
    trace = StepTrace(build_id="b1", step_index=0, agent_name="builder")
    assert collector.record_step(trace) is True
    collector.close()  # flushes writer
    assert collector.row_count() == 1


def test_record_step_closed_collector_returns_false(
    collector: TraceCollector,
) -> None:
    collector.close()
    trace = StepTrace(build_id="b1", step_index=0, agent_name="builder")
    assert collector.record_step(trace) is False


def test_concurrent_writes_all_persisted(tmp_path: Path) -> None:
    c = TraceCollector(tmp_path / "traces.db")

    def worker(build: str, n: int) -> None:
        for i in range(n):
            c.record_step(
                StepTrace(build_id=build, step_index=i, agent_name="builder")
            )

    threads = [
        threading.Thread(target=worker, args=(f"b{i}", 20)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c.close()
    assert c.row_count() == 4 * 20


def test_close_is_idempotent(collector: TraceCollector) -> None:
    collector.record_step(StepTrace(build_id="b1", step_index=0, agent_name="x"))
    collector.close()
    collector.close()  # must not raise
    assert collector.row_count() == 1


# ---------------------------------------------------------------------------
# finalize_build
# ---------------------------------------------------------------------------


def test_finalize_build_marks_all_steps(collector: TraceCollector) -> None:
    for i in range(5):
        collector.record_step(
            StepTrace(build_id="b1", step_index=i, agent_name="builder")
        )
    collector.record_step(
        StepTrace(build_id="b2", step_index=0, agent_name="tester")
    )
    updated = collector.finalize_build("b1", passed=True)
    assert updated == 5

    # Second call is a no-op — all rows already finalized
    again = collector.finalize_build("b1", passed=True)
    assert again == 0

    # b2 untouched
    c = sqlite3.connect(str(collector.db_path))
    row = c.execute(
        "SELECT build_passed FROM traces WHERE build_id = 'b2';"
    ).fetchone()
    c.close()
    assert row[0] is None


def test_finalize_build_records_failure(collector: TraceCollector) -> None:
    collector.record_step(StepTrace(build_id="b3", step_index=0, agent_name="x"))
    updated = collector.finalize_build("b3", passed=False)
    assert updated == 1
    c = sqlite3.connect(str(collector.db_path))
    row = c.execute(
        "SELECT build_passed FROM traces WHERE build_id = 'b3';"
    ).fetchone()
    c.close()
    assert row[0] == 0


# ---------------------------------------------------------------------------
# get_training_data
# ---------------------------------------------------------------------------


def test_training_data_respects_min_builds(collector: TraceCollector) -> None:
    # Only 2 finalized builds, min_builds=50 -> empty
    for b in ("b1", "b2"):
        collector.record_step(StepTrace(build_id=b, step_index=0, agent_name="a"))
        collector.finalize_build(b, passed=True)
    assert collector.get_training_data(min_builds=50) == []


def test_training_data_returns_finalized_rows(collector: TraceCollector) -> None:
    for i in range(3):
        collector.record_step(
            StepTrace(build_id=f"b{i}", step_index=0, agent_name="builder")
        )
        collector.finalize_build(f"b{i}", passed=(i % 2 == 0))
    data = collector.get_training_data(min_builds=1)
    assert len(data) == 3
    assert all("build_passed" in r for r in data)
    assert all(isinstance(r["build_passed"], bool) for r in data)


def test_training_data_excludes_unfinalized(collector: TraceCollector) -> None:
    collector.record_step(StepTrace(build_id="b1", step_index=0, agent_name="a"))
    collector.record_step(StepTrace(build_id="b1", step_index=1, agent_name="b"))
    collector.finalize_build("b1", passed=True)

    # Unfinalized build
    collector.record_step(StepTrace(build_id="b2", step_index=0, agent_name="a"))

    data = collector.get_training_data(min_builds=1)
    assert len(data) == 2
    assert all(r["build_id"] == "b1" for r in data)


def test_build_count_only_counts_finalized(collector: TraceCollector) -> None:
    collector.record_step(StepTrace(build_id="b1", step_index=0, agent_name="a"))
    collector.record_step(StepTrace(build_id="b2", step_index=0, agent_name="a"))
    collector.finalize_build("b1", passed=True)
    collector.close()
    assert collector.build_count() == 1


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def test_export_produces_csv_with_header_only_when_empty(
    collector: TraceCollector, tmp_path: Path,
) -> None:
    out = tmp_path / "probe_data.csv"
    count = collector.export_for_probe_training(out)
    assert count == 0
    content = out.read_text().strip()
    # Header-only file
    assert content.count("\n") == 0
    assert "build_id" in content


def test_export_writes_rows(
    collector: TraceCollector, tmp_path: Path,
) -> None:
    for i in range(3):
        collector.record_step(
            StepTrace(build_id="b1", step_index=i, agent_name="builder")
        )
    collector.finalize_build("b1", passed=True)
    out = tmp_path / "probe_data.csv"
    count = collector.export_for_probe_training(out)
    assert count == 3
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3
    assert rows[0]["agent_name"] == "builder"


# ---------------------------------------------------------------------------
# record_step_from_state helper
# ---------------------------------------------------------------------------


def test_record_step_from_state_populates_build_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = TraceCollector(tmp_path / "traces.db")
    set_default_collector(c)
    try:
        state: dict = {"iteration": 2}
        record_step_from_state(state, agent_name="builder")
        # build_id auto-assigned
        assert state["build_id"].startswith("b-")
        # step_index incremented
        assert state["_step_index"] == 1
        # A second call uses the same build_id and advances step_index
        record_step_from_state(state, agent_name="tester")
        assert state["_step_index"] == 2
        c.close()
        assert c.row_count() == 2
    finally:
        set_default_collector(None)


def test_record_step_from_state_swallows_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Passing a non-dict state should NOT raise.
    record_step_from_state("not a dict", agent_name="builder")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# is_tracing_enabled
# ---------------------------------------------------------------------------


def test_tracing_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BELIEF_ENABLE_TRACE", raising=False)
    assert is_tracing_enabled() is False


def test_tracing_enabled_on_truthy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("1", "true", "TRUE", "on", "yes"):
        monkeypatch.setenv("BELIEF_ENABLE_TRACE", val)
        assert is_tracing_enabled() is True


def test_tracing_disabled_on_falsy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("0", "false", "", "no"):
        monkeypatch.setenv("BELIEF_ENABLE_TRACE", val)
        assert is_tracing_enabled() is False
