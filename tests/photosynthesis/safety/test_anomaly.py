"""Anomaly detectors + watchdog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from belief.photosynthesis.safety.anomaly import (
    mad_alert,
    percentile_alert,
    rolling_zscore_alert,
    run_watchdog,
)
from belief.photosynthesis.safety.cost_tracker import CostTracker
from belief.photosynthesis.safety.kill_switch import (
    ControlStatus,
    KillSwitchState,
)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def test_zscore_alert_fires_on_spike() -> None:
    # Baseline needs nonzero variance or the test is degenerate (any
    # departure from a constant is "infinitely" significant, and both
    # zscore and MAD short-circuit in that case — see mad_alert's
    # baseline-zero branch).
    series = [0.1 + (i % 3) * 0.01 for i in range(30)] + [5.0]
    a = rolling_zscore_alert(series, window=24, z_thresh=3.0)
    assert a is not None and a.detector == "zscore"


def test_zscore_alert_quiet_on_flat_noise() -> None:
    series = [0.1 + (i % 3) * 0.01 for i in range(50)]
    assert rolling_zscore_alert(series, window=24, z_thresh=3.0) is None


def test_zscore_alert_ignores_small_baseline() -> None:
    assert rolling_zscore_alert([0.1, 5.0]) is None


def test_mad_alert_robust_to_one_outlier_in_baseline() -> None:
    """An outlier in the baseline shouldn't mask a real spike."""
    series = [0.1] * 10 + [10.0] + [0.1] * 19 + [5.0]
    # z-score sees mu=~0.4 (outlier inflates it) and may miss
    # mad sees median=0.1, MAD>0 -> robust z is high
    assert mad_alert(series, window=24, k=3.5) is not None


def test_mad_alert_quiet_on_small_jumps() -> None:
    # Baseline needs nonzero MAD — otherwise any nonzero last value
    # counts as "departs from a constant baseline" and trips the
    # mad=0 branch. That's deliberate and desirable in production
    # (first real spike after a flat-zero startup should fire), so we
    # give this test a realistic noisy baseline.
    series = [0.1 + (i % 4) * 0.02 for i in range(24)] + [0.2]
    assert mad_alert(series, window=24, k=3.5) is None


def test_percentile_alert_requires_baseline() -> None:
    assert percentile_alert([1.0] * 3, window=168, pct=99) is None


def test_percentile_alert_fires_on_new_max() -> None:
    baseline = list(range(20))  # 0..19
    series = baseline + [100.0]
    a = percentile_alert(series, window=20, pct=99)
    assert a is not None


# ---------------------------------------------------------------------------
# Watchdog integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def tracker(tmp_path: Path) -> CostTracker:
    return CostTracker(db_path=tmp_path / "costs.db", daily_cap_usd=100.0)


@pytest.fixture()
def ks_state(tmp_path: Path) -> KillSwitchState:
    return KillSwitchState(control_db=tmp_path / "control.db", kill_file=tmp_path / "KILL")


def test_watchdog_ignores_all_zero_series(tracker: CostTracker, ks_state: KillSwitchState) -> None:
    # No calls -> series is all zeros; watchdog must not flip
    result = run_watchdog(tracker, ks_state)
    assert not result.flipped_to_paused
    assert ks_state.current_status() is ControlStatus.RUNNING


def test_watchdog_flips_to_paused_on_mad_spike(
    tracker: CostTracker, ks_state: KillSwitchState
) -> None:
    """Craft a series directly in the calls table that will trip MAD."""
    import time as _t

    now = int(_t.time())
    # 48 background hourly spend points at ~$0.02 each (all in distinct hours)
    with tracker.conn() as c:
        for i in range(48):
            c.execute(
                "INSERT INTO calls(ts, model, in_tok, out_tok, cache_r, cache_w, cost, tag) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?);",
                (
                    now - (48 - i) * 3600,
                    "claude-haiku-4-5-20251001",
                    0,
                    0,
                    0,
                    0,
                    0.02,
                    "background",
                ),
            )
        # Huge spike in the most recent hour
        c.execute(
            "INSERT INTO calls(ts, model, in_tok, out_tok, cache_r, cache_w, cost, tag) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?);",
            (
                now - 30,
                "claude-haiku-4-5-20251001",
                0,
                0,
                0,
                0,
                5.00,
                "spike",
            ),
        )

    sink_events: list[dict[str, Any]] = []
    result = run_watchdog(tracker, ks_state, audit_sink=sink_events.append)
    assert result.flipped_to_paused is True
    assert any(a.detector in {"mad", "zscore", "percentile"} for a in result.alerts)
    assert ks_state.current_status() is ControlStatus.PAUSED
    assert sink_events and sink_events[0]["type"] == "anomaly"
