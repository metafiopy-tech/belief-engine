"""Stdlib-only anomaly detectors for cost-series + a watchdog job.

Three independent detectors (any one can fire):

    rolling_zscore_alert(series, window=24, z_thresh=3.0)
        Parametric. Good at symmetric noise, bad at heavy tails.

    mad_alert(series, window=24, k=3.5)
        Robust — median + median absolute deviation. Survives outliers
        in the window itself. This is the primary detector.

    percentile_alert(series, window=168, pct=99)
        "Have we ever seen a value like this in the last week?"

All three return True iff the most recent sample is anomalous relative
to the trailing `window` samples (excluding the current sample from the
baseline so a single huge spike doesn't raise its own baseline).

The watchdog job:
  1. Pulls hourly spend from costs.db for the last 7 days.
  2. Runs all three detectors.
  3. On any alert, flips control status to 'paused' with a reason
     string identifying which detector fired, so operators can
     unambiguously correlate audit log entries.

`run_watchdog` is factored as a pure function of (tracker, state,
audit_sink) so tests can drive it with synthetic series.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


logger = logging.getLogger("belief.photosynthesis.safety.anomaly")


@dataclass
class AnomalyAlert:
    detector: str
    last_value: float
    baseline_stat: str
    baseline_value: float
    threshold: float
    reason: str = ""

    def __str__(self) -> str:
        return (
            f"{self.detector}: last={self.last_value:.4f} "
            f"{self.baseline_stat}={self.baseline_value:.4f} "
            f"threshold={self.threshold:.4f} ({self.reason})"
        )


def _tail(series: Sequence[float], window: int) -> list[float]:
    if window <= 0:
        return []
    return list(series[-window:])


def _baseline_without_last(series: Sequence[float], window: int) -> list[float]:
    """Last `window` samples, excluding the most recent."""
    if len(series) < 2:
        return []
    return list(series[-(window + 1) : -1])


def rolling_zscore_alert(
    series: Sequence[float],
    *,
    window: int = 24,
    z_thresh: float = 3.0,
) -> Optional[AnomalyAlert]:
    """Classic z-score against the trailing window baseline.

    Returns None when the baseline is too small (<3 samples) or has
    zero stdev — in either case the test is meaningless.
    """
    if not series:
        return None
    baseline = _baseline_without_last(series, window)
    if len(baseline) < 3:
        return None
    mu = statistics.fmean(baseline)
    try:
        sigma = statistics.pstdev(baseline)
    except statistics.StatisticsError:
        return None
    if sigma <= 0:
        return None

    last = float(series[-1])
    z = (last - mu) / sigma
    if abs(z) < z_thresh:
        return None
    return AnomalyAlert(
        detector="zscore",
        last_value=last,
        baseline_stat="mean",
        baseline_value=mu,
        threshold=z_thresh,
        reason=f"|z|={abs(z):.2f}",
    )


def mad_alert(
    series: Sequence[float],
    *,
    window: int = 24,
    k: float = 3.5,
) -> Optional[AnomalyAlert]:
    """Median + median absolute deviation — robust to outliers.

    Fires when |x - median| / (1.4826 * MAD) >= k. The 1.4826 factor
    makes MAD comparable to sigma under a normal distribution.
    """
    if not series:
        return None
    baseline = _baseline_without_last(series, window)
    if len(baseline) < 3:
        return None
    med = statistics.median(baseline)
    abs_devs = [abs(x - med) for x in baseline]
    mad = statistics.median(abs_devs)
    if mad <= 0:
        # Every baseline sample was identical — require last to differ at all
        if float(series[-1]) == med:
            return None
        return AnomalyAlert(
            detector="mad",
            last_value=float(series[-1]),
            baseline_stat="median",
            baseline_value=med,
            threshold=k,
            reason="mad=0 + last differs from baseline",
        )

    last = float(series[-1])
    robust_z = abs(last - med) / (1.4826 * mad)
    if robust_z < k:
        return None
    return AnomalyAlert(
        detector="mad",
        last_value=last,
        baseline_stat="median",
        baseline_value=med,
        threshold=k,
        reason=f"robust_z={robust_z:.2f}",
    )


def percentile_alert(
    series: Sequence[float],
    *,
    window: int = 168,
    pct: int = 99,
) -> Optional[AnomalyAlert]:
    """Fire if the last sample exceeds the pct-th percentile of the window."""
    if not series:
        return None
    baseline = _baseline_without_last(series, window)
    if len(baseline) < 10:
        return None
    sorted_b = sorted(baseline)
    # Percentile rank: round down to avoid off-by-one at small N
    idx = max(0, int(len(sorted_b) * pct / 100.0) - 1)
    cutoff = sorted_b[idx]
    last = float(series[-1])
    if last <= cutoff:
        return None
    return AnomalyAlert(
        detector="percentile",
        last_value=last,
        baseline_stat=f"p{pct}",
        baseline_value=cutoff,
        threshold=float(pct),
        reason=f"last > p{pct} of last {len(baseline)} samples",
    )


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


@dataclass
class WatchdogResult:
    alerts: list[AnomalyAlert] = field(default_factory=list)
    flipped_to_paused: bool = False
    series_length: int = 0


def hourly_cost_series(tracker: Any, hours: int = 168) -> list[float]:
    """Bucket the last `hours` hours of calls into hourly cost sums.

    Works with any object exposing a `conn()` context manager yielding a
    sqlite3 connection to the `calls` table — in production that's the
    Session-5 CostTracker. Tests can pass a shim.
    """
    since = int(time.time()) - int(hours * 3600)
    with tracker.conn() as c:
        rows = c.execute(
            "SELECT (ts / 3600) AS hr, COALESCE(SUM(cost), 0.0) AS s "
            "FROM calls WHERE ts >= ? "
            "GROUP BY hr ORDER BY hr ASC;",
            (since,),
        ).fetchall()
    hour_costs = {int(r["hr"]): float(r["s"]) for r in rows}
    now_hr = int(time.time()) // 3600
    start_hr = now_hr - hours + 1
    return [hour_costs.get(h, 0.0) for h in range(start_hr, now_hr + 1)]


def run_watchdog(
    tracker: Any,
    state: Any,
    *,
    audit_sink: Optional[Callable[[dict[str, Any]], None]] = None,
    min_flag_cost: float = 0.10,
) -> WatchdogResult:
    """Evaluate detectors; flip control to paused if any fires.

    `state` is a KillSwitchState (has set_status). `audit_sink` is a
    one-arg callable (or None) that receives an event dict. We keep
    the watchdog ignorant of the audit module's concrete type so tests
    can swap in a list.append-style sink.

    `min_flag_cost` suppresses noise on tiny-value series: if both
    the baseline and the latest sample are below this floor, we never
    alert (prevents 0.001-vs-0.003 z-score storms).
    """
    series = hourly_cost_series(tracker)
    result = WatchdogResult(series_length=len(series))
    if not series:
        return result

    last = series[-1]
    baseline_last_24 = series[-25:-1] if len(series) >= 2 else []
    if last < min_flag_cost and all(x < min_flag_cost for x in baseline_last_24):
        return result

    detectors = (
        mad_alert(series),
        rolling_zscore_alert(series),
        percentile_alert(series),
    )
    result.alerts = [a for a in detectors if a is not None]

    if result.alerts:
        reasons = "; ".join(a.detector for a in result.alerts)
        try:
            from belief.photosynthesis.safety.kill_switch import ControlStatus

            state.set_status(ControlStatus.PAUSED, reason=f"anomaly:{reasons}")
            result.flipped_to_paused = True
        except Exception:
            logger.exception("failed to flip control to paused")
        if audit_sink is not None:
            for alert in result.alerts:
                try:
                    audit_sink({"type": "anomaly", "alert": str(alert)})
                except Exception:
                    logger.exception("audit_sink raised")
    return result


__all__ = [
    "AnomalyAlert",
    "WatchdogResult",
    "hourly_cost_series",
    "mad_alert",
    "percentile_alert",
    "rolling_zscore_alert",
    "run_watchdog",
]
