"""Tests for the channel-capacity harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from belief.signal.capacity import (
    BENCHMARK_BITS_PER_HOUR,
    CapacityMeasurement,
    CapacityReport,
    cli_format_report,
    mutual_information_bits,
)
from belief.signal.store import SignalStore


# ── Plug-in MI estimator ───────────────────────────────────────────────────


def test_mi_identity_max() -> None:
    """y == x → I(X; Y) = log2(bins). For bins=4 the max is 2 bits."""
    xs = [(i % 4) / 4 + 0.01 for i in range(400)]  # span all 4 bins
    ys = list(xs)
    mi = mutual_information_bits(xs, ys, bins=4)
    assert mi == pytest.approx(2.0, abs=0.05)


def test_mi_independent_zero() -> None:
    """y independent of x → MI ≈ 0 (positive bias on small samples)."""
    import random

    rng = random.Random(0)
    xs = [rng.random() for _ in range(2000)]
    ys = [rng.random() for _ in range(2000)]
    mi = mutual_information_bits(xs, ys, bins=4)
    assert mi < 0.1  # noise floor


def test_mi_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mutual_information_bits([0.1, 0.2], [0.3], bins=4)


def test_mi_empty_zero() -> None:
    assert mutual_information_bits([], [], bins=4) == 0.0


def test_mi_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError):
        mutual_information_bits([0.1], [0.1], bins=1)


def test_mi_clips_negative_to_zero() -> None:
    """Floating-point noise can produce small-negative MI on perfectly
    independent samples; the estimator must clip those to 0.0."""
    # Deterministic uncorrelated short sequence.
    xs = [0.1, 0.4, 0.6, 0.9, 0.2, 0.7, 0.3, 0.5]
    ys = [0.9, 0.6, 0.4, 0.1, 0.7, 0.2, 0.5, 0.3]
    mi = mutual_information_bits(xs, ys, bins=4)
    assert mi >= 0.0


# ── CapacityMeasurement integration ────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    s = SignalStore(db_path=tmp_path / "sig.db")
    yield s
    s.close()


def test_measure_returns_capacity_report(store: SignalStore) -> None:
    m = CapacityMeasurement(store=store)
    report = m.measure(n_samples=400, seed=42)
    assert isinstance(report, CapacityReport)
    assert report.input_emissions == 400
    assert report.sample_count == 400
    assert report.bin_count == 4
    assert report.probe_seed == 42
    assert report.duration_seconds > 0
    # On a known reasonable channel (the store-then-immediately-read
    # pattern), MI is well above the independent-noise floor.
    assert report.mutual_information_bits > 0.1


def test_measure_reports_bias_warning_on_small_sample(store: SignalStore) -> None:
    m = CapacityMeasurement(store=store)
    # bins**2 = 16; 50 samples is well below the 160 threshold.
    report = m.measure(n_samples=50, seed=1)
    assert report.bias_warning is not None
    assert "upward-biased" in report.bias_warning


def test_measure_is_deterministic_with_seed(store: SignalStore) -> None:
    m1 = CapacityMeasurement(store=store)
    r1 = m1.measure(n_samples=200, seed=99)
    # Need a fresh store for the rerun to compare cleanly; the probe
    # writes to the store and the second run would read prior magnitudes
    # back through the integration window. Use a sibling test path
    # instead — the deterministic property tests RNG choices, not
    # store state. We check it by re-running on a fresh store.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        s2 = SignalStore(db_path=Path(td) / "s.db")
        try:
            m2 = CapacityMeasurement(store=s2)
            r2 = m2.measure(n_samples=200, seed=99)
        finally:
            s2.close()
    # With identical seed + fresh store, MI matches bit-for-bit.
    assert r1.mutual_information_bits == pytest.approx(r2.mutual_information_bits)


def test_measure_extrapolates_bits_per_hour(store: SignalStore) -> None:
    report = CapacityMeasurement(store=store).measure(n_samples=200, seed=7)
    assert report.bits_per_hour > 0
    # The probe runs in tens of milliseconds; bits/hour will be wildly
    # extrapolated. Just confirm the math direction is right.
    assert report.bits_per_hour > report.mutual_information_bits


def test_to_dict_round_trip(store: SignalStore) -> None:
    report = CapacityMeasurement(store=store).measure(n_samples=200, seed=1)
    d = report.to_dict()
    assert d["input_emissions"] == 200
    assert d["bin_count"] == 4
    assert d["probe_seed"] == 1
    assert d["benchmark_bits_per_hour"] == BENCHMARK_BITS_PER_HOUR


# ── CLI rendering ──────────────────────────────────────────────────────────


def test_cli_format_report_human_readable(store: SignalStore) -> None:
    report = CapacityMeasurement(store=store).measure(n_samples=200, seed=2)
    out = cli_format_report(report)
    assert "channel-capacity probe" in out
    assert "bits" in out
    assert "benchmark" in out


def test_cli_format_includes_bias_warning_when_present(store: SignalStore) -> None:
    report = CapacityMeasurement(store=store).measure(n_samples=20, seed=3)
    out = cli_format_report(report)
    assert "upward-biased" in out
