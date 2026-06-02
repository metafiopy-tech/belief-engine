"""Hermetic tests for the STARVED-arm variance-decay metrics.

numpy-only, no API key, no scipy/sklearn. Synthetic matrices with known
structure exercise each metric's contract, including the n < dims regime the
real experiment lives in.
"""

from __future__ import annotations

import numpy as np
import pytest

from belief.experiments.variance_decay import (
    DecayFit,
    GenerationMetrics,
    ar1,
    compute_generation_metrics,
    decay_fit,
    hill_q1,
    participation_ratio,
)


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------


def test_pr_isotropic_approaches_dim():
    """Many samples, low dim, equal variance per axis -> PR ~= number of dims."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    pr = participation_ratio(X)
    assert 2.6 <= pr <= 3.0


def test_pr_collapse_to_one_direction():
    """One dominant axis, the rest near-zero variance -> PR ~= 1."""
    rng = np.random.default_rng(1)
    X = np.zeros((500, 5))
    X[:, 0] = rng.normal(scale=100.0, size=500)  # dominant direction
    X[:, 1:] = rng.normal(scale=1e-3, size=(500, 4))
    pr = participation_ratio(X)
    assert 1.0 <= pr <= 1.1


def test_pr_n_less_than_dims_is_rank_ceilinged():
    """With n points in high-dim space, PR cannot exceed n-1 (the experiment regime)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 384))  # 5 nutrients, 384-dim encoder
    pr = participation_ratio(X)
    assert 0.0 < pr <= 5 - 1 + 1e-6


def test_pr_scale_invariance():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 8))
    assert participation_ratio(X) == pytest.approx(participation_ratio(7.5 * X), rel=1e-9)


def test_pr_degenerate_inputs():
    assert participation_ratio(np.zeros((0, 10))) == 0.0
    assert participation_ratio(np.zeros((1, 10))) == 0.0  # single point
    assert participation_ratio(np.ones((50, 10))) == 0.0  # no variance


def test_pr_rejects_non_2d():
    with pytest.raises(ValueError):
        participation_ratio(np.zeros((5,)))


# ---------------------------------------------------------------------------
# Hill q=1
# ---------------------------------------------------------------------------


def _three_blobs(seed: int = 0, per: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    pts = [c + rng.normal(scale=0.5, size=(per, 2)) for c in centers]
    return np.vstack(pts)


def test_hill_three_even_blobs_approaches_three():
    X = _three_blobs()
    h = hill_q1(X, k=3, seed=0)
    assert 2.8 <= h <= 3.0


def test_hill_concentrated_mass_approaches_one():
    """Collapse signature: mass concentrated in one cell, cells starved -> Hill ~= 1.

    A single isotropic blob would split evenly under fixed k (Hill ~= k); the
    collapse signal is instead a dominant dense mass with sparse outliers, where
    k-means strands the outliers as singletons and most proportion lands in one
    cell, dropping Shannon toward 0.
    """
    rng = np.random.default_rng(5)
    dense = rng.normal(scale=0.01, size=(118, 4)) + 50.0
    outliers = np.array([[-500.0, 0.0, 0.0, 0.0], [0.0, 500.0, 0.0, 0.0]])
    X = np.vstack([dense, outliers])
    h = hill_q1(X, k=3, seed=0)
    assert 1.0 <= h <= 1.3


def test_hill_single_isotropic_blob_splits_to_k():
    """Documented property: fixed k splits one blob into k even cells -> Hill ~= k."""
    rng = np.random.default_rng(15)
    X = rng.normal(scale=1.0, size=(150, 4))  # one isotropic cloud
    h = hill_q1(X, k=3, seed=0)
    assert 2.7 <= h <= 3.0


def test_hill_is_deterministic():
    X = _three_blobs(seed=9)
    a = hill_q1(X, k=4, seed=42)
    b = hill_q1(X, k=4, seed=42)
    assert a == b


def test_hill_k_clamped_to_n():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])  # 2 points, ask for k=8
    h = hill_q1(X, k=8, seed=0)
    assert 1.0 <= h <= 2.0


def test_hill_degenerate_inputs():
    assert hill_q1(np.zeros((0, 3)), k=3) == 0.0
    assert hill_q1(np.zeros((1, 3)), k=3) == 1.0
    with pytest.raises(ValueError):
        hill_q1(np.zeros((5, 3)), k=0)


# ---------------------------------------------------------------------------
# AR(1)
# ---------------------------------------------------------------------------


def test_ar1_monotone_series_high_positive():
    assert ar1(list(range(20))) > 0.8


def test_ar1_constant_series_zero():
    assert ar1([3.0] * 10) == 0.0


def test_ar1_short_series_zero():
    assert ar1([1.0]) == 0.0
    assert ar1([]) == 0.0


def test_ar1_white_noise_near_zero():
    rng = np.random.default_rng(7)
    x = rng.normal(size=2000)
    assert abs(ar1(x)) < 0.1


# ---------------------------------------------------------------------------
# Decay fit
# ---------------------------------------------------------------------------


def test_decay_fit_recovers_known_parameters():
    n = np.arange(25, dtype=float)
    y = 5.0 * np.exp(-n / 4.0) + 2.0
    fit = decay_fit(y)
    assert isinstance(fit, DecayFit)
    assert fit.tau == pytest.approx(4.0, rel=0.1)
    assert fit.a == pytest.approx(5.0, rel=0.1)
    assert fit.c == pytest.approx(2.0, abs=0.1)
    assert fit.r_squared > 0.999


def test_decay_fit_flat_series():
    fit = decay_fit([7.0] * 12)
    # A flat series fits with a ~= 0 and c ~= the constant level.
    assert fit.c == pytest.approx(7.0, abs=1e-6)
    assert abs(fit.a) < 1e-6


def test_decay_fit_single_point():
    fit = decay_fit([3.5])
    assert fit.c == 3.5
    assert fit.a == 0.0


def test_decay_fit_length_mismatch_raises():
    with pytest.raises(ValueError):
        decay_fit([1.0, 2.0, 3.0], gens=[0.0, 1.0])


# ---------------------------------------------------------------------------
# Generation metrics record
# ---------------------------------------------------------------------------


def test_compute_generation_metrics_record():
    X = _three_blobs(seed=11)
    m = compute_generation_metrics(
        X, gen=4, arm="STARVED", k=3, seed=0, encoder_fingerprint="minilm@abc123"
    )
    assert isinstance(m, GenerationMetrics)
    assert m.gen == 4
    assert m.arm == "STARVED"
    assert m.n_nutrients == X.shape[0]
    assert m.kmeans_k == 3
    assert m.encoder_fingerprint == "minilm@abc123"
    assert m.participation_ratio > 0.0
    assert 2.8 <= m.hill_q1 <= 3.0


def test_compute_generation_metrics_empty():
    m = compute_generation_metrics(np.zeros((0, 5)), gen=0, arm="FED", k=8)
    assert m.n_nutrients == 0
    assert m.participation_ratio == 0.0
    assert m.hill_q1 == 0.0
