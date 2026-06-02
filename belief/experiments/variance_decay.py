"""Variance-decay metrics for the STARVED-arm experiment.

Pure, offline analytics over per-generation soil-embedding snapshots. Computes
the "effective spread" of a soil embedding cloud and the early-warning /
decay-fit statistics used to adjudicate the experiment described in
``docs/experiments/starved_arm_design.md``.

Design constraints (locked in the design doc, repeated here because they are
load-bearing for correctness):

- **Participation ratio is computed on the centered Gram matrix**
  ``X̃ X̃ᵀ`` (samples × samples), not the feature covariance. The two share the
  same nonzero spectrum, but the Gram form stays well-conditioned when the
  number of nutrients ``n`` is smaller than the embedding dimension ``d`` — the
  regime this experiment lives in for its entire duration (K=4 admits/gen ⇒
  ``n`` tops out near ~100 ≪ ``d``). PR is consequently rank-ceilinged at
  ``n − 1``; that ceiling is *identical across arms per generation* by
  K-matching, so the live signal is **differential** PR (STARVED vs FED at
  matched ``n``), never absolute PR.

- **Hill q=1 co-headlines, with a frozen k.** It is proportion-based over a
  k-means clustering and immune to the PR ceiling, but has its own small-n
  sensitivity, so ``k`` is fixed up front and asserted at compute time. The
  thesis support criterion (applied downstream, not here) is joint-direction:
  differential-PR trend and Hill trend must agree.

- **numpy-only.** ``scipy`` / ``scikit-learn`` are optional extras not present
  in the core hard gate, so this module deliberately avoids them: the decay fit
  uses a τ-search with a closed-form linear solve for ``(a, c)``, and k-means is
  a small deterministic Lloyd implementation with k-means++ seeding under a
  fixed RNG. Determinism (same input + same seed ⇒ same output) is part of the
  contract — Hill must not wobble with a clustering hyperparameter.

Nothing here touches the build pipeline; metrics are computed from snapshots
after a run, so they never perturb it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Default frozen k for the Hill q=1 clustering. The design doc requires this to
# be pinned before the pilot and asserted per-snapshot; callers pass it
# explicitly in the experiment driver. The default exists only so unit tests and
# ad-hoc analysis have a stable value.
DEFAULT_KMEANS_K = 8

# Numerical floor: eigenvalues of a Gram matrix are non-negative in exact
# arithmetic; tiny negative values are floating-point noise and are clipped.
_EIG_FLOOR = 1e-12


@dataclass(frozen=True)
class GenerationMetrics:
    """Per-(generation, arm) metrics record, suitable for serialization.

    ``encoder_fingerprint`` carries the pinned-encoder identity (model +
    revision + backend hash) so a pilot/full-run encoder mismatch is detectable
    offline; ``kmeans_k`` records the frozen clustering hyperparameter.
    """

    gen: int
    arm: str
    n_nutrients: int
    participation_ratio: float
    hill_q1: float
    kmeans_k: int
    encoder_fingerprint: Optional[str] = None


@dataclass(frozen=True)
class DecayFit:
    """Result of fitting ``y(n) = a · e^(−n/τ) + c``.

    ``tau`` is reported as "generations to run down." ``rmse`` is the
    root-mean-square residual of the fit; ``r_squared`` is the coefficient of
    determination (1.0 = perfect, ≤ 0 = worse than predicting the mean).
    """

    a: float
    tau: float
    c: float
    rmse: float
    r_squared: float


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------


def participation_ratio(X: np.ndarray) -> float:
    """Effective number of dimensions the embedding cloud spans.

    ``PR = (Σ λ_i)² / Σ λ_i²`` over the eigenvalues of the **centered Gram
    matrix** of ``X`` (one row per nutrient). Scale-invariant in ``X`` (it is a
    ratio of spectral moments), so a uniform rescale of the embeddings does not
    move it. Returns ``0.0`` for an empty cloud, a single point, or a cloud with
    no variance (all rows identical).

    Computing the spectrum on the ``n × n`` Gram matrix rather than the
    ``d × d`` covariance keeps the eigenproblem well-conditioned when ``n < d``,
    which is the regime of this experiment throughout.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_nutrients, dim); got shape {X.shape}")
    n = X.shape[0]
    if n < 2:
        return 0.0

    # Center per feature (column means) — this is what makes the Gram spectrum
    # match the covariance spectrum (up to the irrelevant (n-1) scale factor).
    Xc = X - X.mean(axis=0, keepdims=True)
    gram = Xc @ Xc.T  # (n, n), symmetric PSD

    eig = np.linalg.eigvalsh(gram)
    eig = eig[eig > _EIG_FLOOR]
    if eig.size == 0:
        return 0.0

    s1 = float(eig.sum())
    s2 = float(np.square(eig).sum())
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


# ---------------------------------------------------------------------------
# Hill number q=1 over k-means clustering
# ---------------------------------------------------------------------------


def _kmeans(X: np.ndarray, k: int, *, seed: int, max_iter: int = 100) -> np.ndarray:
    """Deterministic Lloyd k-means with k-means++ seeding. Returns labels.

    Self-contained (numpy-only) so the metric does not depend on scikit-learn,
    and fully determined by ``seed`` so Hill q=1 is reproducible.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    # k-means++ initialization.
    first = int(rng.integers(n))
    centers = [X[first]]
    # Squared distance to the nearest chosen center, updated incrementally.
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:
            # All remaining points coincide with chosen centers; pad with a
            # repeat so we always return k centers.
            centers.append(X[int(rng.integers(n))])
            continue
        probs = closest_sq / total
        nxt = int(rng.choice(n, p=probs))
        centers.append(X[nxt])
        new_sq = np.sum((X - X[nxt]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    C = np.asarray(centers, dtype=np.float64)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Assign: nearest center by squared Euclidean distance.
        # dists shape (n, k)
        dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        # Update: mean of assigned points; keep old center if a cluster empties.
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                C[j] = X[mask].mean(axis=0)
    return labels


def hill_q1(X: np.ndarray, k: int = DEFAULT_KMEANS_K, *, seed: int = 0) -> float:
    """Effective number of distinct nutrient "species" in the cloud.

    ``Hill_1 = exp(H)`` where ``H = −Σ p_i ln p_i`` is the Shannon entropy of
    the cluster-size proportions from a k-means clustering of ``X`` with frozen
    ``k`` (clamped to ``n`` — you cannot have more clusters than points).

    What this actually measures is the **evenness of the frozen k-partition**,
    not an absolute count of modes. Two properties follow and are intentional:

    - A single isotropic blob clustered with ``k > 1`` is split into ``k`` even
      Voronoi cells, so Hill ≈ ``k`` — k-means does not "detect" that there is
      really one mode. This is a known consequence of a fixed ``k`` (the reason
      the design doc freezes ``k`` and treats it as a hyperparameter, §2.1a).
    - Diversity **collapse** shows up as mass concentrating into a *subset* of
      the ``k`` cells (some cells starve to near-empty), which drops Shannon and
      pulls Hill toward ``1``.

    Because the absolute value is partition-relative, the experiment reads Hill
    as a **trend / differential** across generations and across the FED/STARVED
    contrast (where the shared ``k`` cancels), never as a standalone count.
    Deterministic for a given ``(X, k, seed)``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_nutrients, dim); got shape {X.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    n = X.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    k_eff = min(k, n)
    labels = _kmeans(X, k_eff, seed=seed)
    counts = np.bincount(labels, minlength=k_eff).astype(np.float64)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    shannon = float(-(p * np.log(p)).sum())
    return float(np.exp(shannon))


# ---------------------------------------------------------------------------
# AR(1) early-warning
# ---------------------------------------------------------------------------


def ar1(series: Sequence[float]) -> float:
    """Lag-1 autocorrelation of a time series — critical-slowing-down signal.

    Rising AR(1) over generations means the set is approaching a fixed point
    before the headline metric flatlines. Standard normalized estimator:
    ``Σ (x_t − x̄)(x_{t−1} − x̄) / Σ (x_t − x̄)²``. Returns ``0.0`` for series
    shorter than 2 points or with zero variance.
    """
    x = np.asarray(list(series), dtype=np.float64)
    if x.size < 2:
        return 0.0
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom <= 0.0:
        return 0.0
    num = float(np.sum(xc[1:] * xc[:-1]))
    return num / denom


# ---------------------------------------------------------------------------
# Exponential-decay fit  y(n) = a·e^(−n/τ) + c
# ---------------------------------------------------------------------------


def decay_fit(
    series: Sequence[float],
    gens: Optional[Sequence[float]] = None,
    *,
    tau_grid: Optional[Sequence[float]] = None,
) -> DecayFit:
    """Fit ``y(n) = a·e^(−n/τ) + c`` without scipy.

    For a fixed ``τ`` the model is linear in ``a`` and ``c`` (basis
    ``[e^(−n/τ), 1]``), so we search ``τ`` on a grid and solve the linear least
    squares for ``(a, c)`` in closed form at each candidate, keeping the best
    RMSE. Numpy-only; deterministic.

    ``gens`` defaults to ``0, 1, 2, …``. ``tau_grid`` defaults to a geometric
    sweep scaled to the series length.
    """
    y = np.asarray(list(series), dtype=np.float64)
    m = y.size
    if m == 0:
        raise ValueError("series must be non-empty")
    if gens is None:
        n = np.arange(m, dtype=np.float64)
    else:
        n = np.asarray(list(gens), dtype=np.float64)
        if n.size != m:
            raise ValueError("gens and series must have the same length")

    if m == 1:
        return DecayFit(a=0.0, tau=float("inf"), c=float(y[0]), rmse=0.0, r_squared=1.0)

    if tau_grid is None:
        hi = max(10.0, 5.0 * m)
        tau_grid = np.geomspace(0.5, hi, 400)
    else:
        tau_grid = np.asarray(list(tau_grid), dtype=np.float64)

    ones = np.ones_like(n)
    best: Optional[tuple[float, float, float, float]] = None  # (rmse, a, c, tau)
    for tau in tau_grid:
        basis = np.column_stack([np.exp(-n / tau), ones])
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        a, c = float(coef[0]), float(coef[1])
        resid = y - basis @ coef
        rmse = float(np.sqrt(np.mean(resid * resid)))
        if best is None or rmse < best[0]:
            best = (rmse, a, c, float(tau))

    assert best is not None
    rmse, a, c, tau = best

    ss_res = float(np.sum((y - (a * np.exp(-n / tau) + c)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return DecayFit(a=a, tau=tau, c=c, rmse=rmse, r_squared=r_squared)


# ---------------------------------------------------------------------------
# Convenience: per-generation metric record
# ---------------------------------------------------------------------------


def compute_generation_metrics(
    X: np.ndarray,
    *,
    gen: int,
    arm: str,
    k: int = DEFAULT_KMEANS_K,
    seed: int = 0,
    encoder_fingerprint: Optional[str] = None,
) -> GenerationMetrics:
    """Compute PR + Hill q=1 for one generation's embedding matrix.

    ``k`` is the frozen clustering hyperparameter (recorded in the result) and
    ``encoder_fingerprint`` is the pinned-encoder identity, both carried through
    so downstream adjudication can assert they never drifted.
    """
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0]) if X.ndim == 2 else 0
    return GenerationMetrics(
        gen=gen,
        arm=arm,
        n_nutrients=n,
        participation_ratio=participation_ratio(X) if n else 0.0,
        hill_q1=hill_q1(X, k, seed=seed) if n else 0.0,
        kmeans_k=k,
        encoder_fingerprint=encoder_fingerprint,
    )
