"""Offline exploration of a finished STARVED run — cheap, read-only, no builds.

Runs on the per-generation snapshots (npz) and the admission log (SQLite) that a
completed run already wrote to disk, to chase *where the needle is pointing*
before committing to a larger, longer run. Pure numpy + stdlib; nothing here
launches the engine, a model, or Docker.

Four probes (see docs/experiments/starved_arm_design.md §9):

1. **Hill paradox** — full-n25 showed STARVED's Hill running *higher* than FED's,
   which is backwards from "collapse." Hypothesis: incoherent fictions don't
   cluster, they scatter, so k-means reads them as extra "species" and inflates
   Hill even as the soil degrades. Test geometrically: if the high Hill is
   scatter rather than real structure, it coincides with **low silhouette**
   (clusters that aren't cohesive) and higher nearest-neighbor spread. We cannot
   tag individual fiction nutrients (the influx deposit assigns fresh ids with no
   stored link to admission build_ids), so this geometric signature is the
   available test.

2. **Late-window slope** — the differential PR only dipped negative in the final
   gens. Is the late segment's slope steeper than the whole-run slope, beyond the
   run's own noise? Acceleration ⇒ slow-decay onset is real.

3. **Fiction-rate trajectory** — fictions per generation (not just the total):
   accelerating, or a steady leak?

4. **Success-gap trajectory** — per-generation STARVED − FED build success:
   constant, or widening?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from belief.experiments.starved_report import _aligned_differential, load_arm_snapshots
from belief.experiments.variance_decay import _kmeans, hill_q1

ARMS = ("FED", "STARVED")


# ---------------------------------------------------------------------------
# Geometry helpers (numpy-only)
# ---------------------------------------------------------------------------


def mean_nn_distance(X: np.ndarray) -> float:
    """Mean nearest-neighbor Euclidean distance — a scatter proxy."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return 0.0
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min(axis=1)).mean())


def silhouette(X: np.ndarray, k: int, *, seed: int = 0) -> float:
    """Mean silhouette over a frozen-k k-means clustering (cohesion vs separation).

    High when clusters are tight and well-separated; near 0 / negative when points
    are scattered and the "clusters" are arbitrary. The discriminator for "is the
    Hill diversity real structure or just scatter?". Returns 0.0 when undefined
    (n < 3 or a single occupied cluster).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 3:
        return 0.0
    k_eff = min(k, n)
    labels = _kmeans(X, k_eff, seed=seed)
    occupied = np.unique(labels)
    if occupied.size < 2:
        return 0.0
    dist = np.sqrt(np.maximum(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2), 0.0))
    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = dist[i, same].mean() if same.any() else 0.0
        b = np.inf
        for c in occupied:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.any():
                b = min(b, dist[i, mask].mean())
        denom = max(a, b)
        sil[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(sil.mean())


def slope(series) -> float:
    """Least-squares slope of a series vs its index (0,1,2,...)."""
    y = np.asarray(list(series), dtype=np.float64)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom == 0.0:
        return 0.0
    return float(np.sum(xc * (y - y.mean())) / denom)


# ---------------------------------------------------------------------------
# Probe 1 — Hill paradox
# ---------------------------------------------------------------------------


@dataclass
class CloudStructure:
    arm: str
    gens: list[int] = field(default_factory=list)
    hill: list[float] = field(default_factory=list)
    silhouette: list[float] = field(default_factory=list)
    mean_nn: list[float] = field(default_factory=list)


def cloud_structure(run_dir: Path, arm: str, *, seed: int = 0) -> CloudStructure:
    """Per-generation Hill, silhouette, and NN-scatter for one arm."""
    cs = CloudStructure(arm=arm.upper())
    for snap in load_arm_snapshots(run_dir, arm):
        cs.gens.append(snap.gen)
        cs.hill.append(hill_q1(snap.X, snap.kmeans_k, seed=seed))
        cs.silhouette.append(silhouette(snap.X, snap.kmeans_k, seed=seed))
        cs.mean_nn.append(mean_nn_distance(snap.X))
    return cs


def hill_paradox(run_dir: Path) -> dict:
    """Test whether STARVED's higher Hill is real structure or fiction-scatter.

    Returns per-arm structure plus a verdict: if STARVED's Hill exceeds FED's
    while its silhouette is *lower* (and NN-scatter higher), the diversity is
    scatter, not structure — i.e. Hill is decoupling from quality.
    """
    fed = cloud_structure(run_dir, "FED")
    starved = cloud_structure(run_dir, "STARVED")
    g = min(len(fed.gens), len(starved.gens))

    def _tail_mean(xs: list[float], frac: float = 0.5) -> float:
        if not xs:
            return 0.0
        start = int(len(xs) * (1 - frac))
        return float(np.mean(xs[start:]))

    hill_gap = _tail_mean(starved.hill) - _tail_mean(fed.hill)  # >0 = STARVED more diverse
    sil_gap = _tail_mean(starved.silhouette) - _tail_mean(fed.silhouette)  # <0 = less cohesive
    nn_gap = _tail_mean(starved.mean_nn) - _tail_mean(fed.mean_nn)  # >0 = more scattered

    scatter_signature = hill_gap > 0 and sil_gap < 0
    return {
        "fed": fed,
        "starved": starved,
        "n_gens": g,
        "tail_hill_gap": hill_gap,
        "tail_silhouette_gap": sil_gap,
        "tail_nn_gap": nn_gap,
        "scatter_signature": scatter_signature,
    }


# ---------------------------------------------------------------------------
# Probe 2 — late-window slope
# ---------------------------------------------------------------------------


def late_window_slopes(run_dir: Path, *, split_frac: float = 0.5) -> dict:
    """Slope of the differential PR over the whole run vs its late segment."""
    gens, diff = _aligned_differential(run_dir, "pr")
    if not diff:
        return {"n_gens": 0}
    start = int(len(diff) * split_frac)
    return {
        "n_gens": len(diff),
        "full_slope": slope(diff),
        "late_slope": slope(diff[start:]),
        "late_from_gen": gens[start] if start < len(gens) else None,
        "final_value": diff[-1],
    }


# ---------------------------------------------------------------------------
# Probes 3 & 4 — admission-log trajectories
# ---------------------------------------------------------------------------


def _fetch(experiment_id: str, db_path: Path) -> list[dict]:
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        return []
    from belief.experiments.admission_log import fetch_events

    return fetch_events(experiment_id, db_path=db_path)


def fiction_trajectory(experiment_id: str, db_path: Path) -> dict:
    """Per-generation count of STARVED-admitted builds that failed the external test."""
    events = _fetch(experiment_id, db_path)
    by_gen: dict[int, int] = {}
    for e in events:
        if e["arm"] == "STARVED" and e["admitted"] == 1 and e["external_pass"] == 0:
            by_gen[e["gen"]] = by_gen.get(e["gen"], 0) + 1
    gens = sorted(by_gen)
    counts = [by_gen[g] for g in gens]
    return {"gens": gens, "counts": counts, "slope": slope(counts), "total": sum(counts)}


def success_gap_trajectory(experiment_id: str, db_path: Path) -> dict:
    """Per-generation STARVED − FED build-success rate (mean external_pass)."""
    events = _fetch(experiment_id, db_path)
    rate: dict[str, dict[int, list[int]]] = {"FED": {}, "STARVED": {}}
    for e in events:
        arm = e["arm"]
        if arm in rate:
            rate[arm].setdefault(e["gen"], []).append(int(e["external_pass"]))
    gens = sorted(set(rate["FED"]) & set(rate["STARVED"]))
    gap = []
    for g in gens:
        f = np.mean(rate["FED"][g]) if rate["FED"].get(g) else 0.0
        s = np.mean(rate["STARVED"][g]) if rate["STARVED"].get(g) else 0.0
        gap.append(float(s - f))
    return {
        "gens": gens,
        "gap": gap,
        "slope": slope(gap),
        "mean_gap": float(np.mean(gap)) if gap else 0.0,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(xs) -> str:
    return ", ".join(f"{x:.3f}" for x in xs)


def format_exploration(run_dir: Path, *, experiment_id: str | None = None) -> str:
    run_dir = Path(run_dir).expanduser()
    exp_id = experiment_id or run_dir.name
    db = run_dir / "admissions.db"
    lines: list[str] = []
    lines.append(f"STARVED exploration — {exp_id}")
    lines.append("=" * 60)

    hp = hill_paradox(run_dir)
    lines.append("")
    lines.append("[1] Hill paradox (is STARVED diversity real structure or fiction-scatter?)")
    for arm in ("FED", "STARVED"):
        cs = hp[arm.lower()]
        lines.append(f"  {arm} Hill:       {_fmt(cs.hill)}")
        lines.append(f"  {arm} silhouette: {_fmt(cs.silhouette)}")
        lines.append(f"  {arm} mean_nn:    {_fmt(cs.mean_nn)}")
    lines.append(
        f"  tail gaps (STARVED-FED): Hill={hp['tail_hill_gap']:+.3f} "
        f"silhouette={hp['tail_silhouette_gap']:+.3f} nn={hp['tail_nn_gap']:+.3f}"
    )
    if hp["scatter_signature"]:
        lines.append(
            "  -> SCATTER SIGNATURE: STARVED Hill higher BUT silhouette lower — "
            "diversity is decoupling from structure (fictions scatter, inflating Hill)."
        )
    else:
        lines.append("  -> no scatter signature (higher Hill is matched by cohesion).")

    lw = late_window_slopes(run_dir)
    lines.append("")
    lines.append("[2] Late-window slope of differential PR (STARVED-FED)")
    if lw.get("n_gens"):
        lines.append(
            f"  full-run slope={lw['full_slope']:+.4f}/gen  "
            f"late slope (from gen {lw['late_from_gen']})={lw['late_slope']:+.4f}/gen  "
            f"final={lw['final_value']:+.3f}"
        )
        accel = lw["late_slope"] < lw["full_slope"] and lw["late_slope"] < 0
        lines.append(
            "  -> late slope steeper & negative: decay-onset signal."
            if accel
            else "  -> late slope not clearly accelerating downward."
        )

    ft = fiction_trajectory(exp_id, db)
    lines.append("")
    lines.append("[3] Fiction-rate trajectory (STARVED admitted & failed external, per gen)")
    if ft["gens"]:
        lines.append(f"  per-gen: {ft['counts']}")
        lines.append(
            f"  total={ft['total']}  slope={ft['slope']:+.4f}/gen  "
            f"({'accelerating' if ft['slope'] > 0 else 'flat/declining'})"
        )
    else:
        lines.append("  (no admission log found)")

    sg = success_gap_trajectory(exp_id, db)
    lines.append("")
    lines.append("[4] Build-success gap (STARVED - FED, per gen)")
    if sg["gens"]:
        lines.append(f"  gap: {_fmt(sg['gap'])}")
        lines.append(
            f"  mean gap={sg['mean_gap']:+.3f}  slope={sg['slope']:+.4f}/gen  "
            f"({'widening against STARVED' if sg['slope'] < 0 else 'stable/closing'})"
        )
    else:
        lines.append("  (no admission log found)")

    lines.append("")
    lines.append(
        "NOTE: exploratory, post-hoc — hypothesis-generating only. Any conclusion "
        "needs a fresh pre-registration, not a re-read of this run."
    )
    return "\n".join(lines)
