"""Offline reporting + calibration for the STARVED-arm experiment.

Pure analysis over a finished (or in-progress) run directory: it reads the
per-generation embedding snapshots (npz+json) and the admission/probe SQLite
logs, computes the variance-decay metrics, and renders a report. It never runs
builds or touches a model, so it is fully deterministic and unit-testable.

Two responsibilities, deliberately separated:

1. :func:`compute_run_metrics` / :func:`format_report` — the full picture, BOTH
   arms, for reading the result.

2. :func:`calibrate_fed_band` / :func:`format_preregistration_block` — the
   pre-registration step, which reads **only the FED arm**. The pilot calibrates
   the noise band from the arm that should be stable; looking at the STARVED
   curve's shape to set the band would break falsifiability (design doc §6/§7).
   ``calibrate_fed_band`` structurally refuses to load STARVED snapshots.

Headline rule (design doc §2.1): PR is reported as the centered-Gram value but
adjudicated as a **differential** (STARVED vs FED) trend, jointly with Hill q=1.
This module surfaces both series for both arms; it does not itself declare a
verdict — that is the human pre-registered call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from belief.experiments.soil_snapshot import GenerationSnapshot, load_snapshot
from belief.experiments.variance_decay import DecayFit, ar1, decay_fit, hill_q1, participation_ratio

ARMS = ("FED", "STARVED")


@dataclass
class ArmSeries:
    """Per-generation metric series for one arm."""

    arm: str
    gens: list[int] = field(default_factory=list)
    pr: list[float] = field(default_factory=list)
    hill: list[float] = field(default_factory=list)
    n_nutrients: list[int] = field(default_factory=list)
    pr_ar1: float = 0.0
    hill_ar1: float = 0.0
    pr_decay: DecayFit | None = None
    hill_decay: DecayFit | None = None


@dataclass
class RunMetrics:
    """Computed metrics for a whole run."""

    experiment_id: str
    arms: dict[str, ArmSeries] = field(default_factory=dict)
    fictions: int = 0
    probe: list[dict] = field(default_factory=list)
    encoder_fingerprints: set[str] = field(default_factory=set)

    @property
    def encoder_consistent(self) -> bool:
        """False if snapshots used more than one encoder — voids comparability."""
        return len(self.encoder_fingerprints) <= 1


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------


def _snapshots_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir).expanduser()
    # Accept either the run dir or the snapshots dir directly.
    cand = run_dir / "snapshots"
    return cand if cand.exists() else run_dir


def load_arm_snapshots(run_dir: Path, arm: str) -> list[GenerationSnapshot]:
    """Load one arm's snapshots, sorted by generation."""
    snaps_dir = _snapshots_dir(run_dir)
    out: list[GenerationSnapshot] = []
    for npz in sorted(snaps_dir.glob(f"{arm.upper()}_gen*.npz")):
        out.append(load_snapshot(npz))
    out.sort(key=lambda s: s.gen)
    return out


def _series_from_snapshots(arm: str, snaps: list[GenerationSnapshot]) -> ArmSeries:
    s = ArmSeries(arm=arm.upper())
    for snap in snaps:
        s.gens.append(snap.gen)
        s.n_nutrients.append(snap.n_nutrients)
        s.pr.append(participation_ratio(snap.X))
        s.hill.append(hill_q1(snap.X, snap.kmeans_k))
    if s.pr:
        s.pr_ar1 = ar1(s.pr)
        s.hill_ar1 = ar1(s.hill)
        s.pr_decay = decay_fit(s.pr, s.gens)
        s.hill_decay = decay_fit(s.hill, s.gens)
    return s


# ---------------------------------------------------------------------------
# Full run metrics (both arms)
# ---------------------------------------------------------------------------


def compute_run_metrics(run_dir: Path, *, experiment_id: str | None = None) -> RunMetrics:
    """Compute PR/Hill series + AR(1) + decay fits for both arms, plus fictions."""
    run_dir = Path(run_dir).expanduser()
    exp_id = experiment_id or run_dir.name
    rm = RunMetrics(experiment_id=exp_id)

    for arm in ARMS:
        snaps = load_arm_snapshots(run_dir, arm)
        for snap in snaps:
            rm.encoder_fingerprints.add(snap.encoder_fingerprint)
        rm.arms[arm] = _series_from_snapshots(arm, snaps)

    # Fictions from the admission log (best-effort; absent db -> 0).
    adm_db = run_dir / "admissions.db"
    if adm_db.exists():
        from belief.experiments.admission_log import count_fictions

        rm.fictions = count_fictions(exp_id, db_path=adm_db)

    # Held-out probe curve (best-effort).
    probe_db = run_dir / "probe.db"
    if probe_db.exists():
        from belief.experiments.swebench_probe import fetch_probes

        rm.probe = fetch_probes(exp_id, db_path=probe_db)

    return rm


# ---------------------------------------------------------------------------
# FED-only calibration (pre-registration)
# ---------------------------------------------------------------------------


def calibrate_fed_band(run_dir: Path, *, sigma_mult: float = 2.0) -> dict:
    """Compute the FED-arm PR noise band — reads ONLY the FED arm.

    Returns ``{n_gens, pr_mean, pr_std, sigma_mult, band_low, band_high}``. The
    std is the sample std (ddof=1) of per-generation PR; with <2 generations it
    is undefined and returned as 0.0 with a band collapsed to the mean (the
    caller should not freeze a band off a degenerate pilot).

    By construction this never loads STARVED snapshots — peeking at the STARVED
    curve to set the band would void the pre-registration.
    """
    import numpy as np

    fed = load_arm_snapshots(run_dir, "FED")
    pr = [participation_ratio(s.X) for s in fed]
    if not pr:
        raise ValueError(f"no FED snapshots found under {run_dir}")
    arr = np.asarray(pr, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return {
        "n_gens": len(pr),
        "pr_mean": mean,
        "pr_std": std,
        "sigma_mult": sigma_mult,
        "band_low": mean - sigma_mult * std,
        "band_high": mean + sigma_mult * std,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_floats(xs: list[float]) -> str:
    return ", ".join(f"{x:.3f}" for x in xs)


def format_report(run_dir: Path, *, experiment_id: str | None = None) -> str:
    """Human-readable report over both arms."""
    rm = compute_run_metrics(run_dir, experiment_id=experiment_id)
    lines: list[str] = []
    lines.append(f"STARVED-arm report — {rm.experiment_id}")
    lines.append("=" * 60)
    if not rm.encoder_consistent:
        lines.append(
            f"  !! ENCODER DRIFT: {len(rm.encoder_fingerprints)} distinct fingerprints — "
            "metrics are NOT comparable across these snapshots."
        )
    for arm in ARMS:
        s = rm.arms.get(arm)
        lines.append("")
        lines.append(f"[{arm}]")
        if s is None or not s.gens:
            lines.append("  (no snapshots)")
            continue
        lines.append(f"  generations: {s.gens}")
        lines.append(f"  n_nutrients: {s.n_nutrients}")
        lines.append(f"  PR:   {_fmt_floats(s.pr)}")
        lines.append(f"  Hill: {_fmt_floats(s.hill)}")
        lines.append(f"  AR(1)  PR={s.pr_ar1:.3f}  Hill={s.hill_ar1:.3f}")
        if s.pr_decay is not None:
            lines.append(
                f"  decay  PR: tau={s.pr_decay.tau:.2f} a={s.pr_decay.a:.3f} "
                f"c={s.pr_decay.c:.3f} r2={s.pr_decay.r_squared:.3f}"
            )
            lines.append(
                f"  decay  Hill: tau={s.hill_decay.tau:.2f} a={s.hill_decay.a:.3f} "
                f"c={s.hill_decay.c:.3f} r2={s.hill_decay.r_squared:.3f}"
            )
    lines.append("")
    lines.append(f"STARVED fictions admitted (failed external test): {rm.fictions}")
    lines.append("")
    if rm.probe:
        lines.append("Held-out probe (SWE-bench Verified):")
        for row in rm.probe:
            lines.append(
                f"  gen {row['gen']} {row['arm']}: "
                f"{row['n_resolved']}/{row['n_instances']} ({row['resolve_rate']:.2%})"
            )
    else:
        # A deferred metric and a failed metric must NOT look alike in the
        # artifact: say so explicitly rather than emitting a blank/zero.
        lines.append("Held-out probe (SWE-bench Verified): not measured (probe deferred)")
    lines.append("")
    lines.append(
        "NOTE: adjudicate PR as a DIFFERENTIAL (STARVED vs FED) trend, jointly with Hill. "
        "Absolute PR is rank-ceilinged in the n<dims regime (design doc §2.1)."
    )
    return "\n".join(lines)


def format_preregistration_block(
    run_dir: Path,
    *,
    kill_fraction: float,
    n_full: int = 25,
    sigma_mult: float = 2.0,
) -> str:
    """Emit the frozen §7 pre-registration block from the FED-only band.

    Intended to be pasted into ``docs/experiments/starved_arm_design.md`` §7 as a
    deliberate human act — calibration computes the numbers, the human commits
    them. ``kill_fraction`` must be named explicitly (no default) up front.
    """
    band = calibrate_fed_band(run_dir, sigma_mult=sigma_mult)
    return "\n".join(
        [
            "PRE-REGISTRATION (frozen from pilot FED arm; do not edit after full run begins)",
            "- Co-headline: differential PR (centered-Gram) + Hill q=1, joint-direction.",
            f"- FED-arm PR over pilot: mean={band['pr_mean']:.4f}, "
            f"sigma={band['pr_std']:.4f} (n={band['n_gens']} gens).",
            f"- Noise band: FED mean +/- {sigma_mult:g}sigma = "
            f"[{band['band_low']:.4f}, {band['band_high']:.4f}].",
            f"- Full-run N: {n_full}.",
            f"- Kill criterion (thesis FAILS): STARVED PR stays inside the band for "
            f">= {kill_fraction:.0%} of N={n_full} generations AND held-out success holds.",
            "- Caveat: a flat pilot is uninformative about slow decay (tau > ~6-7); "
            "only the full run adjudicates.",
        ]
    )
