"""Tests for the STARVED offline report + FED-only calibration (Session 5).

Gate-safe: synthetic snapshots written to disk via soil_snapshot.save_snapshot;
no model / ChromaDB. A collapsing STARVED cloud vs a stable FED cloud exercises
the metric trends; a tiny admissions.db exercises the fiction count.
"""

from __future__ import annotations

import numpy as np
import pytest

from belief.experiments import admission_log
from belief.experiments.admission import Candidate, select_admissions
from belief.experiments.soil_snapshot import GenerationSnapshot, save_snapshot
from belief.experiments.starved_report import (
    calibrate_fed_band,
    compute_run_metrics,
    format_preregistration_block,
    format_report,
    load_arm_snapshots,
)

_FP = "minilm@pinned:dim16:norm"


def _write_snap(snap_dir, arm, gen, X):
    save_snapshot(
        GenerationSnapshot(
            gen=gen,
            arm=arm,
            nutrient_ids=[f"{arm}-{gen}-{i}" for i in range(X.shape[0])],
            X=X,
            encoder_fingerprint=_FP,
            kmeans_k=4,
        ),
        snap_dir,
    )


def _make_run(tmp_path, *, n_gens=8, collapse_starved=True):
    """Build a run dir: FED stays spread, STARVED collapses toward one direction."""
    run_dir = tmp_path / "pilot"
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n_pts, dim = 12, 16
    for gen in range(n_gens):
        # FED: full-spread isotropic cloud every generation (stable).
        fed_X = rng.normal(size=(n_pts, dim))
        _write_snap(snap_dir, "FED", gen, fed_X)
        # STARVED: variance along non-dominant axes shrinks with generation.
        starved_X = rng.normal(size=(n_pts, dim))
        if collapse_starved:
            shrink = max(0.02, 1.0 - gen / (n_gens - 1))
            starved_X[:, 1:] *= shrink  # collapse all but the first axis
            starved_X[:, 0] *= 5.0  # dominant direction grows
        _write_snap(snap_dir, "STARVED", gen, starved_X)
    return run_dir


# ---------------------------------------------------------------------------
# Loading + series
# ---------------------------------------------------------------------------


def test_load_arm_snapshots_sorted(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=5)
    fed = load_arm_snapshots(run_dir, "FED")
    assert [s.gen for s in fed] == [0, 1, 2, 3, 4]
    assert all(s.arm == "FED" for s in fed)


def test_compute_run_metrics_both_arms(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=8)
    rm = compute_run_metrics(run_dir)
    assert set(rm.arms) == {"FED", "STARVED"}
    assert len(rm.arms["FED"].pr) == 8
    assert len(rm.arms["STARVED"].hill) == 8
    assert rm.encoder_consistent


def test_starved_pr_collapses_below_fed(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=8, collapse_starved=True)
    rm = compute_run_metrics(run_dir)
    fed_pr = rm.arms["FED"].pr
    starved_pr = rm.arms["STARVED"].pr
    # By the last generation, STARVED's effective spread is well below FED's.
    assert starved_pr[-1] < fed_pr[-1]
    # And STARVED's PR trends down from its start.
    assert starved_pr[-1] < starved_pr[0]


def test_decay_fit_and_ar1_present(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=8)
    rm = compute_run_metrics(run_dir)
    s = rm.arms["STARVED"]
    assert s.pr_decay is not None
    assert s.pr_decay.tau > 0
    assert -1.0 <= s.pr_ar1 <= 1.0


def test_encoder_drift_detected(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=2)
    # Inject a snapshot with a different fingerprint.
    save_snapshot(
        GenerationSnapshot(
            gen=2,
            arm="FED",
            nutrient_ids=["x"],
            X=np.zeros((1, 16)),
            encoder_fingerprint="OTHER@bad:dim16:norm",
            kmeans_k=4,
        ),
        run_dir / "snapshots",
    )
    rm = compute_run_metrics(run_dir)
    assert not rm.encoder_consistent
    assert "ENCODER DRIFT" in format_report(run_dir)


# ---------------------------------------------------------------------------
# Fictions in the report
# ---------------------------------------------------------------------------


def test_report_includes_fiction_count(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=2)
    # STARVED admits a build that fails the external test -> 1 fiction.
    cands = [
        Candidate("a", external_score=0.9, external_pass=True, self_score=0.1),
        Candidate("c", external_score=0.1, external_pass=False, self_score=0.9),
    ]
    res = select_admissions(cands, k=1)
    admission_log.log_arm_generation(
        "pilot", 0, "STARVED", cands, res.admitted_for("STARVED"), db_path=run_dir / "admissions.db"
    )
    report = format_report(run_dir, experiment_id="pilot")
    assert "fictions admitted (failed external test): 1" in report


def test_report_marks_probe_deferred_when_absent(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=2)  # no probe.db written
    report = format_report(run_dir)
    # Deferred must be explicit, not a silent blank or a zero.
    assert "not measured (probe deferred)" in report


# ---------------------------------------------------------------------------
# FED-only calibration
# ---------------------------------------------------------------------------


def test_calibrate_band_reads_only_fed(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=8)
    band = calibrate_fed_band(run_dir, sigma_mult=2.0)
    assert band["n_gens"] == 8  # FED has 8 gens
    assert band["band_low"] <= band["pr_mean"] <= band["band_high"]
    assert band["pr_std"] >= 0.0


def test_calibrate_ignores_starved_generation_count(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=6)
    # Add extra STARVED-only generations; FED count must be unaffected.
    rng = np.random.default_rng(1)
    for gen in range(6, 9):
        _write_snap(run_dir / "snapshots", "STARVED", gen, rng.normal(size=(12, 16)))
    band = calibrate_fed_band(run_dir)
    assert band["n_gens"] == 6  # only FED generations counted


def test_calibrate_raises_without_fed(tmp_path):
    run_dir = tmp_path / "empty"
    (run_dir / "snapshots").mkdir(parents=True)
    with pytest.raises(ValueError):
        calibrate_fed_band(run_dir)


def test_prereg_block_contains_band_and_kill_fraction(tmp_path):
    run_dir = _make_run(tmp_path, n_gens=8)
    block = format_preregistration_block(run_dir, kill_fraction=0.5, n_full=25)
    assert "Noise band" in block
    assert "N=25" in block
    assert "50%" in block
    assert "joint-direction" in block
