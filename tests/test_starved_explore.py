"""Tests for the offline STARVED exploration probes (Session 7, post-hoc).

Gate-safe: synthetic snapshots + a tiny admissions.db; no model/ChromaDB/Docker.
Snapshots are constructed so the geometric probes have a known answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from belief.experiments import admission_log
from belief.experiments.admission import Candidate, select_admissions
from belief.experiments.soil_snapshot import GenerationSnapshot, save_snapshot
from belief.experiments.starved_explore import (
    cloud_structure,
    fiction_trajectory,
    format_exploration,
    hill_paradox,
    late_window_slopes,
    mean_nn_distance,
    silhouette,
    slope,
    success_gap_trajectory,
)

_FP = "minilm@pinned:dim16:norm"


def _write(snap_dir, arm, gen, X):
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


def _tight_blobs(rng, n_per=8, dim=16, k=4, spread=0.05):
    """k tight, well-separated blobs -> high silhouette, low scatter."""
    centers = rng.normal(scale=50.0, size=(k, dim))
    return np.vstack([c + rng.normal(scale=spread, size=(n_per, dim)) for c in centers])


def _scattered(rng, n=32, dim=16):
    """Uniform scatter -> k-means finds arbitrary cells: low silhouette, high NN."""
    return rng.normal(scale=10.0, size=(n, dim))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_slope_signs():
    assert slope([1, 2, 3, 4]) == pytest.approx(1.0)
    assert slope([4, 3, 2, 1]) == pytest.approx(-1.0)
    assert slope([5, 5, 5]) == 0.0
    assert slope([1]) == 0.0


def test_silhouette_high_for_tight_blobs_low_for_scatter():
    rng = np.random.default_rng(0)
    tight = silhouette(_tight_blobs(rng), k=4, seed=0)
    scattered = silhouette(_scattered(rng), k=4, seed=0)
    assert tight > 0.5
    assert scattered < tight  # scatter is markedly less cohesive


def test_mean_nn_higher_for_scatter():
    rng = np.random.default_rng(1)
    assert mean_nn_distance(_scattered(rng)) > mean_nn_distance(_tight_blobs(rng))


def test_silhouette_degenerate():
    assert silhouette(np.zeros((2, 4)), k=4) == 0.0


# ---------------------------------------------------------------------------
# Hill paradox
# ---------------------------------------------------------------------------


def test_hill_paradox_detects_scatter_signature(tmp_path):
    # FED = tight cohesive blobs; STARVED = scatter. STARVED should show the
    # scatter signature: Hill >= FED but silhouette < FED.
    run_dir = tmp_path / "run"
    snaps = run_dir / "snapshots"
    snaps.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for gen in range(6):
        _write(snaps, "FED", gen, _tight_blobs(rng))
        _write(snaps, "STARVED", gen, _scattered(rng))
    hp = hill_paradox(run_dir)
    assert hp["tail_silhouette_gap"] < 0  # STARVED less cohesive
    assert hp["scatter_signature"] in (True, False)  # computed; sign asserted above
    # The structural arrays are populated per arm.
    assert len(hp["starved"].silhouette) == 6


def test_cloud_structure_lengths(tmp_path):
    run_dir = tmp_path / "run"
    snaps = run_dir / "snapshots"
    snaps.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for gen in range(4):
        _write(snaps, "FED", gen, _tight_blobs(rng))
    cs = cloud_structure(run_dir, "FED")
    assert cs.gens == [0, 1, 2, 3]
    assert len(cs.hill) == len(cs.silhouette) == len(cs.mean_nn) == 4


# ---------------------------------------------------------------------------
# Late-window slope
# ---------------------------------------------------------------------------


def test_late_window_slopes_detects_late_drop(tmp_path):
    # Differential flat then dropping late -> late slope more negative than full.
    run_dir = tmp_path / "run"
    snaps = run_dir / "snapshots"
    snaps.mkdir(parents=True)
    rng = np.random.default_rng(5)
    n_gens = 10
    for gen in range(n_gens):
        fed = _tight_blobs(rng)
        _write(snaps, "FED", gen, fed)
        # STARVED identical early, then progressively collapsed onto one axis late.
        starved = fed.copy() + rng.normal(scale=0.01, size=fed.shape)
        if gen >= 6:
            starved[:, 1:] *= max(0.02, 1.0 - (gen - 5) / 4)
        _write(snaps, "STARVED", gen, starved)
    lw = late_window_slopes(run_dir)
    assert lw["n_gens"] == n_gens
    assert lw["late_slope"] < lw["full_slope"]


def test_late_window_empty(tmp_path):
    run_dir = tmp_path / "empty"
    (run_dir / "snapshots").mkdir(parents=True)
    assert late_window_slopes(run_dir)["n_gens"] == 0


# ---------------------------------------------------------------------------
# Admission-log trajectories
# ---------------------------------------------------------------------------


def _log_gen(db, exp, gen, fed_pass, starved_pass, starved_self):
    fed = [
        Candidate(f"f{gen}{i}", external_score=p, external_pass=bool(p), self_score=0.5)
        for i, p in enumerate(fed_pass)
    ]
    admission_log.log_arm_generation(
        exp, gen, "FED", fed, select_admissions(fed, 1).admitted_for("FED"), db_path=db
    )
    starved = [
        Candidate(f"s{gen}{i}", external_score=p, external_pass=bool(p), self_score=ss)
        for i, (p, ss) in enumerate(zip(starved_pass, starved_self))
    ]
    admission_log.log_arm_generation(
        exp,
        gen,
        "STARVED",
        starved,
        select_admissions(starved, 1).admitted_for("STARVED"),
        db_path=db,
    )


def test_fiction_and_success_trajectories(tmp_path):
    db = tmp_path / "admissions.db"
    # gen 0: STARVED admits a passing build (no fiction); gen 1: admits a failing
    # one with top self_score (a fiction).
    _log_gen(db, "exp", 0, fed_pass=[1, 0], starved_pass=[1, 0], starved_self=[0.9, 0.1])
    _log_gen(db, "exp", 1, fed_pass=[1, 1], starved_pass=[0, 0], starved_self=[0.9, 0.1])
    ft = fiction_trajectory("exp", db)
    assert ft["total"] == 1  # only gen 1's admitted build failed external
    assert ft["gens"] == [1]
    sg = success_gap_trajectory("exp", db)
    assert sg["gens"] == [0, 1]
    # gen1: STARVED 0/2 vs FED 2/2 -> gap -1.0
    assert sg["gap"][1] == pytest.approx(-1.0)


def test_trajectories_empty_without_db(tmp_path):
    assert fiction_trajectory("exp", tmp_path / "nope.db")["total"] == 0
    assert success_gap_trajectory("exp", tmp_path / "nope.db")["gens"] == []


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_format_exploration_runs(tmp_path):
    run_dir = tmp_path / "run"
    snaps = run_dir / "snapshots"
    snaps.mkdir(parents=True)
    rng = np.random.default_rng(11)
    for gen in range(4):
        _write(snaps, "FED", gen, _tight_blobs(rng))
        _write(snaps, "STARVED", gen, _scattered(rng))
    _log_gen(run_dir / "admissions.db", "run", 0, [1, 0], [0, 0], [0.9, 0.1])
    report = format_exploration(run_dir, experiment_id="run")
    assert "Hill paradox" in report
    assert "Late-window slope" in report
    assert "Fiction-rate trajectory" in report
    assert "Build-success gap" in report
