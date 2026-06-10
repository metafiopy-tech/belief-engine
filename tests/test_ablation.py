"""Tests for the self-ablation instrument (agent-harness program, stage #3).

Gate-safe: deterministic fake build_fn + metric_fn, no engine/model. The fakes
encode a known per-arm effect so attribution can be asserted exactly.
"""

from __future__ import annotations

import json

import pytest

from belief.experiments.ablation import (
    AblationArm,
    AblationConfig,
    AblationRunner,
    ManifestDriftError,
    RunDirNotEmptyError,
    default_metric_fn,
    format_report,
    soundness_arms,
)

_TASKS = [("t0", "task-0"), ("t1", "task-1"), ("t2", "task-2"), ("t3", "task-3")]


def _config(tmp_path, **kw):
    base = dict(
        experiment_id="abl",
        base_dir=tmp_path,
        arms=[
            AblationArm("baseline", env={}),
            AblationArm("no_soil", env={"BELIEF_EXPERIMENT_CONDITION": "raw_local"}),
        ],
        tasks=_TASKS,
        baseline="baseline",
    )
    base.update(kw)
    return AblationConfig(**base)


def _build_fn(goal, arm_name, soil_dir, env):
    """Outcome = a dict; baseline succeeds 0.9, an ablated arm succeeds 0.6.

    Also asserts the harness set the per-arm soil path in env (isolation).
    """
    assert env["BELIEF_SOIL_PATH"].endswith(f"{arm_name}/soil")
    score = 0.9 if arm_name == "baseline" else 0.6
    return {"arm": arm_name, "score": score}


def _metric(outcome):
    return outcome["score"]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_duplicate_arm_names(tmp_path):
    with pytest.raises(ValueError):
        AblationConfig(
            experiment_id="x",
            base_dir=tmp_path,
            arms=[AblationArm("a"), AblationArm("a")],
            tasks=_TASKS,
            baseline="a",
        )


def test_config_rejects_unknown_baseline(tmp_path):
    with pytest.raises(ValueError):
        AblationConfig(
            experiment_id="x",
            base_dir=tmp_path,
            arms=[AblationArm("a")],
            tasks=_TASKS,
            baseline="missing",
        )


# ---------------------------------------------------------------------------
# Guardrail + manifest
# ---------------------------------------------------------------------------


def test_prepare_creates_soil_dirs_and_manifest(tmp_path):
    cfg = _config(tmp_path)
    AblationRunner(cfg, build_fn=_build_fn, metric_fn=_metric).prepare()
    assert cfg.arm_soil("baseline").exists()
    assert cfg.arm_soil("no_soil").exists()
    manifest = json.loads(cfg.manifest_path.read_text())
    assert manifest["baseline"] == "baseline"
    assert manifest["arms"]["no_soil"] == {"BELIEF_EXPERIMENT_CONDITION": "raw_local"}


def test_guardrail_refuses_nonempty(tmp_path):
    cfg = _config(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "stale").write_text("x")
    with pytest.raises(RunDirNotEmptyError):
        AblationRunner(cfg, build_fn=_build_fn, metric_fn=_metric).prepare()


def test_resume_refuses_arm_drift(tmp_path):
    AblationRunner(_config(tmp_path), build_fn=_build_fn, metric_fn=_metric).prepare()
    drifted = _config(
        tmp_path,
        resume=True,
        arms=[AblationArm("baseline"), AblationArm("no_soil", env={"X": "different"})],
    )
    with pytest.raises(ManifestDriftError):
        AblationRunner(drifted, build_fn=_build_fn, metric_fn=_metric).prepare()


def test_resume_ok_when_unchanged(tmp_path):
    AblationRunner(_config(tmp_path), build_fn=_build_fn, metric_fn=_metric).prepare()
    AblationRunner(_config(tmp_path, resume=True), build_fn=_build_fn, metric_fn=_metric).prepare()


# ---------------------------------------------------------------------------
# Run + attribution
# ---------------------------------------------------------------------------


def test_run_attributes_delta_to_baseline(tmp_path):
    report = AblationRunner(_config(tmp_path), build_fn=_build_fn, metric_fn=_metric).run()
    assert report.results["baseline"].metric_mean == pytest.approx(0.9)
    assert report.results["no_soil"].metric_mean == pytest.approx(0.6)
    # delta is arm - baseline.
    assert report.deltas["no_soil"] == pytest.approx(-0.3)
    assert report.deltas["baseline"] == pytest.approx(0.0)


def test_load_bearing_threshold(tmp_path):
    report = AblationRunner(_config(tmp_path), build_fn=_build_fn, metric_fn=_metric).run()
    # |−0.3| clears a 0.1 band but not a 0.5 band.
    assert report.load_bearing("no_soil", noise_band=0.1) is True
    assert report.load_bearing("no_soil", noise_band=0.5) is False


def test_run_arm_isolates_soil_path(tmp_path):
    # _build_fn asserts the soil path; reaching here means isolation held.
    AblationRunner(_config(tmp_path), build_fn=_build_fn, metric_fn=_metric).run()


def test_three_arm_ablation(tmp_path):
    # Reproduces the STARVED-shaped soundness check: baseline vs two ablations.
    def bf(goal, arm, soil, env):
        return {"score": {"baseline": 0.9, "no_soil": 0.6, "no_decompose": 0.8}[arm]}

    cfg = _config(
        tmp_path,
        arms=[
            AblationArm("baseline"),
            AblationArm("no_soil", env={"BELIEF_EXPERIMENT_CONDITION": "raw_local"}),
            AblationArm("no_decompose", env={"BELIEF_SUPPRESS_DECOMPOSE": "1"}),
        ],
    )
    report = AblationRunner(cfg, build_fn=bf, metric_fn=_metric).run()
    assert report.deltas["no_soil"] == pytest.approx(-0.3)
    assert report.deltas["no_decompose"] == pytest.approx(-0.1)


def test_format_report_marks_load_bearing(tmp_path):
    report = AblationRunner(_config(tmp_path), build_fn=_build_fn, metric_fn=_metric).run()
    text = format_report(report, noise_band=0.1)
    assert "baseline" in text
    assert "LOAD-BEARING" in text
    assert "delta=-0.300" in text


# ---------------------------------------------------------------------------
# seed_soil + preset + metric (Session 2)
# ---------------------------------------------------------------------------


def test_seed_soil_copied_into_arm(tmp_path):
    src = tmp_path / "seed"
    src.mkdir()
    (src / "marker.txt").write_text("seeded")
    cfg = AblationConfig(
        experiment_id="s",
        base_dir=tmp_path / "runs",
        arms=[AblationArm("baseline", seed_soil=src), AblationArm("no_soil", env={"X": "1"})],
        tasks=_TASKS,
        baseline="baseline",
    )
    AblationRunner(cfg, build_fn=_build_fn, metric_fn=_metric).prepare()
    assert (cfg.arm_soil("baseline") / "marker.txt").read_text() == "seeded"
    assert not (cfg.arm_soil("no_soil") / "marker.txt").exists()


def test_seed_soil_missing_raises(tmp_path):
    cfg = AblationConfig(
        experiment_id="s",
        base_dir=tmp_path / "runs",
        arms=[AblationArm("baseline", seed_soil=tmp_path / "nope")],
        tasks=_TASKS,
        baseline="baseline",
    )
    with pytest.raises(FileNotFoundError):
        AblationRunner(cfg, build_fn=_build_fn, metric_fn=_metric).prepare()


def test_seed_soil_drift_detected(tmp_path):
    src = tmp_path / "seed"
    src.mkdir()
    base = dict(experiment_id="s", base_dir=tmp_path / "runs", tasks=_TASKS, baseline="baseline")
    AblationRunner(
        AblationConfig(arms=[AblationArm("baseline", seed_soil=src)], **base),
        build_fn=_build_fn,
        metric_fn=_metric,
    ).prepare()
    drifted = AblationConfig(arms=[AblationArm("baseline", seed_soil=None)], resume=True, **base)
    with pytest.raises(ManifestDriftError):
        AblationRunner(drifted, build_fn=_build_fn, metric_fn=_metric).prepare()


def test_soundness_preset_shape():
    arms = soundness_arms()
    names = [a.name for a in arms]
    assert names == ["baseline", "no_soil", "no_decompose"]
    assert arms[1].env == {"BELIEF_EXPERIMENT_CONDITION": "raw_local"}
    assert arms[2].env == {"BELIEF_SUPPRESS_DECOMPOSE": "1"}


def test_soundness_preset_threads_seed_soil(tmp_path):
    arms = soundness_arms(seed_soil=tmp_path / "seed")
    assert all(a.seed_soil == tmp_path / "seed" for a in arms)


def test_default_metric_reads_weighted_score():
    assert default_metric_fn({"weighted_score": 0.73}) == pytest.approx(0.73)
    assert default_metric_fn({}) == 0.0
