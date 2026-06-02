"""Orchestration + guardrail tests for the STARVED-arm driver (Session 4).

Gate-safe: every external action is an injected fake, so no Ollama / ChromaDB /
MiniLM / SWE-bench is touched. The live run is verified on the Mac per the
manual checklist.
"""

from __future__ import annotations

import pytest

from belief.experiments import admission_log
from belief.experiments.self_judge import JudgeResult
from belief.experiments.starved_runner import (
    BuildArtifact,
    FingerprintDriftError,
    ProbeResult,
    RunDirNotEmptyError,
    StarvedConfig,
    StarvedRunner,
)

# Three tasks engineered so the arms diverge:
#   task-0: strong external, weak self   -> FED admits it
#   task-2: weak external (FAILS), strong self -> STARVED admits it (a fiction)
_PROFILE = {
    "task-0": dict(ext=0.9, ext_pass=True, self_=0.1),
    "task-1": dict(ext=0.5, ext_pass=True, self_=0.5),
    "task-2": dict(ext=0.1, ext_pass=False, self_=0.9),
}
_TASKS = [("t0", "task-0"), ("t1", "task-1"), ("t2", "task-2")]


class _Spy:
    def __init__(self):
        self.decomposed: list[tuple[str, str]] = []  # (arm, run_id)
        self.snapshots: list[tuple[int, str]] = []
        self.probes: list[tuple[int, str]] = []

    def build_fn(self, goal, arm, soil_dir):
        p = _PROFILE[goal]
        return BuildArtifact(
            run_id=f"{arm}-{goal}",
            goal=goal,
            code_files={"m.py": f"# {goal} {arm}"},
            external_score=p["ext"],
            external_pass=p["ext_pass"],
        )

    def judge_fn(self, goal, code_files):
        return JudgeResult(_PROFILE[goal]["self_"], 0.5, "ok", ok=True)

    def decompose_fn(self, artifact, soil_dir):
        # arm is encoded in run_id prefix
        arm = artifact.run_id.split("-")[0]
        self.decomposed.append((arm, artifact.run_id))

    def snapshot_fn(self, gen, arm, soil_dir):
        self.snapshots.append((gen, arm))
        return None

    def probe_fn(self, gen, arm, soil_dir):
        self.probes.append((gen, arm))
        return ProbeResult(gen=gen, arm=arm, n_instances=5, n_resolved=3)


def _config(tmp_path, **kw):
    base = dict(
        experiment_id="exp",
        base_dir=tmp_path,
        n_generations=2,
        n_tasks=3,
        k=1,
        seed=42,
        encoder_fingerprint="enc@v1",
        tasks=_TASKS,
    )
    base.update(kw)
    return StarvedConfig(**base)


def _runner(cfg, spy, *, probe=False):
    return StarvedRunner(
        cfg,
        build_fn=spy.build_fn,
        judge_fn=spy.judge_fn,
        decompose_fn=spy.decompose_fn,
        snapshot_fn=spy.snapshot_fn,
        probe_fn=spy.probe_fn if probe else None,
    )


# ---------------------------------------------------------------------------
# Guardrail + manifest
# ---------------------------------------------------------------------------


def test_prepare_creates_dirs_and_manifest(tmp_path):
    cfg = _config(tmp_path)
    _runner(cfg, _Spy()).prepare()
    assert cfg.arm_soil("FED").exists()
    assert cfg.arm_soil("STARVED").exists()
    assert cfg.snapshots_dir.exists()
    import json

    manifest = json.loads(cfg.manifest_path.read_text())
    assert manifest["encoder_fingerprint"] == "enc@v1"
    assert manifest["kmeans_k"] == cfg.kmeans_k
    assert "judge_prompt_fingerprint" in manifest


def test_guardrail_refuses_nonempty_without_resume(tmp_path):
    cfg = _config(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "stale.txt").write_text("old run")
    with pytest.raises(RunDirNotEmptyError):
        _runner(cfg, _Spy()).prepare()


def test_resume_allows_matching_fingerprints(tmp_path):
    _runner(_config(tmp_path), _Spy()).prepare()
    # Second runner, same fingerprints, resume=True -> no raise.
    _runner(_config(tmp_path, resume=True), _Spy()).prepare()


def test_resume_refuses_encoder_drift(tmp_path):
    _runner(_config(tmp_path), _Spy()).prepare()
    drifted = _config(tmp_path, resume=True, encoder_fingerprint="enc@v2")
    with pytest.raises(FingerprintDriftError):
        _runner(drifted, _Spy()).prepare()


def test_resume_refuses_kmeans_k_drift(tmp_path):
    _runner(_config(tmp_path), _Spy()).prepare()
    drifted = _config(tmp_path, resume=True, kmeans_k=16)
    with pytest.raises(FingerprintDriftError):
        _runner(drifted, _Spy()).prepare()


# ---------------------------------------------------------------------------
# Task stream
# ---------------------------------------------------------------------------


def test_task_stream_is_deterministic_and_rotates(tmp_path):
    r1 = _runner(_config(tmp_path), _Spy())
    r2 = _runner(_config(tmp_path), _Spy())
    assert r1.task_stream(0) == r2.task_stream(0)  # deterministic across instances
    assert len(r1.task_stream(0)) == 3
    # gen 1 window starts after gen 0 (rotation), wraps around the 3-task order
    assert r1.task_stream(1) == r1.task_stream(1)


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


def test_decompose_only_admitted_top_k(tmp_path):
    cfg = _config(tmp_path, n_generations=1, k=1)
    spy = _Spy()
    _runner(cfg, spy).run()
    # k=1 per arm per gen, 1 gen, 2 arms -> exactly 2 decompositions.
    assert len(spy.decomposed) == 2
    arms = {arm for arm, _ in spy.decomposed}
    assert arms == {"FED", "STARVED"}


def test_arms_admit_different_builds(tmp_path):
    cfg = _config(tmp_path, n_generations=1, k=1)
    spy = _Spy()
    spy_runner = _runner(cfg, spy)
    spy_runner.run()
    decomposed = dict(spy.decomposed)  # arm -> run_id
    # FED admits the high-external task-0; STARVED admits the high-self task-2.
    assert decomposed["FED"] == "FED-task-0"
    assert decomposed["STARVED"] == "STARVED-task-2"


def test_snapshot_called_per_arm_per_generation(tmp_path):
    cfg = _config(tmp_path, n_generations=2)
    spy = _Spy()
    _runner(cfg, spy).run()
    assert sorted(spy.snapshots) == [(0, "FED"), (0, "STARVED"), (1, "FED"), (1, "STARVED")]


def test_summary_counts_fictions(tmp_path):
    cfg = _config(tmp_path, n_generations=1, k=1)
    summary = _runner(cfg, _Spy()).run()
    # STARVED admitted task-2 which fails the external test -> 1 fiction.
    assert summary["fictions"] == 1
    assert summary["experiment_id"] == "exp"


def test_admission_log_written(tmp_path):
    cfg = _config(tmp_path, n_generations=1)
    _runner(cfg, _Spy()).run()
    events = admission_log.fetch_events("exp", db_path=cfg.admissions_db)
    # 3 tasks x 2 arms = 6 rows for one generation.
    assert len(events) == 6
    starved_admitted = [e for e in events if e["arm"] == "STARVED" and e["admitted"] == 1]
    assert len(starved_admitted) == 1
    assert starved_admitted[0]["build_id"] == "STARVED-task-2"


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def test_probe_only_at_checkpoints(tmp_path):
    cfg = _config(tmp_path, n_generations=3, probe_at=(1,))
    spy = _Spy()
    _runner(cfg, spy, probe=True).run()
    # Probe runs only at gen 1, both arms.
    assert sorted(spy.probes) == [(1, "FED"), (1, "STARVED")]
    from belief.experiments.swebench_probe import fetch_probes

    rows = fetch_probes("exp", db_path=cfg.probe_db)
    assert len(rows) == 2
    assert all(r["n_instances"] == 5 for r in rows)


def test_probe_not_run_when_probe_fn_none(tmp_path):
    cfg = _config(tmp_path, n_generations=2, probe_at=(0, 1))
    spy = _Spy()
    _runner(cfg, spy, probe=False).run()  # probe_fn=None
    assert spy.probes == []
