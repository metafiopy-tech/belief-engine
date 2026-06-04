"""Tests for the SWE-bench Verified checkpoint probe (store + orchestration).

The real harness (run_instances) is intentionally unimplemented; these tests
cover the result store, the make_probe_fn factory with an injected resolver, and
the loud-failure contract of the un-wired harness.
"""

from __future__ import annotations

import pytest

from belief.experiments import swebench_probe
from belief.experiments.starved_runner import ProbeResult, StarvedConfig


def _config(tmp_path):
    return StarvedConfig(
        experiment_id="exp",
        base_dir=tmp_path,
        n_generations=1,
        n_tasks=2,
        k=1,
    )


def test_record_and_fetch_roundtrip(tmp_path):
    db = tmp_path / "probe.db"
    swebench_probe.record_probe("exp", ProbeResult(2, "STARVED", 10, 4), db_path=db)
    swebench_probe.record_probe("exp", ProbeResult(2, "FED", 10, 9), db_path=db)
    rows = swebench_probe.fetch_probes("exp", db_path=db)
    assert len(rows) == 2
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["STARVED"]["n_resolved"] == 4
    assert by_arm["FED"]["resolve_rate"] == pytest.approx(0.9)


def test_fetch_missing_db_returns_empty(tmp_path):
    assert swebench_probe.fetch_probes("exp", db_path=tmp_path / "nope.db") == []


def test_probe_result_resolve_rate():
    assert ProbeResult(0, "FED", 0, 0).resolve_rate == 0.0
    assert ProbeResult(0, "FED", 4, 1).resolve_rate == 0.25


def test_make_probe_fn_with_injected_runner(tmp_path):
    calls = []

    def fake_runner(ids, soil_dir):
        calls.append((ids, soil_dir))
        return 2

    probe = swebench_probe.make_probe_fn(
        _config(tmp_path), instance_ids=("a", "b", "c"), runner=fake_runner
    )
    res = probe(3, "STARVED", tmp_path / "soil")
    assert isinstance(res, ProbeResult)
    assert res.n_instances == 3
    assert res.n_resolved == 2
    assert res.gen == 3 and res.arm == "STARVED"
    assert len(calls) == 1


def test_make_probe_fn_refuses_empty_instances(tmp_path):
    with pytest.raises(ValueError):
        swebench_probe.make_probe_fn(_config(tmp_path), instance_ids=())


def test_run_instances_composes_seams(tmp_path):
    # Injected loader/predictor/evaluator -> orchestration without Docker/datasets.
    seen = {}

    def fake_loader(ids, **kw):
        seen["ids"] = ids
        return [{"instance_id": i} for i in ids]

    def fake_predictor(inst, soil_dir, *, model):
        return {"instance_id": inst["instance_id"], "model_patch": "diff"}

    def fake_evaluator(preds, ids, *, run_id, **kw):
        # Resolve the first instance only.
        return {ids[0]}

    n = swebench_probe.run_instances(
        ("a", "b", "c"),
        tmp_path / "soil",
        loader=fake_loader,
        predictor=fake_predictor,
        evaluator=fake_evaluator,
    )
    assert n == 1
    assert seen["ids"] == ("a", "b", "c")


def test_run_instances_zero_resolved(tmp_path):
    def empty_patch(inst, soil, *, model):
        return {"instance_id": inst["instance_id"], "model_patch": ""}

    n = swebench_probe.run_instances(
        ("a",),
        tmp_path / "soil",
        loader=lambda ids, **kw: [{"instance_id": i} for i in ids],
        predictor=empty_patch,
        evaluator=lambda preds, ids, *, run_id, **kw: set(),
    )
    assert n == 0


def test_evaluate_predictions_skips_empty_patches(tmp_path):
    # All-empty patches -> harness never invoked, returns empty set.
    resolved = swebench_probe.evaluate_predictions(
        [{"instance_id": "a", "model_patch": "  "}],
        ("a",),
        run_id="t",
        work_dir=tmp_path,
    )
    assert resolved == set()


def test_make_probe_fn_uses_injected_runner_over_real(tmp_path):
    # When a runner is injected, the real run_instances path is not used.
    probe = swebench_probe.make_probe_fn(
        _config(tmp_path), instance_ids=("a", "b"), runner=lambda ids, soil: 2
    )
    res = probe(0, "FED", tmp_path / "soil")
    assert res.n_instances == 2 and res.n_resolved == 2
