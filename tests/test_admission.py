"""Tests for K-matched admission selection + the admission-event log."""

from __future__ import annotations

import pytest

from belief.experiments.admission import (
    Candidate,
    count_fictions,
    select_admissions,
)
from belief.experiments import admission_log


def _c(bid, ext, ext_pass, self_score, conf=0.0):
    return Candidate(
        build_id=bid,
        external_score=ext,
        external_pass=ext_pass,
        self_score=self_score,
        self_confidence=conf,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_arms_admit_same_count_different_members():
    # FED prefers high external_score; STARVED prefers high self_score.
    cands = [
        _c("a", ext=0.9, ext_pass=True, self_score=0.1),
        _c("b", ext=0.8, ext_pass=True, self_score=0.2),
        _c("c", ext=0.1, ext_pass=False, self_score=0.9),
        _c("d", ext=0.2, ext_pass=False, self_score=0.8),
    ]
    res = select_admissions(cands, k=2)
    assert res.k == 2
    assert len(res.fed_admitted) == len(res.starved_admitted) == 2
    assert set(res.fed_admitted) == {"a", "b"}
    assert set(res.starved_admitted) == {"c", "d"}


def test_k_clamped_to_candidate_count():
    cands = [_c("a", 0.5, True, 0.5), _c("b", 0.4, True, 0.4)]
    res = select_admissions(cands, k=10)
    assert res.k == 2
    assert set(res.fed_admitted) == {"a", "b"}
    assert set(res.starved_admitted) == {"a", "b"}


def test_deterministic_tie_break_by_build_id():
    # All equal scores -> ties broken by ascending build_id, stable.
    cands = [_c("z", 0.5, True, 0.5), _c("a", 0.5, True, 0.5), _c("m", 0.5, True, 0.5)]
    res = select_admissions(cands, k=2)
    assert res.fed_admitted == ["a", "m"]
    assert res.starved_admitted == ["a", "m"]


def test_empty_candidates():
    res = select_admissions([], k=4)
    assert res.k == 0
    assert res.fed_admitted == []
    assert res.starved_admitted == []


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        select_admissions([_c("a", 0.5, True, 0.5)], k=-1)


def test_admitted_for_and_is_admitted():
    cands = [_c("a", 0.9, True, 0.1), _c("c", 0.1, False, 0.9)]
    res = select_admissions(cands, k=1)
    assert res.admitted_for("FED") == ["a"]
    assert res.admitted_for("starved") == ["c"]
    assert res.is_admitted("FED", "a")
    assert not res.is_admitted("FED", "c")
    with pytest.raises(ValueError):
        res.admitted_for("MIDDLE")


def test_count_fictions_pure():
    # STARVED admits c,d which both fail the external test -> 2 fictions.
    cands = [
        _c("a", 0.9, True, 0.1),
        _c("b", 0.8, True, 0.2),
        _c("c", 0.1, False, 0.9),
        _c("d", 0.2, False, 0.8),
    ]
    res = select_admissions(cands, k=2)
    assert count_fictions(cands, res) == 2


# ---------------------------------------------------------------------------
# Admission log (SQLite)
# ---------------------------------------------------------------------------


def test_log_generation_writes_two_rows_per_candidate(tmp_path):
    db = tmp_path / "adm.db"
    cands = [
        _c("a", 0.9, True, 0.1, conf=0.3),
        _c("c", 0.1, False, 0.9, conf=0.7),
    ]
    res = select_admissions(cands, k=1)
    rows = admission_log.log_generation("exp1", 0, cands, res, db_path=db)
    assert rows == 4  # 2 candidates x 2 arms
    events = admission_log.fetch_events("exp1", db_path=db)
    assert len(events) == 4
    arms = {(e["build_id"], e["arm"]): e for e in events}
    # FED admitted a, STARVED admitted c.
    assert arms[("a", "FED")]["admitted"] == 1
    assert arms[("a", "STARVED")]["admitted"] == 0
    assert arms[("c", "STARVED")]["admitted"] == 1
    assert arms[("c", "FED")]["admitted"] == 0


def test_count_fictions_from_log(tmp_path):
    db = tmp_path / "adm.db"
    cands = [
        _c("a", 0.9, True, 0.1),
        _c("c", 0.1, False, 0.9),  # STARVED admits this; it fails external -> fiction
    ]
    res = select_admissions(cands, k=1)
    admission_log.log_generation("exp1", 0, cands, res, db_path=db)
    assert admission_log.count_fictions("exp1", db_path=db) == 1


def test_count_fictions_scoped_by_experiment(tmp_path):
    db = tmp_path / "adm.db"
    cands = [_c("c", 0.1, False, 0.9)]
    res = select_admissions(cands, k=1)
    admission_log.log_generation("expA", 0, cands, res, db_path=db)
    assert admission_log.count_fictions("expB", db_path=db) == 0


def test_init_db_idempotent(tmp_path):
    db = tmp_path / "adm.db"
    admission_log.init_db(db)
    admission_log.init_db(db)  # second call must not raise
    assert admission_log.fetch_events("none", db_path=db) == []
