"""Session C (handoff Q2): planned-vs-produced coverage gate.

Hermetic — the coverage gate is a pure, dependency-free module. These tests
cover the hollow-file detector, the coverage computation, and the verdict
downgrade.
"""

from __future__ import annotations

import pytest

from belief.validators.coverage_gate import (
    HOLLOW_SCORE_CAP,
    compute_coverage,
    find_hollow_files,
    gate_validation_result,
    is_hollow_file,
    planned_filenames,
)


# ── is_hollow_file ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,expected",
    [
        ("", True),
        ("   \n  ", True),
        ("import os\nimport sys\n", True),
        ("def f():\n    pass\n", True),
        ('"""just a docstring"""\n', True),
        ("def f():\n    raise NotImplementedError\n", True),
        ("class C:\n    pass\n", True),  # bare stub class
        ("x = 1\n", False),  # an assignment is substantive
        ("def f():\n    return 1\n", False),
        ("class C:\n    x: int = 0\n", False),
        ("print('hi')\n", False),
        ("def (oops\n", False),  # parse error → not our concern (compile_gate)
        # Research-brief follow-up: trivial / placeholder returns are stubs.
        ("def f():\n    return None\n", True),
        ("def f():\n    return\n", True),
        ("def f():\n    return []\n", True),
        ("def f():\n    return {}\n", True),
        ("def f():\n    return 'TODO: implement'\n", True),
        ("def f():\n    return 'hello'\n", False),  # a real string return
        # Research-brief caveat: subclass-only modules are NOT hollow.
        ("class NotFoundError(Exception):\n    pass\n", False),  # legit exception
        (
            "from abc import ABC, abstractmethod\n"
            "class Repo(ABC):\n    @abstractmethod\n    def get(self): ...\n",
            False,
        ),  # ABC interface
        (
            "from typing import Protocol\nclass P(Protocol):\n    def m(self) -> int: ...\n",
            False,
        ),  # Protocol interface
        ("import enum\nclass Color(enum.Enum):\n    RED = 1\n", False),  # enum w/ member
    ],
)
def test_is_hollow_file(content, expected):
    assert is_hollow_file(content) is expected


def test_exception_module_not_flagged_hollow():
    # A whole module of pure-`pass` exception classes is a complete deliverable
    # (the architect fallback emits exactly this as exceptions.py).
    code = {
        "exceptions.py": (
            "class NotFoundError(Exception):\n    pass\n"
            "class DuplicateError(Exception):\n    pass\n"
        )
    }
    assert find_hollow_files(code) == []


def test_all_trivial_return_module_is_hollow():
    code = {"app.py": "def get():\n    return None\n\ndef fetch():\n    return []\n"}
    assert find_hollow_files(code) == ["app.py"]


# ── planned_filenames ────────────────────────────────────────────────────────


def test_planned_filenames_from_dict():
    manifest = {"files": [{"filename": "a.py"}, {"filename": "b.py"}]}
    assert planned_filenames(manifest) == ["a.py", "b.py"]


def test_planned_filenames_none():
    assert planned_filenames(None) == []


# ── compute_coverage ─────────────────────────────────────────────────────────


def test_coverage_partial():
    frac, missing = compute_coverage(["a.py", "b.py"], {"a.py": "x=1"})
    assert frac == pytest.approx(0.5)
    assert missing == ["b.py"]


def test_coverage_empty_plan_is_full():
    frac, missing = compute_coverage([], {"a.py": "x=1"})
    assert frac == pytest.approx(1.0)
    assert missing == []


def test_coverage_basename_fallback():
    # planned under a dir, produced at root → not a miss (dir reshuffle)
    frac, missing = compute_coverage(["src/a.py"], {"a.py": "def f():\n return 1\n"})
    assert frac == pytest.approx(1.0)
    assert missing == []


# ── find_hollow_files ────────────────────────────────────────────────────────


def test_find_hollow_excludes_init_and_tests():
    produced = {
        "app.py": "def f():\n    pass\n",  # hollow
        "__init__.py": "",  # excluded
        "test_x.py": "x = 1\n",  # test file, excluded
        "real.py": "def g():\n    return 1\n",  # substantive
    }
    assert find_hollow_files(produced) == ["app.py"]


# ── gate_validation_result ───────────────────────────────────────────────────


def _passing() -> dict:
    return {"verdict": "pass", "weighted_score": 1.0, "correctness_score": 1.0, "issues": []}


def test_gate_missing_files_forces_fail_and_caps():
    manifest = {"files": [{"filename": "a.py"}, {"filename": "b.py"}]}
    code = {"a.py": "def f():\n    return 1\n"}
    result, frac, missing, hollow = gate_validation_result(manifest, code, _passing())
    assert frac == pytest.approx(0.5)
    assert missing == ["b.py"]
    assert result["verdict"] == "fail_fixable"
    assert result["weighted_score"] <= 0.5
    assert any("not produced" in i for i in result["issues"])


def test_gate_hollow_file_caught_even_when_all_present():
    manifest = {"files": [{"filename": "a.py"}]}
    code = {"a.py": "def f():\n    pass\n"}  # present but hollow
    result, frac, missing, hollow = gate_validation_result(manifest, code, _passing())
    assert frac == pytest.approx(1.0)
    assert hollow == ["a.py"]
    assert result["verdict"] == "fail_fixable"
    assert result["weighted_score"] <= HOLLOW_SCORE_CAP
    assert any("hollow stub" in i for i in result["issues"])


def test_gate_complete_build_untouched():
    manifest = {"files": [{"filename": "a.py"}]}
    code = {"a.py": "def f():\n    return 1\n"}
    validation = _passing()
    result, frac, missing, hollow = gate_validation_result(manifest, code, validation)
    assert frac == pytest.approx(1.0)
    assert missing == [] and hollow == []
    # Returned unchanged (same object) — no downgrade.
    assert result is validation
    assert result["verdict"] == "pass"
    assert result["weighted_score"] == pytest.approx(1.0)


def test_gate_no_manifest_only_hollow_check():
    # No plan → coverage can't be assessed (1.0), but hollow check still runs.
    code = {"a.py": "def f():\n    return 1\n"}
    result, frac, missing, hollow = gate_validation_result(None, code, _passing())
    assert frac == pytest.approx(1.0)
    assert missing == [] and hollow == []
    assert result["verdict"] == "pass"
