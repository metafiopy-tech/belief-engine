"""Hermetic tests for the terminal compile gate.

The gate is the post-refinement backstop that prevents a build from reporting
``pass`` / archiving a non-zero score while shipping a ``.py`` file that does
not parse. Two layers under test:

  1. The pure functions (``find_uncompilable_files``, ``gate_validation_result``)
     — no graph, no langgraph, no I/O.
  2. The graph node (``_compile_gate_node``) operating on dict state, mirroring
     how the compiled pipeline invokes it.
"""

from __future__ import annotations

from belief.validators.compile_gate import (
    find_uncompilable_files,
    gate_validation_result,
)

_GOOD = "def f(x):\n    return x + 1\n"
# Ends mid-call, exactly like a max_tokens-truncated builder file.
_TRUNCATED = "def f(x):\n    return SynthesisResult(\n"


# ── find_uncompilable_files ────────────────────────────────────────────────


def test_clean_files_report_nothing() -> None:
    assert find_uncompilable_files({"a.py": _GOOD, "b.py": _GOOD}) == []


def test_truncated_file_is_flagged() -> None:
    broken = find_uncompilable_files({"ok.py": _GOOD, "bad.py": _TRUNCATED})
    assert [f for f, _ in broken] == ["bad.py"]
    assert "line" in broken[0][1]


def test_non_python_files_ignored() -> None:
    # A .txt / .md / .j2 file with "invalid python" must not be parsed.
    assert find_uncompilable_files({"notes.md": "def (((", "t.j2": "{{ x"}) == []


def test_empty_and_none_inputs() -> None:
    assert find_uncompilable_files({}) == []
    assert find_uncompilable_files(None) == []


# ── gate_validation_result ─────────────────────────────────────────────────


def test_clean_build_passes_through_unchanged() -> None:
    vr = {"verdict": "pass", "weighted_score": 1.0, "issues": []}
    result, broken = gate_validation_result({"a.py": _GOOD}, vr)
    assert broken == []
    assert result is vr  # untouched, same object


def test_broken_build_is_downgraded_and_floored() -> None:
    vr = {
        "verdict": "pass",
        "weighted_score": 1.0,
        "correctness_score": 1.0,
        "completeness_score": 1.0,
        "issues": ["pre-existing"],
    }
    result, broken = gate_validation_result({"ok.py": _GOOD, "bad.py": _TRUNCATED}, vr)
    assert [f for f, _ in broken] == ["bad.py"]
    assert result["verdict"] == "fail_fixable"
    assert result["weighted_score"] == 0.0
    assert result["correctness_score"] == 0.0
    assert result["completeness_score"] == 0.0
    # Original issue preserved, gate issue appended naming the file.
    assert "pre-existing" in result["issues"]
    assert any("bad.py" in i and "compile_gate" in i for i in result["issues"])
    # Pure: caller's dict not mutated in place.
    assert vr["verdict"] == "pass"


def test_none_validation_result_yields_failing_dict() -> None:
    result, broken = gate_validation_result({"bad.py": _TRUNCATED}, None)
    assert broken
    assert result["verdict"] == "fail_fixable"
    assert result["weighted_score"] == 0.0


# ── graph node ─────────────────────────────────────────────────────────────


def test_node_downgrades_state_on_broken_file() -> None:
    from belief.graph import _compile_gate_node

    state = {
        "code_files": {"ok.py": _GOOD, "bad.py": _TRUNCATED},
        "validation_result": {"verdict": "pass", "weighted_score": 1.0, "issues": []},
    }
    out = _compile_gate_node(state)
    assert out["validation_result"]["verdict"] == "fail_fixable"
    assert out["validation_result"]["weighted_score"] == 0.0


def test_node_passthrough_on_clean_build() -> None:
    from belief.graph import _compile_gate_node

    vr = {"verdict": "pass", "weighted_score": 1.0, "issues": []}
    state = {"code_files": {"a.py": _GOOD}, "validation_result": vr}
    out = _compile_gate_node(state)
    assert out["validation_result"]["verdict"] == "pass"
    assert out["validation_result"]["weighted_score"] == 1.0


def test_node_handles_missing_validation_result() -> None:
    from belief.graph import _compile_gate_node

    state = {"code_files": {"bad.py": _TRUNCATED}}
    out = _compile_gate_node(state)
    assert out["validation_result"]["verdict"] == "fail_fixable"
