"""Tests for the three-tier decomposition (mycorrhizal Stage 7, Area 4)."""

from __future__ import annotations

from belief.memory.decomposers import (
    decompose_failed_build,
    extract_clean_fragments,
    extract_composition_edges,
    extract_failure_signature,
)


# ── Easy tier ────────────────────────────────────────────────────────────────


def test_easy_extracts_clean_functions() -> None:
    code = {
        "mod.py": (
            "def good_one():\n"
            "    return 1\n\n"
            "class Helper:\n"
            "    def method(self):\n"
            "        return 2\n"
        )
    }
    frags = extract_clean_fragments(code, source_build_id="b1")
    names = {f.name for f in frags}
    assert "good_one" in names
    assert "Helper" in names
    assert all(f.source_build_id == "b1" for f in frags)


def test_easy_skips_unparseable_file() -> None:
    code = {"broken.py": "def x(:\n  pass"}  # syntax error
    assert extract_clean_fragments(code) == []


def test_easy_recovers_from_partial_file() -> None:
    """A file that parses as a whole yields all its clean top-level defs,
    even if one function is logically broken (runtime, not syntax)."""
    code = {
        "m.py": (
            "def works():\n    return 42\n\n"
            "def also_works():\n    return undefined_name  # runtime error, parses fine\n"
        )
    }
    frags = extract_clean_fragments(code)
    assert {f.name for f in frags} == {"works", "also_works"}


def test_easy_ignores_non_python() -> None:
    code = {"data.json": '{"x": 1}', "readme.md": "# hi"}
    assert extract_clean_fragments(code) == []


# ── Structural tier ──────────────────────────────────────────────────────────


def test_structural_extracts_imports_and_calls() -> None:
    code = {
        "app.py": (
            "import os\n"
            "from fastapi import FastAPI\n"
            "def main():\n"
            "    app = FastAPI()\n"
            "    os.getcwd()\n"
        )
    }
    edges = extract_composition_edges(code, source_build_id="b1", failure_annotation="boom")
    imports = {e.dst for e in edges if e.kind == "import"}
    calls = {e.dst for e in edges if e.kind == "call"}
    assert "os" in imports
    assert "fastapi" in imports
    assert "FastAPI" in calls
    assert "os.getcwd" in calls
    assert all(e.failure_annotation == "boom" for e in edges)


def test_structural_skips_unparseable() -> None:
    assert extract_composition_edges({"x.py": "import (("}) == []


# ── Recalcitrant tier ────────────────────────────────────────────────────────


def test_recalcitrant_builds_signature() -> None:
    sig = extract_failure_signature(
        errors=["Traceback...", "RecursionError: maximum recursion depth exceeded"],
        source_build_id="b1",
    )
    assert sig is not None
    assert sig.error_type == "RecursionError"
    assert "recursion" in sig.systemic_markers
    assert sig.source_build_id == "b1"


def test_recalcitrant_none_on_clean() -> None:
    assert extract_failure_signature(errors=[], exec_error="") is None


def test_recalcitrant_signature_stable_across_volatile_bits() -> None:
    """Two instances of the same failure mode (differing only in line
    numbers / addresses) hash to the same signature_id."""
    a = extract_failure_signature(errors=["ValueError at line 42 in /tmp/x/foo.py"])
    b = extract_failure_signature(errors=["ValueError at line 99 in /tmp/y/bar.py"])
    assert a is not None and b is not None
    assert a.signature_id == b.signature_id


# ── Dispatcher ───────────────────────────────────────────────────────────────


def test_dispatcher_skips_passing_build() -> None:
    state = {
        "run_id": "b-pass",
        "validation_result": {"verdict": "pass"},
        "code_files": {"m.py": "def f():\n    return 1\n"},
    }
    result = decompose_failed_build(state)
    assert result.build_passed is True
    assert result.tiers_run == []
    assert result.recovered_anything is False


def test_dispatcher_runs_tiers_on_failed_build() -> None:
    state = {
        "run_id": "b-fail",
        "validation_result": {"verdict": "fail"},
        "execution_result": {"success": False, "error_summary": "ImportError: no module named foo"},
        "code_files": {
            "app.py": "import foo\ndef handler():\n    return foo.run()\n",
        },
        "errors": ["ImportError: no module named foo"],
    }
    result = decompose_failed_build(state)
    assert result.build_passed is False
    assert "easy" in result.tiers_run  # handler() parses cleanly
    assert "structural" in result.tiers_run  # import + call edges
    assert "recalcitrant" in result.tiers_run  # error trail present
    assert result.recovered_anything is True
    assert result.failure_signature is not None


def test_dispatcher_summary_serializable() -> None:
    state = {
        "run_id": "b1",
        "validation_result": {"verdict": "fail"},
        "code_files": {"m.py": "def f():\n    return 1\n"},
        "errors": ["RuntimeError: boom"],
    }
    summary = decompose_failed_build(state).summary()
    assert summary["build_id"] == "b1"
    assert summary["build_passed"] is False
    assert isinstance(summary["clean_fragments"], int)
