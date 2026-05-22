"""Session D (handoff Q1): single-canonical-structure coherence gate.

Hermetic — the structure gate is a pure module; the fallback-manifest check
imports the architect helper (no LLM call). Covers the duplicate-implementation
detector, the verdict downgrade, and the guarantee that the architect fallback
emits exactly one canonical structure.
"""

from __future__ import annotations

from belief.agents.architect import _fallback_manifest
from belief.validators.structure_gate import (
    COHERENCE_SCORE_CAP,
    find_duplicate_implementations,
    gate_validation_result,
)


# ── detector ─────────────────────────────────────────────────────────────────


def test_flags_duplicate_symbol_across_root_and_package():
    code = {
        "models.py": "class User:\n    pass\n",
        "idea_capsule/models.py": "class User:\n    name = ''\n",
        "main.py": "def main():\n    return 1\n",
    }
    findings = find_duplicate_implementations(code)
    assert any("User" in f and "two parallel implementations" in f for f in findings)


def test_flags_competing_entry_points():
    code = {
        "main.py": "def run():\n    pass\n\nif __name__ == '__main__':\n    run()\n",
        "pkg/app.py": "def go():\n    pass\n\nif __name__ == '__main__':\n    go()\n",
    }
    findings = find_duplicate_implementations(code)
    assert any("competing entry points" in f for f in findings)


def test_clean_flat_build_not_flagged():
    code = {
        "models.py": "class User:\n    pass\n",
        "services.py": "def create():\n    return 1\n",
        "main.py": "def main():\n    return 1\n",
    }
    assert find_duplicate_implementations(code) == []


def test_clean_package_build_not_flagged():
    code = {
        "pkg/__init__.py": "",
        "pkg/models.py": "class User:\n    pass\n",
        "pkg/services.py": "def create():\n    return 1\n",
        "pkg/main.py": "def main():\n    return 1\n",
    }
    assert find_duplicate_implementations(code) == []


def test_legit_root_main_plus_package_not_flagged():
    # Common, coherent layout: a root entry that imports from the package and
    # defines NO duplicate symbols. Must not be a false positive.
    code = {
        "main.py": "from pkg.app import App\n\ndef main():\n    App().run()\n",
        "pkg/__init__.py": "",
        "pkg/app.py": "class App:\n    def run(self):\n        return 1\n",
        "pkg/models.py": "class User:\n    pass\n",
    }
    assert find_duplicate_implementations(code) == []


def test_test_files_ignored_by_detector():
    # A test file legitimately re-uses a symbol name; not a duplicate impl.
    code = {
        "models.py": "class User:\n    pass\n",
        "tests/test_models.py": "class User:\n    pass\n",
    }
    assert find_duplicate_implementations(code) == []


# ── gate downgrade ───────────────────────────────────────────────────────────


def _passing() -> dict:
    return {"verdict": "pass", "weighted_score": 1.0, "correctness_score": 1.0, "issues": []}


def test_gate_forces_fail_and_caps_on_two_implementations():
    code = {
        "models.py": "class User:\n    pass\n",
        "idea_capsule/models.py": "class User:\n    name = ''\n",
    }
    result, findings = gate_validation_result(code, _passing())
    assert findings
    assert result["verdict"] == "fail_fixable"
    assert result["weighted_score"] <= COHERENCE_SCORE_CAP
    assert any("structure_gate" in i for i in result["issues"])


def test_gate_passes_clean_build_unchanged():
    code = {"models.py": "class User:\n    pass\n", "main.py": "def main():\n    return 1\n"}
    validation = _passing()
    result, findings = gate_validation_result(code, validation)
    assert findings == []
    assert result is validation
    assert result["verdict"] == "pass"


# ── architect fallback emits a single canonical structure ────────────────────


def test_fallback_manifest_is_single_canonical_structure():
    manifest = _fallback_manifest("Build a FastAPI CRUD service with a database and tests")
    filenames = [f.filename for f in manifest.files]
    # Exactly one entry point.
    entry_points = [f for f in manifest.files if f.is_entry_point]
    assert len(entry_points) == 1
    # No duplicate basenames (no competing same-named modules).
    basenames = [name.rsplit("/", 1)[-1] for name in filenames]
    assert len(basenames) == len(set(basenames))
    # The produced manifest, treated as a build, has no coherence defect.
    synthetic = {f.filename: "x = 1\n" for f in manifest.files if f.filename.endswith(".py")}
    assert find_duplicate_implementations(synthetic) == []
