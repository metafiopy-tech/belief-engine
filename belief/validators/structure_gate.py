"""Terminal structure gate — single-canonical-structure coherence floor.

Runs at the end of the pipeline, after the compile and coverage gates. Catches
the failure mode behind handoff Q1: the architect's skeleton repair failed, the
fallback fired, and the builder never converged on one structure — producing
TWO parallel implementations (a clean ``idea_capsule/`` package AND a competing
root-level ``main.py`` monolith with its own duplicate ``models`` / ``config`` /
``exceptions``). A reader's first question was "which one is real?".

The detector is deliberately precise: it flags a coherence failure only when the
*same* top-level symbol (class or function) is defined BOTH at the project root
AND inside a sub-package, or when there are competing ``__main__`` entry guards
split across root and package. That coexistence is the two-implementation smell.
It does NOT fire on a legitimate single structure — a flat build (everything at
root) or a package build (everything nested), including the common
root-``main.py``-plus-package layout where the entry imports from the package
but defines no duplicate symbols.

Detection only — no judgement about whether either implementation is *good*.

Pure module: no I/O, no belief imports, so it stays cheap to test and reuse
(same shape as compile_gate / coverage_gate).
"""

from __future__ import annotations

import ast
import logging
from typing import Mapping

logger = logging.getLogger(__name__)

# Forced verdict on failure (matches ValidationVerdict.FAIL_FIXABLE.value; kept
# as a literal so this stays dependency-free).
_FAIL_VERDICT = "fail_fixable"

# A two-implementation build is seriously incoherent ("which one is real?"), so
# cap its score well below pass. Not floored to 0.0 — the code may still parse
# and partly work; this is a coherence defect, not a catastrophic one.
COHERENCE_SCORE_CAP = 0.5


def _is_test_path(fname: str) -> bool:
    return fname.startswith("test") or "/test" in fname


def _is_root(fname: str) -> bool:
    """A root-level module has no directory component."""
    return "/" not in fname.replace("\\", "/").lstrip("./")


def _top_level_symbols(content: str) -> set[str]:
    """Top-level class and function names defined in a module (AST)."""
    try:
        tree = ast.parse(content or "")
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _has_main_guard(content: str) -> bool:
    """True if the module has an ``if __name__ == "__main__":`` entry guard."""
    try:
        tree = ast.parse(content or "")
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
            if test.left.id == "__name__":
                for comp in test.comparators:
                    if isinstance(comp, ast.Constant) and comp.value == "__main__":
                        return True
    return False


def find_duplicate_implementations(code_files: Mapping[str, str] | None) -> list[str]:
    """Return human-readable findings for two-implementation coherence defects.

    Two signals, both keyed on root-vs-package coexistence:

    1. A top-level class/function defined in a root module AND in a packaged
       (nested) module — the duplicate-core-module signature.
    2. A ``__main__`` entry guard present in BOTH a root module and a nested
       module — competing entry points.
    """
    files = {
        f: c
        for f, c in (code_files or {}).items()
        if f.endswith(".py") and not _is_test_path(f) and f.rsplit("/", 1)[-1] != "__init__.py"
    }

    # symbol -> (set of root files, set of nested files) defining it
    root_syms: dict[str, str] = {}
    nested_syms: dict[str, str] = {}
    for fname, content in files.items():
        target = root_syms if _is_root(fname) else nested_syms
        for sym in _top_level_symbols(content):
            target.setdefault(sym, fname)

    findings: list[str] = []
    for sym in sorted(set(root_syms) & set(nested_syms)):
        findings.append(
            f"symbol '{sym}' implemented at root ({root_syms[sym]}) and in package "
            f"({nested_syms[sym]}) — two parallel implementations"
        )

    root_entries = sorted(f for f, c in files.items() if _is_root(f) and _has_main_guard(c))
    nested_entries = sorted(f for f, c in files.items() if not _is_root(f) and _has_main_guard(c))
    if root_entries and nested_entries:
        findings.append(
            f"competing entry points: root {root_entries} and package {nested_entries} "
            f"both define a __main__ guard"
        )

    return findings


def gate_validation_result(
    code_files: Mapping[str, str] | None,
    validation_result: object | None,
) -> tuple[object, list[str]]:
    """Downgrade ``validation_result`` when two parallel implementations exist.

    Returns ``(validation_result, findings)``. When ``findings`` is empty the
    result is returned unchanged. On a coherence failure the verdict is forced
    to ``fail_fixable``, the score is capped at :data:`COHERENCE_SCORE_CAP`, and
    one issue is appended per finding. The returned result is a plain ``dict``
    (graph-state form) when a downgrade occurs.
    """
    findings = find_duplicate_implementations(code_files)
    if not findings:
        return validation_result, findings

    if isinstance(validation_result, dict):
        result: dict = dict(validation_result)
    elif validation_result is not None and hasattr(validation_result, "model_dump"):
        result = validation_result.model_dump()
    else:
        result = {}

    existing = float(result.get("weighted_score", 0.0) or 0.0)
    new_score = round(min(existing, COHERENCE_SCORE_CAP), 4) if existing else COHERENCE_SCORE_CAP
    if existing == 0.0:
        new_score = 0.0

    result["verdict"] = _FAIL_VERDICT
    result["weighted_score"] = new_score
    result["correctness_score"] = min(float(result.get("correctness_score", 0.0) or 0.0), new_score)
    result["completeness_score"] = new_score

    issues = list(result.get("issues") or [])
    for finding in findings:
        issues.append(f"structure_gate: {finding}")
    result["issues"] = issues
    result["summary"] = (
        f"Structure gate failed: {len(findings)} coherence defect(s) — "
        f"two parallel implementations detected"
    )
    return result, findings
