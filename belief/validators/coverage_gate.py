"""Terminal coverage gate — planned-vs-produced completeness floor.

Runs at the end of the pipeline, right after the compile gate. Where the
compile gate catches files that do not *parse*, this gate catches files that
*should exist and don't* (the planner specified them, the builder never
produced them) and files that exist but are *hollow* (a stub: only ``pass``,
only imports, or no substantive content).

This is the structural-incompleteness detector behind handoff Q2: the
`belief-9ecc581f` hollow build scored 1.00 while producing a fraction of the
planned files. The gate records the produced-vs-planned coverage fraction and,
when coverage is below threshold or any produced file is hollow, downgrades the
verdict to ``fail_fixable`` and caps the score at what actually shipped.

Scope discipline (handoff Q2): DETECTION of structural mismatch only. It makes
NO judgement about whether the produced files are *good* — only whether the
planned ones are *present and non-hollow*. Whether present code is correct is
the deferred research question; this gate does not touch it.

Pure module: no I/O, no belief imports, so it stays cheap to test and reuse.
"""

from __future__ import annotations

import ast
import logging
from typing import Mapping

logger = logging.getLogger(__name__)

# Forced verdict on failure. Matches ValidationVerdict.FAIL_FIXABLE.value; kept
# as a literal so this stays a dependency-free pure module (same choice as
# compile_gate).
_FAIL_VERDICT = "fail_fixable"

# Default: every planned file must be produced (coverage 1.0). Override via the
# threshold argument (the node reads BELIEF_COVERAGE_THRESHOLD).
DEFAULT_COVERAGE_THRESHOLD = 1.0

# When all planned files are present but some produced file is hollow, the
# build still can't claim a perfect score. Cap it here.
HOLLOW_SCORE_CAP = 0.6


def _norm(path: str) -> str:
    """Normalise a path for comparison: forward slashes, no leading ``./``."""
    return path.replace("\\", "/").lstrip("./")


def _basename(path: str) -> str:
    return _norm(path).rsplit("/", 1)[-1]


def _is_test_path(fname: str) -> bool:
    return fname.startswith("test") or "/test" in fname


def _is_stub_body(body: list[ast.stmt]) -> bool:
    """True if a def/class body is only filler: docstring, ``pass``, ``...``,
    or ``raise NotImplementedError``."""
    for node in body:
        if isinstance(node, ast.Pass):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring or bare constant (incl. ``...``)
        if isinstance(node, ast.Raise):
            exc = node.exc
            name = ""
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                name = exc.func.id
            elif isinstance(exc, ast.Name):
                name = exc.id
            if name == "NotImplementedError":
                continue
        return False
    return True


def is_hollow_file(content: str | None) -> bool:
    """True if a Python source has no substantive content.

    Hollow = empty, or every top-level statement is filler (imports, a module
    docstring, ``pass``, ``...``) and every def/class present has only a stub
    body. A file with even one real function/class body, assignment, or
    executable statement is NOT hollow.

    A file that fails to parse is NOT treated as hollow here — that is the
    compile gate's job. Returns False so the two gates don't double-report.
    """
    src = (content or "").strip()
    if not src:
        return True
    try:
        tree = ast.parse(content or "")
    except SyntaxError:
        return False

    substantive = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring / bare constant
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _is_stub_body(node.body):
                continue  # a stub def is not substantive
            substantive += 1
            continue
        # assignments, if/for/while/with/try, real expression-calls, etc.
        substantive += 1
    return substantive == 0


def planned_filenames(file_manifest: object | None) -> list[str]:
    """Extract the planned filenames from a FileManifestPlan (model or dict)."""
    if file_manifest is None:
        return []
    if isinstance(file_manifest, dict):
        files = file_manifest.get("files") or []
    else:
        files = getattr(file_manifest, "files", []) or []
    names: list[str] = []
    for f in files:
        if isinstance(f, dict):
            name = f.get("filename", "")
        else:
            name = getattr(f, "filename", "")
        if name:
            names.append(name)
    return names


def compute_coverage(
    planned: list[str],
    produced: Mapping[str, str] | None,
) -> tuple[float, list[str]]:
    """Return ``(coverage_fraction, missing)``.

    A planned file counts as produced when its normalised path matches a
    produced path, or (fallback) when its basename matches a produced basename
    — so a directory reshuffle isn't reported as a miss. Coverage is 1.0 when
    nothing was planned (can't assess → don't penalise).
    """
    produced = produced or {}
    if not planned:
        return 1.0, []
    produced_norm = {_norm(p) for p in produced}
    produced_base = {_basename(p) for p in produced}
    missing: list[str] = []
    for pf in planned:
        n = _norm(pf)
        if n in produced_norm or _basename(pf) in produced_base:
            continue
        missing.append(pf)
    fraction = (len(planned) - len(missing)) / len(planned)
    return round(fraction, 4), missing


def find_hollow_files(produced: Mapping[str, str] | None) -> list[str]:
    """Produced ``.py`` files that are hollow stubs.

    Excludes ``__init__.py`` (legitimately empty) and test files (a test file
    isn't application logic).
    """
    hollow: list[str] = []
    for fname, content in (produced or {}).items():
        if not fname.endswith(".py"):
            continue
        if _basename(fname) == "__init__.py" or _is_test_path(fname):
            continue
        if is_hollow_file(content):
            hollow.append(fname)
    return hollow


def gate_validation_result(
    file_manifest: object | None,
    code_files: Mapping[str, str] | None,
    validation_result: object | None,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[object, float, list[str], list[str]]:
    """Downgrade ``validation_result`` on structural incompleteness.

    Returns ``(validation_result, coverage_fraction, missing, hollow)``.

    The coverage fraction is always computed (so the build record can show it
    even on a complete build). When coverage is below ``threshold`` OR any
    produced file is hollow, the verdict is forced to ``fail_fixable`` and the
    numeric scores are capped at what actually shipped — the score can't exceed
    the produced-vs-planned fraction, and a build with hollow files can't beat
    ``HOLLOW_SCORE_CAP``. Issues are appended per missing/hollow file. On a
    downgrade the returned result is a plain ``dict`` (graph-state form).
    """
    planned = planned_filenames(file_manifest)
    coverage_fraction, missing = compute_coverage(planned, code_files)
    hollow = find_hollow_files(code_files)

    coverage_fail = bool(planned) and coverage_fraction < threshold
    if not coverage_fail and not hollow:
        return validation_result, coverage_fraction, missing, hollow

    if isinstance(validation_result, dict):
        result: dict = dict(validation_result)
    elif validation_result is not None and hasattr(validation_result, "model_dump"):
        result = validation_result.model_dump()
    else:
        result = {}

    # Honest ceiling: can't claim more than the fraction of planned work that
    # shipped; hollow files drag it down further.
    effective = coverage_fraction
    if hollow:
        effective = min(effective, HOLLOW_SCORE_CAP)
    existing = float(result.get("weighted_score", 0.0) or 0.0)
    new_score = round(min(existing, effective), 4) if existing else round(effective, 4)
    # If existing was 0 (e.g. compile gate already floored it), keep it at 0.
    if existing == 0.0:
        new_score = 0.0

    result["verdict"] = _FAIL_VERDICT
    result["weighted_score"] = new_score
    result["correctness_score"] = min(float(result.get("correctness_score", 0.0) or 0.0), new_score)
    result["completeness_score"] = new_score

    issues = list(result.get("issues") or [])
    for fname in missing:
        issues.append(f"coverage_gate: planned file not produced: {fname}")
    for fname in hollow:
        issues.append(f"coverage_gate: produced file is a hollow stub: {fname}")
    result["issues"] = issues
    parts = []
    if coverage_fail:
        parts.append(f"{len(missing)} planned file(s) missing (coverage {coverage_fraction:.0%})")
    if hollow:
        parts.append(f"{len(hollow)} hollow file(s)")
    result["summary"] = "Coverage gate failed: " + "; ".join(parts)
    return result, coverage_fraction, missing, hollow
