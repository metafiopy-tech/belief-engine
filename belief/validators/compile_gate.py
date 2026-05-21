"""Terminal compile gate — deterministic syntax floor for generated builds.

Runs at the end of the pipeline, after the refinement / polish pass and right
before the build's verdict is finalised. Guarantees that no build can report
``pass`` (or a non-zero score) while shipping a ``.py`` file that does not even
parse.

Why this exists separately from ``belief.agents.validator``: the per-stage
validator runs *before* the refinement water-cycle, so it cannot see truncation
introduced by the polish pass. This gate is the post-refinement backstop and the
single place that makes the final verdict honest regardless of which upstream
step (builder output cap, refinement rewrite, ...) truncated a file.

Deliberately NOT a completeness check. It catches files that fail ``ast.parse``
(the catastrophic, won't-even-import case). A file that parses but is missing
methods it calls is out of scope here — that is addressed upstream by the
builder's output-token handling and by requiring real tests. Doing one thing
reliably beats a gate that pretends to verify completeness and gives false
confidence.
"""

from __future__ import annotations

import ast
import logging
from typing import Mapping

logger = logging.getLogger(__name__)

# Forced verdict on failure. Matches ValidationVerdict.FAIL_FIXABLE.value; kept
# as a literal so this stays a dependency-free pure module.
_FAIL_VERDICT = "fail_fixable"


def find_uncompilable_files(
    code_files: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    """Return ``[(filename, error)]`` for every ``.py`` that fails to parse.

    Pure: no I/O, no mutation. Non-``.py`` files are ignored.
    """
    broken: list[tuple[str, str]] = []
    for fname, content in (code_files or {}).items():
        if not fname.endswith(".py"):
            continue
        try:
            ast.parse(content or "")
        except SyntaxError as exc:
            broken.append((fname, f"line {exc.lineno}: {exc.msg}"))
    return broken


def gate_validation_result(
    code_files: Mapping[str, str] | None,
    validation_result: object | None,
) -> tuple[object, list[tuple[str, str]]]:
    """Downgrade ``validation_result`` if any ``.py`` file fails to parse.

    ``validation_result`` is the model-dumped dict carried in graph state (a
    ``ValidationResult`` object or ``None`` are also tolerated). Returns
    ``(validation_result, broken)``.

    When ``broken`` is empty the result is returned unchanged. On failure the
    verdict is forced to ``fail_fixable``, the numeric scores are floored to
    0.0, and one issue is appended per broken file. The returned result is
    always a plain ``dict`` (graph-state form) when a downgrade occurs.
    """
    broken = find_uncompilable_files(code_files)
    if not broken:
        return validation_result, broken

    if isinstance(validation_result, dict):
        result: dict = dict(validation_result)
    elif validation_result is not None and hasattr(validation_result, "model_dump"):
        result = validation_result.model_dump()
    else:
        result = {}

    result["verdict"] = _FAIL_VERDICT
    result["weighted_score"] = 0.0
    result["correctness_score"] = 0.0
    result["completeness_score"] = 0.0

    issues = list(result.get("issues") or [])
    for fname, err in broken:
        issues.append(f"compile_gate: {fname} does not parse ({err})")
    result["issues"] = issues
    result["summary"] = f"Compile gate failed: {len(broken)} generated file(s) do not parse"
    return result, broken
