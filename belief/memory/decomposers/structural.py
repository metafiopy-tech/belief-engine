"""Structural decomposition — the "hemicellulase" path (Stage 7, Area 4).

For builds that failed at the *integration* level — the parts worked
individually but didn't compose — the recoverable value is the *attempted
composition*: the import graph and the call graph. Each edge is a candidate
composition with a failure annotation. Even though the composition failed,
the fact that the build *tried* to wire module A to module B is information
a future build can use (or avoid).

Pure function: extracts edges, no soil writes.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("belief.memory.decomposers.structural")


@dataclass(frozen=True)
class CompositionEdge:
    """One attempted composition edge: import or call."""

    kind: str  # "import" | "call"
    src: str  # source file or caller name
    dst: str  # imported module / called name
    file: str
    source_build_id: str = ""
    failure_annotation: str = ""


def extract_composition_edges(
    code_files: dict[str, str],
    source_build_id: str = "",
    failure_annotation: str = "",
) -> list[CompositionEdge]:
    """Extract import edges and call edges from each parseable file.

    Import edges: ``import x`` / ``from x import y`` → (file → x).
    Call edges: top-level + nested ``name(...)`` → (file → name).

    A file that doesn't parse is skipped (the recalcitrant tier handles
    fully-opaque builds). Never raises.
    """
    edges: list[CompositionEdge] = []
    for fname, content in code_files.items():
        if not content or not fname.endswith(".py"):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.debug("structural: %s did not parse; skipping", fname)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.append(
                        CompositionEdge(
                            kind="import",
                            src=fname,
                            dst=alias.name,
                            file=fname,
                            source_build_id=source_build_id,
                            failure_annotation=failure_annotation,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                edges.append(
                    CompositionEdge(
                        kind="import",
                        src=fname,
                        dst=mod,
                        file=fname,
                        source_build_id=source_build_id,
                        failure_annotation=failure_annotation,
                    )
                )
            elif isinstance(node, ast.Call):
                callee = _call_name(node.func)
                if callee:
                    edges.append(
                        CompositionEdge(
                            kind="call",
                            src=fname,
                            dst=callee,
                            file=fname,
                            source_build_id=source_build_id,
                            failure_annotation=failure_annotation,
                        )
                    )
    return edges


def _call_name(func: ast.AST) -> Optional[str]:
    """Best-effort name for a call target. ``foo()`` → 'foo';
    ``a.b.c()`` → 'a.b.c'; complex expressions → None."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        cur: ast.AST = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return func.attr
    return None
