"""Easy decomposition — the "cellulase" path (mycorrhizal Stage 7, Area 4).

For builds whose failure is *local* — a typo, a wrong import, a single broken
function — most of the code actually parsed and works. This extractor
AST-walks each file, harvests the top-level functions and classes that parse
cleanly, and returns them as reusable primitives with provenance. The
insight from the brief: most failed builds contain more working code than
the developer intuits, and recovering it directly accelerates future builds.

Pure function: no soil writes, no side effects. The caller (dispatcher)
decides what to persist.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("belief.memory.decomposers.easy")


@dataclass
class CleanFragment:
    """One cleanly-parsed code fragment recovered from a failed build."""

    name: str  # function/class name
    kind: str  # "function" | "async_function" | "class"
    source: str  # the fragment's source text
    file: str  # which file it came from
    lineno: int
    source_build_id: str = ""
    docstring: Optional[str] = None


def _segment_source(source: str) -> list[str]:
    return source.splitlines()


def extract_clean_fragments(
    code_files: dict[str, str], source_build_id: str = ""
) -> list[CleanFragment]:
    """Harvest cleanly-parseable top-level defs/classes from each file.

    A file that fails to parse as a whole still yields nothing here — but
    the *structural* tier can attempt partial recovery. The easy tier only
    claims fragments from files that parse, where each top-level def/class
    is independently extractable.

    Returns a flat list of ``CleanFragment``. Never raises — a malformed
    file is skipped with a debug log.
    """
    fragments: list[CleanFragment] = []
    for fname, content in code_files.items():
        if not content or not fname.endswith(".py"):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Whole-file parse failed — easy tier can't help here; the
            # structural/recalcitrant tiers handle this substrate.
            logger.debug("easy: %s did not parse; skipping", fname)
            continue
        lines = _segment_source(content)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = (
                    "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                    if isinstance(node, ast.FunctionDef)
                    else "class"
                )
                seg = _node_source(node, lines)
                if not seg.strip():
                    continue
                fragments.append(
                    CleanFragment(
                        name=node.name,
                        kind=kind,
                        source=seg,
                        file=fname,
                        lineno=node.lineno,
                        source_build_id=source_build_id,
                        docstring=ast.get_docstring(node),
                    )
                )
    return fragments


def _node_source(node: ast.AST, lines: list[str]) -> str:
    """Slice the source lines spanning a node. Uses end_lineno (3.8+);
    falls back to a single line if unavailable."""
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", None)
    if end is None:
        return lines[start] if 0 <= start < len(lines) else ""
    return "\n".join(lines[start:end])
