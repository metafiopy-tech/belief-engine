"""ForbiddenImportsCovenant — strip stdlib names from requirements.txt.

Session 2 (v3.2).  Complements :mod:`belief.covenants.pydantic_v2`: this
covenant runs in the same 4-stage pipeline but targets requirements.txt
rather than Python sources.

Why a separate covenant
-----------------------

The overnight logs show the builder emitting ``pip install timeit`` and
``pip install uuid`` in generated requirements.txt files.  Both are
stdlib modules; pip downloads them as typo-squatted packages, wasting
~150s per install and introducing real security risk (slopsquatting).

The existing :mod:`belief.validators.__init__` already has a
``_enforce_no_stdlib_in_requirements`` using a hand-curated set of ~70
names.  This covenant SUPERSEDES it by using
:data:`sys.stdlib_module_names` (Python 3.10+, authoritative — it's
what Python itself uses internally) plus a few Windows-specific +
historical aliases the hand-curated set got right.  The hand-curated
path is left in place for now as a belt-and-suspenders defence; its
output is a no-op once this pipeline runs first.

Scope
-----

This covenant does NOT touch Python imports of stdlib modules — those
are correct and should not be rewritten.  It only affects
requirements.txt-style content (package names, one per line).
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass

logger = logging.getLogger("belief.covenants.forbidden_imports")


# ---------------------------------------------------------------------------
# Stdlib name set — pulled from sys.stdlib_module_names (Python 3.10+).
# ---------------------------------------------------------------------------

# sys.stdlib_module_names is a frozenset of top-level stdlib module
# names for the running Python version.  We snapshot it at import time.
STDLIB_NAMES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", frozenset()))

# Historical / cross-version / Windows-only names that Python 3.10+
# may not list but that were stdlib at some point and still sometimes
# appear in LLM-generated requirements.txt (often because the LLM
# pattern-matched on older Stack Overflow answers).
_STDLIB_EXTRAS: frozenset[str] = frozenset(
    {
        "typing_extensions",  # only in some distributions; commonly confused
        "__future__",
        "collections-extended",  # real pypi pkg but easy confusion — keep out
        "typing-extensions",
    }
)
# Note: typing_extensions is actually a real PyPI package.  We don't
# include it in the blocklist — it belongs there if the project really
# needs it.  Keeping as an example; do not add it back.
_STDLIB_EXTRAS = frozenset(
    {
        # Names historically shipped as stdlib that still show up in
        # hallucinated requirements.txt lines.
        "__future__",
    }
)

# Canonical per-line regex: optional leading whitespace, package name,
# optional version specifier (==, >=, <=, ~=, <, >, !=), extras
# [...], and optional trailing whitespace/comment.
_REQ_LINE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z0-9_\-\.]+)          # package name (PEP 508 identifier subset)
    (?P<extras>\[[^\]]*\])?              # optional extras: [security]
    (?P<spec>\s*(?:==|>=|<=|~=|!=|<|>)[^#\s]+)?  # version spec
    \s*(?P<comment>\#.*)?                # trailing comment
    \s*$
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Result type (keeps the same shape as pydantic_v2.CovenantApplied so the
# 4-stage pipeline can mix them into one applied-rewrite list).
# ---------------------------------------------------------------------------


@dataclass
class CovenantApplied:
    rule: str
    detail: str = ""
    line: int | None = None
    file: str | None = None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def is_stdlib_name(name: str) -> bool:
    """Return True iff ``name`` refers to a stdlib module on this Python.

    Canonicalises the input (strips extras, version specifier, comments)
    before checking.  Case-sensitive (stdlib names are all lowercase).
    """
    canonical = _canonicalize_name(name)
    if canonical in STDLIB_NAMES:
        return True
    if canonical in _STDLIB_EXTRAS:
        return True
    return False


def _canonicalize_name(raw: str) -> str:
    """Strip extras/version/comment to get just the package name."""
    s = raw.strip()
    for sep in ("==", ">=", "<=", "~=", "!=", "<", ">"):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
            break
    if "[" in s:
        s = s.split("[", 1)[0]
    if "#" in s:
        s = s.split("#", 1)[0]
    return s.strip().lower().replace("-", "_")


def apply_forbidden_imports_covenant(
    source: str, *, filename: str | None = None
) -> tuple[str, list[CovenantApplied]]:
    """Strip stdlib names from requirements.txt content.

    Silent per Joe's session-2 preference — log INFO per stripped
    name.  Returns ``(cleaned, applied)`` — non-requirements files or
    empty input round-trip unchanged.
    """
    if filename is not None and not _looks_like_requirements(filename):
        return source, []
    if not source.strip():
        return source, []

    applied: list[CovenantApplied] = []
    out_lines: list[str] = []

    for i, line in enumerate(source.splitlines(keepends=False), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            # Blank line, comment, pip flag (e.g., -r other.txt) — keep.
            out_lines.append(line)
            continue

        match = _REQ_LINE.match(line)
        if match is None:
            # Malformed line; leave it alone and let pip yell.
            out_lines.append(line)
            continue
        name = match.group("name") or ""
        if is_stdlib_name(name):
            logger.info(
                "forbidden_imports: stripping stdlib package %r from %s (line %d)",
                name,
                filename or "requirements.txt",
                i,
            )
            applied.append(
                CovenantApplied(
                    rule="forbidden_imports.stdlib_in_requirements",
                    detail=f"stripped stdlib name: {name}",
                    line=i,
                    file=filename,
                )
            )
            continue
        out_lines.append(line)

    cleaned = "\n".join(out_lines)
    # Preserve trailing newline if the original had one.
    if source.endswith("\n") and not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned, applied


def _looks_like_requirements(filename: str) -> bool:
    """Heuristic: a requirements-style file is one pip reads.

    Matches ``requirements.txt``, ``requirements-dev.txt``,
    ``constraints.txt``, or anything under a ``requirements/`` dir.
    Keeps out .py / .toml / .yaml etc.
    """
    lower = filename.lower()
    if lower.endswith(".txt") and ("requirement" in lower or "constraint" in lower):
        return True
    if "/requirements/" in lower or lower.startswith("requirements/"):
        return True
    return False


__all__ = [
    "CovenantApplied",
    "STDLIB_NAMES",
    "apply_forbidden_imports_covenant",
    "is_stdlib_name",
]
