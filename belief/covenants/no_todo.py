"""NoTodoCovenant — refuse to emit TODO / FIXME markers in generated code.

Session 8.5b (v3.2).  Generated source files that contain ``TODO``,
``FIXME``, ``XXX``, or ``pass  # TODO`` are a recurring failure mode:
the LLM writes a reasonable-looking scaffold with placeholder comments
thinking a human will fill them in, but the downstream executor runs
the file as-is and either silently passes (``pass  # TODO``) or hits
a confusing error later.

This covenant is the deterministic answer: scan every generated Python
file after the other rewrites land; if a TODO/FIXME/XXX marker appears
in a **comment**, either rewrite the host statement to
``raise NotImplementedError(...)`` (when it's ``pass # TODO`` /
``... # TODO``) or strip the comment.  String-literal occurrences are
left alone — a generated agent may legitimately render a TODO in a
user-facing output.

Scope
-----

* Applies to Python source files emitted by the engine (skeleton,
  builder, tester, brownfield patcher, covenant proposer).
* Does **not** apply to the engine's own source tree.  The boundary
  is the caller — ``enforce_python_covenants`` only hits generated
  files passed through it.

What counts as a TODO marker
----------------------------

Case-insensitive match on whole-word ``TODO``, ``FIXME``, ``XXX``
appearing in a comment (after ``#``).  ``TODO:`` and ``FIX_ME`` (with
underscore) are treated as equivalent.  String literals are ignored;
a naïve split-on-``#`` is enough to filter the common cases without
reaching for a full Python tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result type (same shape as other covenants)
# ---------------------------------------------------------------------------


@dataclass
class CovenantApplied:
    rule: str
    detail: str = ""
    line: int | None = None
    file: str | None = None


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches a TODO/FIXME/XXX marker inside a comment.  We require the
# comment marker ``#`` somewhere earlier on the line (the split below
# enforces that) and the marker-word bounded to avoid false positives
# in variable names like ``todo_count``.
_TODO_WORD_RE = re.compile(r"\b(?:TODO|FIX[_-]?ME|XXX)\b", re.IGNORECASE)

# Statement whose entire body is ``pass  # TODO …`` or ``...  # TODO …``.
# We rewrite these to NotImplementedError so a silent fall-through
# can't masquerade as a real implementation.
_STUB_STMT_RE = re.compile(
    r"""
    ^(?P<indent>[ \t]*)
    (?P<body>pass|\.\.\.)
    [ \t]*
    \#[ \t]*
    (?P<marker>TODO|FIX[_-]?ME|XXX)
    \b
    (?P<rest>[^\n]*)
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def apply_no_todo_covenant(
    source: str,
    *,
    filename: str | None = None,
) -> tuple[str, list[CovenantApplied]]:
    """Strip TODO/FIXME/XXX markers from generated Python source.

    * ``pass  # TODO …`` / ``...  # TODO …`` → ``raise NotImplementedError(...)``
    * ``<code>  # TODO …`` (non-stub statements) → ``<code>`` (comment stripped)

    Returns ``(new_source, applied_list)``.  If nothing matched, the
    original source is returned unchanged and ``applied_list`` is empty.
    """
    applied: list[CovenantApplied] = []
    out_lines: list[str] = []

    for lineno, line in enumerate(source.splitlines(), start=1):
        # Stub-statement rewrite: the whole body is pass/... + # TODO
        stub_match = _STUB_STMT_RE.match(line)
        if stub_match:
            indent = stub_match.group("indent")
            rest = stub_match.group("rest").strip(" :;-")
            msg = (
                rest.strip() or f"{stub_match.group('marker').upper()} marker stripped by covenant"
            )
            out_lines.append(
                f'{indent}raise NotImplementedError("covenant-patched stub: {_escape(msg)}")'
            )
            applied.append(
                CovenantApplied(
                    rule="no_todo.stub_to_raise",
                    detail=f"Rewrote stub '{stub_match.group('body')}  # TODO' to NotImplementedError",
                    line=lineno,
                    file=filename,
                )
            )
            continue

        # Generic comment strip: any `# TODO ...` tail on a line of code
        if "#" in line:
            code_part, _, comment_part = line.partition("#")
            if _TODO_WORD_RE.search(comment_part):
                # Strip trailing whitespace left behind after removing
                # the comment; preserve leading indent + code verbatim.
                stripped = code_part.rstrip()
                if stripped:
                    out_lines.append(stripped)
                # If stripped is empty the entire line was a TODO comment;
                # drop the line entirely (don't leave a blank gap).
                applied.append(
                    CovenantApplied(
                        rule="no_todo.comment_strip",
                        detail=f"Stripped '# {comment_part.strip()}' marker",
                        line=lineno,
                        file=filename,
                    )
                )
                continue

        out_lines.append(line)

    # Preserve a trailing newline iff the input had one.
    suffix = "\n" if source.endswith("\n") else ""
    return "\n".join(out_lines) + suffix, applied


def _escape(s: str) -> str:
    """Make ``s`` safe inside a double-quoted Python string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


__all__ = ["CovenantApplied", "apply_no_todo_covenant"]
