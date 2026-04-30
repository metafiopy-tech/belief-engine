"""Shared HumanEval/MBPP code-stub adapter helpers.

Used by both ``scripts/bigcode_shim.py`` (FastAPI front for the engine)
and ``scripts/generate_humaneval_completions.py`` (the BigCode-format
completion generator).

Why a shared module: BigCode's HumanEval and MBPP tasks send the model
a code stub — optional imports, then ``def NAME(args):`` with a
docstring — and expect the *body* of that function as the completion.
The harness assembles ``prompt + completion`` and runs the test suite
on the result.

The Belief Engine's intake agent expects an English instruction, not a
code prefix. The pipeline is:

  1. Detect a stub-shaped prompt (``looks_like_code_stub``).
  2. Rewrite to an English goal (``rewrite_stub_to_goal``).
  3. Run the engine and read its full Python file.
  4. Extract just the body of the function with the matching name
     (``extract_function_body``).
  5. Return that body so harness-side ``prompt + completion`` is valid.

The helpers here are pure (no I/O, no state) so they're cheap to test
hermetically and safe to import from any context.
"""

from __future__ import annotations

import ast
import re

__all__ = [
    "looks_like_code_stub",
    "detect_function_name",
    "rewrite_stub_to_goal",
    "extract_function_body",
]


_DEF_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def looks_like_code_stub(text: str) -> bool:
    """Return True iff ``text`` parses as Python AND has a top-level
    ``def`` whose first statement is a docstring.

    HumanEval and MBPP prompts always satisfy this. English goals
    don't, so the adapter is a no-op for natural-language input.
    """
    if not text or "def " not in text:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    return True
    return False


def detect_function_name(stub: str) -> str | None:
    """Return the name of the *last* top-level ``def`` in ``stub``.

    HumanEval prompts often have helper imports, occasionally a small
    helper function, and the target function comes last. Picking the
    last def is more robust than picking the first.

    Falls back to a regex scan if the stub doesn't quite parse.
    """
    try:
        tree = ast.parse(stub)
    except SyntaxError:
        m = list(_DEF_RE.finditer(stub))
        return m[-1].group(1) if m else None
    name: str | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
    return name


def rewrite_stub_to_goal(stub: str) -> str:
    """Wrap a code stub in an English instruction the intake agent can
    act on. Engine should produce a single Python file with the
    function fully implemented; no test/main code."""
    return (
        "Implement the function described below. Return a single Python "
        "file containing the complete function definition (including the "
        "signature and docstring as given). Do not include __main__ "
        "blocks, example usage, or test code outside the function.\n\n"
        f"{stub.rstrip()}\n"
    )


def extract_function_body(source: str, fn_name: str, *, indent: str = "    ") -> str:
    """Return the body of ``fn_name`` from ``source``, indented for
    insertion after the original stub.

    Drops the docstring (the harness's stub already has it). Falls
    back to returning the whole source if parsing fails or the
    function can't be found — better to give the harness *something*
    than a blank completion.
    """
    if not source:
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    target: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            target = node
            break
    if target is None:
        return source
    body = list(target.body)
    # Drop a leading docstring expression if present.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if not body:
        return f"{indent}pass\n"
    lines: list[str] = []
    for stmt in body:
        try:
            stmt_src = ast.unparse(stmt)
        except Exception:  # pragma: no cover — ast.unparse is robust on 3.9+
            continue
        for line in stmt_src.split("\n"):
            lines.append(indent + line if line else "")
    return "\n".join(lines) + "\n"
