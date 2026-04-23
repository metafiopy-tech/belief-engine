"""Tests for Session 8.5b — NoTodoCovenant.

The covenant rewrites TODO/FIXME/XXX markers out of generated Python
source.  These tests pin the transformation contract so future sessions
can't silently weaken it:

* ``pass  # TODO …`` / ``...  # TODO …`` → ``raise NotImplementedError(...)``
* ``<code>  # TODO …`` (non-stub) → ``<code>`` (comment stripped)
* String-literal occurrences are left alone.
* Marker words inside identifiers (``todo_count``) are not matched.
* Case-insensitive; ``fixme`` / ``FIX_ME`` both trigger.
"""

from __future__ import annotations

from belief.covenants.no_todo import apply_no_todo_covenant


# ---------------------------------------------------------------------------
# Stub rewrites (pass / ... + # TODO → NotImplementedError)
# ---------------------------------------------------------------------------


class TestStubRewrite:
    def test_pass_todo_becomes_raise(self) -> None:
        src = "def f():\n    pass  # TODO: implement\n"
        out, applied = apply_no_todo_covenant(src)
        assert "pass" not in out
        assert "# TODO" not in out
        assert "raise NotImplementedError" in out
        assert any(a.rule == "no_todo.stub_to_raise" for a in applied)

    def test_ellipsis_todo_becomes_raise(self) -> None:
        src = "def f():\n    ...  # TODO: implement f\n"
        out, applied = apply_no_todo_covenant(src)
        assert "..." not in out
        assert "raise NotImplementedError" in out
        assert len(applied) == 1

    def test_fixme_with_underscore_variant(self) -> None:
        src = "def f():\n    pass  # FIX_ME: later\n"
        out, _ = apply_no_todo_covenant(src)
        assert "raise NotImplementedError" in out
        assert "FIX_ME" not in out

    def test_xxx_marker_also_rewrites(self) -> None:
        src = "def f():\n    pass  # XXX revisit\n"
        out, _ = apply_no_todo_covenant(src)
        assert "raise NotImplementedError" in out

    def test_case_insensitive(self) -> None:
        src = "def f():\n    pass  # todo implement\n"
        out, _ = apply_no_todo_covenant(src)
        assert "raise NotImplementedError" in out

    def test_indent_preserved(self) -> None:
        src = "class C:\n    def m(self):\n        pass  # TODO\n"
        out, _ = apply_no_todo_covenant(src)
        # The indent must be preserved — 8 spaces for a method body.
        assert "        raise NotImplementedError" in out


# ---------------------------------------------------------------------------
# Trailing-comment strip (TODO on otherwise-fine code)
# ---------------------------------------------------------------------------


class TestCommentStrip:
    def test_trailing_todo_comment_is_stripped(self) -> None:
        src = "x = 1  # TODO: clean this up later\n"
        out, applied = apply_no_todo_covenant(src)
        assert out.strip() == "x = 1"
        assert any(a.rule == "no_todo.comment_strip" for a in applied)

    def test_todo_only_line_is_dropped(self) -> None:
        src = "def f():\n    x = 1\n    # TODO: refactor\n    return x\n"
        out, _ = apply_no_todo_covenant(src)
        assert "# TODO" not in out
        # The code around the comment must stay intact.
        assert "x = 1" in out
        assert "return x" in out

    def test_non_todo_comment_is_preserved(self) -> None:
        src = "x = 1  # regular comment\n"
        out, applied = apply_no_todo_covenant(src)
        assert "# regular comment" in out
        assert applied == []


# ---------------------------------------------------------------------------
# Things we must NOT touch
# ---------------------------------------------------------------------------


class TestFalsePositiveSafety:
    def test_string_literals_untouched(self) -> None:
        src = 'MSG = "User has a TODO item"\nprint(MSG)\n'
        out, applied = apply_no_todo_covenant(src)
        # Note: the naïve split-on-# heuristic doesn't tokenize strings
        # perfectly, but a TODO inside a string on a line with NO '#'
        # should pass through unchanged.
        assert "TODO item" in out
        assert applied == []

    def test_identifier_containing_todo_is_untouched(self) -> None:
        src = "todo_count = 3\nprint(todo_count)\n"
        out, applied = apply_no_todo_covenant(src)
        assert "todo_count = 3" in out
        assert applied == []

    def test_clean_file_roundtrips(self) -> None:
        src = "def f(x: int) -> int:\n    return x + 1\n"
        out, applied = apply_no_todo_covenant(src)
        assert out == src
        assert applied == []


# ---------------------------------------------------------------------------
# Applied metadata
# ---------------------------------------------------------------------------


class TestAppliedMetadata:
    def test_filename_recorded(self) -> None:
        src = "x = 1  # TODO\n"
        _, applied = apply_no_todo_covenant(src, filename="foo.py")
        assert applied[0].file == "foo.py"

    def test_line_numbers_are_1_based(self) -> None:
        src = "\n\nx = 1  # TODO strip\n"
        _, applied = apply_no_todo_covenant(src)
        assert applied[0].line == 3


# ---------------------------------------------------------------------------
# Integration with enforce_python_covenants pipeline
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_no_todo_runs_before_pydantic_prepass(self) -> None:
        """The NoTodo covenant must run on every .py file, including
        files that don't import pydantic.  If it only ran for pydantic
        files, the 95% of generated code that's not pydantic-related
        would escape the check.
        """
        from belief.covenants import enforce_python_covenants

        # No pydantic anywhere — this would short-circuit out of the
        # old pipeline before Stage 0 was added.
        src = "def f():\n    pass  # TODO later\n"
        out, applied = enforce_python_covenants(src, filename="plain.py")

        assert "# TODO" not in out
        assert "raise NotImplementedError" in out
        assert any(a.rule.startswith("no_todo.") for a in applied)
