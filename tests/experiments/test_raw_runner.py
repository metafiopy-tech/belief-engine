"""Unit tests for belief.experiments.raw_runner.

All tests are pure-function — no network, no subprocess.
"""

from __future__ import annotations

import pytest

from belief.experiments.raw_runner import parse_file_blocks, parse_pytest_output


# ---------------------------------------------------------------------------
# parse_file_blocks
# ---------------------------------------------------------------------------


class TestParseFileBlocks:

    def test_single_python_file(self):
        text = (
            "### FILE: main.py\n"
            "```python\n"
            "print('hello')\n"
            "```\n"
        )
        files = parse_file_blocks(text)
        assert list(files.keys()) == ["main.py"]
        assert "print('hello')" in files["main.py"]

    def test_multiple_files(self):
        text = (
            "### FILE: main.py\n"
            "```python\n"
            "x = 1\n"
            "```\n"
            "\n"
            "### FILE: requirements.txt\n"
            "```\n"
            "fastapi\n"
            "```\n"
            "\n"
            "### FILE: test_main.py\n"
            "```python\n"
            "def test_x(): assert 1 == 1\n"
            "```\n"
        )
        files = parse_file_blocks(text)
        assert set(files.keys()) == {"main.py", "requirements.txt", "test_main.py"}
        assert "fastapi" in files["requirements.txt"]

    def test_returns_empty_on_no_blocks(self):
        text = "Here is some text with no file blocks at all."
        assert parse_file_blocks(text) == {}

    def test_strips_trailing_whitespace_from_content(self):
        text = "### FILE: a.py\n```python\nx = 1   \n```\n"
        files = parse_file_blocks(text)
        assert files["a.py"] == "x = 1"

    def test_language_tag_is_optional(self):
        text = "### FILE: notes.txt\n```\nhello world\n```\n"
        files = parse_file_blocks(text)
        assert "notes.txt" in files
        assert files["notes.txt"] == "hello world"

    def test_extra_whitespace_around_file_keyword(self):
        text = "###  FILE:  data.json  \n```json\n{}\n```\n"
        files = parse_file_blocks(text)
        assert "data.json" in files

    def test_multiline_content(self):
        text = (
            "### FILE: app.py\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "\n"
            "def bar():\n"
            "    return foo() * 2\n"
            "```\n"
        )
        files = parse_file_blocks(text)
        assert "def foo():" in files["app.py"]
        assert "def bar():" in files["app.py"]

    def test_ignores_prose_between_blocks(self):
        text = (
            "Here is the implementation:\n\n"
            "### FILE: main.py\n```python\nx = 1\n```\n\n"
            "And here are the tests:\n\n"
            "### FILE: test_main.py\n```python\ndef test_x(): pass\n```\n"
        )
        files = parse_file_blocks(text)
        assert set(files.keys()) == {"main.py", "test_main.py"}

    def test_nested_path_filename(self):
        text = "### FILE: src/utils.py\n```python\npass\n```\n"
        files = parse_file_blocks(text)
        assert "src/utils.py" in files


# ---------------------------------------------------------------------------
# parse_pytest_output
# ---------------------------------------------------------------------------


class TestParsePytestOutput:

    def test_all_passed(self):
        out = "3 passed in 0.12s"
        assert parse_pytest_output(out) == (3, 3)

    def test_passed_and_failed(self):
        out = "2 passed, 1 failed in 0.20s"
        assert parse_pytest_output(out) == (2, 3)

    def test_zero_passed_one_failed(self):
        out = "1 failed in 0.15s"
        assert parse_pytest_output(out) == (0, 1)

    def test_errors_counted_in_total(self):
        out = "1 passed, 2 errors in 0.30s"
        assert parse_pytest_output(out) == (1, 3)

    def test_no_tests_collected(self):
        out = "no tests ran"
        assert parse_pytest_output(out) == (0, 0)

    def test_empty_output(self):
        assert parse_pytest_output("") == (0, 0)

    def test_combined_failed_and_errors(self):
        out = "1 passed, 1 failed, 1 errors in 0.50s"
        assert parse_pytest_output(out) == (1, 3)

    def test_large_numbers(self):
        out = "100 passed, 5 failed in 12.3s"
        assert parse_pytest_output(out) == (100, 105)
