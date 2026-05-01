"""Hermetic tests for ``scripts/generate_humaneval_completions.py``.

Mocks the ``datasets`` library, ``httpx`` (raw backend), and
``subprocess.run`` + filesystem (engine backend) so every test is
fully self-contained.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_generator():
    """Load ``scripts/generate_humaneval_completions.py`` by path so
    we don't have to make ``scripts/`` a real package."""
    path = _REPO_ROOT / "scripts" / "generate_humaneval_completions.py"
    spec = importlib.util.spec_from_file_location("gen_humaneval", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def gen():
    return _load_generator()


_HUMANEVAL_STUB = (
    "from typing import List\n"
    "\n"
    "\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    '    """ Check if any two numbers in the list are closer than the threshold."""\n'
)


# ---------------------------------------------------------------------------
# load_humaneval_problems — datasets shim
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Iterable of dicts, the shape `datasets.load_dataset` returns
    when iterated."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _install_fake_datasets(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    """Stub `datasets.load_dataset` to avoid HuggingFace network IO."""
    import types

    fake_module = types.ModuleType("datasets")

    def _load_dataset(name: str, split: str | None = None, **_: Any) -> _FakeDataset:
        return _FakeDataset(rows)

    fake_module.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


class TestLoadHumanevalProblems:
    def test_returns_all_rows_when_no_limit(self, gen, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [
            {"task_id": "HumanEval/0", "prompt": "p0", "entry_point": "f0"},
            {"task_id": "HumanEval/1", "prompt": "p1", "entry_point": "f1"},
            {"task_id": "HumanEval/2", "prompt": "p2", "entry_point": "f2"},
        ]
        _install_fake_datasets(monkeypatch, rows)
        out = gen.load_humaneval_problems()
        assert len(out) == 3
        assert out[0]["task_id"] == "HumanEval/0"
        assert out[2]["entry_point"] == "f2"

    def test_limit_truncates(self, gen, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [
            {"task_id": f"HumanEval/{i}", "prompt": f"p{i}", "entry_point": f"f{i}"}
            for i in range(10)
        ]
        _install_fake_datasets(monkeypatch, rows)
        out = gen.load_humaneval_problems(limit=3)
        assert len(out) == 3

    def test_missing_fields_default_to_blank(self, gen, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_datasets(monkeypatch, [{"task_id": "x"}])
        out = gen.load_humaneval_problems()
        assert out[0] == {"task_id": "x", "prompt": "", "entry_point": ""}


# ---------------------------------------------------------------------------
# generate_raw_completion — Ollama HTTP path
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Mock httpx.Client. Captures request body for assertions."""

    def __init__(self, response_payload: dict[str, Any]) -> None:
        self._response = _FakeResponse(response_payload)
        self.last_url: str | None = None
        self.last_body: dict[str, Any] | None = None

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def post(self, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        self.last_url = url
        self.last_body = json
        return self._response


class _FakeHttpx:
    def __init__(self, response_payload: dict[str, Any]) -> None:
        self._payload = response_payload
        self.client_instance: _FakeClient | None = None

    def Client(self, **_: Any) -> _FakeClient:  # noqa: N802 — match httpx API
        c = _FakeClient(self._payload)
        self.client_instance = c
        return c


class TestGenerateRawCompletion:
    def test_returns_message_content_unchanged_when_body_only(self, gen) -> None:
        """If Qwen obeys the system prompt and returns just the body,
        we pass it through."""
        fake = _FakeHttpx({"message": {"content": "    return sum(numbers)\n"}})
        text = gen.generate_raw_completion(_HUMANEVAL_STUB, httpx_module=fake)
        assert text == "    return sum(numbers)\n"
        assert fake.client_instance is not None
        body = fake.client_instance.last_body
        assert body is not None
        assert body["model"] == gen.DEFAULT_OLLAMA_MODEL
        assert body["options"]["seed"] == 42
        assert body["options"]["temperature"] == 0.0

    def test_strips_full_function_redefinition(self, gen) -> None:
        """If Qwen redefines the function, we extract just the body."""
        full = (
            "def has_close_elements(numbers, threshold):\n"
            '    """Check pairwise distances."""\n'
            "    for a in numbers:\n"
            "        for b in numbers:\n"
            "            if a != b and abs(a - b) < threshold:\n"
            "                return True\n"
            "    return False\n"
        )
        fake = _FakeHttpx({"message": {"content": full}})
        text = gen.generate_raw_completion(_HUMANEVAL_STUB, httpx_module=fake)
        assert "def has_close_elements" not in text
        assert "Check pairwise distances" not in text
        assert "return False" in text

    def test_fallback_to_response_field(self, gen) -> None:
        """Some Ollama versions put the text in `response` not `message.content`."""
        fake = _FakeHttpx({"response": "    return True\n"})
        text = gen.generate_raw_completion(_HUMANEVAL_STUB, httpx_module=fake)
        assert text == "    return True\n"

    def test_empty_response_returns_blank(self, gen) -> None:
        fake = _FakeHttpx({})
        assert gen.generate_raw_completion(_HUMANEVAL_STUB, httpx_module=fake) == ""

    def test_markdown_fences_are_stripped(self, gen) -> None:
        """Qwen wraps body in ```python ... ``` despite the system
        prompt asking it not to. Strip the fences."""
        fenced = (
            "```python\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
            "```"
        )
        fake = _FakeHttpx({"message": {"content": fenced}})
        text = gen.generate_raw_completion(_HUMANEVAL_STUB, httpx_module=fake)
        assert "```" not in text
        assert "for i in range" in text
        assert "return False" in text


class TestStripMarkdownFences:
    def test_strips_python_fence(self, gen) -> None:
        out = gen._strip_markdown_fences("```python\nreturn 1\n```")
        assert out.strip() == "return 1"

    def test_strips_bare_fence(self, gen) -> None:
        out = gen._strip_markdown_fences("```\nreturn 1\n```")
        assert out.strip() == "return 1"

    def test_passes_through_when_no_fence(self, gen) -> None:
        out = gen._strip_markdown_fences("    return 1\n")
        assert out == "    return 1\n"

    def test_handles_unclosed_fence(self, gen) -> None:
        """If only the opening fence is present, drop just that line."""
        out = gen._strip_markdown_fences("```python\nreturn 1\n")
        assert "```" not in out
        assert "return 1" in out

    def test_blank_input_unchanged(self, gen) -> None:
        assert gen._strip_markdown_fences("") == ""


# ---------------------------------------------------------------------------
# generate_engine_completion — subprocess + filesystem
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestGenerateEngineCompletion:
    def test_extracts_body_from_engine_output(self, gen, tmp_path: Path) -> None:
        engine_output_dir = tmp_path / "belief-abc123"
        engine_output_dir.mkdir()
        (engine_output_dir / "test_x.py").write_text("def test_y(): pass\n")
        (engine_output_dir / "has_close_elements.py").write_text(
            "from typing import List\n\n"
            "def has_close_elements(numbers, threshold):\n"
            '    """drop me."""\n'
            "    return any(abs(a - b) < threshold for a in numbers for b in numbers if a != b)\n"
        )

        def fake_run(*args: Any, **kwargs: Any) -> _FakeProc:
            return _FakeProc(stdout="Run ID: belief-abc123\n", returncode=0)

        text = gen.generate_engine_completion(
            _HUMANEVAL_STUB,
            subprocess_run=fake_run,
            output_root_fn=lambda: tmp_path,
            code_reader=gen._read_first_code_file,
        )
        assert "def has_close_elements" not in text
        assert "drop me" not in text
        # ast.unparse may add extra parens around the genexpr.
        assert "abs(a - b)" in text
        assert "for a in numbers" in text

    def test_falls_back_to_json_summary_for_run_id(self, gen, tmp_path: Path) -> None:
        run_dir = tmp_path / "belief-fromjson"
        run_dir.mkdir()
        (run_dir / "f.py").write_text(
            "def has_close_elements(numbers, threshold):\n    return False\n"
        )

        def fake_run(*args: Any, **kwargs: Any) -> _FakeProc:
            return _FakeProc(
                stdout='{"run_id": "belief-fromjson", "verdict": "pass"}\n',
                returncode=0,
            )

        text = gen.generate_engine_completion(
            _HUMANEVAL_STUB,
            subprocess_run=fake_run,
            output_root_fn=lambda: tmp_path,
            code_reader=gen._read_first_code_file,
        )
        assert "return False" in text

    def test_no_run_id_returns_blank(self, gen, tmp_path: Path) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> _FakeProc:
            return _FakeProc(stdout="garbage with no run id\n", returncode=0)

        text = gen.generate_engine_completion(
            _HUMANEVAL_STUB,
            subprocess_run=fake_run,
            output_root_fn=lambda: tmp_path,
            code_reader=gen._read_first_code_file,
        )
        assert text == ""

    def test_subprocess_timeout_returns_blank(self, gen, tmp_path: Path) -> None:
        import subprocess as sp

        def fake_run(*args: Any, **kwargs: Any) -> _FakeProc:
            raise sp.TimeoutExpired(cmd="belief", timeout=1)

        text = gen.generate_engine_completion(
            _HUMANEVAL_STUB,
            subprocess_run=fake_run,
            output_root_fn=lambda: tmp_path,
            code_reader=gen._read_first_code_file,
        )
        assert text == ""

    def test_run_dir_missing_returns_blank(self, gen, tmp_path: Path) -> None:
        """Run id parsed but output dir doesn't exist."""

        def fake_run(*args: Any, **kwargs: Any) -> _FakeProc:
            return _FakeProc(stdout="Run ID: belief-nodir\n", returncode=0)

        text = gen.generate_engine_completion(
            _HUMANEVAL_STUB,
            subprocess_run=fake_run,
            output_root_fn=lambda: tmp_path,
            code_reader=gen._read_first_code_file,
        )
        assert text == ""


# ---------------------------------------------------------------------------
# Resumable I/O
# ---------------------------------------------------------------------------


class TestResumableIO:
    def test_partial_round_trip(self, gen, tmp_path: Path) -> None:
        partial = tmp_path / "out.json.partial.jsonl"
        gen._append_partial(partial, 0, "    return 1\n")
        gen._append_partial(partial, 2, "    return 3\n")
        loaded = gen._load_partial(partial)
        assert loaded == {0: "    return 1\n", 2: "    return 3\n"}

    def test_load_partial_missing_file_returns_empty(self, gen, tmp_path: Path) -> None:
        assert gen._load_partial(tmp_path / "nope.jsonl") == {}

    def test_load_partial_skips_malformed_lines(self, gen, tmp_path: Path) -> None:
        partial = tmp_path / "out.partial.jsonl"
        partial.write_text(
            'not json\n{"index": 0, "completion": "ok"}\n{}\n\n{"index": 5, "completion": "x"}\n'
        )
        loaded = gen._load_partial(partial)
        assert loaded == {0: "ok", 5: "x"}

    def test_write_final_pads_missing_indices_with_blanks(self, gen, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        gen._write_final(out, {0: "a", 2: "c"}, n_problems=4)
        payload = json.loads(out.read_text())
        assert payload == [["a"], [""], ["c"], [""]]

    def test_write_final_prepends_prompt_when_given(self, gen, tmp_path: Path) -> None:
        """When prompts_by_index is supplied, each non-empty body must
        be prefixed with its task's prompt — required by BigCode harness."""
        out = tmp_path / "with_prompt.json"
        gen._write_final(
            out,
            {0: "    return 1\n", 1: "    return 2\n"},
            n_problems=2,
            prompts_by_index={0: "def a():\n", 1: "def b():\n"},
        )
        payload = json.loads(out.read_text())
        assert payload == [
            ["def a():\n    return 1\n"],
            ["def b():\n    return 2\n"],
        ]

    def test_write_final_skips_prompt_for_empty_bodies(self, gen, tmp_path: Path) -> None:
        """Empty completions stay empty even when prompts are supplied —
        the harness uses empty completions to mark 'engine produced
        nothing usable' which scores as fail (correctly)."""
        out = tmp_path / "empty_skip.json"
        gen._write_final(
            out,
            {0: "", 1: "    return 2\n"},
            n_problems=2,
            prompts_by_index={0: "def a():\n", 1: "def b():\n"},
        )
        payload = json.loads(out.read_text())
        assert payload == [
            [""],
            ["def b():\n    return 2\n"],
        ]

    def test_write_final_atomically_via_tmp(self, gen, tmp_path: Path) -> None:
        out = tmp_path / "atomic.json"
        gen._write_final(out, {0: "a"}, n_problems=1)
        # No leftover .tmp file.
        assert not out.with_suffix(out.suffix + ".tmp").exists()
        assert json.loads(out.read_text()) == [["a"]]


# ---------------------------------------------------------------------------
# Driver — full happy path with both backends mocked
# ---------------------------------------------------------------------------


class TestRunDriver:
    def test_raw_backend_writes_bigcode_format(
        self, gen, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            {
                "task_id": "HumanEval/0",
                "prompt": _HUMANEVAL_STUB,
                "entry_point": "has_close_elements",
            },
            {
                "task_id": "HumanEval/1",
                "prompt": _HUMANEVAL_STUB,
                "entry_point": "has_close_elements",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        # Replace the raw call so we don't open httpx.
        monkeypatch.setattr(gen, "generate_raw_completion", lambda *a, **k: "    return True\n")

        out = tmp_path / "raw.json"
        rc = gen.run(
            backend="raw",
            output=out,
            limit=None,
            ollama_url=gen.DEFAULT_OLLAMA_URL,
            ollama_model=gen.DEFAULT_OLLAMA_MODEL,
            seed=42,
            temperature=0.0,
            engine_timeout_s=60,
            resume=False,
        )
        assert rc == 0
        payload = json.loads(out.read_text())
        # The driver now prepends each task's prompt so the harness's
        # postprocess (which strips len(prompt) chars) round-trips
        # cleanly. Bare-body files would score 0 against BigCode's
        # HumanEval task.
        expected = _HUMANEVAL_STUB + "    return True\n"
        assert payload == [[expected], [expected]]

    def test_resume_skips_done_indices(
        self, gen, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            {
                "task_id": f"HumanEval/{i}",
                "prompt": _HUMANEVAL_STUB,
                "entry_point": "has_close_elements",
            }
            for i in range(3)
        ]
        _install_fake_datasets(monkeypatch, rows)

        # Pre-seed the partial file so index 1 is already "done".
        out = tmp_path / "resume.json"
        partial = gen._partial_path_for(out)
        gen._append_partial(partial, 1, "PRE-EXISTING\n")

        calls: list[int] = []

        def _fake_raw(*args: Any, **kwargs: Any) -> str:
            calls.append(len(calls))
            return f"    return {len(calls)}\n"

        monkeypatch.setattr(gen, "generate_raw_completion", _fake_raw)

        rc = gen.run(
            backend="raw",
            output=out,
            limit=None,
            ollama_url=gen.DEFAULT_OLLAMA_URL,
            ollama_model=gen.DEFAULT_OLLAMA_MODEL,
            seed=42,
            temperature=0.0,
            engine_timeout_s=60,
            resume=True,
        )
        assert rc == 0
        # 2 fresh calls (index 0 and 2), index 1 came from the partial file.
        assert len(calls) == 2
        payload = json.loads(out.read_text())
        # All entries are now prompt-prefixed in the final JSON.
        assert payload[1] == [_HUMANEVAL_STUB + "PRE-EXISTING\n"]

    def test_empty_completions_signal_nonzero_exit(
        self, gen, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rows = [
            {"task_id": "HumanEval/0", "prompt": _HUMANEVAL_STUB, "entry_point": "x"},
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(gen, "generate_raw_completion", lambda *a, **k: "")

        out = tmp_path / "empty.json"
        rc = gen.run(
            backend="raw",
            output=out,
            limit=None,
            ollama_url=gen.DEFAULT_OLLAMA_URL,
            ollama_model=gen.DEFAULT_OLLAMA_MODEL,
            seed=42,
            temperature=0.0,
            engine_timeout_s=60,
            resume=False,
        )
        assert rc == 2  # "completed but had empties"
        # Output file is still written so the harness has something to score.
        assert json.loads(out.read_text()) == [[""]]
