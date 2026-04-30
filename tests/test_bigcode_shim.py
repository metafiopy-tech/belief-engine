"""Hermetic tests for ``scripts/bigcode_shim.py``.

No network, no real subprocess, no real engine. Every test
monkeypatches ``subprocess.run`` (via ``asyncio.to_thread``) and the
output-directory reader so the shim exercises only its own routing,
parsing, and code-extraction logic.

Skipped wholesale if FastAPI isn't installed (it lives in the [bench]
extra, opt-in). That keeps the default ``pip install -e .[dev]`` test
run exactly as it was.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# scripts/ is not a regular package and isn't on the import path by
# default — pull it in via importlib so we can exercise the shim
# without packaging it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# FastAPI lives in the [bench] extra. Skip cleanly when it's missing
# rather than blowing up the whole test suite.
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_SHIM_PATH = _REPO_ROOT / "scripts" / "bigcode_shim.py"


def _load_shim():
    spec = importlib.util.spec_from_file_location("bigcode_shim", _SHIM_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def shim():
    return _load_shim()


@pytest.fixture()
def client(shim):
    return TestClient(shim.app)


# ---------------------------------------------------------------------------
# _parse_summary_line
# ---------------------------------------------------------------------------


class TestParseSummaryLine:
    def test_extracts_summary_at_end_of_stdout(self, shim) -> None:
        stdout = (
            "============================================================\n"
            "  BUILD COMPLETE — 187.3s\n"
            "  Verdict: pass\n"
            '{"run_id": "belief-abcdef12", "verdict": "pass", "weighted_score": 1.0}\n'
        )
        s = shim._parse_summary_line(stdout)
        assert s["run_id"] == "belief-abcdef12"
        assert s["verdict"] == "pass"

    def test_ignores_non_json_braces(self, shim) -> None:
        """A line like ``{some debug repr}`` is not JSON and must not
        be misparsed as the summary."""
        stdout = (
            'DEBUG: state = {state object stuff that looks like}\n{"run_id": "belief-deadbeef"}\n'
        )
        s = shim._parse_summary_line(stdout)
        assert s["run_id"] == "belief-deadbeef"

    def test_returns_empty_when_no_summary(self, shim) -> None:
        stdout = "nothing useful here\nplain text\n"
        assert shim._parse_summary_line(stdout) == {}

    def test_returns_empty_on_blank(self, shim) -> None:
        assert shim._parse_summary_line("") == {}


# ---------------------------------------------------------------------------
# _extract_first_code_file
# ---------------------------------------------------------------------------


class TestExtractFirstCodeFile:
    def test_prefers_non_test_python_file(self, shim, tmp_path: Path) -> None:
        d = tmp_path / "run_a"
        d.mkdir()
        (d / "test_fizzbuzz.py").write_text("def test_x(): pass\n")
        (d / "fizzbuzz.py").write_text("def fizzbuzz(n):\n    return 'foo'\n")
        out = shim._extract_first_code_file(d)
        assert "def fizzbuzz" in out
        assert "def test_x" not in out

    def test_skips_conftest_and_tests_module(self, shim, tmp_path: Path) -> None:
        d = tmp_path / "run_b"
        d.mkdir()
        (d / "conftest.py").write_text("import pytest\n")
        (d / "tests.py").write_text("def test_z(): pass\n")
        (d / "main.py").write_text("MAIN = 1\n")
        out = shim._extract_first_code_file(d)
        assert "MAIN = 1" in out

    def test_falls_back_to_only_test_file(self, shim, tmp_path: Path) -> None:
        """If literally everything is a test file, return the first
        anyway — better to give the harness something than nothing."""
        d = tmp_path / "run_c"
        d.mkdir()
        (d / "test_a.py").write_text("# only file in dir\n")
        out = shim._extract_first_code_file(d)
        assert "only file in dir" in out

    def test_returns_empty_for_missing_dir(self, shim, tmp_path: Path) -> None:
        assert shim._extract_first_code_file(tmp_path / "nope") == ""

    def test_returns_empty_for_empty_dir(self, shim, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert shim._extract_first_code_file(d) == ""


# ---------------------------------------------------------------------------
# /v1/models — startup pings
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_lists_belief_engine_local(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert "belief-engine-local" in ids

    def test_healthz(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


# ---------------------------------------------------------------------------
# /v1/completions — full path with monkeypatched engine
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_engine(
    monkeypatch: pytest.MonkeyPatch,
    shim,
    *,
    proc: _FakeProc,
    code_files: dict[str, str] | None = None,
    output_root: Path | None = None,
) -> Path:
    """Replace subprocess.run + output dir resolution with synthetic
    state. Returns the synthetic output root so callers can assert on
    it if they care.
    """
    import subprocess as _sp

    def _fake_run(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(_sp, "run", _fake_run)

    if output_root is None:
        output_root = Path("/tmp/__shim_test_output__")
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(shim, "_output_root", lambda: output_root)

    # Pre-populate run_id directory if asked.
    if code_files is not None:
        # Pull run_id out of the proc's stdout summary.
        summary = shim._parse_summary_line(proc.stdout)
        run_id = summary.get("run_id") or "belief-test"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, body in code_files.items():
            (run_dir / name).write_text(body, encoding="utf-8")
    return output_root


class TestCompletionsEndpoint:
    def test_single_prompt_returns_extracted_code(
        self, client: TestClient, shim, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        proc = _FakeProc(
            stdout='{"run_id": "belief-fizz", "verdict": "pass", "weighted_score": 1.0}\n',
        )
        _install_fake_engine(
            monkeypatch,
            shim,
            proc=proc,
            output_root=tmp_path,
            code_files={
                "test_fizzbuzz.py": "def test_x(): pass\n",
                "fizzbuzz.py": "def fizzbuzz(n):\n    return 'output'\n",
            },
        )
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Build a fizzbuzz", "model": "belief-engine-local"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["object"] == "text_completion"
        assert len(body["choices"]) == 1
        text = body["choices"][0]["text"]
        assert "def fizzbuzz" in text
        assert "def test_x" not in text
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_list_of_prompts_returns_a_choice_per_prompt(
        self, client: TestClient, shim, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        proc = _FakeProc(
            stdout='{"run_id": "belief-batch", "verdict": "pass"}\n',
        )
        _install_fake_engine(
            monkeypatch,
            shim,
            proc=proc,
            output_root=tmp_path,
            code_files={"main.py": "VALUE = 42\n"},
        )
        resp = client.post(
            "/v1/completions",
            json={"prompt": ["task A", "task B", "task C"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["choices"]) == 3
        for i, c in enumerate(body["choices"]):
            assert c["index"] == i
            assert "VALUE = 42" in c["text"]

    def test_engine_produces_nothing_returns_content_filter(
        self, client: TestClient, shim, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No run_id parsed and no code on disk → empty completion +
        finish_reason=content_filter so the harness can distinguish
        engine-produced-nothing from a normal stop."""
        proc = _FakeProc(stdout="some chatter without a json summary\n")
        _install_fake_engine(monkeypatch, shim, proc=proc, output_root=tmp_path)
        resp = client.post("/v1/completions", json={"prompt": "Build a fizzbuzz"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == ""
        assert body["choices"][0]["finish_reason"] == "content_filter"

    def test_empty_prompt_list_400s(self, client: TestClient) -> None:
        resp = client.post("/v1/completions", json={"prompt": []})
        assert resp.status_code == 400

    def test_extra_fields_are_tolerated(
        self, client: TestClient, shim, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """BigCode sends temperature, top_p, logit_bias, etc. The shim
        must accept and ignore them."""
        proc = _FakeProc(
            stdout='{"run_id": "belief-extras", "verdict": "pass"}\n',
        )
        _install_fake_engine(
            monkeypatch,
            shim,
            proc=proc,
            output_root=tmp_path,
            code_files={"out.py": "X = 1\n"},
        )
        resp = client.post(
            "/v1/completions",
            json={
                "prompt": "anything",
                "temperature": 0.0,
                "top_p": 1.0,
                "logit_bias": {"foo": 1},
                "seed": 42,
                "stop": ["\n\n"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["choices"][0]["text"] == "X = 1\n"


# ---------------------------------------------------------------------------
# HumanEval/MBPP stub adapter
# ---------------------------------------------------------------------------


_HUMANEVAL_STUB = (
    "from typing import List\n"
    "\n"
    "\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
    "    given threshold.\n"
    "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
    "    False\n"
    "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
    "    True\n"
    '    """\n'
)


class TestLooksLikeCodeStub:
    def test_humaneval_prompt_is_a_stub(self, shim) -> None:
        assert shim._looks_like_code_stub(_HUMANEVAL_STUB) is True

    def test_english_goal_is_not_a_stub(self, shim) -> None:
        assert shim._looks_like_code_stub("Build a fizzbuzz function in Python.") is False

    def test_function_without_docstring_is_not_a_stub(self, shim) -> None:
        # Pure code without a docstring isn't HumanEval-shaped — treat
        # as ambiguous and pass through.
        assert shim._looks_like_code_stub("def f(x):\n    return x + 1\n") is False

    def test_blank_prompt_is_not_a_stub(self, shim) -> None:
        assert shim._looks_like_code_stub("") is False
        assert shim._looks_like_code_stub("   ") is False

    def test_syntactically_invalid_prompt_is_not_a_stub(self, shim) -> None:
        # English text that happens to contain "def " shouldn't trigger.
        assert shim._looks_like_code_stub("the def of insanity is...") is False


class TestDetectFunctionName:
    def test_picks_target_function_from_humaneval_stub(self, shim) -> None:
        assert shim._detect_function_name(_HUMANEVAL_STUB) == "has_close_elements"

    def test_picks_last_def_when_multiple_present(self, shim) -> None:
        """HumanEval sometimes has a small helper above the target."""
        src = (
            "def _helper(x):\n    return x * 2\n\n"
            "def actual_target(n):\n"
            '    """The function the tests check."""\n'
        )
        assert shim._detect_function_name(src) == "actual_target"

    def test_returns_none_when_no_def(self, shim) -> None:
        assert shim._detect_function_name("just some text") is None

    def test_regex_fallback_on_syntax_error(self, shim) -> None:
        # Stub that doesn't quite parse but has a recognizable signature.
        src = "def busted(x:\n    pass\n"
        # Either None or "busted" is acceptable; never raises.
        result = shim._detect_function_name(src)
        assert result in (None, "busted")


class TestRewriteStubToGoal:
    def test_rewrite_includes_original_stub(self, shim) -> None:
        out = shim._rewrite_stub_to_goal(_HUMANEVAL_STUB)
        assert "has_close_elements" in out
        assert "Implement the function" in out

    def test_rewrite_says_no_test_code(self, shim) -> None:
        out = shim._rewrite_stub_to_goal(_HUMANEVAL_STUB)
        # Engine should not generate __main__ or test functions —
        # those would mismatch the harness's own test suite.
        assert "test" in out.lower() or "__main__" in out


class TestExtractFunctionBody:
    def test_extracts_body_drops_docstring(self, shim) -> None:
        source = (
            "def has_close_elements(numbers, threshold):\n"
            '    """drop me."""\n'
            "    for i, a in enumerate(numbers):\n"
            "        for b in numbers[i + 1:]:\n"
            "            if abs(a - b) < threshold:\n"
            "                return True\n"
            "    return False\n"
        )
        body = shim._extract_function_body(source, "has_close_elements")
        assert "drop me" not in body
        # ast.unparse may add parens around tuple targets — accept either.
        assert "enumerate(numbers)" in body
        assert "return False" in body
        # Every non-empty line must be indented (so harness can append
        # to the original stub cleanly).
        for line in body.splitlines():
            if line.strip():
                assert line.startswith("    ")

    def test_function_not_found_returns_whole_source(self, shim) -> None:
        source = "def something_else():\n    return 42\n"
        assert shim._extract_function_body(source, "missing") == source

    def test_syntax_error_returns_whole_source(self, shim) -> None:
        source = "def busted(:\n    pass\n"
        assert shim._extract_function_body(source, "busted") == source

    def test_empty_body_after_docstring_returns_pass(self, shim) -> None:
        source = 'def stub():\n    """only a docstring."""\n'
        body = shim._extract_function_body(source, "stub")
        assert body.strip() == "pass"

    def test_blank_source_returns_blank(self, shim) -> None:
        assert shim._extract_function_body("", "anything") == ""


class TestStubAdapterEndToEnd:
    def test_stub_prompt_returns_function_body_only(
        self, client: TestClient, shim, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Send a HumanEval-style stub. The engine produces a full
        file. The shim must return just the body so harness-side
        prompt + completion concatenates to a valid program."""
        engine_output = (
            "from typing import List\n"
            "\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """Check pairwise distances."""\n'
            "    for i, a in enumerate(numbers):\n"
            "        for b in numbers[i + 1:]:\n"
            "            if abs(a - b) < threshold:\n"
            "                return True\n"
            "    return False\n"
        )
        proc = _FakeProc(stdout='{"run_id": "belief-stub-1", "verdict": "pass"}\n')
        _install_fake_engine(
            monkeypatch,
            shim,
            proc=proc,
            output_root=tmp_path,
            code_files={"has_close_elements.py": engine_output},
        )
        resp = client.post("/v1/completions", json={"prompt": _HUMANEVAL_STUB})
        assert resp.status_code == 200
        text = resp.json()["choices"][0]["text"]
        # No signature redefinition (harness already has it).
        assert "def has_close_elements" not in text
        # No docstring (harness already has it).
        assert "Check pairwise distances" not in text
        # Body lines present (ast.unparse may reformat tuple targets).
        assert "enumerate(numbers)" in text
        assert "return False" in text
        # Body is indented for direct concatenation.
        for line in text.splitlines():
            if line.strip():
                assert line.startswith("    ")

    def test_natural_language_prompt_returns_full_file(
        self, client: TestClient, shim, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Non-stub prompts are unchanged: the shim returns the
        engine's full file, no body extraction."""
        engine_output = "def fizzbuzz(n):\n    return 'fizz' if n % 3 == 0 else str(n)\n"
        proc = _FakeProc(stdout='{"run_id": "belief-nl-1", "verdict": "pass"}\n')
        _install_fake_engine(
            monkeypatch,
            shim,
            proc=proc,
            output_root=tmp_path,
            code_files={"fizzbuzz.py": engine_output},
        )
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Build a fizzbuzz function in Python."},
        )
        assert resp.status_code == 200
        text = resp.json()["choices"][0]["text"]
        # Full file returned (signature + body) — no body extraction.
        assert "def fizzbuzz" in text
        assert "return 'fizz'" in text
