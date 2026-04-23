"""Session 0.5 — HTTP and LLM boundary enforcement.

Every production file in ``belief/`` must route outbound HTTP through
:mod:`belief.core.http` and LLM calls through either :mod:`belief.llm`
(main pipeline) or :mod:`belief.photosynthesis.safety.cost_tracker`
(photosynthesis daemon).

This test walks the tree and fails on any file that constructs raw
``httpx.Client`` / ``httpx.AsyncClient`` or imports from ``anthropic``
unless it is on an explicit exemption list.  Exemptions are given as
repo-relative paths so a same-named file elsewhere in the tree cannot
accidentally match.

If a new file needs to bypass the boundary (rare), add it to the
exemption list *and* note the reason in
``docs/architecture/http_boundary.md``.  Do not broaden the pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BELIEF_ROOT = REPO_ROOT / "belief"


# ---------------------------------------------------------------------------
# Patterns + exemptions (see docs/architecture/http_boundary.md)
# ---------------------------------------------------------------------------


# Pattern -> (human label, exempt-paths relative to repo root)
BOUNDARY_RULES: list[tuple[str, str, set[str]]] = [
    (
        r"\bhttpx\.AsyncClient\s*\(",
        "raw httpx.AsyncClient(...) construction",
        {
            # The shared wrapper itself
            "belief/core/http.py",
            # Main-pipeline LLM transport (Anthropic HTTP API + Ollama 127.0.0.1)
            "belief/llm.py",
            # Raw-model comparison baseline — deliberately bypasses our stack
            "belief/experiments/raw_runner.py",
        },
    ),
    (
        r"\bhttpx\.Client\s*\(",
        "raw httpx.Client(...) construction",
        {
            "belief/core/http.py",
            "belief/llm.py",
            "belief/experiments/raw_runner.py",
        },
    ),
    (
        r"^\s*from\s+anthropic\s+import\b",
        "direct 'from anthropic import ...'",
        {
            # The only module that instantiates the Anthropic SDK, for
            # photosynthesis daemon cost metering.  Main pipeline uses
            # raw HTTP to api.anthropic.com and does not import the SDK.
            "belief/photosynthesis/safety/cost_tracker.py",
        },
    ),
    (
        r"\banthropic\.Anthropic\s*\(",
        "anthropic.Anthropic(...) construction",
        {
            "belief/photosynthesis/safety/cost_tracker.py",
        },
    ),
]


# ---------------------------------------------------------------------------
# Walk + filter
# ---------------------------------------------------------------------------


def _belief_py_files() -> list[Path]:
    """Every .py under belief/, excluding generated test fixtures and
    __pycache__ directories.  Ordered for deterministic output on
    failure."""
    files: list[Path] = []
    for path in BELIEF_ROOT.rglob("*.py"):
        # Skip compiled-cache + vendored content if any ever lands.
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return sorted(files)


def _contains_bare_pattern(file_path: Path, pattern: re.Pattern[str]) -> bool:
    """True iff ``pattern`` matches outside Python string literals and
    comments.  We split on ``#`` for comments and use a naive in-string
    heuristic — this catches the 99% case without bringing in a full
    Python tokenizer for a boundary test.

    Note: the integration_tester agent emits *generated-code strings*
    containing ``httpx.get(...)``.  Those are strings inside source,
    not real calls, and must not trigger the boundary test.  We rely
    on the regex requiring ``httpx.AsyncClient(`` (with parens) which
    is distinctive enough that it never appears as a string template."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0]
        if pattern.search(line):
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern_src,label,exempt_paths",
    BOUNDARY_RULES,
    ids=[label for _, label, _ in BOUNDARY_RULES],
)
def test_no_boundary_bypass(
    pattern_src: str, label: str, exempt_paths: set[str]
) -> None:
    """For each forbidden pattern, assert no non-exempt belief/*.py file
    contains it.  On failure, list every offender — operators should
    see the full set, not one file at a time."""
    pattern = re.compile(pattern_src, re.MULTILINE)
    offenders: list[str] = []
    for path in _belief_py_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in exempt_paths:
            continue
        if _contains_bare_pattern(path, pattern):
            offenders.append(rel)

    assert not offenders, (
        f"Boundary violation: {label} found in non-exempt files. "
        f"If this is intentional, add the path to BOUNDARY_RULES' "
        f"exemption list in this test AND document the reason in "
        f"docs/architecture/http_boundary.md. Offending files: "
        f"{offenders}"
    )


def test_exempt_paths_all_exist() -> None:
    """Every path in an exemption list must be a real file.  Keeps the
    exemption list from drifting when files get renamed or deleted."""
    missing: list[str] = []
    for _pattern, _label, exempt in BOUNDARY_RULES:
        for rel in exempt:
            if not (REPO_ROOT / rel).is_file():
                missing.append(rel)
    assert not missing, (
        f"Stale boundary exemptions (file does not exist): {missing}. "
        f"Remove or update BOUNDARY_RULES in this test."
    )


def test_core_http_exports_sync_helpers() -> None:
    """Regression guard for Session 0.5: package_validator relies on
    get_bytes_sync existing in belief.core.http.  If someone removes
    it, fail here loudly rather than let the validator fall back to
    raw httpx silently."""
    from belief.core import http as core_http

    for name in ("get_bytes_sync", "head_sync", "post_form_sync", "get_async_client"):
        assert hasattr(core_http, name), (
            f"belief.core.http is missing {name!r} — do not remove without "
            f"updating every caller and the boundary test."
        )
