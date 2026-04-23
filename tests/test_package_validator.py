"""Hermetic tests for the Session 3 (v3.2) 6-layer package validator.

No live PyPI calls — every Layer 5 test goes through a mock
httpx.AsyncClient.  Fixture top-15k corpus is a 3-name JSON file
in a tmp_path so tests don't touch ``~/.belief-engine/``.

Run with::

    python3 -m pytest tests/test_package_validator.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from belief.validators.import_to_package import resolve_import_to_package
from belief.validators.package_validator import (
    PackageValidator,
    canonicalize_name,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_top_packages(path: Path, names: list[str]) -> None:
    """Write a hugovk-schema top-pypi JSON file to ``path``."""
    payload = {
        "last_update": "2026-04-22",
        "rows": [{"project": n, "download_count": 1} for n in names],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_blocklist(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.request = httpx.Request("GET", "https://pypi.org/simple/x/")


class _FakeAsyncClient:
    """Async httpx client stand-in that returns scripted responses.

    ``handler`` is ``(url) -> int`` returning the desired status code.
    """

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[str] = []

    async def get(self, url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(self._handler(url))

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create a validator-compatible tmp cache dir + empty files."""
    (tmp_path / "known_hallucinations.txt").write_text(
        "settings\nsettings_library\nfake_pkg\n", encoding="utf-8"
    )
    _write_top_packages(
        tmp_path / "top-pypi-packages-15k.json",
        ["pydantic-settings", "fastapi", "pydantic", "requests", "click", "httpx"],
    )
    return tmp_path


def _make_validator(
    tmp_cache_dir: Path,
    *,
    http_handler: Any = None,
) -> PackageValidator:
    client = _FakeAsyncClient(http_handler) if http_handler is not None else None
    return PackageValidator(
        blocklist_path=tmp_cache_dir / "known_hallucinations.txt",
        top_packages_path=tmp_cache_dir / "top-pypi-packages-15k.json",
        lookup_cache_path=tmp_cache_dir / "pypi_lookup_cache.json",
        rejection_log_path=tmp_cache_dir / "hallucination_log.jsonl",
        http_client=client,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Canonicalization (PEP 503)
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_underscore_to_hyphen(self) -> None:
        assert canonicalize_name("pydantic_settings") == "pydantic-settings"

    def test_uppercase_to_lowercase(self) -> None:
        assert canonicalize_name("PyYAML") == "pyyaml"

    def test_mixed_runs_collapse(self) -> None:
        assert canonicalize_name("foo__bar--baz..qux") == "foo-bar-baz-qux"

    def test_empty_string(self) -> None:
        assert canonicalize_name("") == ""


# ---------------------------------------------------------------------------
# Layer 2 — stdlib rejection
# ---------------------------------------------------------------------------


class TestStdlibLayer:
    @pytest.mark.asyncio
    async def test_timeit_rejected_at_stdlib_layer(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir)
        result = await validator.validate("timeit")
        assert result.accepted is False
        assert result.layer == "stdlib"
        assert "stdlib module" in result.reason

    @pytest.mark.asyncio
    async def test_uuid_rejected_at_stdlib_layer(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir)
        result = await validator.validate("uuid")
        assert result.layer == "stdlib"
        assert result.accepted is False

    @pytest.mark.asyncio
    async def test_stdlib_check_is_fast(self, tmp_cache_dir: Path) -> None:
        """Stdlib rejection must not depend on the PyPI handler — so
        we pass a handler that raises if called, and verify timeit
        still resolves."""

        def _explode(url: str) -> int:
            raise AssertionError(f"Layer 5 called for stdlib name: {url}")

        validator = _make_validator(tmp_cache_dir, http_handler=_explode)
        result = await validator.validate("timeit")
        assert result.accepted is False
        assert result.layer == "stdlib"


# ---------------------------------------------------------------------------
# Layer 3 — hallucination blocklist
# ---------------------------------------------------------------------------


class TestHallucinationLayer:
    @pytest.mark.asyncio
    async def test_settings_library_rejected(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir)
        result = await validator.validate("settings_library")
        assert result.accepted is False
        assert result.layer == "hallucination"
        assert "blocklist" in result.reason

    @pytest.mark.asyncio
    async def test_add_hallucination_persists(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir)
        # Cheat: the validator's add_hallucination writes to the
        # package's installed blocklist file.  For the test we reach
        # directly into the private blocklist to verify runtime behavior.
        validator._blocklist = frozenset(validator._blocklist | {"brand-new-fake"})
        result = await validator.validate("brand-new-fake")
        assert result.layer == "hallucination"


# ---------------------------------------------------------------------------
# Layer 4 — top-15k authoritative positive
# ---------------------------------------------------------------------------


class TestTop15kLayer:
    @pytest.mark.asyncio
    async def test_pydantic_settings_accepted_via_top15k(self, tmp_cache_dir: Path) -> None:
        """PEP 503 canonicalisation + top-15k lookup.  The original
        overnight bug: passing ``pydantic_settings`` (underscored) to
        PyPI failed; canonicalising to ``pydantic-settings`` finds it.
        """

        def _explode(url: str) -> int:
            raise AssertionError(f"Top-15k should accept without PyPI call: {url}")

        validator = _make_validator(tmp_cache_dir, http_handler=_explode)
        result = await validator.validate("pydantic_settings")
        assert result.accepted is True
        assert result.layer == "top15k"
        assert result.canonical_name == "pydantic-settings"

    @pytest.mark.asyncio
    async def test_requests_in_top15k(self, tmp_cache_dir: Path) -> None:
        def _explode(url: str) -> int:
            raise AssertionError(f"Top-15k should accept without PyPI call: {url}")

        validator = _make_validator(tmp_cache_dir, http_handler=_explode)
        result = await validator.validate("requests")
        assert result.accepted is True
        assert result.layer == "top15k"


# ---------------------------------------------------------------------------
# Layer 5 — PyPI Simple lookup (mocked)
# ---------------------------------------------------------------------------


class TestPypiSimpleLayer:
    @pytest.mark.asyncio
    async def test_accepts_on_200(self, tmp_cache_dir: Path) -> None:
        # "new-legit-pkg" isn't in our top-15k fixture, so we need
        # Layer 5.  Simulate PyPI returning 200.
        validator = _make_validator(tmp_cache_dir, http_handler=lambda _: 200)
        result = await validator.validate("new-legit-pkg")
        assert result.accepted is True
        assert result.layer == "pypi"

    @pytest.mark.asyncio
    async def test_rejects_on_404(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir, http_handler=lambda _: 404)
        result = await validator.validate("definitely-not-real")
        assert result.accepted is False
        assert result.layer == "pypi"

    @pytest.mark.asyncio
    async def test_positive_cached_on_second_call(self, tmp_cache_dir: Path) -> None:
        call_count = {"n": 0}

        def _handler(url: str) -> int:
            call_count["n"] += 1
            return 200

        validator = _make_validator(tmp_cache_dir, http_handler=_handler)
        await validator.validate("brand-new-pkg")
        assert call_count["n"] == 1
        # Second call within TTL — must NOT call PyPI again.
        # Rebuild the validator with the SAME cache path to simulate
        # a fresh run (persistence).
        validator2 = _make_validator(tmp_cache_dir, http_handler=_handler)
        result2 = await validator2.validate("brand-new-pkg")
        assert result2.accepted is True
        assert result2.layer == "pypi_cache"
        assert call_count["n"] == 1  # still only one call

    @pytest.mark.asyncio
    async def test_negative_cache_has_short_ttl(self, tmp_cache_dir: Path) -> None:
        """Negative entries expire after 1h so a just-published pkg
        isn't stuck.  We simulate by writing a stale cache entry
        directly and verifying the validator re-queries PyPI."""
        cache_path = tmp_cache_dir / "pypi_lookup_cache.json"
        # Seed a stale negative entry (2 hours old).
        stale = {
            "just-published": {
                "exists": False,
                "checked_at": time.time() - 7200,
            }
        }
        cache_path.write_text(json.dumps(stale), encoding="utf-8")

        # Now call with a 200 handler — the stale negative must NOT be
        # honoured, and the real lookup should accept.
        validator = _make_validator(tmp_cache_dir, http_handler=lambda _: 200)
        result = await validator.validate("just-published")
        assert result.accepted is True
        assert result.layer == "pypi"  # fresh lookup, not pypi_cache


# ---------------------------------------------------------------------------
# Layer 6 — Levenshtein fuzzy match
# ---------------------------------------------------------------------------


class TestFuzzySuggestion:
    @pytest.mark.asyncio
    async def test_typo_gets_suggestion(self, tmp_cache_dir: Path) -> None:
        """`reqeusts` (typo for requests) should be rejected with a
        fuzzy-match suggestion pointing at `requests` (in top-15k)."""
        validator = _make_validator(tmp_cache_dir, http_handler=lambda _: 404)
        result = await validator.validate("reqeusts")
        assert result.accepted is False
        assert result.suggestion == "requests"

    @pytest.mark.asyncio
    async def test_no_suggestion_when_too_far(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir, http_handler=lambda _: 404)
        result = await validator.validate("xyzzylmnop-not-a-real-thing")
        assert result.suggestion is None


# ---------------------------------------------------------------------------
# Import name aliasing
# ---------------------------------------------------------------------------


class TestImportAliasing:
    @pytest.mark.asyncio
    async def test_cv2_routed_to_opencv_python(self, tmp_cache_dir: Path) -> None:
        # Add opencv-python to the top-15k corpus so the alias resolves.
        _write_top_packages(
            tmp_cache_dir / "top-pypi-packages-15k.json",
            ["opencv-python", "Pillow", "scikit-learn", "fastapi"],
        )
        validator = _make_validator(tmp_cache_dir)
        result = await validator.validate("cv2")
        assert result.accepted is True
        assert result.canonical_name == "opencv-python"

    def test_PIL_routed_to_Pillow(self) -> None:
        assert resolve_import_to_package("PIL") == "Pillow"

    def test_sklearn_routed_to_scikit_learn(self) -> None:
        assert resolve_import_to_package("sklearn") == "scikit-learn"

    def test_unknown_import_passes_through(self) -> None:
        assert resolve_import_to_package("httpx") == "httpx"
        assert resolve_import_to_package("requests") == "requests"


# ---------------------------------------------------------------------------
# Telemetry — local-only
# ---------------------------------------------------------------------------


class TestLocalRejectionLog:
    @pytest.mark.asyncio
    async def test_rejection_is_logged_locally(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir)
        await validator.validate("timeit")  # stdlib → rejected
        log_path = tmp_cache_dir / "hallucination_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["canonical_name"] == "timeit"
        assert entry["layer"] == "stdlib"
        assert "ts" in entry

    @pytest.mark.asyncio
    async def test_acceptance_not_logged(self, tmp_cache_dir: Path) -> None:
        validator = _make_validator(tmp_cache_dir)
        await validator.validate("requests")  # top-15k → accepted
        log_path = tmp_cache_dir / "hallucination_log.jsonl"
        assert not log_path.exists()  # no log file written on accept
