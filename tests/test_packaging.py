"""Tests for Session 18: packaging + distribution.

These assert only on files and metadata — no heavy imports, no
network, no chromadb.  They belong to the "lint your own artefacts"
band: catch simple packaging mistakes (wrong version string,
missing extras group, a README that dropped a section) before
they reach PyPI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


# ── Locate the repo root relative to this test file ─────────────────────────


REPO_ROOT = Path(__file__).resolve().parents[1]


# ── tomllib / tomli shim (3.10 fallback) ────────────────────────────────────


try:
    import tomllib as _toml_mod  # type: ignore
except ImportError:  # pragma: no cover - only when running on Python 3.10
    try:
        import tomli as _toml_mod  # type: ignore
    except ImportError:
        _toml_mod = None  # type: ignore


def _load_pyproject() -> dict:
    if _toml_mod is None:
        pytest.skip("neither tomllib nor tomli is available")
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return _toml_mod.load(fh)


# ── Version + optional dependencies ─────────────────────────────────────────


class TestPyprojectVersion:
    def test_version_bumped_to_3_2_0(self):
        cfg = _load_pyproject()
        assert cfg["project"]["version"] == "3.2.0", "Audit ship: bump to 3.2.0"

    def test_project_name_unchanged(self):
        cfg = _load_pyproject()
        assert cfg["project"]["name"] == "belief-engine"

    def test_python_requires_untouched(self):
        """Session 18 must not tighten python-requires — PyPI users
        on 3.11 / 3.12 / 3.13 shouldn't suddenly be frozen out."""
        cfg = _load_pyproject()
        assert cfg["project"]["requires-python"].startswith(">=3.")

    def test_belief_cli_entry_point(self):
        cfg = _load_pyproject()
        assert cfg["project"]["scripts"]["belief"] == "belief.cli:app"


class TestOptionalDependencies:
    def test_spec_groups_present(self):
        """The spec names [local], [photosynthesis], [full] — all
        three must exist after Session 18."""
        cfg = _load_pyproject()
        groups = cfg["project"]["optional-dependencies"]
        assert "local" in groups
        assert "photosynthesis" in groups
        assert "full" in groups

    def test_local_group_has_ollama(self):
        cfg = _load_pyproject()
        local = " ".join(cfg["project"]["optional-dependencies"]["local"])
        assert "ollama" in local

    def test_photosynthesis_has_harvester_stack(self):
        cfg = _load_pyproject()
        photo = " ".join(cfg["project"]["optional-dependencies"]["photosynthesis"])
        # Items that drove Session 3-5's harvester:
        for pkg in ("apscheduler", "feedparser", "tenacity"):
            assert pkg in photo, f"[photosynthesis] missing {pkg}"

    def test_full_group_covers_local_photosynthesis_optimize(self):
        """`pip install belief-engine[full]` must pull the major
        optional stacks — otherwise the 'full v3.1 experience'
        label in the pyproject comment is a lie."""
        cfg = _load_pyproject()
        full = " ".join(cfg["project"]["optional-dependencies"]["full"])
        for pkg in (
            "ollama",
            "apscheduler",
            "feedparser",
            "dspy",
            "scikit-learn",
            "sentence-transformers",
        ):
            assert pkg in full, f"[full] missing {pkg}"


# ── Setup + demo shell scripts ──────────────────────────────────────────────


SETUP_SCRIPT = REPO_ROOT / "scripts" / "belief-setup.sh"
DEMO_SCRIPT = REPO_ROOT / "scripts" / "belief-demo.sh"


class TestSetupScript:
    def test_file_exists(self):
        assert SETUP_SCRIPT.exists(), "scripts/belief-setup.sh must be present"

    def test_shebang_is_bash(self):
        first = SETUP_SCRIPT.read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("#!"), "setup script needs a shebang"
        assert "bash" in first, "setup script should use bash (set -euo pipefail)"

    def test_syntax_is_valid(self):
        """bash -n catches unclosed quotes, missing fi, etc."""
        result = subprocess.run(
            ["bash", "-n", str(SETUP_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    def test_covers_all_spec_steps(self):
        """Session 18 Task 1 lists five required steps."""
        body = SETUP_SCRIPT.read_text(encoding="utf-8").lower()
        # 1. ollama install
        assert "ollama" in body
        # 2. pull a model
        assert "ollama pull" in body
        # 3. soil directory
        assert "soil" in body
        # 4. smoke build
        assert "smoke" in body or "belief --goal" in body
        # 5. grinder hint or invocation
        assert "grinder" in body

    def test_cross_platform_support(self):
        """Setup must work on macOS AND Linux (spec constraint)."""
        body = SETUP_SCRIPT.read_text(encoding="utf-8").lower()
        assert "darwin" in body or "macos" in body
        assert "linux" in body

    def test_set_flags(self):
        """set -euo pipefail is the standard for fail-fast shell scripts."""
        body = SETUP_SCRIPT.read_text(encoding="utf-8")
        assert "set -euo pipefail" in body or "set -eu" in body


class TestDemoScript:
    def test_file_exists(self):
        assert DEMO_SCRIPT.exists(), "scripts/belief-demo.sh must be present"

    def test_syntax_is_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(DEMO_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    def test_covers_spec_flow(self):
        """Session 18 Task 4 lists the recording flow."""
        body = DEMO_SCRIPT.read_text(encoding="utf-8").lower()
        for cmd in ("progression", "dashboard", "library"):
            assert cmd in body, f"demo should call belief {cmd}"


# ── README ─────────────────────────────────────────────────────────────────


README = REPO_ROOT / "README.md"


class TestReadme:
    def _text(self) -> str:
        return README.read_text(encoding="utf-8")

    def test_readme_exists(self):
        assert README.exists()

    def test_local_only_section(self):
        body = self._text().lower()
        assert "local-only quick start" in body

    def test_soil_compounds_section(self):
        body = self._text().lower()
        assert "soil compounds" in body

    def test_progression_per_vertical_section(self):
        body = self._text().lower()
        assert "progression per vertical" in body

    def test_photosynthesis_section(self):
        body = self._text().lower()
        assert "photosynthesis" in body

    def test_hybrid_mode_section(self):
        body = self._text().lower()
        assert "hybrid mode" in body

    def test_manifold_cli_referenced(self):
        """v3.1 added `belief manifold` — new users should find it here."""
        assert "belief manifold" in self._text()
