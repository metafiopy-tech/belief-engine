"""Hermetic tests for the Session 7 (v3.2) repo-map.

No network, no ChromaDB, no Ollama.  Every test builds a tiny fixture
repo in a tmp_path and checks parse / rank / budget / cache
behaviour.
"""

from __future__ import annotations

from pathlib import Path


from belief.repomap import RepoMap


def _fixture_repo(root: Path) -> None:
    """Tiny 3-file repo.

    ``helpers.foo`` is called from ``app`` and ``bar`` (3 refs); it
    should rank highest.  ``unused_func`` has no callers anywhere.
    """
    (root / "helpers.py").write_text(
        "def foo():\n    return 42\n\n"
        "def unused_func():\n    return 0\n\n"
        "class Helper:\n    def method(self):\n        return foo()\n"
    )
    (root / "app.py").write_text(
        "from helpers import foo\n\ndef main():\n    return foo() + foo() + foo()\n"
    )
    (root / "bar.py").write_text(
        "from helpers import foo\n\nclass Bar:\n    def run(self):\n        return foo()\n"
    )


# ---------------------------------------------------------------------------
# Parse + basic shape
# ---------------------------------------------------------------------------


class TestParse:
    def test_parses_all_definitions(self, tmp_path: Path) -> None:
        _fixture_repo(tmp_path)
        rm = RepoMap(root=tmp_path, cache_dir=tmp_path / ".cache")
        out = rm.get_ranked_tags_map(max_tokens=500)
        assert "helpers.py" in out
        assert "app.py" in out
        assert "bar.py" in out
        assert "foo" in out

    def test_non_py_files_ignored(self, tmp_path: Path) -> None:
        _fixture_repo(tmp_path)
        (tmp_path / "README.md").write_text("# not python\n")
        rm = RepoMap(root=tmp_path, cache_dir=tmp_path / ".cache")
        out = rm.get_ranked_tags_map(max_tokens=500)
        assert "README.md" not in out


# ---------------------------------------------------------------------------
# Ranking — referenced function > unused function
# ---------------------------------------------------------------------------


class TestRanking:
    def test_referenced_symbol_appears(self, tmp_path: Path) -> None:
        _fixture_repo(tmp_path)
        rm = RepoMap(root=tmp_path, cache_dir=tmp_path / ".cache")
        out = rm.get_ranked_tags_map(max_tokens=2000)
        # foo is referenced from two other files; it must be in the
        # output.  unused_func should also appear when budget allows.
        assert "foo" in out


# ---------------------------------------------------------------------------
# Personalization — chat_fnames biases ranking
# ---------------------------------------------------------------------------


class TestPersonalization:
    def test_chat_fnames_biases_output(self, tmp_path: Path) -> None:
        _fixture_repo(tmp_path)
        rm = RepoMap(root=tmp_path, cache_dir=tmp_path / ".cache")
        out_biased = rm.get_ranked_tags_map(
            chat_fnames=["bar.py"],
            max_tokens=2000,
        )
        # Both outputs mention bar.py, but biased version should list
        # it at or near the top (before app.py).
        assert "bar.py" in out_biased
        # Weak check: bar.py appears before app.py in biased output.
        bar_idx = out_biased.find("bar.py")
        app_idx = out_biased.find("app.py")
        if bar_idx != -1 and app_idx != -1:
            assert bar_idx < app_idx, (
                f"chat_fnames=['bar.py'] should rank bar.py before app.py:\n{out_biased}"
            )


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


class TestBudget:
    def test_tight_budget_truncates(self, tmp_path: Path) -> None:
        _fixture_repo(tmp_path)
        rm = RepoMap(root=tmp_path, cache_dir=tmp_path / ".cache")
        tight = rm.get_ranked_tags_map(max_tokens=20)  # ~80 chars
        generous = rm.get_ranked_tags_map(max_tokens=2000)  # ~8000 chars
        assert len(tight) <= len(generous)
        # Tight should still return something non-empty.
        assert tight

    def test_empty_repo_returns_empty(self, tmp_path: Path) -> None:
        rm = RepoMap(root=tmp_path, cache_dir=tmp_path / ".cache")
        out = rm.get_ranked_tags_map(max_tokens=2000)
        assert out == ""


# ---------------------------------------------------------------------------
# Cache — second call hits cached parse
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_file_created(self, tmp_path: Path) -> None:
        _fixture_repo(tmp_path)
        cache_dir = tmp_path / ".cache"
        rm = RepoMap(root=tmp_path, cache_dir=cache_dir)
        rm.get_ranked_tags_map(max_tokens=500)
        # At least one cache file should exist after the first call.
        # Accept either .json or .pkl — implementation detail.
        has_cache = any(cache_dir.glob("*.json")) or any(cache_dir.glob("*.pkl"))
        assert has_cache, "cache files not written"

    def test_mtime_invalidates_cache(self, tmp_path: Path) -> None:
        import os

        _fixture_repo(tmp_path)
        cache_dir = tmp_path / ".cache"
        rm = RepoMap(root=tmp_path, cache_dir=cache_dir)
        # Warm the cache (return value unused — we only care about the
        # subsequent call, which should re-read after mtime bump).
        rm.get_ranked_tags_map(max_tokens=2000)
        # Modify helpers.py — bump mtime + content.
        helpers = tmp_path / "helpers.py"
        helpers.write_text("def foo():\n    return 99\n\ndef brand_new():\n    return 'hello'\n")
        # Ensure mtime actually changes (filesystem-resolution safe).
        now = helpers.stat().st_mtime + 2
        os.utime(helpers, (now, now))
        out2 = rm.get_ranked_tags_map(max_tokens=2000)
        assert "brand_new" in out2, (
            "cache should invalidate on mtime change but brand_new is missing"
        )
