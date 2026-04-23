"""Tests for the caching additions in validation Session 1.

Covers:
  - get_or_generate_skeleton — file + in-process two-tier cache
  - cached_ast_parse — lru-cached ast.parse
  - clear_caches — wipes both memos
"""

from __future__ import annotations


import pytest


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "skeleton_cache"


# ── get_or_generate_skeleton ──────────────────────────────────────────────


def test_get_or_generate_stores_on_miss(cache_dir):
    from belief.cache.skeleton_cache import clear_caches, get_or_generate_skeleton

    clear_caches()
    spec = {"file_tree": [{"path": "a.py"}], "framework": "fastapi"}
    calls = {"n": 0}

    def gen():
        calls["n"] += 1
        return {"files": {"a.py": "def f(): ...\n"}}

    payload, hit = get_or_generate_skeleton(spec, gen, base_dir=cache_dir)
    assert hit is False
    assert calls["n"] == 1
    assert "a.py" in payload["files"]


def test_get_or_generate_hits_on_second_call(cache_dir):
    from belief.cache.skeleton_cache import clear_caches, get_or_generate_skeleton

    clear_caches()
    spec = {"file_tree": [{"path": "a.py"}], "framework": "fastapi"}
    calls = {"n": 0}

    def gen():
        calls["n"] += 1
        return {"files": {"a.py": "def f(): ...\n"}}

    get_or_generate_skeleton(spec, gen, base_dir=cache_dir)
    _, hit = get_or_generate_skeleton(spec, gen, base_dir=cache_dir)
    assert hit is True
    assert calls["n"] == 1  # generator must not run again


def test_get_or_generate_different_spec_misses_cache(cache_dir):
    from belief.cache.skeleton_cache import clear_caches, get_or_generate_skeleton

    clear_caches()
    calls = {"n": 0}

    def gen():
        calls["n"] += 1
        return {"files": {"a.py": "x"}}

    get_or_generate_skeleton({"framework": "fastapi"}, gen, base_dir=cache_dir)
    get_or_generate_skeleton({"framework": "flask"}, gen, base_dir=cache_dir)
    assert calls["n"] == 2


def test_get_or_generate_disk_backs_memo_after_clear(cache_dir):
    """clear_caches() wipes the in-process memo but the on-disk entry
    remains, so the next call still hits — without re-running gen()."""
    from belief.cache.skeleton_cache import clear_caches, get_or_generate_skeleton

    clear_caches()
    spec = {"framework": "fastapi"}
    calls = {"n": 0}

    def gen():
        calls["n"] += 1
        return {"files": {"a.py": "x"}}

    get_or_generate_skeleton(spec, gen, base_dir=cache_dir)
    clear_caches()
    _, hit = get_or_generate_skeleton(spec, gen, base_dir=cache_dir)
    assert hit is True
    assert calls["n"] == 1  # generator never ran a second time


def test_get_or_generate_propagates_generator_exceptions(cache_dir):
    """A generator that raises must not poison the cache with a partial
    entry — the next call should retry."""
    from belief.cache.skeleton_cache import clear_caches, get_or_generate_skeleton

    clear_caches()
    spec = {"framework": "fastapi"}
    calls = {"n": 0}

    def gen_fail():
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        get_or_generate_skeleton(spec, gen_fail, base_dir=cache_dir)

    def gen_ok():
        calls["n"] += 1
        return {"files": {"a.py": "x"}}

    payload, hit = get_or_generate_skeleton(spec, gen_ok, base_dir=cache_dir)
    assert hit is False
    assert payload["files"]["a.py"] == "x"
    assert calls["n"] == 2


# ── cached_ast_parse ──────────────────────────────────────────────────────


def test_cached_ast_parse_returns_cached_object():
    from belief.cache.skeleton_cache import cached_ast_parse, clear_caches

    clear_caches()

    src = "x = 1\ndef f():\n    return 42\n"
    t1 = cached_ast_parse(src)
    t2 = cached_ast_parse(src)
    assert t1 is t2


def test_cached_ast_parse_different_sources_different_trees():
    from belief.cache.skeleton_cache import cached_ast_parse, clear_caches

    clear_caches()
    assert cached_ast_parse("a = 1") is not cached_ast_parse("b = 2")


def test_cached_ast_parse_rejects_non_string():
    from belief.cache.skeleton_cache import cached_ast_parse

    with pytest.raises(TypeError):
        cached_ast_parse(123)


def test_cached_ast_parse_raises_on_invalid_syntax():
    from belief.cache.skeleton_cache import cached_ast_parse, clear_caches

    clear_caches()
    with pytest.raises(SyntaxError):
        cached_ast_parse("def :::")
