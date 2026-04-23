"""Regression tests for the skeleton_pass1 cache refactor.

The cache refactor in Session 1 moved the live ``SymbolRegistry`` into
a closure inside ``_generate()``.  A stale reference to the outer
``registry`` variable survived in one log line and caused the node's
except handler to fire on every build (the deterministic skeleton
generator ran, but its output was discarded because the following
``len(registry.all_files())`` call raised NameError).

These tests exercise both the cache-miss and cache-hit paths with a
minimal SkeletonArtifact and assert:
  - no warnings leak the word ``registry`` (would indicate a NameError)
  - skeleton_registry_context is populated
  - calling twice hits the cache on the second call
"""

from __future__ import annotations


import pytest


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the skeleton_cache module at a fresh temp dir for the test."""
    from belief.cache import skeleton_cache as sc

    sc.clear_caches()
    monkeypatch.setattr(sc, "DEFAULT_CACHE_DIR", tmp_path)
    yield tmp_path
    sc.clear_caches()


def _minimal_skeleton():
    """The smallest SkeletonArtifact the node accepts.

    We use FileRole.IMPLEMENTATION (any real member that exists in
    the current enum) because the validator is strict; the role value
    itself doesn't matter for this regression.
    """
    from belief.models.skeleton import (
        FileRole,
        FileTreeEntry,
        SkeletonArtifact,
    )

    return SkeletonArtifact(
        project_name="reg_test",
        description="regression fixture — no registry NameError",
        file_tree=[
            FileTreeEntry(
                path="mod.py",
                role=FileRole.IMPLEMENTATION,
                description="module",
                skeleton=True,
            ),
        ],
        external_dependencies=[],
        framework="python3",
        language="python",
    )


def _run_node(state):
    import asyncio
    from belief.agents.skeleton_pass1 import skeleton_pass1_node

    return asyncio.run(skeleton_pass1_node(state))


def test_skeleton_pass1_does_not_raise_registry_nameerror_on_miss(isolated_cache):
    """Cache miss: _generate() runs, writes to disk, returns payload.

    If any code path in the node tries to read a bare `registry`
    variable, the except handler will append a warning containing the
    word 'registry'.  Surface that failure directly.
    """
    state = {
        "skeleton_artifact": _minimal_skeleton().model_dump(),
        "code_files": {},
    }
    result = _run_node(state)
    warnings = result.get("warnings", [])
    offenders = [w for w in warnings if "registry" in str(w).lower()]
    assert not offenders, (
        f"skeleton_pass1 is leaking the registry NameError via warnings: {offenders}"
    )


def test_skeleton_pass1_does_not_raise_on_cache_hit(isolated_cache):
    """Cache hit: the second call reads from disk instead of generating.

    Both paths must be clean of the registry NameError.
    """
    state = {
        "skeleton_artifact": _minimal_skeleton().model_dump(),
        "code_files": {},
    }
    _run_node(state)  # miss (writes to cache)
    result = _run_node(state)  # hit (reads from cache)
    warnings = result.get("warnings", [])
    offenders = [w for w in warnings if "registry" in str(w).lower()]
    assert not offenders, f"skeleton_pass1 cache-hit path is leaking the NameError: {offenders}"


def test_skeleton_pass1_populates_registry_context(isolated_cache):
    """Positive assertion: after the node runs, downstream state must
    have the registry context string set.  If the except handler
    swallowed the generator output, this would be missing."""
    state = {
        "skeleton_artifact": _minimal_skeleton().model_dump(),
        "code_files": {},
    }
    result = _run_node(state)
    assert "skeleton_registry_context" in result
    # The exact contents vary, but the field must exist and be a string.
    assert isinstance(result["skeleton_registry_context"], str)


def test_skeleton_pass1_second_call_is_cache_hit(isolated_cache, caplog):
    """Second identical call logs 'cache HIT' — confirms the lru memo
    + on-disk cache are both wired up correctly."""
    import logging

    caplog.set_level(logging.INFO, logger="belief.agents.skeleton_pass1")

    state = {
        "skeleton_artifact": _minimal_skeleton().model_dump(),
        "code_files": {},
    }
    _run_node(state)
    caplog.clear()
    _run_node(state)

    log_lines = [rec.getMessage() for rec in caplog.records]
    hit_lines = [ln for ln in log_lines if "cache HIT" in ln]
    assert hit_lines, f"second call did not report a cache hit. logs: {log_lines}"
