"""Repo-root pytest configuration — Session 2 (v3.2).

This file exists for ONE reason: silence the langchain_core internal
``Core Pydantic V1 functionality isn't compatible with Python 3.14``
warning that langchain emits from its own v1 compatibility shim.  The
warning is harmless at the langchain_core level (that's just langchain
probing) but it pollutes test output with one noise line per import
of langchain-adjacent modules.

We scope the filter strictly to the ``langchain_core`` module to avoid
papering over the SAME warning from our own code — if the Belief
Engine ever emits ``from pydantic.v1 import …``, the warning must
still fire so the Session 2 covenant enforcer catches it (via the
existing ``filterwarnings = ["error"]`` set by pyproject.toml, which
this conftest does NOT weaken).

Why at the repo root and not tests/?
    pytest's conftest.py is discovered by walking UP from the test
    file toward the root, stopping at the first directory without a
    conftest.py.  A repo-root conftest is picked up by every test
    regardless of subdir, which is what we want — tests/photosynthesis,
    tests/grinder, and tests/milestone* all hit the same langchain_core
    probe.
"""

from __future__ import annotations

import warnings

# Scope the ignore-filter to langchain_core only.  The `module` regex
# matches ``langchain_core`` and its submodules (``langchain_core.schemas``,
# etc.).  Our own v1 imports would fail the ``module=`` predicate and
# therefore still error out under the ``filterwarnings = ["error"]``
# elsewhere (pyproject.toml's [tool.pytest.ini_options]).
warnings.filterwarnings(
    action="ignore",
    message=r"^Core Pydantic V1 functionality isn't compatible with Python 3\.14",
    category=UserWarning,
    module=r"langchain_core(\..*)?",
)

# If langsmith ever emits the same warning (it imports langchain_core.v1
# internally), the filter above catches it too because the message regex
# is what's checked first.  No separate rule needed.


# ---------------------------------------------------------------------------
# Session 8.5a (2026-04-23): skip Finder / iCloud duplicate files
# ---------------------------------------------------------------------------
#
# macOS Finder and iCloud Drive will occasionally fork a file into
# ``foo 2.py`` / ``foo 3.py`` when a process rewrites it while sync is
# active.  ``.gitignore`` already matches the pattern (``*\ [0-9]*``) so
# they stay out of git, but *pytest does not read ``.gitignore``* — it
# will collect those duplicates as real test modules if we don't tell it
# otherwise.  In session 8.5a a pre-commit rewrite triggered the fork on
# 243 files, blowing the boundary test and doubling the pytest runtime.
#
# ``collect_ignore_glob`` is a repo-root conftest hook that pytest
# consults during collection.  Globs are relative to this conftest's
# directory.
collect_ignore_glob = [
    "**/* [0-9].py",  # `foo 2.py`, `bar 10.py`
    "**/* [0-9]*.py",  # covers multi-digit + trailing-descriptor forms
]
