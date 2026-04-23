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
