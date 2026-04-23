"""Small, targeted cheatsheets injected into agent system prompts.

Session 2 introduces a Pydantic v2 cheatsheet.  Future sessions may
add others (e.g., SQLAlchemy 2.0, FastAPI dependency injection, etc.)
so the helpers are factored to handle more than one cheatsheet
without growing :class:`belief.agents.builder.BuilderAgent`.

Design constraint — append-only:
    Cheatsheets are always appended to a system prompt, never
    prepended.  This is what keeps the Session 1 ``num_keep=512``
    prefix-cache behaviour intact: the leading bytes of the builder's
    BUILDER_SYSTEM constant stay byte-stable, and Ollama's KV cache
    only loses the cached suffix (where the cheatsheet sits).  If you
    add a new cheatsheet, keep to this convention.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("belief.prompts.cheatsheets")


_PROMPTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Pydantic v2 cheatsheet
# ---------------------------------------------------------------------------

# The builder triggers cheatsheet injection when the file under generation
# plausibly touches pydantic's surface area.  These strings are scanned
# against (a) the file's planned imports if present, (b) the filename
# itself (fallback), and (c) the build's user goal (last-resort signal).
_PYDANTIC_TRIGGER_IMPORTS: tuple[str, ...] = (
    "pydantic",
    "pydantic_settings",
    "langchain",  # matches langchain, langchain_core, langchain_community, …
    "fastapi",  # pydantic is a transitive dep; LLMs routinely emit v1 code here
)

# Filename hints that strongly suggest pydantic usage even if we don't
# know the planned imports yet.  The builder gets these "for free"
# from the SkeletonArtifact's declared files.
_PYDANTIC_TRIGGER_FILENAMES: tuple[str, ...] = (
    "models.py",
    "schemas.py",
    "settings.py",
    "config.py",
)


@lru_cache(maxsize=1)
def load_pydantic_v2_cheatsheet() -> str:
    """Read pydantic_v2_cheatsheet.md once and cache it.

    Returns an empty string if the cheatsheet file is missing (which
    shouldn't happen at runtime — the file ships with the package —
    but we tolerate it so a partial checkout doesn't crash the
    builder).
    """
    cheatsheet = _PROMPTS_DIR / "pydantic_v2_cheatsheet.md"
    try:
        return cheatsheet.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("pydantic_v2_cheatsheet.md not found at %s", cheatsheet)
        return ""


def should_inject_pydantic_v2_cheatsheet(
    *,
    filename: str | None = None,
    planned_imports: Iterable[str] | None = None,
    user_goal: str | None = None,
) -> bool:
    """Return True iff the builder should append the Pydantic v2 cheatsheet.

    Any single positive signal is enough — we err toward inclusion
    because the cheatsheet is ~1 KB and the cost of a false-positive
    injection is trivial compared to the cost of the v1↔v2 debug
    thrash it prevents.

    Signals checked (in priority order):

    1. ``planned_imports`` explicitly contains any pydantic-adjacent
       module name.  This is the strongest signal — the skeleton
       already committed to importing it.
    2. ``filename`` matches one of the common pydantic-code filenames
       (``models.py``, ``schemas.py``, ``settings.py``, ``config.py``).
    3. ``user_goal`` contains the substring "pydantic", "fastapi",
       "langchain", or "basesettings" (case-insensitive).
    """
    if planned_imports:
        for imp in planned_imports:
            low = str(imp).lower()
            if any(trigger in low for trigger in _PYDANTIC_TRIGGER_IMPORTS):
                return True

    if filename:
        base = filename.rsplit("/", 1)[-1].lower()
        if base in _PYDANTIC_TRIGGER_FILENAMES:
            return True

    if user_goal:
        goal_low = user_goal.lower()
        # Fast substring checks — broader than _PYDANTIC_TRIGGER_IMPORTS
        # because we want to catch "Build a FastAPI service" even
        # though the goal never says "pydantic" explicitly.
        for kw in ("pydantic", "fastapi", "langchain", "basesettings"):
            if kw in goal_low:
                return True

    return False


def pydantic_v2_cheatsheet_if_needed(
    *,
    filename: str | None = None,
    planned_imports: Iterable[str] | None = None,
    user_goal: str | None = None,
) -> str:
    """Return the cheatsheet text if the trigger fires, else empty string.

    Convenience wrapper — callers can just ``system_prompt += text``
    without a conditional.
    """
    if not should_inject_pydantic_v2_cheatsheet(
        filename=filename,
        planned_imports=planned_imports,
        user_goal=user_goal,
    ):
        return ""
    text = load_pydantic_v2_cheatsheet()
    if not text:
        return ""
    # Wrap with a stable header so the builder's prompt stays legible.
    return (
        "\n\n---\n"
        "## PYDANTIC V2 CHEATSHEET (injected by covenant enforcer)\n\n"
        "The covenant enforcer WILL rewrite v1 patterns you emit; keep "
        "your output in v2 shape to avoid wasted debug rounds.\n\n"
        f"{text.strip()}\n"
    )


def _planned_imports_from_file_spec(file_spec: Any) -> list[str]:
    """Extract a list of planned imports from a :class:`~belief.models.skeleton.FileSpec`-shaped object.

    The skeleton models vary across sessions; we duck-type to survive
    minor renames.  An empty list is returned if nothing useful is
    present — the filename/goal fallbacks still run.
    """
    for attr in ("planned_imports", "imports", "dependencies", "deps"):
        val = getattr(file_spec, attr, None)
        if val is None:
            continue
        try:
            return [str(x) for x in val]
        except TypeError:
            continue
    return []


def pydantic_v2_cheatsheet_for_file_spec(
    file_spec: Any,
    user_goal: str | None = None,
) -> str:
    """High-level helper for :class:`belief.agents.builder.BuilderAgent`.

    Given a FileSpec and an optional user_goal, returns the cheatsheet
    string if the trigger fires, else empty string.  The builder's
    call site is literally::

        system = f"{system}{pydantic_v2_cheatsheet_for_file_spec(file_spec, state.user_goal)}"
    """
    filename = getattr(file_spec, "filename", None) or getattr(file_spec, "path", None)
    planned_imports = _planned_imports_from_file_spec(file_spec)
    return pydantic_v2_cheatsheet_if_needed(
        filename=filename,
        planned_imports=planned_imports,
        user_goal=user_goal,
    )


__all__ = [
    "load_pydantic_v2_cheatsheet",
    "pydantic_v2_cheatsheet_for_file_spec",
    "pydantic_v2_cheatsheet_if_needed",
    "should_inject_pydantic_v2_cheatsheet",
]
