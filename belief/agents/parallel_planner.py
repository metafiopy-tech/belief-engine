"""
Parallel Build Planner — Milestone 1 of Tier 4-5 Scaling

Organizes files from the architect's manifest into dependency levels:
  Level 0: models, config, exceptions, __init__.py (no internal deps)
  Level 1: base classes, utilities (depend only on level 0)
  Level N: files that depend on levels 0..N-1
  Last level: test files (depend on everything)

Within each level, files are independent and can be generated in parallel.
The builder generates level by level, so each file's dependencies are
already generated when it's being built.

Source: TIER_4_5_SCALING_PLAN.md Milestone 1
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger("belief.agents.parallel_planner")


class FileLevel(BaseModel):
    """A group of files at the same dependency depth — parallelizable."""
    level: int
    files: list[str]  # Filenames in this level


class ParallelBuildPlan(BaseModel):
    """Files organized into dependency levels for ordered generation."""
    levels: list[FileLevel] = Field(default_factory=list)
    total_files: int = 0

    def file_order(self) -> list[str]:
        """Flat list of files in correct build order (level 0 first)."""
        return [f for level in self.levels for f in level.files]

    def files_at_level(self, level: int) -> list[str]:
        """Files that can be built in parallel at this level."""
        for fl in self.levels:
            if fl.level == level:
                return fl.files
        return []

    def completed_files_before(self, level: int) -> list[str]:
        """All files completed before this level (for context injection)."""
        return [f for fl in self.levels if fl.level < level for f in fl.files]


# ── Heuristic classification ────────────────────────────────────────────────

# Files that should be generated first (no internal dependencies)
_LEVEL_0_PATTERNS = [
    r"__init__\.py$",
    r"models\.py$",
    r"schemas\.py$",
    r"config\.py$",
    r"settings\.py$",
    r"exceptions\.py$",
    r"errors\.py$",
    r"types\.py$",
    r"constants\.py$",
    r"enums\.py$",
]

# Files that typically depend only on level 0
_LEVEL_1_PATTERNS = [
    r"base\.py$",
    r"utils?\.py$",
    r"helpers?\.py$",
    r"interfaces?\.py$",
    r"protocols?\.py$",
    r"mixins?\.py$",
    r"decorators?\.py$",
]

# Files that are always last (entry points)
_ENTRY_PATTERNS = [
    r"^main\.py$",
    r"^app\.py$",
    r"^run\.py$",
    r"server\.py$",
    r"^cli\.py$",
]

# Test files are always the very last level
_TEST_PATTERNS = [
    r"^tests?/",
    r"^test_",
    r"/test_",
]


def _classify_file(filename: str) -> int:
    """Classify a file into a heuristic level (0-4, or 99 for tests)."""
    # Tests always last
    for p in _TEST_PATTERNS:
        if re.search(p, filename):
            return 99

    # Level 0: models, config, exceptions
    for p in _LEVEL_0_PATTERNS:
        if re.search(p, filename):
            return 0

    # Level 1: base classes, utilities
    for p in _LEVEL_1_PATTERNS:
        if re.search(p, filename):
            return 1

    # Entry points near last (before tests)
    for p in _ENTRY_PATTERNS:
        if re.search(p, filename):
            return 4

    # Everything else is middle tier
    return 2


def _resolve_level_from_deps(
    filename: str,
    depends_on: list[str],
    file_levels: dict[str, int],
) -> int:
    """Resolve a file's level from its explicit dependencies.

    A file's level = max(level of all its dependencies) + 1.
    If it has no resolved dependencies, fall back to heuristic.
    """
    if not depends_on:
        return _classify_file(filename)

    max_dep_level = -1
    for dep in depends_on:
        if dep in file_levels:
            max_dep_level = max(max_dep_level, file_levels[dep])
        else:
            # Dependency not yet resolved — use heuristic for dep
            max_dep_level = max(max_dep_level, _classify_file(dep))

    return max_dep_level + 1


# ── Plan builders ────────────────────────────────────────────────────────────

def build_plan_from_manifest(manifest) -> ParallelBuildPlan:
    """Build a ParallelBuildPlan from a FileManifestPlan.

    Uses explicit depends_on when available, falls back to heuristic
    classification by filename pattern.

    Args:
        manifest: FileManifestPlan with .files list of FileManifest objects
    """
    if not manifest or not manifest.files:
        return ParallelBuildPlan()

    # Phase 1: classify each file
    file_levels: dict[str, int] = {}
    file_deps: dict[str, list[str]] = {}

    for f in manifest.files:
        fname = f.filename
        deps = f.depends_on or []
        file_deps[fname] = deps

        if deps:
            # Has explicit dependencies — resolve from them
            file_levels[fname] = _resolve_level_from_deps(fname, deps, file_levels)
        else:
            # No explicit deps — use heuristic
            file_levels[fname] = _classify_file(fname)

    # Phase 2: iterative refinement — resolve levels that depend on other files
    # Run 3 passes to handle transitive dependencies
    for _ in range(3):
        changed = False
        for fname, deps in file_deps.items():
            if deps:
                new_level = _resolve_level_from_deps(fname, deps, file_levels)
                if new_level != file_levels.get(fname):
                    file_levels[fname] = new_level
                    changed = True
        if not changed:
            break

    # Phase 3: group into levels
    level_groups: dict[int, list[str]] = {}
    for fname, level in sorted(file_levels.items(), key=lambda x: (x[1], x[0])):
        level_groups.setdefault(level, []).append(fname)

    # Phase 4: renumber levels to be contiguous (0, 1, 2, ...)
    sorted_levels = sorted(level_groups.keys())
    levels = [
        FileLevel(level=i, files=level_groups[old_level])
        for i, old_level in enumerate(sorted_levels)
    ]

    plan = ParallelBuildPlan(
        levels=levels,
        total_files=sum(len(fl.files) for fl in levels),
    )

    logger.info(
        f"BuildPlan: {plan.total_files} files across {len(levels)} levels, "
        f"max parallelism: {max(len(fl.files) for fl in levels)}"
    )
    for fl in levels:
        logger.debug(f"  Level {fl.level}: {', '.join(fl.files)}")

    return plan


def build_plan_from_files(filenames: list[str]) -> ParallelBuildPlan:
    """Build a plan from just a list of filenames (no dependency info).

    Pure heuristic classification — used when no manifest is available.
    """
    file_levels: dict[str, int] = {}
    for fname in filenames:
        file_levels[fname] = _classify_file(fname)

    level_groups: dict[int, list[str]] = {}
    for fname, level in sorted(file_levels.items(), key=lambda x: (x[1], x[0])):
        level_groups.setdefault(level, []).append(fname)

    sorted_levels = sorted(level_groups.keys())
    levels = [
        FileLevel(level=i, files=level_groups[old_level])
        for i, old_level in enumerate(sorted_levels)
    ]

    return ParallelBuildPlan(
        levels=levels,
        total_files=len(filenames),
    )
