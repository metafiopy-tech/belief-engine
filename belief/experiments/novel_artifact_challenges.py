"""Novel-artifact challenge specs + runner integration for the substrate-transfer experiment.

The 5 novel-artifact challenges (see docs/experiments/substrate_transfer_challenges.md)
test whether the engine's engineering loop transfers to artifacts the soil never saw
during the in-distribution buckets. They produce non-Python files (Sokoban grids,
SMT-LIB encodings, crosswords, TLA+ specs, regexes) that pytest cannot validate.

This module bridges the gap between:
  - belief.experiments.novel_artifact_validators (the pure validator functions)
  - belief.experiments.ab_runner (which invokes the engine pipeline)

For challenge ids matching the ``novel-*`` pattern, the runner calls
``apply_novel_artifact_validation`` after the build completes. This:
  1. Locates the produced file(s) (from raw_runner's in-memory dict, or
     from the engine's output directory by mtime)
  2. Reads the expected artifact files
  3. Runs the matching validator
  4. Returns ``(passed, message, weighted_score)`` — the runner uses this
     to override the engine's pytest-derived score, which is meaningless
     for non-Python artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from belief.experiments.novel_artifact_validators import (
    validate_crossword,
    validate_regex,
    validate_smtlib,
    validate_sokoban,
    validate_tlaplus,
)

# ---------------------------------------------------------------------------
# Challenge specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NovelArtifactSpec:
    """Specification for one novel-artifact challenge.

    Attributes:
        challenge_id: Stable identifier (must start with ``novel-``).
        goal: Natural-language goal text sent to the engine. Pulled verbatim
            from docs/experiments/substrate_transfer_challenges.md.
        expected_files: Filenames the engine must produce. Tuple to support
            multi-file challenges (TLA+ needs both .tla and .cfg).
        validator: Function called with the file contents.
        needs_wordlist: True for the crossword validator which requires the
            5000-word fixture.
    """

    challenge_id: str
    goal: str
    expected_files: tuple[str, ...]
    validator: Callable
    needs_wordlist: bool = False


_SOKOBAN_GOAL = (
    "Design a Sokoban puzzle level on a 6x6 grid. The level must contain "
    "exactly one player, one box, and one target square, surrounded by walls. "
    "The shortest solution must require exactly 14 player moves (counting both "
    "pushes and non-push moves).\n\n"
    "Output a file `level.txt` containing 6 lines of 6 characters each, using:\n"
    "- `#` for wall\n"
    "- ` ` (space) for empty floor\n"
    "- `@` for player starting position\n"
    "- `$` for box starting position\n"
    "- `.` for target square\n\n"
    "The outer border of the grid must be all walls. The level must be "
    "solvable. The shortest solution (in player-moves) must be exactly 14."
)

_SMTLIB_GOAL = (
    "Encode the following logic puzzle in SMT-LIB v2 such that Z3 returns "
    "`sat` and the unique solution can be extracted from the model.\n\n"
    "Puzzle: Three houses sit in a row at positions 1, 2, and 3 (left to "
    "right). Each house has exactly one color (red, green, or blue), exactly "
    "one owner (Alice, Bob, or Carol), and exactly one pet (cat, dog, or "
    "fish). Every color, owner, and pet appears exactly once across the three "
    "houses. The following constraints hold:\n\n"
    "1. The green house is immediately to the right of the red house.\n"
    "2. The blue house is at position 1.\n"
    "3. Alice owns the dog.\n"
    "4. The cat lives in the red house.\n"
    "5. Bob lives at position 3.\n\n"
    "Question: Who owns the fish, and at which position?\n"
    "Expected answer: Bob, at position 3.\n\n"
    "Output a file `puzzle.smt2` containing an SMT-LIB v2 encoding. The file "
    "must: declare appropriate variables/functions for color, owner, pet at "
    "each position; assert all five constraints AND the implicit uniqueness "
    "constraints; end with `(check-sat)` and `(get-model)`."
)

_CROSSWORD_GOAL = (
    "Construct a 7x7 American-style crossword puzzle.\n\n"
    "Output two files:\n"
    "- `grid.txt` -- exactly 7 lines, each line exactly 7 characters. Use "
    "uppercase A-Z for letter cells and `#` for black squares.\n"
    "- `clues.txt` -- one clue per line, format `[A|D][number]: [clue text]`.\n\n"
    "Constraints the grid must satisfy:\n"
    "1. 180-degree rotational symmetry of black squares.\n"
    "2. Every horizontal run of letter cells of length >= 3 is an Across "
    "entry; every vertical run of letter cells of length >= 3 is a Down entry.\n"
    "3. No entries shorter than 3 letters.\n"
    "4. All Across and Down entries must be valid English words from the "
    "provided wordlist at `wordlist.txt`.\n"
    "5. At least 15 entries total."
)

_TLAPLUS_GOAL = (
    "Write a TLA+ specification for a two-process mutual exclusion problem "
    "implementing Peterson's algorithm.\n\n"
    "Output two files:\n"
    "- `Mutex.tla` -- the TLA+ module\n"
    "- `Mutex.cfg` -- the TLC configuration\n\n"
    "The module must contain: module header `---- MODULE Mutex ----` and "
    "footer `====`; `EXTENDS Naturals`; `VARIABLES pc1, pc2, turn, flag1, "
    "flag2`; `Init`, `Next`, `Spec`, and `MutualExclusion` definitions. "
    "Primed-variable syntax must be e.g. `pc1'` not `pc'1`. Every action "
    "must include `UNCHANGED` clauses for variables it doesn't modify.\n\n"
    "The Mutex.cfg file must contain `SPECIFICATION Spec` and "
    "`INVARIANT MutualExclusion`.\n\n"
    "When run with TLC, the spec must complete model-checking without "
    "reporting a violation of MutualExclusion."
)

_REGEX_GOAL = (
    "Produce a single Python-compatible regular expression that matches "
    "valid IPv4 addresses in dotted-quad notation and rejects all invalid "
    "strings.\n\n"
    "Definition of valid: four decimal octets separated by dots. Each octet "
    "is an integer from 0 to 255 inclusive, written without leading zeros "
    "(so `0` is valid, `01` is not, `255` is valid, `256` is not).\n\n"
    "Examples that must match: 0.0.0.0, 255.255.255.255, 192.168.1.1, "
    "8.8.8.8, 127.0.0.1.\n\n"
    "Examples that must NOT match: 256.1.1.1, 01.1.1.1, 1.1.1, 1.1.1.1.1, "
    "1.1.1.a, leading/trailing whitespace, 1..1.1.\n\n"
    "Output a file `solution.regex` containing only the regex pattern as a "
    'single line -- no quotes, no leading `r"..."`, no flags, no surrounding '
    "`re.compile(...)`. The pattern will be loaded as "
    "`re.compile(open('solution.regex').read().strip())`.\n\n"
    "Your regex must achieve F1 >= 0.95 on a held-out test set drawn from "
    "the same distribution as the examples above."
)


NOVEL_ARTIFACT_SPECS: dict[str, NovelArtifactSpec] = {
    "novel-sokoban": NovelArtifactSpec(
        challenge_id="novel-sokoban",
        goal=_SOKOBAN_GOAL,
        expected_files=("level.txt",),
        validator=validate_sokoban,
    ),
    "novel-smtlib": NovelArtifactSpec(
        challenge_id="novel-smtlib",
        goal=_SMTLIB_GOAL,
        expected_files=("puzzle.smt2",),
        validator=validate_smtlib,
    ),
    "novel-crossword": NovelArtifactSpec(
        challenge_id="novel-crossword",
        goal=_CROSSWORD_GOAL,
        expected_files=("grid.txt", "clues.txt"),
        validator=validate_crossword,
        needs_wordlist=True,
    ),
    "novel-tlaplus": NovelArtifactSpec(
        challenge_id="novel-tlaplus",
        goal=_TLAPLUS_GOAL,
        expected_files=("Mutex.tla", "Mutex.cfg"),
        validator=validate_tlaplus,
    ),
    "novel-regex": NovelArtifactSpec(
        challenge_id="novel-regex",
        goal=_REGEX_GOAL,
        expected_files=("solution.regex",),
        validator=validate_regex,
    ),
}


def is_novel_artifact_id(challenge_id: str) -> bool:
    """Return True if ``challenge_id`` is one of the novel-artifact challenges."""
    return challenge_id in NOVEL_ARTIFACT_SPECS


# ---------------------------------------------------------------------------
# File-discovery helpers
# ---------------------------------------------------------------------------


def find_build_output_dir(
    start_time: float,
    output_root: Path,
) -> Optional[Path]:
    """Find the engine output directory created after ``start_time``.

    The engine writes builds to ``output/belief-{hash}/``. We pick the
    most-recently-modified directory matching that pattern whose mtime
    is >= ``start_time``. For sequential builds (as the runner does)
    this is unambiguous.

    Returns None if no matching directory exists (build failed before
    writing anything, or output_root doesn't exist).
    """
    if not output_root.exists() or not output_root.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for entry in output_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("belief-"):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime >= start_time:
            candidates.append((mtime, entry))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def read_files_from_dir(
    build_dir: Path,
    expected_files: tuple[str, ...],
) -> tuple[Optional[list[str]], Optional[str]]:
    """Read the expected files from a build directory.

    Tries the build dir root first, then a few common subdirs the engine
    might write into (``src/``, ``app/``, ``main/``).

    Returns ``(contents, error)``. On success, contents is a list of file
    contents in the same order as expected_files. On failure, error is a
    descriptive string and contents is None.
    """
    contents: list[str] = []
    search_dirs = [build_dir, build_dir / "src", build_dir / "app", build_dir / "main"]
    for fname in expected_files:
        found: Optional[Path] = None
        for sd in search_dirs:
            candidate = sd / fname
            if candidate.exists() and candidate.is_file():
                found = candidate
                break
        if found is None:
            return None, f"expected file '{fname}' not found under {build_dir}"
        try:
            contents.append(found.read_text(encoding="utf-8"))
        except OSError as exc:
            return None, f"failed to read {found}: {exc}"
    return contents, None


def read_files_from_dict(
    code_files: dict[str, str],
    expected_files: tuple[str, ...],
) -> tuple[Optional[list[str]], Optional[str]]:
    """Read expected files from an in-memory ``code_files`` dict.

    Used for raw_local builds where the model output was parsed into a
    dict in memory and never written to a stable directory.
    """
    contents: list[str] = []
    for fname in expected_files:
        # Allow either exact-match or basename-match (model might prefix paths)
        match: Optional[str] = None
        for k in code_files.keys():
            if k == fname or k.endswith("/" + fname):
                match = k
                break
        if match is None:
            return None, f"expected file '{fname}' not found in code_files"
        contents.append(code_files[match])
    return contents, None


# ---------------------------------------------------------------------------
# Wordlist loader (crossword only)
# ---------------------------------------------------------------------------


_WORDLIST_CACHE: Optional[set[str]] = None


def _load_default_wordlist() -> set[str]:
    """Load the crossword wordlist from docs/experiments/fixtures/wordlist.txt."""
    global _WORDLIST_CACHE
    if _WORDLIST_CACHE is not None:
        return _WORDLIST_CACHE
    fixture_path = (
        Path(__file__).resolve().parent.parent.parent
        / "docs"
        / "experiments"
        / "fixtures"
        / "wordlist.txt"
    )
    if not fixture_path.exists():
        _WORDLIST_CACHE = set()
        return _WORDLIST_CACHE
    _WORDLIST_CACHE = {
        line.strip().lower()
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return _WORDLIST_CACHE


# ---------------------------------------------------------------------------
# Top-level validation entrypoint
# ---------------------------------------------------------------------------


def apply_novel_artifact_validation(
    challenge_id: str,
    build_start_time: Optional[float] = None,
    output_root: Optional[Path] = None,
    code_files: Optional[dict[str, str]] = None,
    wordlist: Optional[set[str]] = None,
) -> tuple[bool, str, float]:
    """Run the novel-artifact validator for a completed build.

    Exactly one of ``code_files`` (for raw_local builds) or
    ``build_start_time`` (for engine builds) must be supplied:

    - If ``code_files`` is given, validate from the in-memory dict.
    - If ``build_start_time`` is given, locate the engine output dir
      created after that time and validate from disk.

    Returns ``(passed, message, weighted_score)``. ``weighted_score`` is
    binary: 1.0 if passed else 0.0 (novel-artifact challenges are
    pass/fail; partial credit doesn't apply to "did Z3 say sat?" or
    "does the level have exactly 14-move solution?").
    """
    spec = NOVEL_ARTIFACT_SPECS.get(challenge_id)
    if spec is None:
        return False, f"unknown novel-artifact challenge: {challenge_id}", 0.0

    # Acquire file contents
    if code_files is not None:
        contents, err = read_files_from_dict(code_files, spec.expected_files)
    else:
        if build_start_time is None:
            return (
                False,
                "either code_files or build_start_time must be provided",
                0.0,
            )
        if output_root is None:
            output_root = Path.home() / "Desktop" / "belief-engine" / "output"
        build_dir = find_build_output_dir(build_start_time, output_root)
        if build_dir is None:
            return (
                False,
                f"no engine output directory found after build_start_time={build_start_time}",
                0.0,
            )
        contents, err = read_files_from_dir(build_dir, spec.expected_files)

    if contents is None:
        return False, err or "file read failed", 0.0

    # Dispatch validator with correct signature per challenge
    if spec.needs_wordlist:
        wl = wordlist if wordlist is not None else _load_default_wordlist()
        passed, msg = spec.validator(contents[0], contents[1], wl)
    elif len(spec.expected_files) == 2:
        # Two-file challenges other than crossword: TLA+
        passed, msg = spec.validator(contents[0], contents[1])
    else:
        passed, msg = spec.validator(contents[0])

    return passed, msg, (1.0 if passed else 0.0)
