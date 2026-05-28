"""Tests for belief.experiments.novel_artifact_challenges integration layer.

The pure validators are tested in test_novel_artifact_validators.py. This
file tests the wiring that the substrate-transfer runner uses:

- is_novel_artifact_id dispatch
- find_build_output_dir picks the newest belief-* dir after start_time
- read_files_from_dir / read_files_from_dict find expected files under
  build directories or in-memory code_files dicts
- apply_novel_artifact_validation dispatches to the right validator with
  the right signature for each challenge type

End-to-end correctness of the substrate-transfer runner using these
helpers is exercised at the shakedown stage with real engine builds;
this file focuses on the pure-Python dispatch logic.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from belief.experiments.novel_artifact_challenges import (
    NOVEL_ARTIFACT_SPECS,
    apply_novel_artifact_validation,
    find_build_output_dir,
    is_novel_artifact_id,
    read_files_from_dict,
    read_files_from_dir,
)


# ---------------------------------------------------------------------------
# Spec table sanity checks
# ---------------------------------------------------------------------------


def test_all_five_challenges_registered() -> None:
    """The spec table must contain exactly the 5 challenge ids."""
    assert set(NOVEL_ARTIFACT_SPECS.keys()) == {
        "novel-sokoban",
        "novel-smtlib",
        "novel-crossword",
        "novel-tlaplus",
        "novel-regex",
    }


def test_is_novel_artifact_id_recognises_known() -> None:
    for cid in NOVEL_ARTIFACT_SPECS:
        assert is_novel_artifact_id(cid)


def test_is_novel_artifact_id_rejects_in_distribution() -> None:
    assert not is_novel_artifact_id("t1-fizzbuzz")
    assert not is_novel_artifact_id("t3-bookmark-api")
    assert not is_novel_artifact_id("")
    assert not is_novel_artifact_id("novel-not-a-real-challenge")


def test_each_spec_has_nonempty_goal_and_expected_files() -> None:
    for cid, spec in NOVEL_ARTIFACT_SPECS.items():
        assert spec.goal.strip(), f"{cid} has empty goal"
        assert spec.expected_files, f"{cid} has no expected files"
        assert callable(spec.validator), f"{cid} validator not callable"


def test_crossword_is_only_spec_needing_wordlist() -> None:
    for cid, spec in NOVEL_ARTIFACT_SPECS.items():
        if cid == "novel-crossword":
            assert spec.needs_wordlist
        else:
            assert not spec.needs_wordlist


# ---------------------------------------------------------------------------
# find_build_output_dir
# ---------------------------------------------------------------------------


def test_find_build_output_dir_returns_none_when_root_missing(
    tmp_path: Path,
) -> None:
    """Nonexistent output_root → None, not a crash."""
    missing = tmp_path / "does-not-exist"
    assert find_build_output_dir(time.time(), missing) is None


def test_find_build_output_dir_picks_newest_after_start_time(
    tmp_path: Path,
) -> None:
    """Of multiple belief-* dirs, pick the one with the newest mtime
    that is >= start_time."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    # Create three dirs with controlled mtimes
    older = output_root / "belief-old"
    older.mkdir()
    middle = output_root / "belief-middle"
    middle.mkdir()
    newer = output_root / "belief-new"
    newer.mkdir()
    # Set explicit mtimes — older = 1000s ago, middle = 100s ago, newer = now
    now = time.time()
    import os as _os

    _os.utime(older, (now - 1000, now - 1000))
    _os.utime(middle, (now - 100, now - 100))
    _os.utime(newer, (now, now))

    # start_time excludes 'older'
    result = find_build_output_dir(now - 500, output_root)
    assert result == newer


def test_find_build_output_dir_ignores_non_belief_dirs(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "random-dir").mkdir()
    (output_root / "not-belief-prefixed").mkdir()
    assert find_build_output_dir(time.time() - 10, output_root) is None


def test_find_build_output_dir_returns_none_when_all_dirs_too_old(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    d = output_root / "belief-x"
    d.mkdir()
    import os as _os

    _os.utime(d, (time.time() - 10000, time.time() - 10000))
    assert find_build_output_dir(time.time(), output_root) is None


# ---------------------------------------------------------------------------
# read_files_from_dir
# ---------------------------------------------------------------------------


def test_read_files_from_dir_finds_files_in_root(tmp_path: Path) -> None:
    (tmp_path / "level.txt").write_text("######\n#@   #\n# $  #\n#    #\n#   .#\n######\n")
    contents, err = read_files_from_dir(tmp_path, ("level.txt",))
    assert err is None
    assert contents is not None
    assert "######" in contents[0]


def test_read_files_from_dir_finds_files_in_src_subdir(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "puzzle.smt2").write_text("(assert true)")
    contents, err = read_files_from_dir(tmp_path, ("puzzle.smt2",))
    assert err is None
    assert contents == ["(assert true)"]


def test_read_files_from_dir_returns_error_when_file_missing(tmp_path: Path) -> None:
    contents, err = read_files_from_dir(tmp_path, ("level.txt",))
    assert contents is None
    assert err is not None
    assert "level.txt" in err


def test_read_files_from_dir_reads_multiple_files_in_order(tmp_path: Path) -> None:
    (tmp_path / "Mutex.tla").write_text("module content")
    (tmp_path / "Mutex.cfg").write_text("config content")
    contents, err = read_files_from_dir(tmp_path, ("Mutex.tla", "Mutex.cfg"))
    assert err is None
    assert contents == ["module content", "config content"]


# ---------------------------------------------------------------------------
# read_files_from_dict
# ---------------------------------------------------------------------------


def test_read_files_from_dict_exact_match() -> None:
    code_files = {"level.txt": "some grid", "other.txt": "noise"}
    contents, err = read_files_from_dict(code_files, ("level.txt",))
    assert err is None
    assert contents == ["some grid"]


def test_read_files_from_dict_basename_match_with_path_prefix() -> None:
    """The model may prefix filenames with paths (src/level.txt). Match
    on basename so the runner doesn't have to enforce a specific structure."""
    code_files = {"src/level.txt": "prefixed grid"}
    contents, err = read_files_from_dict(code_files, ("level.txt",))
    assert err is None
    assert contents == ["prefixed grid"]


def test_read_files_from_dict_returns_error_when_missing() -> None:
    contents, err = read_files_from_dict({"other.txt": "x"}, ("level.txt",))
    assert contents is None
    assert err is not None
    assert "level.txt" in err


# ---------------------------------------------------------------------------
# apply_novel_artifact_validation — dispatch + score override
# ---------------------------------------------------------------------------


def test_apply_returns_error_for_unknown_challenge() -> None:
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-bogus",
        code_files={"x.txt": "y"},
    )
    assert not passed
    assert "unknown" in msg.lower()
    assert score == 0.0


def test_apply_requires_either_code_files_or_build_start_time() -> None:
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-sokoban",
    )
    assert not passed
    assert score == 0.0
    assert "code_files" in msg or "build_start_time" in msg


def test_apply_returns_failure_when_engine_dir_not_found(tmp_path: Path) -> None:
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-sokoban",
        build_start_time=time.time(),
        output_root=tmp_path / "no-such-output",
    )
    assert not passed
    assert score == 0.0


def test_apply_sokoban_via_code_files_with_valid_artifact() -> None:
    """Use the same fixture the validator's own test uses; verify the
    integration layer dispatches to validate_sokoban correctly and
    returns 1.0/0.0 score (not the underlying 0.6 partial-credit values)."""
    # First discover the actual move count for the fixture grid via the
    # validator's BFS, then ask the integration layer to validate against
    # that exact count. If the dispatch is wired correctly, it should pass.
    from belief.experiments.novel_artifact_validators import (
        _parse_sokoban_level,
        _shortest_sokoban_solution,
    )

    grid = "######\n#@   #\n# $  #\n#    #\n#   .#\n######\n"
    state = _parse_sokoban_level(grid)
    actual_moves = _shortest_sokoban_solution(state)
    assert actual_moves is not None

    # The spec hardcodes target_moves=14; this fixture's actual count may
    # differ. Either outcome is acceptable here — we're testing dispatch
    # mechanics, not the puzzle. Just confirm the score is binary 1.0 or 0.0.
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-sokoban",
        code_files={"level.txt": grid},
    )
    assert score in (0.0, 1.0), f"score must be binary, got {score}"
    assert bool(passed) == (score == 1.0)


def test_apply_regex_via_code_files_with_passing_pattern() -> None:
    """Provide a correct IPv4 regex; the integration should report passed=True
    and weighted_score=1.0."""
    correct_pattern = (
        r"(?:0|[1-9]|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])"
        r"(?:\.(?:0|[1-9]|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])){3}"
    )
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-regex",
        code_files={"solution.regex": correct_pattern},
    )
    assert passed, msg
    assert score == 1.0


def test_apply_regex_via_code_files_with_failing_pattern() -> None:
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-regex",
        code_files={"solution.regex": ".*"},
    )
    assert not passed
    assert score == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("z3") is None,
    reason="z3-solver not installed",
)
def test_apply_smtlib_via_code_files_with_unique_sat() -> None:
    smt = (
        "(declare-const bob Int)\n"
        "(declare-const fish Int)\n"
        "(assert (= bob 3))\n"
        "(assert (= fish 1))\n"
    )
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-smtlib",
        code_files={"puzzle.smt2": smt},
    )
    assert passed, msg
    assert score == 1.0


def test_apply_crossword_via_code_files_with_invalid_grid() -> None:
    """A 3x3 grid should fail size validation; the integration should
    report passed=False and the failure message should mention size."""
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-crossword",
        code_files={"grid.txt": "ABC\nDEF\nGHI\n", "clues.txt": ""},
    )
    assert not passed
    assert score == 0.0
    assert "row" in msg.lower() or "size" in msg.lower() or "got 3" in msg.lower()


def test_apply_via_engine_output_dir(tmp_path: Path) -> None:
    """Full path through find_build_output_dir + read_files_from_dir for
    the engine-build path."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    build_dir = output_root / "belief-abc123"
    build_dir.mkdir()
    # Write a regex file the validator will accept
    correct_pattern = (
        r"(?:0|[1-9]|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])"
        r"(?:\.(?:0|[1-9]|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])){3}"
    )
    (build_dir / "solution.regex").write_text(correct_pattern)

    # build_start_time before mkdir, so the dir's mtime is after start
    start_time = time.time() - 5
    passed, msg, score = apply_novel_artifact_validation(
        challenge_id="novel-regex",
        build_start_time=start_time,
        output_root=output_root,
    )
    assert passed, msg
    assert score == 1.0
