"""Validators for the novel-artifact challenges in the substrate-transfer experiment.

Each validator takes the engine's produced artifact (as text), runs a mechanical
check, and returns ``(passed: bool, message: str)``. No LLM-as-judge. No fuzzy
matching. All criteria are programmatically decidable.

Five validators:

- :func:`validate_sokoban` -- Sokoban level with exact-move-count solution
- :func:`validate_smtlib`  -- SMT-LIB encoding of a logic puzzle (needs ``z3-solver``)
- :func:`validate_crossword` -- 7x7 crossword with structural constraints
- :func:`validate_tlaplus` -- TLA+ specification (needs ``tla2tools.jar``)
- :func:`validate_regex`   -- IPv4 regex with F1 >= 0.95 on held-out set

External-dependency validators fail with an informative ``(False, msg)`` rather
than raising, so test runs are robust to missing tools.

See ``docs/experiments/substrate_transfer_challenges.md`` for the goal text of
each challenge.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path

ValidationResult = tuple[bool, str]


# ---------------------------------------------------------------------------
# Challenge 1: Sokoban level with exact-move solution
# ---------------------------------------------------------------------------


def _parse_sokoban_level(text: str) -> dict:
    """Parse the level text into a structured state."""
    rows = text.rstrip("\n").split("\n")
    walls: set[tuple[int, int]] = set()
    boxes: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            elif ch == "$":
                boxes.add((r, c))
            elif ch == ".":
                targets.add((r, c))
            elif ch == "@":
                player = (r, c)
            elif ch == "*":  # box on target
                boxes.add((r, c))
                targets.add((r, c))
            elif ch == "+":  # player on target
                player = (r, c)
                targets.add((r, c))
            # ' ' (space) -> empty floor; anything else ignored
    return {
        "walls": walls,
        "boxes": boxes,
        "targets": targets,
        "player": player,
        "rows": len(rows),
        "cols": max((len(r) for r in rows), default=0),
    }


def _shortest_sokoban_solution(state: dict, max_depth: int = 60) -> int | None:
    """BFS for shortest player-move solution. Returns move count, or None."""
    walls = state["walls"]
    targets = frozenset(state["targets"])
    if state["player"] is None:
        return None
    start = (state["player"], frozenset(state["boxes"]))
    if start[1] == targets:
        return 0
    visited: set[tuple[tuple[int, int], frozenset]] = {start}
    queue: deque = deque([(start, 0)])
    while queue:
        (player, boxes), depth = queue.popleft()
        if depth >= max_depth:
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_player = (player[0] + dr, player[1] + dc)
            if new_player in walls:
                continue
            new_boxes = boxes
            if new_player in boxes:
                box_dest = (new_player[0] + dr, new_player[1] + dc)
                if box_dest in walls or box_dest in boxes:
                    continue
                new_boxes = (boxes - {new_player}) | {box_dest}
            new_state = (new_player, new_boxes)
            if new_state in visited:
                continue
            if new_boxes == targets:
                return depth + 1
            visited.add(new_state)
            queue.append((new_state, depth + 1))
    return None


def validate_sokoban(
    level_text: str,
    target_moves: int = 14,
    expected_size: tuple[int, int] = (6, 6),
) -> ValidationResult:
    """Validate a Sokoban level requires exactly ``target_moves`` to solve.

    Args:
        level_text: Grid as text. ``#`` walls, ``@`` player, ``$`` box, ``.``
            target, ``*`` box-on-target, ``+`` player-on-target, space empty.
        target_moves: Expected shortest-solution player-move count.
        expected_size: Required (rows, cols).

    Returns:
        ``(passed, message)``.
    """
    state = _parse_sokoban_level(level_text)
    if state["rows"] != expected_size[0] or state["cols"] != expected_size[1]:
        return (
            False,
            f"expected {expected_size[0]}x{expected_size[1]} grid, "
            f"got {state['rows']}x{state['cols']}",
        )
    if state["player"] is None:
        return False, "no player (@) in grid"
    if len(state["boxes"]) != 1:
        return False, f"expected 1 box ($), got {len(state['boxes'])}"
    if len(state["targets"]) != 1:
        return False, f"expected 1 target (.), got {len(state['targets'])}"
    moves = _shortest_sokoban_solution(state)
    if moves is None:
        return False, "level is unsolvable"
    if moves != target_moves:
        return False, (f"shortest solution is {moves} moves, expected {target_moves}")
    return True, f"valid: shortest solution = {moves} moves"


# ---------------------------------------------------------------------------
# Challenge 2: SMT-LIB logic puzzle encoding
# ---------------------------------------------------------------------------


def validate_smtlib(
    smt_text: str,
    required_terms: tuple[str, ...] = ("bob", "fish"),
    check_uniqueness: bool = True,
) -> ValidationResult:
    """Validate an SMT-LIB encoding of the 3-houses logic puzzle.

    Checks (in order):
      1. ``z3-solver`` is importable.
      2. The text parses as SMT-LIB v2.
      3. The encoding is satisfiable.
      4. (Optional) The model is unique under block-and-resolve.
      5. ``required_terms`` (case-insensitive) appear in the stringified model.

    Args:
        smt_text: The engine's SMT-LIB v2 output.
        required_terms: Substrings that must appear in the model (lowercase).
        check_uniqueness: If True, verify only one satisfying model exists.

    Returns:
        ``(passed, message)``.
    """
    try:
        import z3  # type: ignore
    except ImportError:
        return False, "z3-solver not installed (pip install z3-solver)"

    # Strip explicit (check-sat)/(get-model) directives so we can drive solving
    cleaned_lines = []
    for line in smt_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(check-sat") or stripped.startswith("(get-model"):
            continue
        cleaned_lines.append(line)
    assertions_text = "\n".join(cleaned_lines)

    try:
        assertions = z3.parse_smt2_string(assertions_text)
    except z3.Z3Exception as exc:
        return False, f"z3 parse error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"unexpected z3 parse error: {exc}"

    solver = z3.Solver()
    for assertion in assertions:
        solver.add(assertion)

    if solver.check() != z3.sat:
        return False, "encoding is unsat (expected sat)"

    model = solver.model()

    if check_uniqueness:
        decls = [d for d in model.decls() if d.arity() == 0]
        block_terms = []
        for decl in decls:
            value = model[decl]
            if value is not None:
                block_terms.append(decl() != value)
        if block_terms:
            solver.push()
            solver.add(z3.Or(block_terms))
            second = solver.check()
            solver.pop()
            if second == z3.sat:
                return (
                    False,
                    "encoding has multiple satisfying models (puzzle should be unique)",
                )

    model_str = str(model).lower()
    missing = [term for term in required_terms if term.lower() not in model_str]
    if missing:
        return False, f"model is missing required terms: {missing}"

    return True, "z3 sat, unique model, required terms present"


# ---------------------------------------------------------------------------
# Challenge 3: 7x7 American-style crossword
# ---------------------------------------------------------------------------


def _crossword_runs(rows: list[str], horizontal: bool) -> list[tuple[int, int, int, str]]:
    """Extract maximal runs of letter cells. Returns (r, c, length, word) tuples."""
    out: list[tuple[int, int, int, str]] = []
    n = len(rows)
    if horizontal:
        for r in range(n):
            c = 0
            while c < len(rows[r]):
                if rows[r][c] != "#":
                    start = c
                    while c < len(rows[r]) and rows[r][c] != "#":
                        c += 1
                    out.append((r, start, c - start, rows[r][start:c]))
                else:
                    c += 1
    else:
        cols = max(len(r) for r in rows) if rows else 0
        for c in range(cols):
            r = 0
            while r < n:
                ch = rows[r][c] if c < len(rows[r]) else "#"
                if ch != "#":
                    start = r
                    while r < n and c < len(rows[r]) and rows[r][c] != "#":
                        r += 1
                    word = "".join(rows[i][c] for i in range(start, r))
                    out.append((start, c, r - start, word))
                else:
                    r += 1
    return out


def validate_crossword(
    grid_text: str,
    clues_text: str | None,
    wordlist: set[str],
    size: int = 7,
    min_entries: int = 15,
) -> ValidationResult:
    """Validate a crossword grid against structural constraints + a wordlist.

    Args:
        grid_text: ``size`` lines of ``size`` chars each (uppercase A-Z or ``#``).
        clues_text: Clue list. Not validated (would require LLM judgement).
        wordlist: Lowercased valid-word set.
        size: Grid side length.
        min_entries: Minimum total Across+Down entries.

    Returns:
        ``(passed, message)``.
    """
    del clues_text  # accepted for API compatibility; not validated

    rows = [r.rstrip() for r in grid_text.strip("\n").split("\n")]
    if len(rows) != size:
        return False, f"expected {size} rows, got {len(rows)}"
    widths = [len(r) for r in rows]
    if any(w != size for w in widths):
        return False, f"all rows must be {size} chars; got widths {widths}"

    # 1. 180-degree rotational symmetry of black squares
    for r in range(size):
        for c in range(size):
            if (rows[r][c] == "#") != (rows[size - 1 - r][size - 1 - c] == "#"):
                return (
                    False,
                    f"asymmetric black square at ({r},{c}) vs ({size - 1 - r},{size - 1 - c})",
                )

    horizontal_runs = _crossword_runs(rows, horizontal=True)
    vertical_runs = _crossword_runs(rows, horizontal=False)

    # 2. Every letter cell must be in a >=3 run in at least one direction
    isolated: list[tuple[int, int]] = []
    for r in range(size):
        for c in range(size):
            if rows[r][c] == "#":
                continue
            h_len = next(
                (
                    length
                    for (rr, cc, length, _) in horizontal_runs
                    if rr == r and cc <= c < cc + length
                ),
                0,
            )
            v_len = next(
                (
                    length
                    for (rr, cc, length, _) in vertical_runs
                    if cc == c and rr <= r < rr + length
                ),
                0,
            )
            if h_len < 3 and v_len < 3:
                isolated.append((r, c))
    if isolated:
        return False, f"isolated or short-run letter cells: {isolated[:5]}"

    across = [(r, c, w) for (r, c, length, w) in horizontal_runs if length >= 3]
    down = [(r, c, w) for (r, c, length, w) in vertical_runs if length >= 3]

    # 3. Entry count
    total = len(across) + len(down)
    if total < min_entries:
        return False, f"only {total} entries (Across+Down); need {min_entries}+"

    # 4. Wordlist membership
    for r, c, word in across + down:
        if word.lower() not in wordlist:
            return False, f"entry '{word}' (at {r},{c}) not in wordlist"

    return True, f"valid: {len(across)} across + {len(down)} down"


# ---------------------------------------------------------------------------
# Challenge 4: TLA+ specification + TLC model check
# ---------------------------------------------------------------------------


def _find_tla2tools_jar() -> str | None:
    """Locate ``tla2tools.jar`` from env var or common install paths."""
    env = os.environ.get("TLC_JAR")
    if env and Path(env).exists():
        return env
    candidates = [
        "/usr/local/lib/tla2tools.jar",
        "/opt/tla2tools.jar",
        Path.home() / "lib" / "tla2tools.jar",
        Path.home() / "Downloads" / "tla2tools.jar",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    which = shutil.which("tla2tools.jar")
    return which


def validate_tlaplus(
    tla_text: str,
    cfg_text: str,
    timeout_s: int = 120,
) -> ValidationResult:
    """Validate a TLA+ Mutex specification + TLC config.

    Args:
        tla_text: Contents of ``Mutex.tla``.
        cfg_text: Contents of ``Mutex.cfg``.
        timeout_s: TLC process timeout.

    Returns:
        ``(passed, message)``.
    """
    if not shutil.which("java"):
        return False, "java not on PATH; cannot run TLC"
    tlc_jar = _find_tla2tools_jar()
    if tlc_jar is None:
        return False, (
            "tla2tools.jar not found; set TLC_JAR env var or install to ~/lib/tla2tools.jar"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "Mutex.tla").write_text(tla_text)
        (tmp_path / "Mutex.cfg").write_text(cfg_text)
        try:
            result = subprocess.run(
                ["java", "-jar", tlc_jar, "-config", "Mutex.cfg", "Mutex.tla"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(tmp_path),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"TLC timeout after {timeout_s}s"
        except FileNotFoundError:
            return False, "java executable not found at run time"

        output = (result.stdout or "") + (result.stderr or "")

        # Parse-time errors
        if "Was expecting" in output or "lexical error" in output.lower():
            return False, f"TLC parse error: {output[:400].strip()}"
        if "Module not found" in output or "cannot find module" in output.lower():
            return False, f"TLC could not find module: {output[:400].strip()}"

        # Invariant violation
        if "is violated" in output:
            return False, "MutualExclusion invariant violated"

        success_markers = (
            "Model checking completed",
            "No error has been found",
            "states generated",
        )
        if any(marker in output for marker in success_markers) and result.returncode == 0:
            return True, "TLC completed; MutualExclusion holds"

        return False, (f"TLC exit {result.returncode}: {output[:400].strip() or '(no output)'}")


# ---------------------------------------------------------------------------
# Challenge 5: IPv4 regex synthesis
# ---------------------------------------------------------------------------


_IPV4_TEST_POSITIVES: tuple[str, ...] = (
    "0.0.0.0",
    "1.1.1.1",
    "10.0.0.1",
    "127.0.0.1",
    "169.254.0.1",
    "172.16.0.1",
    "192.168.1.1",
    "203.0.113.42",
    "255.255.255.255",
    "224.0.0.1",
    "100.64.0.1",
    "198.51.100.7",
    "8.8.8.8",
    "9.9.9.9",
    "1.2.3.4",
    "99.99.99.99",
    "100.100.100.100",
    "200.200.200.200",
    "249.249.249.249",
    "254.254.254.254",
)

_IPV4_TEST_NEGATIVES: tuple[str, ...] = (
    "256.1.1.1",
    "1.256.1.1",
    "1.1.256.1",
    "1.1.1.256",
    "300.1.1.1",
    "999.999.999.999",
    "-1.1.1.1",
    "1.-1.1.1",
    "01.1.1.1",
    "001.1.1.1",
    "1.01.1.1",
    "1.1.1",
    "1.1.1.1.1",
    "1.1.1.",
    ".1.1.1.1",
    "1..1.1",
    "1.1..1",
    "1.1.1.a",
    "a.1.1.1",
    "1.1.1.1a",
    "",
    "1",
    "1.1",
    "1.1.1.1 ",
    " 1.1.1.1",
    " 1.1.1.1 ",
    "1.1.1.1\n",
    "\n1.1.1.1",
    "1,1,1,1",
    "1:1:1:1",
    "1.1.1.1.",
    "..1.1",
    "1.1.1.1/24",
)


def validate_regex(
    pattern_text: str,
    target_f1: float = 0.95,
) -> ValidationResult:
    """Validate an IPv4 regex against held-out positive/negative test sets.

    Args:
        pattern_text: The regex pattern as written by the engine. Single line.
        target_f1: Minimum F1 score required to pass.

    Returns:
        ``(passed, message)``.
    """
    pattern_text = pattern_text.strip()
    if not pattern_text:
        return False, "empty pattern"
    if "\n" in pattern_text:
        return False, "pattern must be a single line"
    try:
        compiled = re.compile(pattern_text)
    except re.error as exc:
        return False, f"invalid regex: {exc}"

    tp = sum(1 for s in _IPV4_TEST_POSITIVES if compiled.fullmatch(s))
    fp = sum(1 for s in _IPV4_TEST_NEGATIVES if compiled.fullmatch(s))
    fn = len(_IPV4_TEST_POSITIVES) - tp

    if tp == 0:
        return False, f"zero true positives (fp={fp}, fn={fn})"

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if (precision + recall) == 0:
        return False, "precision and recall both zero"
    f1 = 2 * precision * recall / (precision + recall)

    if f1 >= target_f1:
        return True, f"F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})"
    return False, (
        f"F1={f1:.3f} below target {target_f1:.2f} "
        f"(P={precision:.3f}, R={recall:.3f}, tp={tp}, fp={fp}, fn={fn})"
    )


# ---------------------------------------------------------------------------
# Convenience: dispatch by challenge id
# ---------------------------------------------------------------------------


NOVEL_ARTIFACT_CHALLENGE_IDS: tuple[str, ...] = (
    "novel-sokoban",
    "novel-smtlib",
    "novel-crossword",
    "novel-tlaplus",
    "novel-regex",
)


def is_novel_artifact_challenge(challenge_id: str) -> bool:
    """Check whether a challenge id targets one of the novel-artifact validators."""
    return challenge_id in NOVEL_ARTIFACT_CHALLENGE_IDS
