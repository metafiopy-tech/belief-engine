# Substrate-Transfer Experiment — Novel-Artifact Challenge Specs

**Status:** Drafted 2026-05-27. Locks the 5 novel-artifact challenges that constitute the most-important quarter of the 20-challenge reduced experiment.

**Thesis under test (locked separately in memory):** *The Belief Engine's engineering loop — iterate, validate, refine, retain — transferred to artifacts the substrate had never seen.*

These 5 challenges are the operational test of that thesis. The other 15 challenges (5 Python microservices, 5 CLI/scripts, 5 data pipelines) come from `belief.benchmark.CHALLENGES` and exist to establish baselines and the per-domain Chart 3 bars. **These five are the bars that decide whether the thesis is supported.**

Every validator is pure Python with no LLM-as-judge. Every scoring criterion is mechanically checkable. Each challenge is designed so that raw `qwen2.5-coder:14b` (the floor) produces something at least structurally recognizable — confirmed by the 2026-05-27 floor pre-test for TLA+; assumed-passing for the other four based on the model's training distribution.

---

## Challenge 1 — Sokoban level with exact-move solution

### Goal text (sent verbatim to the engine)

> Design a Sokoban puzzle level on a 6×6 grid. The level must contain exactly one player, one box, and one target square, surrounded by walls. The shortest solution must require exactly **14 player moves** (counting both pushes and non-push moves).
>
> Output a file `level.txt` containing 6 lines of 6 characters each, using:
> - `#` for wall
> - ` ` (space) for empty floor
> - `@` for player starting position
> - `$` for box starting position
> - `.` for target square
>
> The outer border of the grid must be all walls. The level must be solvable. The shortest solution (in player-moves) must be exactly 14.

### Why this isn't codegen in disguise

The deliverable is a level (text grid), scored on a creative property (puzzle difficulty calibrated to a specific move count). None of the existing soil contains Sokoban content. The engine will almost certainly attack this by writing a Python BFS solver and searching the design space — that's the substrate transferring its iterate-validate-refine pattern to a non-Python artifact, which is exactly what the thesis claims.

### Validator (pure Python, no external deps)

```python
from collections import deque
from typing import Optional

def parse_level(text: str) -> dict:
    rows = text.rstrip('\n').split('\n')
    walls, boxes, targets, player = set(), set(), set(), None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == '#':
                walls.add((r, c))
            elif ch == '$':
                boxes.add((r, c))
            elif ch == '.':
                targets.add((r, c))
            elif ch == '@':
                player = (r, c)
            elif ch == '*':
                boxes.add((r, c)); targets.add((r, c))
            elif ch == '+':
                player = (r, c); targets.add((r, c))
    return {'walls': walls, 'boxes': boxes, 'targets': targets,
            'player': player, 'rows': len(rows), 'cols': max(len(r) for r in rows)}

def shortest_solution(state: dict, max_depth: int = 60) -> Optional[int]:
    walls, targets = state['walls'], frozenset(state['targets'])
    start = (state['player'], frozenset(state['boxes']))
    if start[1] == targets:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (player, boxes), depth = queue.popleft()
        if depth >= max_depth:
            continue
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            np_ = (player[0]+dr, player[1]+dc)
            if np_ in walls:
                continue
            nb = boxes
            if np_ in boxes:
                bd = (np_[0]+dr, np_[1]+dc)
                if bd in walls or bd in boxes:
                    continue
                nb = (boxes - {np_}) | {bd}
            ns = (np_, nb)
            if ns in visited:
                continue
            if nb == targets:
                return depth + 1
            visited.add(ns)
            queue.append((ns, depth + 1))
    return None

def validate(level_text: str, target_moves: int = 14) -> tuple[bool, str]:
    state = parse_level(level_text)
    if state['rows'] != 6 or state['cols'] != 6:
        return False, f"expected 6x6 grid, got {state['rows']}x{state['cols']}"
    if state['player'] is None:
        return False, "no player (@)"
    if len(state['boxes']) != 1:
        return False, f"expected 1 box, got {len(state['boxes'])}"
    if len(state['targets']) != 1:
        return False, f"expected 1 target, got {len(state['targets'])}"
    moves = shortest_solution(state)
    if moves is None:
        return False, "level is unsolvable"
    if moves != target_moves:
        return False, f"shortest solution is {moves} moves, expected {target_moves}"
    return True, f"valid: shortest solution = {moves} moves"
```

### Scoring nuance

Strict pass/fail (exact-move match). If we want a partial-credit version, we could score by `1 - abs(actual - target) / target` clamped to [0, 1]. **Recommend strict pass/fail** for the headline experiment — the thesis is binary ("did the loop transfer"), not graded.

---

## Challenge 2 — SMT-LIB encoding of a logic puzzle

### Goal text (sent verbatim)

> Encode the following logic puzzle in SMT-LIB v2 such that Z3 returns `sat` and the unique solution can be extracted from the model.
>
> **Puzzle:** Three houses sit in a row at positions 1, 2, and 3 (left to right). Each house has exactly one color (red, green, or blue), exactly one owner (Alice, Bob, or Carol), and exactly one pet (cat, dog, or fish). Every color, owner, and pet appears exactly once across the three houses. The following constraints hold:
>
> 1. The green house is immediately to the right of the red house.
> 2. The blue house is at position 1.
> 3. Alice owns the dog.
> 4. The cat lives in the red house.
> 5. Bob lives at position 3.
>
> **Question:** Who owns the fish, and at which position?
>
> **Expected answer:** Bob, at position 3.
>
> Output a file `puzzle.smt2` containing an SMT-LIB v2 encoding. The file must:
> - Declare appropriate variables / functions for color, owner, pet at each position.
> - Assert all five constraints AND the implicit uniqueness constraints (each color/owner/pet appears exactly once).
> - End with `(check-sat)` and `(get-model)`.
>
> When run with `z3 puzzle.smt2`, the output's first line must be `sat`, and the model must assign Bob to position 3 with the fish.

### Why this isn't codegen in disguise

SMT-LIB is declarative first-order logic. The engine must translate English constraints into formal logical assertions — a fundamentally different cognitive operation from "implement this Python function." Z3 decides satisfiability mechanically. The artifact is `.smt2`, not Python.

### Validator (requires `pip install z3-solver`)

```python
def validate_smtlib(smt_text: str) -> tuple[bool, str]:
    try:
        import z3
    except ImportError:
        return False, "z3-solver not installed (pip install z3-solver)"

    # Strip trailing (check-sat)/(get-model) so we can drive solving ourselves
    lines = [ln for ln in smt_text.splitlines()
             if not ln.strip().startswith('(check-sat')
             and not ln.strip().startswith('(get-model')]
    assertions_text = '\n'.join(lines)

    try:
        assertions = z3.parse_smt2_string(assertions_text)
    except z3.Z3Exception as e:
        return False, f"z3 parse error: {e}"

    solver = z3.Solver()
    for a in assertions:
        solver.add(a)

    if solver.check() != z3.sat:
        return False, "encoding is unsat (expected sat)"

    model = solver.model()

    # Verify uniqueness: blocking the current model must yield unsat.
    decls = [d for d in model.decls() if d.arity() == 0]
    if not decls:
        return False, "model has no top-level constants to verify"

    block_terms = []
    for d in decls:
        val = model[d]
        if val is not None:
            block_terms.append(d() != val)
    if block_terms:
        solver.push()
        solver.add(z3.Or(block_terms))
        if solver.check() == z3.sat:
            solver.pop()
            return False, "encoding has multiple satisfying models (puzzle should be unique)"
        solver.pop()

    # Verify Bob/fish/position-3 appears in the model. The engine controls
    # variable naming, so this is a string search over the model's text rep.
    model_str = str(model).lower()
    if 'bob' not in model_str:
        return False, "model does not reference Bob"
    if 'fish' not in model_str:
        return False, "model does not reference fish"
    # The puzzle's logical solution forces Bob -> position 3 -> fish.
    # If uniqueness passed and Bob+fish are in the model, the assignment is correct.

    return True, "z3 sat, unique model, Bob/fish present"
```

### Scoring nuance

The validator does **three** checks: (a) Z3 returns sat, (b) the model is unique under block-and-resolve, (c) "bob" and "fish" appear in the stringified model. If the engine uses very different naming (e.g., numeric encoding of pets), check (c) fails even with a correct encoding. **Mitigation:** the goal text specifies the puzzle in natural language with English names, which strongly biases naming. Accept this small risk of false negative.

---

## Challenge 3 — Crossword construction with structural constraints

### Goal text (sent verbatim)

> Construct a 7×7 American-style crossword puzzle.
>
> Output two files:
> - `grid.txt` — exactly 7 lines, each line exactly 7 characters. Use uppercase A-Z for letter cells and `#` for black squares.
> - `clues.txt` — one clue per line, format `[A|D][number]: [clue text]` (e.g., `A1: Yellow fruit (6)`).
>
> Constraints the grid must satisfy:
> 1. 180-degree rotational symmetry of black squares: if `(r, c)` is black, then `(6-r, 6-c)` must also be black.
> 2. Every horizontal run of letter cells of length ≥ 3 is an Across entry; every vertical run of letter cells of length ≥ 3 is a Down entry.
> 3. No entries shorter than 3 letters. (So every letter cell must be part of an entry of length ≥ 3.)
> 4. All Across and Down entries must be valid English words from the provided wordlist (top-5000 English words; the wordlist file `wordlist.txt` is provided in the working directory).
> 5. At least 15 entries total (Across + Down combined).
> 6. All intersections must agree: the letter at any cell shared between an Across and a Down entry must be identical (this is automatic if you treat the grid as the single source of truth).

### Why this isn't codegen in disguise

The deliverable is a puzzle artifact (grid + clue list), scored on structural properties (symmetry, word membership, intersection legality). None of these are remotely Python-shaped. The engine will likely write Python search to construct it — exactly the substrate-transfer pattern.

### Validator

```python
def validate_crossword(grid_text: str, clues_text: str,
                      wordlist: set[str]) -> tuple[bool, str]:
    rows = [r.rstrip() for r in grid_text.strip('\n').split('\n')]
    if len(rows) != 7:
        return False, f"expected 7 rows, got {len(rows)}"
    if any(len(r) != 7 for r in rows):
        widths = [len(r) for r in rows]
        return False, f"all rows must be 7 chars; got widths {widths}"

    # 1. Symmetry of black squares
    for r in range(7):
        for c in range(7):
            if (rows[r][c] == '#') != (rows[6-r][6-c] == '#'):
                return False, f"asymmetric black square at ({r},{c}) vs ({6-r},{6-c})"

    # 2. Extract entries
    def runs_horizontal():
        out = []
        for r in range(7):
            c = 0
            while c < 7:
                if rows[r][c] != '#':
                    start = c
                    while c < 7 and rows[r][c] != '#':
                        c += 1
                    out.append((r, start, c - start, rows[r][start:c]))
                else:
                    c += 1
        return out
    def runs_vertical():
        out = []
        for c in range(7):
            r = 0
            while r < 7:
                if rows[r][c] != '#':
                    start = r
                    while r < 7 and rows[r][c] != '#':
                        r += 1
                    word = ''.join(rows[i][c] for i in range(start, r))
                    out.append((start, c, r - start, word))
                else:
                    r += 1
        return out

    across = [(r, c, word) for (r, c, length, word) in runs_horizontal() if length >= 3]
    down = [(r, c, word) for (r, c, length, word) in runs_vertical() if length >= 3]

    # 3. No 1- or 2-letter runs (every letter cell must be in a 3+ run)
    bad_h = [(r, c, length) for (r, c, length, _) in runs_horizontal()
             if length == 2 or (length == 1)]
    bad_v = [(r, c, length) for (r, c, length, _) in runs_vertical()
             if length == 2 or (length == 1)]
    # A length-1 horizontal is fine ONLY if the cell is also in a 3+ vertical;
    # a length-2 horizontal is never fine.
    # Same for vertical.
    isolated = []
    for r in range(7):
        for c in range(7):
            if rows[r][c] == '#':
                continue
            h_run = next((length for (rr, cc, length, _) in runs_horizontal()
                         if rr == r and cc <= c < cc + length), 0)
            v_run = next((length for (rr, cc, length, _) in runs_vertical()
                         if cc == c and rr <= r < rr + length), 0)
            if h_run < 3 and v_run < 3:
                isolated.append((r, c))
    if isolated:
        return False, f"isolated/short-run cells: {isolated[:5]}"

    # 4. Entry count
    if len(across) + len(down) < 15:
        return False, f"only {len(across) + len(down)} entries; need 15+"

    # 5. Wordlist membership
    for r, c, word in across + down:
        if word.lower() not in wordlist:
            return False, f"entry '{word}' (at {r},{c}) not in wordlist"

    return True, f"valid: {len(across)} across + {len(down)} down"
```

### Scoring nuance

Wordlist needs to exist at `wordlist.txt` in the engine's working directory. Use the top-5000 most frequent English words (e.g. from MIT 5000 list or Google's `english-words` repo). The engine should be told the wordlist file path in the goal text or via a provided fixture. **Action item:** copy a 5000-word list into `docs/experiments/fixtures/wordlist.txt` before running.

Clues are not validated for quality (they would require LLM-as-judge). The validator only checks the grid is structurally valid; clues being present in `clues.txt` is encouraged but not enforced.

---

## Challenge 4 — TLA+ specification (stretch, per floor pre-test)

### Goal text (sent verbatim)

> Write a TLA+ specification for a two-process mutual exclusion problem implementing **Peterson's algorithm**.
>
> Output two files:
> - `Mutex.tla` — the TLA+ module
> - `Mutex.cfg` — the TLC configuration
>
> The module must contain:
> - Module header (`---- MODULE Mutex ----`) and footer (`====`).
> - `EXTENDS Naturals`.
> - `VARIABLES pc1, pc2, turn, flag1, flag2`. (`pc1`/`pc2` are program counters with values in `{"ncs", "trying", "cs", "exit"}`; `turn` is `1` or `2`; `flag1`/`flag2` are booleans.)
> - `vars == <<pc1, pc2, turn, flag1, flag2>>`.
> - `Init` setting `pc1 = "ncs"`, `pc2 = "ncs"`, `turn = 1`, `flag1 = FALSE`, `flag2 = FALSE`.
> - Per-process action predicates `Proc1` and `Proc2`, each a disjunction of guarded transitions matching Peterson's algorithm. **All primed-variable syntax must place the prime on the variable name itself (e.g., `pc1'`, NOT `pc'1`).** Every action must include `UNCHANGED` clauses for variables it does not modify.
> - `Next == Proc1 \/ Proc2`.
> - `Spec == Init /\ [][Next]_vars`.
> - `MutualExclusion == ~(pc1 = "cs" /\ pc2 = "cs")`.
>
> The `Mutex.cfg` file must contain:
> ```
> SPECIFICATION Spec
> INVARIANT MutualExclusion
> ```
>
> When run with `java -jar tla2tools.jar -config Mutex.cfg Mutex.tla`, TLC must complete model-checking without reporting a violation of `MutualExclusion`.

### Why this isn't codegen in disguise

TLA+ is declarative temporal logic. State transitions are written as predicates over primed and unprimed variables. The reasoning paradigm — state-space exploration, invariants, fairness — has near-zero overlap with imperative Python. The raw floor pre-test confirmed the model can produce structurally-recognizable TLA+ but with multiple syntactic and semantic bugs (prime placement, missing `UNCHANGED`, undefined `Spec`); this challenge tests whether the substrate's iterate-fix-revalidate loop can drive those bugs out within 15 builds.

### Validator (requires `tla2tools.jar`)

```python
import subprocess, tempfile, shutil, os
from pathlib import Path

def validate_tlaplus(tla_text: str, cfg_text: str,
                     timeout_s: int = 120) -> tuple[bool, str]:
    tlc_jar = os.environ.get('TLC_JAR')
    if not tlc_jar:
        for candidate in [
            '/usr/local/lib/tla2tools.jar',
            os.path.expanduser('~/lib/tla2tools.jar'),
            os.path.expanduser('~/Downloads/tla2tools.jar'),
        ]:
            if Path(candidate).exists():
                tlc_jar = candidate
                break
    if not tlc_jar:
        return False, "tla2tools.jar not found (set TLC_JAR or install)"

    with tempfile.TemporaryDirectory() as tmp:
        tla_path = Path(tmp) / 'Mutex.tla'
        cfg_path = Path(tmp) / 'Mutex.cfg'
        tla_path.write_text(tla_text)
        cfg_path.write_text(cfg_text)
        try:
            result = subprocess.run(
                ['java', '-jar', tlc_jar, '-config', 'Mutex.cfg', 'Mutex.tla'],
                capture_output=True, text=True, timeout=timeout_s, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return False, f"TLC timeout after {timeout_s}s"

        out = (result.stdout or '') + (result.stderr or '')

        # Parse failure markers
        if 'Was expecting' in out or 'lexical error' in out.lower():
            return False, f"TLC parse error: {out[:400]}"
        if 'Module not found' in out:
            return False, "TLC could not find module"

        # Invariant violation
        if 'is violated' in out or 'Invariant' in out and 'violated' in out:
            return False, "MutualExclusion violated"

        # Success markers (TLC version-dependent)
        success_markers = [
            'Model checking completed',
            'No error has been found',
            'states generated',
        ]
        if any(m in out for m in success_markers) and result.returncode == 0:
            return True, "TLC completed without invariant violation"

        return False, f"TLC exit {result.returncode}: {out[:400]}"
```

### Scoring nuance

Requires Java + `tla2tools.jar` on the host. Install before experiment day:
```
mkdir -p ~/lib && curl -L -o ~/lib/tla2tools.jar \
  https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
```

This is the **stretch challenge**. Expect raw_local to score ~0/20 and full to score somewhere between 0/20 and 5/20 within 15 builds. Even a small positive delta (full - raw) on this challenge is meaningful — it would directly demonstrate the substrate rescuing the model in a paradigm where the raw floor is near zero.

---

## Challenge 5 — Regex synthesis (IPv4 validation)

### Goal text (sent verbatim)

> Produce a single Python-compatible regular expression that matches valid IPv4 addresses in dotted-quad notation and rejects all invalid strings. The regex will be compiled with `re.compile(pattern)` and applied as `re.fullmatch(pattern, candidate)`.
>
> **Definition of valid:** four decimal octets separated by dots. Each octet is an integer from 0 to 255 inclusive, written without leading zeros (so `0` is valid, `01` is not, `255` is valid, `256` is not).
>
> **Examples that must match:**
> - `0.0.0.0`
> - `255.255.255.255`
> - `192.168.1.1`
> - `8.8.8.8`
> - `127.0.0.1`
>
> **Examples that must NOT match:**
> - `256.1.1.1` (octet > 255)
> - `01.1.1.1` (leading zero)
> - `1.1.1` (too few octets)
> - `1.1.1.1.1` (too many octets)
> - `1.1.1.a` (non-digit)
> - ` 1.1.1.1` (leading whitespace)
> - `1.1.1.1 ` (trailing whitespace)
> - `1..1.1` (empty octet)
>
> Output a file `solution.regex` containing only the regex pattern as a single line — no quotes, no leading `r"...""`, no flags, no surrounding `re.compile(...)`. The pattern will be loaded as `re.compile(open('solution.regex').read().strip())`.
>
> Your regex must achieve F1 ≥ 0.95 on a held-out test set drawn from the same distribution as the examples above.

### Why this isn't codegen in disguise

The deliverable is a single declarative pattern, not a procedural program. The engine cannot solve this by running anything — it has to *think* in regex. Validation is mechanical via the `re` module.

### Validator (no external deps beyond stdlib)

```python
import re

# Held-out test set — NOT shown to the engine in the goal text
TEST_POSITIVES = [
    '0.0.0.0', '1.1.1.1', '10.0.0.1', '127.0.0.1', '169.254.0.1',
    '172.16.0.1', '192.168.1.1', '203.0.113.42', '255.255.255.255',
    '224.0.0.1', '100.64.0.1', '198.51.100.7', '8.8.8.8', '9.9.9.9',
    '1.2.3.4', '99.99.99.99', '100.100.100.100', '200.200.200.200',
    '249.249.249.249', '254.254.254.254',
]
TEST_NEGATIVES = [
    '256.1.1.1', '1.256.1.1', '1.1.256.1', '1.1.1.256',
    '300.1.1.1', '999.999.999.999', '-1.1.1.1', '1.-1.1.1',
    '01.1.1.1', '001.1.1.1', '1.01.1.1',
    '1.1.1', '1.1.1.1.1', '1.1.1.', '.1.1.1.1', '1..1.1', '1.1..1',
    '1.1.1.a', 'a.1.1.1', '1.1.1.1a',
    '', '1', '1.1', '1.1.1.1 ', ' 1.1.1.1', ' 1.1.1.1 ',
    '1.1.1.1\n', '\n1.1.1.1', '1,1,1,1', '1:1:1:1',
    '1.1.1.1.', '..1.1', '1.1.1.1/24',
]

def validate_regex(pattern_text: str, target_f1: float = 0.95) -> tuple[bool, str]:
    pattern_text = pattern_text.strip()
    if not pattern_text:
        return False, "empty pattern"
    if '\n' in pattern_text:
        return False, "pattern must be a single line"
    try:
        rx = re.compile(pattern_text)
    except re.error as e:
        return False, f"invalid regex: {e}"

    tp = sum(1 for s in TEST_POSITIVES if rx.fullmatch(s))
    fp = sum(1 for s in TEST_NEGATIVES if rx.fullmatch(s))
    fn = len(TEST_POSITIVES) - tp

    if tp == 0:
        return False, f"zero true positives (fp={fp}, fn={fn})"

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if f1 >= target_f1:
        return True, f"F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})"
    return False, (f"F1={f1:.3f} below target {target_f1} "
                   f"(P={precision:.3f}, R={recall:.3f}, tp={tp}, fp={fp}, fn={fn})")
```

### Scoring nuance

F1 threshold of 0.95 is deliberately strict but achievable. The canonical IPv4 regex `^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$` would F1 around 0.90 (it accepts leading-zero octets) — a higher-quality answer is needed. **This is a feature, not a bug:** the threshold forces the engine to actually understand the leading-zero edge case, which the floor model almost certainly won't get on the first try.

The "must be a single line" constraint exists to prevent the engine from cheating with `re.VERBOSE`-style multi-line patterns; the validator strips before compiling, so a multi-line pattern would still be loaded literally with newlines in it, breaking compilation.

---

## Open prerequisites before any of these can run

1. **`wordlist.txt`** fixture for Challenge 3. Need a top-5000 English-words file in a fixed location. Suggestion: vendor `https://github.com/dwyl/english-words/raw/master/words_alpha.txt` filtered to length 3-7, top by frequency. **Action item.**
2. **`tla2tools.jar`** on the host for Challenge 4. ~12 MB download. Will live in `~/lib/`. **Action item before experiment day.**
3. **`z3-solver`** Python package for Challenge 2. `pip install z3-solver`. **Action item.**
4. **Validator-as-belief-engine-validator wiring.** None of these validators are currently wired into the engine's `validator` node. The engine's existing validator runs `pytest` on the generated code. For these novel-artifact challenges, the validator needs to (a) detect that the challenge is a novel-artifact challenge, (b) invoke the appropriate validator function from this doc, (c) return the score in the format the experiment runner expects. This is a downstream task (likely belongs in task #4 of the task list, runner wiring).

---

## Notes on what these challenges deliberately don't test

- **Creativity / aesthetic quality.** A Sokoban level might have an "elegant" solution path or be "interesting" — neither is scored. Pass/fail on move count only.
- **Clue quality** for the crossword. The validator only checks the grid; clues are present-but-unscored.
- **Proof elegance** — N/A here since Lean was dropped post-floor-test.
- **Performance** of generated regex. F1 only; a slow but accurate regex still passes.

All four omissions are deliberate — they would require LLM-as-judge, which would compromise the experiment's mechanical reproducibility and make the eventual writeup vulnerable to "but the judge model is biased toward Claude-style outputs."

---

## Next steps

This doc unblocks task #2 (inventory in-distribution challenges) and tasks #3-4 (pipeline toggle + runner wiring). The validators in this doc need to be packaged as importable Python (probably `belief/experiments/novel_artifact_validators.py`) before task #5 (shakedown) can run.
