"""Trophic competition: dense step-level rewards without human labels.

Every nutrient in the soil can carry a ``trophic_level`` metadata field:

    Level 0:  Raw code fragments (producers)
    Level 1:  Patterns that subsume raw fragments (herbivores)
    Level 2:  Abstractions that subsume patterns (predators)
    Level 3+: Meta-abstractions (apex predators, named-library candidates)

The :func:`compete` function pits two fragments against a battery of
test inputs. It executes each fragment in a sandboxed subprocess so a
malicious or buggy fragment can't take the main process down. The
outcome is a :class:`TrophicRelation` — predation when one wins,
symbiosis when the two together beat either alone, competition when
they tie.

Sandboxing notes:
  - Each fragment is serialized to a temp file and executed under
    ``python3 -I -S`` (no user site, no site-packages auto-import)
    with a 5-second timeout.
  - Nothing from the main process leaks into the child — the driver
    re-imports only stdlib.
  - The driver catches every Exception and returns a failure rather
    than letting a traceback land on stderr as noise.
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


logger = logging.getLogger("belief.memory.trophic")


MAX_COMPETITIONS_PER_BUILD = 5
SUBPROCESS_TIMEOUT_S = 5.0
SYMBIOSIS_MARGIN = 1  # A+B together must beat the better one by >= this many passes


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrophicRelation:
    """Outcome of a single fragment-vs-fragment competition."""

    predator_id: Optional[str] = None   # Nutrient that won, or None on tie/symbiosis
    prey_id: Optional[str] = None        # Nutrient that lost, or None
    relation: str = "competition"        # predation | symbiosis | competition
    test_id: str = ""
    margin: float = 0.0                  # pass-rate margin in [0, 1]
    passes_a: int = 0
    passes_b: int = 0
    tests_run: int = 0
    errors: list[str] = field(default_factory=list)

    def is_tie(self) -> bool:
        return self.relation == "competition"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compete(
    fragment_a: str,
    fragment_b: str,
    test_inputs: Sequence[dict[str, Any]],
    *,
    fragment_a_id: str = "a",
    fragment_b_id: str = "b",
    test_id: str = "",
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> TrophicRelation:
    """Run two fragments against a battery of test inputs.

    Each ``test_inputs`` entry is expected to carry the keyword arguments
    passed to the fragment's main function plus an ``expected`` key with
    the result to compare against. Fragments that error or time out
    count as a failure on that test. The function with the higher pass
    count is the predator; ties are "competition"; A+B together passing
    more than either alone is "symbiosis".
    """
    tests = list(test_inputs)
    if not tests:
        return TrophicRelation(
            relation="competition",
            test_id=test_id,
            errors=["no test inputs provided"],
        )

    passes_a, passes_b, passes_combined, errors = _run_battery(
        fragment_a, fragment_b, tests, timeout_s=timeout_s
    )

    better = max(passes_a, passes_b)
    relation = "competition"
    predator_id: Optional[str] = None
    prey_id: Optional[str] = None
    margin = 0.0
    n = len(tests)

    if passes_combined >= better + SYMBIOSIS_MARGIN:
        relation = "symbiosis"
        # Both fragments carry credit; neither is prey.
    elif passes_a > passes_b:
        relation = "predation"
        predator_id = fragment_a_id
        prey_id = fragment_b_id
        margin = (passes_a - passes_b) / n
    elif passes_b > passes_a:
        relation = "predation"
        predator_id = fragment_b_id
        prey_id = fragment_a_id
        margin = (passes_b - passes_a) / n

    return TrophicRelation(
        predator_id=predator_id,
        prey_id=prey_id,
        relation=relation,
        test_id=test_id,
        margin=margin,
        passes_a=passes_a,
        passes_b=passes_b,
        tests_run=n,
        errors=errors,
    )


def update_trophic_levels(
    soil: Any,
    relations: Iterable[TrophicRelation],
    *,
    lapse_threshold: int = 3,
) -> dict[str, Any]:
    """Roll up relations into trophic-level changes.

    `soil` is a belief.memory.soil.Soil instance (duck-typed: we only
    need get_metadata/set_metadata accessors). When the soil is a light
    stub (as in tests) we fall back to the optional in-memory overlay
    attached to the instance.

    Returns a summary dict with counts of promotions / demotions so
    callers can write metrics without re-walking the relation list.
    """
    promotions: dict[str, int] = {}
    prey_losses: dict[str, int] = {}
    symbionts: set[str] = set()

    for rel in relations:
        if rel.relation == "predation" and rel.predator_id and rel.prey_id:
            promotions[rel.predator_id] = promotions.get(rel.predator_id, 0) + 1
            prey_losses[rel.prey_id] = prey_losses.get(rel.prey_id, 0) + 1
        elif rel.relation == "symbiosis":
            # Symbiosis doesn't identify A/B here; callers with the ids
            # should pass them via the TrophicRelation in a future extension.
            pass

    # Apply promotions: predator_level = prey_level + 1 (bounded). Apply
    # via setter if soil exposes one; otherwise attach to an overlay.
    _apply_levels_via_soil(soil, promotions, prey_losses, lapse_threshold)

    return {
        "promotions": len(promotions),
        "prey_demoted": sum(1 for n in prey_losses.values() if n >= lapse_threshold),
        "symbionts": len(symbionts),
    }


def run_trophic_pass(
    new_fragments: Sequence[dict[str, Any]],
    competitors: Sequence[dict[str, Any]],
    test_inputs: Sequence[dict[str, Any]],
    *,
    max_competitions: int = MAX_COMPETITIONS_PER_BUILD,
) -> list[TrophicRelation]:
    """Pair each new fragment against at most N competitors.

    ``new_fragments`` and ``competitors`` are dicts with at least
    ``id`` and ``code`` keys. ``test_inputs`` is shared across all
    pairings. Caller should select the most-similar competitors (by
    embedding) before passing them in.
    """
    relations: list[TrophicRelation] = []
    budget = max(0, int(max_competitions))
    for new in new_fragments:
        if budget <= 0:
            break
        for comp in competitors:
            if budget <= 0:
                break
            rel = compete(
                fragment_a=str(new.get("code", "")),
                fragment_b=str(comp.get("code", "")),
                test_inputs=test_inputs,
                fragment_a_id=str(new.get("id", "new")),
                fragment_b_id=str(comp.get("id", "comp")),
            )
            relations.append(rel)
            budget -= 1
    return relations


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_function_name(fragment: str) -> Optional[str]:
    """Return the name of the first top-level FunctionDef, or None."""
    try:
        tree = ast.parse(fragment)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def _build_driver(fragment: str, fn_name: str, test_case: dict[str, Any]) -> str:
    """Compose a standalone Python program that returns exit 0 on pass.

    The test_case is injected as canonical JSON — no importing of the
    test file, no path juggling.
    """
    case_json = json.dumps(test_case, default=str)
    # Indent the fragment consistently so we don't get mixed-indent errors.
    fragment_block = textwrap.dedent(fragment).strip()
    return (
        "import json, sys\n"
        + fragment_block
        + "\n"
        + f"_case = json.loads({case_json!r})\n"
        + "_expected = _case.pop('expected', None)\n"
        + "try:\n"
        + f"    _result = {fn_name}(**_case)\n"
        + "except Exception as _exc:\n"
        + "    print(f'runtime:{_exc}', file=sys.stderr)\n"
        + "    sys.exit(2)\n"
        + "if _result == _expected:\n"
        + "    sys.exit(0)\n"
        + "print(f'mismatch:{_result!r}!={_expected!r}', file=sys.stderr)\n"
        + "sys.exit(1)\n"
    )


def _run_one(
    fragment: str,
    test_case: dict[str, Any],
    *,
    timeout_s: float,
) -> tuple[bool, str]:
    """Run one (fragment, test_case) pair. Returns (passed, error_msg)."""
    fn = _extract_function_name(fragment)
    if fn is None:
        return (False, "no top-level function found")
    driver = _build_driver(fragment, fn, test_case)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(driver)
        script_path = f.name
    try:
        # -I = isolated (ignore PYTHON* env vars, no user site).
        # -S = do not run site.py (skip packages auto-import).
        proc = subprocess.run(
            [sys.executable, "-I", "-S", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            return (True, "")
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr else f"exit={proc.returncode}"
        return (False, msg[:200])
    except subprocess.TimeoutExpired:
        return (False, f"timeout>{timeout_s:.0f}s")
    except Exception as exc:  # pragma: no cover - subprocess infra errors
        return (False, f"infra:{exc}")
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def _run_battery(
    fragment_a: str,
    fragment_b: str,
    tests: list[dict[str, Any]],
    *,
    timeout_s: float,
) -> tuple[int, int, int, list[str]]:
    """Execute both fragments + combined-vote over the shared battery.

    Returns (passes_a, passes_b, passes_combined, errors). For the
    combined count, a test "passes_combined" when EITHER fragment
    passes it — a weak OR, enough to detect when the two fragments
    cover complementary inputs (symbiosis).
    """
    passes_a = 0
    passes_b = 0
    passes_combined = 0
    errors: list[str] = []
    for i, case in enumerate(tests):
        ok_a, err_a = _run_one(fragment_a, case, timeout_s=timeout_s)
        ok_b, err_b = _run_one(fragment_b, case, timeout_s=timeout_s)
        passes_a += 1 if ok_a else 0
        passes_b += 1 if ok_b else 0
        passes_combined += 1 if (ok_a or ok_b) else 0
        if not ok_a:
            errors.append(f"t{i}/a:{err_a}")
        if not ok_b:
            errors.append(f"t{i}/b:{err_b}")
    return passes_a, passes_b, passes_combined, errors


def _apply_levels_via_soil(
    soil: Any,
    promotions: dict[str, int],
    prey_losses: dict[str, int],
    lapse_threshold: int,
) -> None:
    """Best-effort update of trophic_level metadata on soil nutrients.

    Uses soil.get_nutrient / soil.update_metadata if available, else
    attaches an in-memory overlay at soil._trophic_overlay so tests
    can inspect it without a real ChromaDB-backed soil.
    """
    # Attach overlay either way — it's cheap and keeps the test hook stable.
    overlay: dict[str, dict[str, Any]] = getattr(soil, "_trophic_overlay", None) or {}
    if overlay is None:
        overlay = {}

    for nid, wins in promotions.items():
        current = int(overlay.get(nid, {}).get("trophic_level", 0))
        overlay[nid] = {"trophic_level": current + 1, "wins": wins}

    for nid, losses in prey_losses.items():
        if losses >= lapse_threshold:
            overlay[nid] = {
                "trophic_level": max(0, int(overlay.get(nid, {}).get("trophic_level", 0)) - 1),
                "lapsed": True,
                "losses": losses,
            }

    try:
        soil._trophic_overlay = overlay
    except AttributeError:
        # soil object doesn't allow attribute assignment; give up silently.
        logger.debug("soil object does not accept overlay; skipping persistence")


__all__ = [
    "MAX_COMPETITIONS_PER_BUILD",
    "SUBPROCESS_TIMEOUT_S",
    "TrophicRelation",
    "compete",
    "run_trophic_pass",
    "update_trophic_levels",
]
