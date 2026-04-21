"""belief.memory.trophic — compete() + update_trophic_levels + run_trophic_pass."""

from __future__ import annotations

import pytest

from belief.memory.trophic import (
    MAX_COMPETITIONS_PER_BUILD,
    TrophicRelation,
    compete,
    run_trophic_pass,
    update_trophic_levels,
)


ADD_CORRECT = "def add(a, b):\n    return a + b\n"
ADD_WRONG = "def add(a, b):\n    return a - b\n"
ADD_IDENTICAL = "def add(a, b):\n    return a + b\n"
ADD_BROKEN = "def add(a, b):\n    return 1/0\n"
ADD_SYNTAX_ERROR = "def add(a, b) return a + b\n"


BASIC_TESTS: list[dict] = [
    {"a": 1, "b": 2, "expected": 3},
    {"a": 0, "b": 0, "expected": 0},
    {"a": -1, "b": 1, "expected": 0},
]


# ---------------------------------------------------------------------------
# compete()
# ---------------------------------------------------------------------------


class TestCompete:
    def test_correct_vs_wrong_predation(self) -> None:
        rel = compete(ADD_CORRECT, ADD_WRONG, BASIC_TESTS)
        assert rel.relation == "predation"
        assert rel.predator_id == "a"
        assert rel.prey_id == "b"
        assert rel.passes_a == 3
        # ADD_WRONG gets "a-b" right only when a==b==0 (1/3 passes)
        assert rel.passes_b == 1
        assert rel.margin == pytest.approx(2 / 3)

    def test_wrong_vs_correct_marks_b_as_predator(self) -> None:
        rel = compete(ADD_WRONG, ADD_CORRECT, BASIC_TESTS)
        assert rel.predator_id == "b"
        assert rel.prey_id == "a"
        assert rel.margin == pytest.approx(2 / 3)

    def test_tie_is_competition(self) -> None:
        rel = compete(ADD_CORRECT, ADD_IDENTICAL, BASIC_TESTS)
        # A and B both pass all 3 tests; A+B combined also passes all 3
        # — no symbiosis margin either.
        assert rel.relation == "competition"
        assert rel.predator_id is None
        assert rel.prey_id is None

    def test_both_broken_is_competition(self) -> None:
        rel = compete(ADD_BROKEN, ADD_BROKEN, BASIC_TESTS)
        assert rel.passes_a == 0
        assert rel.passes_b == 0
        assert rel.relation == "competition"

    def test_syntax_error_counts_as_all_fail(self) -> None:
        rel = compete(ADD_SYNTAX_ERROR, ADD_CORRECT, BASIC_TESTS)
        assert rel.passes_a == 0
        assert rel.relation == "predation"
        assert rel.predator_id == "b"

    def test_no_test_inputs_returns_competition(self) -> None:
        rel = compete(ADD_CORRECT, ADD_WRONG, [])
        assert rel.relation == "competition"
        assert rel.passes_a == 0
        assert rel.passes_b == 0

    def test_custom_fragment_ids_propagate(self) -> None:
        rel = compete(
            ADD_CORRECT, ADD_WRONG, BASIC_TESTS,
            fragment_a_id="nutrient-42", fragment_b_id="nutrient-43",
            test_id="battery-1",
        )
        assert rel.predator_id == "nutrient-42"
        assert rel.prey_id == "nutrient-43"
        assert rel.test_id == "battery-1"

    def test_symbiosis_when_fragments_are_complementary(self) -> None:
        """A handles even a; B handles odd a. Neither alone beats the union."""
        a_even_only = (
            "def decide(a):\n"
            "    if a % 2 == 0:\n"
            "        return 'even'\n"
            "    raise ValueError('odd')\n"
        )
        b_odd_only = (
            "def decide(a):\n"
            "    if a % 2 == 1:\n"
            "        return 'odd'\n"
            "    raise ValueError('even')\n"
        )
        tests = [
            {"a": 2, "expected": "even"},
            {"a": 4, "expected": "even"},
            {"a": 1, "expected": "odd"},
            {"a": 3, "expected": "odd"},
        ]
        rel = compete(a_even_only, b_odd_only, tests)
        assert rel.passes_a == 2
        assert rel.passes_b == 2
        # Union passes 4/4 — strictly more than either alone
        assert rel.relation == "symbiosis"
        assert rel.predator_id is None
        assert rel.prey_id is None


# ---------------------------------------------------------------------------
# update_trophic_levels() + overlay
# ---------------------------------------------------------------------------


class _SoilStub:
    """Duck-typed stand-in for Soil — accepts the overlay attribute."""

    pass


class TestUpdateTrophicLevels:
    def test_predator_gets_trophic_boost(self) -> None:
        soil = _SoilStub()
        relations = [
            TrophicRelation(
                predator_id="n-1", prey_id="n-2",
                relation="predation", passes_a=3, passes_b=1, margin=0.67,
            )
        ]
        summary = update_trophic_levels(soil, relations)
        assert summary["promotions"] == 1
        overlay = getattr(soil, "_trophic_overlay", {})
        assert overlay["n-1"]["trophic_level"] == 1

    def test_prey_flagged_only_after_repeated_losses(self) -> None:
        soil = _SoilStub()
        losing = TrophicRelation(
            predator_id="n-p", prey_id="n-q",
            relation="predation", passes_a=3, passes_b=0, margin=1.0,
        )
        # One loss — below lapse threshold of 3 by default
        summary = update_trophic_levels(soil, [losing])
        assert summary["prey_demoted"] == 0
        # Three losses — hits threshold
        summary = update_trophic_levels(soil, [losing] * 3)
        assert summary["prey_demoted"] == 1
        overlay = soil._trophic_overlay
        assert overlay["n-q"]["lapsed"] is True


# ---------------------------------------------------------------------------
# run_trophic_pass() budgeting
# ---------------------------------------------------------------------------


class TestRunTrophicPass:
    def test_budget_respected(self) -> None:
        new_fragments = [
            {"id": "n-new", "code": ADD_CORRECT},
        ]
        # 10 potential competitors
        competitors = [
            {"id": f"n-c{i}", "code": ADD_WRONG} for i in range(10)
        ]
        relations = run_trophic_pass(
            new_fragments, competitors, BASIC_TESTS, max_competitions=3,
        )
        assert len(relations) == 3
        assert all(r.relation == "predation" for r in relations)

    def test_default_budget_is_spec(self) -> None:
        assert MAX_COMPETITIONS_PER_BUILD == 5

    def test_zero_competitors_is_noop(self) -> None:
        relations = run_trophic_pass(
            [{"id": "n-new", "code": ADD_CORRECT}], [], BASIC_TESTS,
        )
        assert relations == []

    def test_zero_new_fragments_is_noop(self) -> None:
        relations = run_trophic_pass(
            [], [{"id": "n-c", "code": ADD_WRONG}], BASIC_TESTS,
        )
        assert relations == []
