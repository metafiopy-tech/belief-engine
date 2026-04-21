"""Tests for Session 15: Active-Inference trigger + Assembly-Theory promotion.

Active inference:
  - should_trigger_jitterbug returns (bool, phase) matching the spec's
    threshold logic for contract / expand / equilibrium
  - pragmatic (pass-rate) pressure takes precedence over epistemic
    (novelty) pressure when both are above threshold
  - compute_pragmatic_pressure handles empty input ("cold start"
    defaults to maximum pragmatic pressure)
  - compute_epistemic_pressure returns novelty and pressure
  - EFESignalSource.from_archive produces usable signals from a
    duck-typed archive stub
  - EFETrigger is callable and returns the same (bool, phase) tuple

Assembly theory:
  - extract_signatures produces stable hashes; identical source ->
    identical sets; unparseable input -> empty set
  - assembly_index == 0.0 with empty library, empty tool, or
    unparseable source
  - assembly_index for truly shared structure is > 0.3; for
    structurally-distinct tools <= 0.1
  - tool is excluded from its own library (no self-reinforcement)
  - copy_numbers returns per-signature counts
  - should_promote covers all four quadrants (building_block,
    creative_novelty, unique_workhorse, trivial)
  - scan_library_for_promotions batch-runs should_promote
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from belief.evolution.active_inference import (
    DEFAULT_CONTRACT_THRESHOLD,
    DEFAULT_EXPAND_THRESHOLD,
    DEFAULT_NOVELTY_FLOOR,
    DEFAULT_TARGET_PASS_RATE,
    EFESignalSource,
    EFETrigger,
    compute_epistemic_pressure,
    compute_pragmatic_pressure,
    should_trigger_jitterbug,
)
from belief.evolution.assembly_theory import (
    DEFAULT_AI_HIGH,
    DEFAULT_USAGE_HIGH,
    PromotionVerdict,
    assembly_index,
    copy_numbers,
    extract_signatures,
    scan_library_for_promotions,
    should_promote,
)


# ── Active inference — pure function ──────────────────────────────────────


class TestShouldTriggerJitterbug:
    def test_contract_when_pass_rate_drops(self):
        assert should_trigger_jitterbug(
            recent_pass_rate=0.50, target_pass_rate=0.80,
            recent_novelty=0.50, archive_coverage=0.50,
        ) == (True, "contract")

    def test_expand_when_novelty_drops(self):
        assert should_trigger_jitterbug(
            recent_pass_rate=0.78, target_pass_rate=0.80,
            recent_novelty=0.05, archive_coverage=0.80,
        ) == (True, "expand")

    def test_equilibrium_when_both_healthy(self):
        assert should_trigger_jitterbug(
            recent_pass_rate=0.85, target_pass_rate=0.80,
            recent_novelty=0.50, archive_coverage=0.80,
        ) == (False, "equilibrium")

    def test_pragmatic_wins_over_epistemic(self):
        """When both pressures fire, contract runs first (spec order)."""
        assert should_trigger_jitterbug(
            recent_pass_rate=0.50, target_pass_rate=0.80,
            recent_novelty=0.05, archive_coverage=0.50,
        ) == (True, "contract")

    def test_pressure_below_threshold_is_equilibrium(self):
        # pragmatic = 0.8 - 0.66 = 0.14, below 0.15 threshold
        assert should_trigger_jitterbug(
            recent_pass_rate=0.66, target_pass_rate=0.80,
            recent_novelty=0.50, archive_coverage=0.50,
        ) == (False, "equilibrium")

    def test_pass_rate_above_target_does_not_fire(self):
        """No pragmatic pressure when the engine is beating its target."""
        assert should_trigger_jitterbug(
            recent_pass_rate=0.95, target_pass_rate=0.80,
            recent_novelty=0.50, archive_coverage=0.50,
        ) == (False, "equilibrium")

    def test_custom_thresholds_respected(self):
        # Loosen contract threshold to 0.30 — the drop from 0.80 to
        # 0.70 is now below threshold even though default would fire.
        assert should_trigger_jitterbug(
            recent_pass_rate=0.70, target_pass_rate=0.80,
            recent_novelty=0.50, archive_coverage=0.50,
            contract_threshold=0.30,
        ) == (False, "equilibrium")

    def test_defaults_exposed(self):
        """Constants are exported for callers who want to read them."""
        assert DEFAULT_CONTRACT_THRESHOLD == 0.15
        assert DEFAULT_EXPAND_THRESHOLD == 0.15
        assert DEFAULT_NOVELTY_FLOOR == 0.3
        assert DEFAULT_TARGET_PASS_RATE == 0.80


# ── Signal extraction helpers ─────────────────────────────────────────────


@dataclass
class _FakeResult:
    passed: bool
    score: float = 0.0


class TestPragmaticPressure:
    def test_mixed_results(self):
        rate, pressure = compute_pragmatic_pressure(
            [_FakeResult(True), _FakeResult(True),
             _FakeResult(False), _FakeResult(True)],
            target=0.80,
        )
        assert rate == 0.75
        assert pressure == pytest.approx(0.05)

    def test_all_pass_no_pressure(self):
        rate, pressure = compute_pragmatic_pressure(
            [_FakeResult(True), _FakeResult(True)], target=0.80,
        )
        assert rate == 1.0
        assert pressure == 0.0

    def test_empty_results_cold_start(self):
        """Empty history defaults to maximum pragmatic pressure."""
        rate, pressure = compute_pragmatic_pressure([], target=0.80)
        assert rate == 0.0
        assert pressure == pytest.approx(0.80)


class TestEpistemicPressure:
    def test_basic_ratio(self):
        novelty, pressure = compute_epistemic_pressure(
            recent_count=10, novel_count=2,
        )
        assert novelty == pytest.approx(0.2)
        assert pressure == pytest.approx(0.1)

    def test_saturated_novelty_no_pressure(self):
        novelty, pressure = compute_epistemic_pressure(
            recent_count=10, novel_count=5,
        )
        assert novelty == 0.5
        assert pressure == 0.0

    def test_empty_window_defaults_to_max_pressure(self):
        novelty, pressure = compute_epistemic_pressure(0, 0)
        assert novelty == 0.0
        assert pressure == pytest.approx(DEFAULT_NOVELTY_FLOOR)


class TestEFESignalSource:
    def test_from_archive_stub(self):
        class _Arc:
            def get_all_results_recent(self, n):
                return [_FakeResult(True, 0.3), _FakeResult(True, 0.9),
                        _FakeResult(False, 0.1), _FakeResult(True, 0.5)]

            def get_niche_map(self):
                return {"a": 1, "b": 2}

        src = EFESignalSource.from_archive(_Arc(), window=20)
        assert src.window == 4
        assert src.recent_pass_rate == 0.75
        assert src.archive_coverage > 0.0

    def test_from_archive_handles_errors(self):
        class _BrokenArc:
            def get_all_results_recent(self, n):
                raise RuntimeError("db down")

            def get_niche_map(self):
                raise RuntimeError("db down")

        src = EFESignalSource.from_archive(_BrokenArc(), window=10)
        assert src.recent_pass_rate == 0.0
        assert src.recent_novelty == 0.0
        assert src.archive_coverage == 0.0


class TestEFETrigger:
    def test_callable_returns_decision(self):
        trig = EFETrigger(source=EFESignalSource(
            recent_pass_rate=0.50, recent_novelty=0.50,
            archive_coverage=0.80, window=20,
        ))
        assert trig() == (True, "contract")

    def test_evaluate_and_call_match(self):
        src = EFESignalSource(recent_pass_rate=0.90, recent_novelty=0.50,
                              archive_coverage=0.80, window=20)
        trig = EFETrigger(source=src)
        assert trig.evaluate() == trig() == (False, "equilibrium")


# ── Assembly theory ───────────────────────────────────────────────────────


_PROCESS_TOOL = '''
def process(items):
    result = []
    for item in items:
        result.append(item)
    return result
'''

_COLLECT_TOOL = '''
def collect(items):
    result = []
    for item in items:
        result.append(item)
    return result
'''

_CLASS_TOOL = 'class X: pass\n'


class TestExtractSignatures:
    def test_deterministic(self):
        a = extract_signatures(_PROCESS_TOOL)
        b = extract_signatures(_PROCESS_TOOL)
        assert a == b
        assert len(a) > 0

    def test_unparseable_returns_empty(self):
        assert extract_signatures("def (( this is not valid python") == set()

    def test_empty_returns_empty(self):
        assert extract_signatures("") == set()

    def test_trivial_source_has_few_signatures(self):
        # With min_nodes=3, `pass` alone generates no substructures of
        # size ≥ 3 apart from the module/class roots themselves.
        sigs = extract_signatures(_CLASS_TOOL)
        assert len(sigs) <= 3


class TestAssemblyIndex:
    def test_shared_loop_structure(self):
        """Two tools with the same loop-and-append shape share substructure."""
        ai = assembly_index(_PROCESS_TOOL, [_COLLECT_TOOL])
        assert ai > 0.3, f"expected meaningful overlap, got {ai}"

    def test_unrelated_tools(self):
        ai = assembly_index(_CLASS_TOOL, [_PROCESS_TOOL, _COLLECT_TOOL])
        assert ai <= 0.1, f"unrelated tool should have tiny AI, got {ai}"

    def test_empty_library(self):
        assert assembly_index(_PROCESS_TOOL, []) == 0.0

    def test_unparseable_tool(self):
        assert assembly_index("def (( this is not valid python", [_PROCESS_TOOL]) == 0.0

    def test_tool_excluded_from_own_library(self):
        """A tool must not reinforce itself."""
        # The tool itself is in the library — only its presence as a
        # peer library entry should be filtered out.  With only the
        # tool itself in `library`, assembly_index must be 0.
        assert assembly_index(_PROCESS_TOOL, [_PROCESS_TOOL]) == 0.0

    def test_library_can_include_unparseable_entries(self):
        """Bad library entries don't crash the score."""
        ai = assembly_index(_PROCESS_TOOL, ["def (( bad syntax", _COLLECT_TOOL])
        assert 0.0 <= ai <= 1.0


class TestCopyNumbers:
    def test_basic_count(self):
        """When library contains three identical tools, shared signatures
        should have copy_number 3."""
        cn = copy_numbers(_PROCESS_TOOL, [_COLLECT_TOOL, _COLLECT_TOOL,
                                           _COLLECT_TOOL])
        assert any(v == 3 for v in cn.values())
        assert all(v >= 0 for v in cn.values())

    def test_unparseable_tool_returns_empty(self):
        assert copy_numbers("def (( this is not valid python", [_PROCESS_TOOL]) == {}


class TestShouldPromote:
    def test_building_block(self):
        v = should_promote(_PROCESS_TOOL, [_COLLECT_TOOL] * 5,
                           usage_count=DEFAULT_USAGE_HIGH + 1)
        assert isinstance(v, PromotionVerdict)
        assert v.category == "building_block"
        assert v.should_promote

    def test_creative_novelty(self):
        v = should_promote(_PROCESS_TOOL, [_COLLECT_TOOL] * 5, usage_count=1)
        assert v.category == "creative_novelty"
        assert not v.should_promote

    def test_unique_workhorse(self):
        v = should_promote(_CLASS_TOOL, [_PROCESS_TOOL, _COLLECT_TOOL],
                           usage_count=DEFAULT_USAGE_HIGH + 1)
        assert v.category == "unique_workhorse"
        assert v.should_promote

    def test_trivial(self):
        v = should_promote(_CLASS_TOOL, [_PROCESS_TOOL], usage_count=1)
        assert v.category == "trivial"
        assert not v.should_promote

    def test_rationale_is_populated(self):
        v = should_promote(_PROCESS_TOOL, [_COLLECT_TOOL] * 5, usage_count=10)
        assert v.rationale  # non-empty string
        assert "AI=" in v.rationale


class TestScanLibraryForPromotions:
    def test_batch_scan_classifies_each(self):
        tools = [
            ("a", _PROCESS_TOOL, 10),
            ("b", _COLLECT_TOOL, 1),
            ("c", _CLASS_TOOL, 1),
        ]
        results = scan_library_for_promotions(tools)
        assert len(results) == 3
        by_id = {tid: v for tid, v in results}
        # `a` with high usage and shared structure → building_block.
        assert by_id["a"].category == "building_block"
        assert by_id["a"].should_promote
        # `b` shares structure with a but usage is low → novelty.
        assert by_id["b"].category == "creative_novelty"
        # `c` is unrelated and low-usage → trivial.
        assert by_id["c"].category == "trivial"

    def test_empty_input(self):
        assert scan_library_for_promotions([]) == []
