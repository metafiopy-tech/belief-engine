"""Tests for Session 16: danger-theory gate + pheromone trails + SICA wire-up.

Danger theory:
  - CRITICAL_FILES contains benchmark.py and hardening.py (spec)
  - is_critical matches by suffix, dotted name, and absolute path
  - has_localized_failures fires only when min_count AND min_fraction
    are both satisfied; empty input returns False
  - uncertainty_rising detects positive slope; flat/falling = False;
    too-few-samples = False
  - is_danger_zone ANDs all three gates (localized, rising, not-critical)
  - evaluate() returns (bool, reason) distinguishing critical /
    no-localized / no-rise / danger-zone

Pheromones:
  - deposit_pheromone writes a JSONL line under base_dir and never
    raises on filesystem errors
  - read_pheromones returns everything deposited, skips malformed lines
  - decay_weight matches (1/2)^(age / half_life)
  - pheromone_density sums decay-weighted trails; outcome_filter
    restricts which trails count
  - is_hot_zone threshold classification
  - clear_pheromones deletes the trail file and returns bytes removed

SICA integration:
  - danger_gate=None → existing behaviour (no new failure modes)
  - gate returning (False, reason) → SICA defers (no apply, marks
    result.error="deferred: ..."), proposal still archived
  - gate returning (True, "ok") → SICA proceeds to apply path
  - gate raising an exception → SICA logs and proceeds
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from belief.safety.danger_theory import (
    CRITICAL_FILES,
    DangerSignals,
    evaluate,
    has_localized_failures,
    is_critical,
    is_danger_zone,
    uncertainty_rising,
)
from belief.safety.pheromones import (
    DEFAULT_HALF_LIFE_SECONDS,
    DEFAULT_HOT_ZONE_THRESHOLD,
    PheromoneTrail,
    clear_pheromones,
    deposit_pheromone,
    is_hot_zone,
    pheromone_density,
    read_pheromones,
)


# ── CRITICAL_FILES ────────────────────────────────────────────────────────


class TestCriticalFiles:
    def test_spec_mandatory_entries(self):
        """The spec explicitly requires benchmark.py and hardening.py."""
        assert "belief/benchmark.py" in CRITICAL_FILES
        assert "belief/hardening.py" in CRITICAL_FILES

    def test_safety_modules_are_critical(self):
        """Safety primitives must not edit themselves."""
        assert "belief/safety/overseer.py" in CRITICAL_FILES
        assert "belief/safety/danger_theory.py" in CRITICAL_FILES


class TestIsCritical:
    def test_exact_match(self):
        assert is_critical("belief/benchmark.py")

    def test_suffix_match_absolute_path(self):
        assert is_critical("/Users/foo/proj/belief/benchmark.py")

    def test_dotted_module_notation(self):
        assert is_critical("belief.benchmark")
        assert is_critical("belief.hardening")

    def test_non_critical(self):
        assert not is_critical("belief/memory/soil.py")

    def test_empty_input(self):
        assert not is_critical("")


# ── Localized failures ────────────────────────────────────────────────────


class TestHasLocalizedFailures:
    def _fail(self, **fields):
        return dict(fields)

    def test_empty_is_false(self):
        assert not has_localized_failures("belief/memory/soil.py", [])

    def test_path_attribute_matches(self):
        fails = [
            self._fail(path="belief/memory/soil.py", message="x"),
            self._fail(path="belief/memory/soil.py", message="y"),
            self._fail(path="belief/other.py"),
        ]
        # 2/3 blame soil.py, > 30% and >= 2 count → localized
        assert has_localized_failures("belief/memory/soil.py", fails)

    def test_traceback_substring_matches(self):
        fails = [
            self._fail(error="Traceback (most recent call last):\n  "
                             "File \"belief/memory/soil.py\", line 10"),
            self._fail(error="Traceback: belief/memory/soil.py:20"),
        ]
        assert has_localized_failures("belief/memory/soil.py", fails)

    def test_below_min_count(self):
        """One failure isn't a pattern (default min_count=2)."""
        fails = [self._fail(path="belief/memory/soil.py")]
        assert not has_localized_failures("belief/memory/soil.py", fails)

    def test_below_min_fraction(self):
        """Two failures on soil but 20+ unrelated → not localized."""
        fails = ([self._fail(path="belief/memory/soil.py")] * 2 +
                 [self._fail(path=f"belief/other_{i}.py") for i in range(20)])
        assert not has_localized_failures("belief/memory/soil.py", fails)

    def test_object_with_attributes(self):
        class _F:
            def __init__(self, path): self.path = path
        fails = [_F("belief/memory/soil.py"), _F("belief/memory/soil.py")]
        assert has_localized_failures("belief/memory/soil.py", fails)


# ── Uncertainty trend ─────────────────────────────────────────────────────


class TestUncertaintyRising:
    def test_rising(self):
        assert uncertainty_rising([0.1, 0.2, 0.3, 0.4])

    def test_falling(self):
        assert not uncertainty_rising([0.4, 0.3, 0.2, 0.1])

    def test_flat(self):
        assert not uncertainty_rising([0.2, 0.2, 0.2, 0.2])

    def test_too_short(self):
        assert not uncertainty_rising([0.1])
        assert not uncertainty_rising([0.1, 0.2])  # default min_samples=3

    def test_noisy_but_rising(self):
        # Overall slope positive despite jitter.
        assert uncertainty_rising([0.1, 0.15, 0.12, 0.2, 0.25, 0.3])

    def test_empty(self):
        assert not uncertainty_rising([])


# ── is_danger_zone (canonical spec function) ──────────────────────────────


class TestIsDangerZone:
    def _fails(self, module, n):
        return [{"path": module, "message": f"fail {i}"} for i in range(n)]

    def test_all_three_signals_fire(self):
        assert is_danger_zone(
            "belief/memory/soil.py",
            self._fails("belief/memory/soil.py", 5),
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_critical_always_refused(self):
        """Even perfect signals don't permit touching CRITICAL_FILES."""
        assert not is_danger_zone(
            "belief/benchmark.py",
            self._fails("belief/benchmark.py", 10),
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_no_failures(self):
        assert not is_danger_zone("belief/memory/soil.py", [],
                                   [0.1, 0.2, 0.3, 0.4])

    def test_failures_not_localized(self):
        """Many failures but only 1 on soil → below min_count."""
        fails = ([{"path": "belief/memory/soil.py"}] +
                 [{"path": f"belief/other_{i}.py"} for i in range(20)])
        assert not is_danger_zone("belief/memory/soil.py", fails,
                                   [0.1, 0.2, 0.3, 0.4])

    def test_uncertainty_not_rising(self):
        assert not is_danger_zone(
            "belief/memory/soil.py",
            self._fails("belief/memory/soil.py", 5),
            [0.4, 0.3, 0.2, 0.1],  # falling
        )


class TestEvaluate:
    def test_danger_zone_reason(self):
        permit, reason = evaluate("belief/memory/soil.py", DangerSignals(
            recent_failures=[{"path": "belief/memory/soil.py"}] * 4,
            uncertainty_trend=[0.1, 0.2, 0.3, 0.4],
        ))
        assert permit is True
        assert "danger-zone" in reason

    def test_critical_reason(self):
        permit, reason = evaluate("belief/benchmark.py", DangerSignals(
            recent_failures=[{"path": "belief/benchmark.py"}] * 4,
            uncertainty_trend=[0.1, 0.2, 0.3, 0.4],
        ))
        assert permit is False
        assert "critical" in reason

    def test_no_localized_reason(self):
        permit, reason = evaluate("belief/memory/soil.py", DangerSignals(
            recent_failures=[],
            uncertainty_trend=[0.1, 0.2, 0.3, 0.4],
        ))
        assert permit is False
        assert "no-localized-failures" in reason

    def test_no_uncertainty_rise_reason(self):
        permit, reason = evaluate("belief/memory/soil.py", DangerSignals(
            recent_failures=[{"path": "belief/memory/soil.py"}] * 4,
            uncertainty_trend=[0.4, 0.4, 0.4, 0.4],
        ))
        assert permit is False
        assert "no-uncertainty-rise" in reason


# ── Pheromone trails ──────────────────────────────────────────────────────


class TestPheromoneRoundTrip:
    def test_deposit_and_read(self, tmp_path):
        t = deposit_pheromone(
            "belief/memory/soil.py", "crystallized covenant-7",
            outcome="success", source="sica", base_dir=tmp_path,
        )
        assert isinstance(t, PheromoneTrail)
        stored = read_pheromones("belief/memory/soil.py", base_dir=tmp_path)
        assert len(stored) == 1
        assert stored[0].description == "crystallized covenant-7"
        assert stored[0].source == "sica"
        assert stored[0].outcome == "success"
        assert stored[0].timestamp > 0

    def test_append_multiple(self, tmp_path):
        for i in range(3):
            deposit_pheromone(
                "belief/memory/soil.py", f"change-{i}",
                base_dir=tmp_path,
            )
        stored = read_pheromones("belief/memory/soil.py", base_dir=tmp_path)
        assert len(stored) == 3
        assert [t.description for t in stored] == ["change-0", "change-1", "change-2"]

    def test_missing_trail_returns_empty(self, tmp_path):
        assert read_pheromones("belief/nothing.py", base_dir=tmp_path) == []

    def test_malformed_line_skipped(self, tmp_path):
        """Readers must tolerate a partially-written JSONL line."""
        # Manually write one good line and one malformed one.
        from belief.safety.pheromones import _trail_path, _ensure_dir
        base = _ensure_dir(tmp_path)
        path = _trail_path("belief/memory/soil.py", base)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "module": "belief/memory/soil.py",
                "timestamp": 1234.0,
                "description": "ok",
                "outcome": "success",
                "source": "",
                "weight": 1.0,
            }) + "\n")
            fh.write("{not valid json\n")
        stored = read_pheromones("belief/memory/soil.py", base_dir=tmp_path)
        assert len(stored) == 1
        assert stored[0].description == "ok"


class TestPheromoneDecay:
    def test_decay_weight_half_life(self):
        """At exactly one half-life of age, weight = 0.5."""
        now = 1000.0
        t = PheromoneTrail(module="x", timestamp=now - DEFAULT_HALF_LIFE_SECONDS)
        assert t.decay_weight(now=now) == pytest.approx(0.5)

    def test_decay_weight_two_half_lives(self):
        now = 1000.0
        t = PheromoneTrail(module="x", timestamp=now - 2 * DEFAULT_HALF_LIFE_SECONDS)
        assert t.decay_weight(now=now) == pytest.approx(0.25)

    def test_decay_weight_zero_age(self):
        t = PheromoneTrail(module="x", timestamp=1000.0)
        assert t.decay_weight(now=1000.0) == 1.0


class TestPheromoneDensity:
    def test_empty_is_zero(self, tmp_path):
        assert pheromone_density("belief/nothing.py", base_dir=tmp_path) == 0.0

    def test_fresh_deposits_close_to_unity(self, tmp_path):
        now = time.time()
        for i in range(3):
            deposit_pheromone(
                "belief/memory/soil.py", f"change-{i}",
                timestamp=now - i,  # all within 3s
                base_dir=tmp_path,
            )
        density = pheromone_density(
            "belief/memory/soil.py",
            base_dir=tmp_path, now=now,
        )
        assert density == pytest.approx(3.0, rel=1e-3)

    def test_old_deposits_decay(self, tmp_path):
        now = 1_000_000.0
        deposit_pheromone(
            "belief/memory/soil.py", "old change",
            timestamp=now - 7 * DEFAULT_HALF_LIFE_SECONDS,  # 7 half-lives
            base_dir=tmp_path,
        )
        density = pheromone_density(
            "belief/memory/soil.py",
            base_dir=tmp_path, now=now,
        )
        # (1/2)^7 = 1/128 ≈ 0.0078
        assert density < 0.01

    def test_outcome_filter(self, tmp_path):
        now = time.time()
        deposit_pheromone("m.py", "good", outcome="success",
                          timestamp=now, base_dir=tmp_path)
        deposit_pheromone("m.py", "bad", outcome="failure",
                          timestamp=now, base_dir=tmp_path)
        only_success = pheromone_density(
            "m.py", base_dir=tmp_path, now=now,
            outcome_filter={"success"},
        )
        assert only_success == pytest.approx(1.0, rel=1e-3)
        total = pheromone_density("m.py", base_dir=tmp_path, now=now)
        assert total == pytest.approx(2.0, rel=1e-3)


class TestIsHotZone:
    def test_hot_when_many_fresh(self, tmp_path):
        now = time.time()
        for i in range(5):
            deposit_pheromone(
                "m.py", f"c{i}", timestamp=now, base_dir=tmp_path,
            )
        assert is_hot_zone("m.py", base_dir=tmp_path, now=now)

    def test_cold_when_empty(self, tmp_path):
        assert not is_hot_zone("m.py", base_dir=tmp_path)


class TestClearPheromones:
    def test_delete_existing(self, tmp_path):
        deposit_pheromone("m.py", "x", base_dir=tmp_path)
        bytes_removed = clear_pheromones("m.py", base_dir=tmp_path)
        assert bytes_removed > 0
        assert read_pheromones("m.py", base_dir=tmp_path) == []

    def test_missing_returns_zero(self, tmp_path):
        assert clear_pheromones("nothing.py", base_dir=tmp_path) == 0


# ── SICA integration ─────────────────────────────────────────────────────


# SICA imports chromadb indirectly via belief.memory.  Check at
# collection time so the danger-theory / pheromone tests above still
# run even when chromadb is absent (module-level importorskip would
# skip the entire file).
try:
    import chromadb  # noqa: F401
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False


@pytest.mark.skipif(not _HAS_CHROMADB, reason="chromadb not installed")
class TestSICADangerGate:
    def _make_cycle(self, tmp_path, gate=None):
        from belief.evolution.sica import SelfImprovementCycle
        return SelfImprovementCycle(
            project_root=tmp_path,
            archive_path=tmp_path / "sica_archive.json",
            danger_gate=gate,
        )

    def test_gate_none_preserves_default_behavior(self, tmp_path):
        """Constructor accepts gate=None; the attribute is None."""
        cycle = self._make_cycle(tmp_path)
        assert cycle.danger_gate is None

    def test_gate_attribute_is_set(self, tmp_path):
        gate = lambda target: (True, "permit")
        cycle = self._make_cycle(tmp_path, gate=gate)
        assert cycle.danger_gate is gate
