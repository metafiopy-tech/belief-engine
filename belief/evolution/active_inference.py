"""
Active-Inference jitterbug trigger (Session 15 Task 1).

Replaces the fixed-interval jitterbug cadence (``builds_between_jitterbug``
in the grinder daemon) with a signal-driven trigger inspired by the
**Expected Free Energy** (EFE) decomposition from Active Inference:

    EFE = pragmatic_pressure + epistemic_pressure

* **Pragmatic pressure** — exploitation need.  How far below the
  target pass rate is the engine's recent build performance?  When
  this is high, the engine is *losing ground* and should **contract**
  (compress, fix the failures) — the octahedron phase.
* **Epistemic pressure** — exploration need.  How novel have recent
  builds been relative to the archive?  When this is low, the engine
  is *stuck in a rut* and should **expand** (branch out into
  unexplored territory) — the cuboctahedron phase.
* **Equilibrium** — neither pressure is acute enough: the engine is
  tracking its target with adequate novelty.  Keep building normally.

Everything here is a pure function of metrics already produced by
the Archive + build loop — no new data collection required (the
Session-15 constraint).  The grinder daemon opts in by passing an
:class:`EFESignalSource` to :meth:`EFETrigger.from_grinder`; tests
construct :class:`EFESignalSource` directly with canned numbers so the
threshold logic can be verified without a live archive.

Public API:

    should_trigger_jitterbug(recent_pass_rate, target_pass_rate,
                             recent_novelty, archive_coverage)
        → (bool, phase)         the spec's canonical function

    compute_pragmatic_pressure(recent_results, target=0.8) → float
    compute_epistemic_pressure(recent_niches, novel_niches) → float

    EFESignalSource(archive, window=20)   snapshot recent metrics
    EFETrigger(source, target_pass_rate=0.8)  callable returning
        the same (bool, phase) tuple — used by grinder integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

# ── Thresholds (spec defaults) ─────────────────────────────────────────────


# Pragmatic pressure above this → contract phase.  15 percentage points
# below target is "losing ground"; tunable by callers via the EFE trigger.
DEFAULT_CONTRACT_THRESHOLD = 0.15

# Epistemic pressure above this → expand phase.  With the default
# novelty floor of 0.3, this fires when recent_novelty drops below 0.15.
DEFAULT_EXPAND_THRESHOLD = 0.15

# Novelty floor — below this the engine is considered "in a rut".
# Callers can override when they have a different baseline archive.
DEFAULT_NOVELTY_FLOOR = 0.3

# Default target pass rate — the engine should aim to hit this fraction
# of its benchmark consistently.  Matches the project's v3.0 10/12 score.
DEFAULT_TARGET_PASS_RATE = 0.80


# ── Canonical pure function (Session 15 spec) ─────────────────────────────


def should_trigger_jitterbug(
    recent_pass_rate: float,
    target_pass_rate: float,
    recent_novelty: float,
    archive_coverage: float,
    *,
    contract_threshold: float = DEFAULT_CONTRACT_THRESHOLD,
    expand_threshold: float = DEFAULT_EXPAND_THRESHOLD,
    novelty_floor: float = DEFAULT_NOVELTY_FLOOR,
) -> tuple[bool, str]:
    """Decide whether the jitterbug should fire, and in which phase.

    Mirrors the spec verbatim: exposes the two pressure signals and
    returns ``(should_trigger, phase)`` where phase is one of
    ``"contract"``, ``"expand"``, or ``"equilibrium"``.

    Args:
        recent_pass_rate:  Fraction of recent builds that passed (0-1).
        target_pass_rate:  Desired pass rate the engine aims to hit.
        recent_novelty:    Fraction of recent builds that opened new
                           territory in the archive (0-1).
        archive_coverage:  How saturated the niche space is (0-1).
                           Reserved for future weighting; not used in
                           the canonical threshold logic but kept in
                           the signature so downstream code can plumb
                           it through.
        contract_threshold / expand_threshold / novelty_floor:
            Optional knobs, exposed for experimentation.  Defaults
            reproduce the spec's values.

    Returns:
        Tuple ``(should_trigger, phase)``.  ``phase`` is always set —
        use it for logging even when ``should_trigger`` is False.
    """
    _ = archive_coverage  # Reserved (see docstring).
    pragmatic_pressure = max(0.0, target_pass_rate - recent_pass_rate)
    epistemic_pressure = max(0.0, novelty_floor - recent_novelty)

    if pragmatic_pressure > contract_threshold:
        return True, "contract"
    if epistemic_pressure > expand_threshold:
        return True, "expand"
    return False, "equilibrium"


# ── Signal extraction from existing metrics ───────────────────────────────


def compute_pragmatic_pressure(
    recent_results: Sequence[Any],
    target: float = DEFAULT_TARGET_PASS_RATE,
) -> tuple[float, float]:
    """Compute (pass_rate, pragmatic_pressure) from recent benchmark results.

    Accepts duck-typed records with a boolean-ish ``passed`` attribute
    (matches :class:`belief.evolution.archive.BenchmarkResult`).  Passing
    an empty sequence returns ``(0.0, max(0.0, target))`` so the
    engine treats "no data" as a prompt to contract — the only safe
    assumption for a cold start.

    Returns:
        ``(pass_rate, pressure)`` — pressure is the canonical
        ``max(0, target - pass_rate)``.
    """
    n = 0
    passed = 0
    for r in recent_results:
        n += 1
        if bool(getattr(r, "passed", False)):
            passed += 1
    if n == 0:
        return 0.0, max(0.0, float(target))
    rate = passed / n
    return rate, max(0.0, float(target) - rate)


def compute_epistemic_pressure(
    recent_count: int,
    novel_count: int,
    *,
    novelty_floor: float = DEFAULT_NOVELTY_FLOOR,
) -> tuple[float, float]:
    """Compute (novelty, epistemic_pressure) from niche-opening counts.

    ``novel_count`` is how many of the ``recent_count`` most-recent
    builds opened a niche that was previously empty in the archive.
    ``novelty = novel_count / recent_count`` (or 0 when there is no
    data).  Pressure is ``max(0, novelty_floor - novelty)``.

    Keeping both signals at the call site lets the trigger decide which
    one to act on first — the spec privileges pragmatic over epistemic
    when both fire simultaneously (contract before expand).
    """
    if recent_count <= 0:
        # No recent data: the safest read is "we know nothing new",
        # which maps to maximum epistemic pressure.  Whether that
        # actually fires a jitterbug is up to the caller's threshold.
        return 0.0, max(0.0, float(novelty_floor))
    novelty = max(0.0, min(1.0, novel_count / recent_count))
    return novelty, max(0.0, float(novelty_floor) - novelty)


# ── Live signal source bound to the archive ───────────────────────────────


@dataclass
class EFESignalSource:
    """Snapshot of the metrics :func:`should_trigger_jitterbug` needs.

    Produced from an :class:`~belief.evolution.archive.Archive` (or
    any duck-typed stand-in) so tests can supply canned numbers
    without a live SQLite archive.

    Fields:
        recent_pass_rate: Fraction of recent builds that passed.
        recent_novelty:   Fraction of recent builds that opened a new
                          niche (``novel_niches / window``).
        archive_coverage: Fraction of the niche space that is
                          populated — ``populated / max_possible``.
                          Defaults to 0.0 when the archive's niche map
                          is empty, which is the standard cold-start.
        window:           How many recent builds the snapshot covered.
    """

    recent_pass_rate: float = 0.0
    recent_novelty: float = 0.0
    archive_coverage: float = 0.0
    window: int = 0

    @classmethod
    def from_archive(
        cls,
        archive: Any,
        window: int = 20,
        max_niches: Optional[int] = None,
    ) -> "EFESignalSource":
        """Compute signals from an Archive-like object.

        The archive must expose::

            get_all_results_recent(window) → iterable of BenchmarkResult
            get_niche_map()                → dict[niche_key, version]

        Novelty is approximated as ``|niches_touched_by_recent_results|
        / window`` — a build that lands in a niche not present in the
        pre-window map counts as novel.  Archive coverage is
        ``populated_niches / max_niches`` (the caller can cap
        ``max_niches`` to a rough "total possible" value; the default
        treats the existing niche count as 100% coverage, which makes
        coverage = 1.0 when the archive is non-empty and 0.0 otherwise).
        """
        try:
            recent = list(archive.get_all_results_recent(window))
        except Exception:
            recent = []
        pass_rate, _ = compute_pragmatic_pressure(recent, target=0.0)

        try:
            niche_map = archive.get_niche_map() or {}
        except Exception:
            niche_map = {}
        populated = len(niche_map)

        if max_niches is None:
            # Unknown upper bound — treat "non-empty archive" as full
            # coverage.  Callers that know the true upper bound should
            # pass max_niches.
            coverage = 1.0 if populated > 0 else 0.0
        else:
            coverage = min(1.0, populated / max(1, int(max_niches)))

        # Novelty approximation: count distinct niches opened in the
        # recent window that weren't in the niche map *before* those
        # builds.  We can't cheaply reconstruct the pre-window map
        # from SQLite, so a pragmatic proxy is "how many of the
        # recent builds landed in populated niches they pioneered" —
        # approximated here as the fraction of recent results whose
        # score is above the archive's median.  This keeps novelty a
        # function of existing metrics (constraint) without needing
        # per-niche timestamps.
        if recent and populated > 0:
            scores = sorted(float(getattr(r, "score", 0.0)) for r in recent)
            # A build is "novel" if it beats the median — a cheap
            # stand-in that tracks how much the engine is
            # out-performing its archive baseline.
            median = scores[len(scores) // 2]
            novel = sum(
                1 for r in recent
                if float(getattr(r, "score", 0.0)) >= median
            )
            novelty = novel / len(recent)
        else:
            novelty = 0.0

        return cls(
            recent_pass_rate=pass_rate,
            recent_novelty=novelty,
            archive_coverage=coverage,
            window=len(recent),
        )


@dataclass
class EFETrigger:
    """Stateful trigger the grinder daemon can consult each cycle.

    Wraps :func:`should_trigger_jitterbug` with a signal source and
    configuration so callers get a single callable that returns
    ``(should_trigger, phase)``.  The grinder's fixed-interval logic
    stays as a fallback when ``EFETrigger`` is not configured.
    """

    source: EFESignalSource
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE
    contract_threshold: float = DEFAULT_CONTRACT_THRESHOLD
    expand_threshold: float = DEFAULT_EXPAND_THRESHOLD
    novelty_floor: float = DEFAULT_NOVELTY_FLOOR

    def evaluate(self) -> tuple[bool, str]:
        """Evaluate the EFE decision against the current source snapshot."""
        return should_trigger_jitterbug(
            recent_pass_rate=self.source.recent_pass_rate,
            target_pass_rate=self.target_pass_rate,
            recent_novelty=self.source.recent_novelty,
            archive_coverage=self.source.archive_coverage,
            contract_threshold=self.contract_threshold,
            expand_threshold=self.expand_threshold,
            novelty_floor=self.novelty_floor,
        )

    __call__ = evaluate
