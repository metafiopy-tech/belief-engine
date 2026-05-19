"""Channel-capacity measurement harness (mycorrhizal Stage 4, Area 10).

The cell-signaling channel-capacity literature is unambiguous on a
counterintuitive point: per-event capacity is small. Cheong et al. 2011
(*Science* 334:354) measured ~0.92 bits per NF-κB pulse in mammalian
cells; ATF-2 alone yielded ≤ 0.85 bits per event. Selimkhanov et al. 2014
(*Science* 346:1370) showed *dynamics* — temporal trajectories — carry
significantly more capacity than scalar single-timepoint responses.
Nałęcz-Jawecki et al. 2023 (*PLOS Comput. Biol.*) pushed this to 6 bits/h
for the MAPK/ERK pathway with random-pulse-train probes.

The mycorrhizal brief proposes the Belief Engine treat its signal
protocol the same way: don't speculate about how much information the
five-token alphabet can carry; *measure* it. This module is the
measurement harness.

Methodology
-----------

The classical Shannon channel-capacity estimator:

    I(X; Y) = Σ_xy p(x, y) log2( p(x, y) / (p(x) p(y)) )

For a probe-driven measurement we treat the input distribution as known
(we choose the input sequence) and the output distribution as observed
(what the receiver reports). With small alphabets a binned plug-in
estimator is adequate — no need to pull in sklearn or scipy. The
estimator's bias on small samples is well-characterised; we report the
naive plug-in number and let the operator size the probe to reduce
bias if they want tighter bounds.

Caveat
------

The numbers reported here are mammalian-cell guidance, not direct
prescriptions. The brief itself flags this: "treat dynamics-over-events
as a strong default; treat the specific bit counts as benchmarks to
measure against, not targets to engineer toward."
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from belief.signal.alphabet import Signal, SignalToken
from belief.signal.store import (
    SignalStore,
    get_default_store,
)

logger = logging.getLogger("belief.signal.capacity")


# ── Target benchmark figures ───────────────────────────────────────────────

#: Cell-signaling per-event ceiling — Cheong et al. 2011 NF-κB figure.
#: Reference value only; the Belief Engine should not engineer toward it.
BENCHMARK_PER_EVENT_BITS = 0.92

#: Biology's best engineered channel — MAPK/ERK random-pulse-train probes
#: (Nałęcz-Jawecki et al. 2023). Reported in bits/hour.
BENCHMARK_BITS_PER_HOUR = 6.0


# ── Report type ────────────────────────────────────────────────────────────


@dataclass
class CapacityReport:
    """Result of one capacity-measurement run.

    Fields:
      mutual_information_bits — naive plug-in MI over the probe.
      duration_seconds         — wall-clock the probe ran for.
      bits_per_hour            — extrapolated rate.
      input_emissions          — how many signals the probe injected.
      sample_count             — input/output pairs the estimator saw.
      bin_count                — discretization granularity for MI.
      probe_seed               — seed used; lets the run be replayed.
      benchmark_bits_per_hour  — biology comparable, for context only.
      bias_warning             — set if sample/bin ratio is too small.
    """

    mutual_information_bits: float
    duration_seconds: float
    bits_per_hour: float
    input_emissions: int
    sample_count: int
    bin_count: int
    probe_seed: int
    benchmark_bits_per_hour: float = BENCHMARK_BITS_PER_HOUR
    bias_warning: Optional[str] = None
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mutual_information_bits": round(self.mutual_information_bits, 6),
            "duration_seconds": round(self.duration_seconds, 3),
            "bits_per_hour": round(self.bits_per_hour, 4),
            "input_emissions": self.input_emissions,
            "sample_count": self.sample_count,
            "bin_count": self.bin_count,
            "probe_seed": self.probe_seed,
            "benchmark_bits_per_hour": self.benchmark_bits_per_hour,
            "bias_warning": self.bias_warning,
            "notes": self.notes,
        }


# ── Plug-in MI estimator ───────────────────────────────────────────────────


def _discretize(values: list[float], bins: int) -> list[int]:
    """Map continuous magnitudes to integer bins in [0, bins).

    Assumes values are in [0, 1] (the magnitude range). Out-of-range
    values are clipped; a value of exactly 1.0 maps to ``bins - 1``.
    """
    if bins < 2:
        raise ValueError(f"need at least 2 bins, got {bins}")
    out: list[int] = []
    for v in values:
        if v < 0.0:
            v = 0.0
        elif v >= 1.0:
            out.append(bins - 1)
            continue
        out.append(int(v * bins))
    return out


def mutual_information_bits(xs: list[float], ys: list[float], bins: int = 4) -> float:
    """Naive plug-in MI estimator over binned (x, y) pairs.

    Returns the discrete I(X; Y) in bits. Bias is positive on small
    samples; for our purposes a few thousand pairs are enough for
    1-bit-scale comparisons.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: len(xs)={len(xs)} len(ys)={len(ys)}")
    n = len(xs)
    if n == 0:
        return 0.0
    xb = _discretize(xs, bins)
    yb = _discretize(ys, bins)
    # Joint histogram.
    joint: dict[tuple[int, int], int] = {}
    px: dict[int, int] = {}
    py: dict[int, int] = {}
    for a, b in zip(xb, yb):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        px[a] = px.get(a, 0) + 1
        py[b] = py.get(b, 0) + 1
    mi = 0.0
    for (a, b), c_ab in joint.items():
        p_ab = c_ab / n
        p_a = px[a] / n
        p_b = py[b] / n
        # By construction p_a > 0 and p_b > 0 for any pair we've seen.
        mi += p_ab * math.log2(p_ab / (p_a * p_b))
    # Clip tiny negatives from floating-point noise.
    return max(0.0, mi)


# ── Probe runner ───────────────────────────────────────────────────────────


@dataclass
class CapacityMeasurement:
    """Inject controlled signals into a store, read out concentrations,
    compute mutual information.

    Default probe: STRESS magnitudes drawn from a uniform discrete set
    over four levels, emitted at fixed intervals; after a settling delay
    the receiver-side concentration is read back. The MI between the
    input level (4 values) and the discretized output concentration is
    the measured per-event channel capacity for STRESS.

    The probe deliberately uses a *much shorter* half-life than the
    production defaults (``DEFAULT_HALF_LIFE = 2m``) — its emissions are
    spaced microseconds apart in synthetic time, and we want each new
    emission's contribution to dominate the readout (the channel-capacity
    question is "how much input information survives the receiver's
    integration step?"). With production-scale half-lives, the
    concentration becomes a running sum, accumulation dominates, and
    measured capacity collapses to near zero — that's a probe design
    bug, not a property of the channel. Operators measuring
    production-scale capacity should construct their own measurement
    with the actual receiver's half-life and emission cadence.
    """

    store: Optional[SignalStore] = None
    bins: int = 4
    probe_agent_id: str = "capacity-probe"
    probe_token: SignalToken = "STRESS"
    # Probe-scale defaults: half_life << inter-emission gap so the
    # receiver's read-back is dominated by the just-emitted magnitude.
    half_life: timedelta = timedelta(microseconds=100)
    read_window: timedelta = timedelta(milliseconds=10)

    def _store(self) -> SignalStore:
        return self.store if self.store is not None else get_default_store()

    def measure(
        self,
        n_samples: int = 200,
        interval_seconds: float = 0.0,
        seed: int = 1,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> CapacityReport:
        """Run the probe and return a ``CapacityReport``.

        ``interval_seconds`` lets the harness pace its emissions; for
        unit tests we use 0 (back-to-back synthetic emissions with
        explicit timestamps). The ``clock`` callable is for tests that
        want deterministic timing without sleeping.
        """
        store = self._store()
        rng = random.Random(seed)
        levels = [(i + 0.5) / self.bins for i in range(self.bins)]  # midpoints
        emitted: list[float] = []
        observed: list[float] = []
        t0 = time.monotonic()
        clock_fn = clock or (lambda: datetime.now(timezone.utc))
        base_ts = clock_fn()

        for i in range(n_samples):
            x = rng.choice(levels)
            ts = base_ts + timedelta(milliseconds=i)
            # Emit at the probe timestamp.
            sig = Signal(
                agent_id=self.probe_agent_id,
                token=self.probe_token,
                magnitude=x,
                timestamp=ts,
                idempotency_key=f"probe:{seed}:{i}",
            )
            store.emit(sig)
            # Read receiver concentration just after the emission with
            # the configured half_life and window. Concentration close
            # to the input magnitude implies high channel capacity.
            y = store.concentration(
                self.probe_agent_id,
                self.probe_token,
                window=self.read_window,
                half_life=self.half_life,
                now=ts + timedelta(microseconds=1),
            )
            # Normalise back into [0, 1] by clipping; the absolute scale
            # depends on accumulation but the *information* in the
            # relationship is what the MI estimator extracts.
            y = min(y, 1.0)
            emitted.append(x)
            observed.append(y)
            if interval_seconds > 0:
                time.sleep(interval_seconds)

        duration = max(time.monotonic() - t0, 1e-6)
        mi = mutual_information_bits(emitted, observed, bins=self.bins)
        # Bias check: with `bins**2` joint cells, we want >> that many
        # samples for the plug-in estimator to behave.
        bias_warning: Optional[str] = None
        if n_samples < self.bins * self.bins * 10:
            bias_warning = (
                f"sample_count={n_samples} is small relative to "
                f"bin_count^2={self.bins * self.bins}; MI estimate is "
                f"upward-biased — increase n_samples for tighter bounds"
            )
        return CapacityReport(
            mutual_information_bits=mi,
            duration_seconds=duration,
            bits_per_hour=mi * 3600.0 / duration if duration > 0 else 0.0,
            input_emissions=n_samples,
            sample_count=len(emitted),
            bin_count=self.bins,
            probe_seed=seed,
            bias_warning=bias_warning,
            notes={
                "probe_token": self.probe_token,
                "half_life_seconds": self.half_life.total_seconds(),
                "read_window_seconds": self.read_window.total_seconds(),
            },
        )


# ── CLI rendering ──────────────────────────────────────────────────────────


def cli_format_report(report: CapacityReport) -> str:
    lines = [
        "Signal channel-capacity probe",
        f"  probe_seed:           {report.probe_seed}",
        f"  emissions:            {report.input_emissions}",
        f"  bin_count:            {report.bin_count}",
        f"  duration:             {report.duration_seconds:.3f}s",
        "",
        f"  mutual information:   {report.mutual_information_bits:.4f} bits",
        f"  extrapolated rate:    {report.bits_per_hour:.2f} bits/hour",
        "",
        f"  per-event benchmark:  {BENCHMARK_PER_EVENT_BITS:.2f} bits (Cheong 2011 NF-κB)",
        f"  per-hour benchmark:   {report.benchmark_bits_per_hour:.2f} "
        f"bits/hour (Nałęcz-Jawecki 2023 MAPK/ERK)",
    ]
    if report.bias_warning:
        lines.append("")
        lines.append(f"  ⚠ {report.bias_warning}")
    return "\n".join(lines)


def cli_run_capacity(n_samples: int = 500, seed: int = 1) -> str:
    """`belief signal capacity` entry point."""
    measurement = CapacityMeasurement()
    report = measurement.measure(n_samples=n_samples, seed=seed)
    return cli_format_report(report)
