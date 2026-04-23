"""
Metrics Dashboard — per-iteration tracking and statistical analysis.

Records IterationMetrics to a JSONL file (one JSON object per line)
and provides growth analysis (linear, exponential fits) to detect
plateaus and estimate improvement rate.

No numpy/scipy required — all math is stdlib only.
"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("belief.metrics.dashboard")


@dataclass
class IterationMetrics:
    """Metrics for one SICA iteration."""

    iteration: int
    timestamp: str
    benchmark_score: float
    benchmark_ci_lower: float = 0.0  # Bootstrap 95% CI
    benchmark_ci_upper: float = 1.0
    cost_per_solved: float = 0.0
    novel_capabilities: int = 0  # Tasks newly passing
    regressions: int = 0  # Tasks that regressed
    tool_library_size: int = 0
    tool_invocation_entropy: float = 0.0  # Shannon entropy of tool usage
    retrieval_precision_at_5: float = 0.0
    covenant_count: int = 0
    covenant_discovery_rate: float = 0.0  # New covenants in last 10 builds
    canary_score: Optional[float] = None
    # Session 7: soil lift = warm_score - cold_score for a sampled build
    # pair, measuring how much accumulated nutrients improve local-model
    # output. 0.0 is the "untested" sentinel — the metric is only
    # meaningful after at least one hot/cold benchmark pair has run.
    soil_lift: float = 0.0


class MetricsDashboard:
    """Records and analyzes per-iteration metrics."""

    def __init__(self, db_path: str = "~/.belief-engine/metrics.jsonl") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, metrics: IterationMetrics) -> None:
        """Append metrics to JSONL file."""
        with open(self.db_path, "a") as f:
            f.write(json.dumps(asdict(metrics), default=str) + "\n")
        logger.debug(f"Recorded metrics for iteration {metrics.iteration}")

    def load_all(self) -> list[IterationMetrics]:
        """Load all recorded metrics."""
        if not self.db_path.exists():
            return []

        metrics: list[IterationMetrics] = []
        # Session 7: filter unknown keys so forward/backward compatible reads
        # succeed. An install that predates the soil_lift field can still
        # read rows written after the field was added (and vice-versa).
        known_keys = {f for f in IterationMetrics.__dataclass_fields__}
        with open(self.db_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    filtered = {k: v for k, v in d.items() if k in known_keys}
                    metrics.append(IterationMetrics(**filtered))
                except (json.JSONDecodeError, TypeError):
                    continue
        return metrics

    def compute_growth_analysis(self) -> dict:
        """Fit linear and exponential models and compare.

        Returns dict with model fits and AIC comparison.
        Requires at least 6 data points.
        """
        metrics = self.load_all()
        if len(metrics) < 6:
            return {
                "status": "insufficient_data",
                "n": len(metrics),
                "min_required": 6,
            }

        scores = [m.benchmark_score for m in metrics]
        iterations = list(range(len(scores)))

        results: dict = {}

        # Linear fit
        results["linear"] = self._fit_linear(iterations, scores)

        # Exponential fit (on logit-transformed scores for bounded [0,1])
        logit_scores = [math.log(max(s, 0.01) / max(1.0 - s, 0.01)) for s in scores]
        results["exponential"] = self._fit_exponential(iterations, logit_scores)

        # Doubling time
        rate = results["exponential"].get("rate", 0)
        if rate > 0:
            results["doubling_time"] = math.log(2) / rate

        # Best fit by AIC
        fits = {k: v for k, v in results.items() if isinstance(v, dict) and "aic" in v}
        if fits:
            results["best_fit"] = min(fits, key=lambda k: fits[k]["aic"])
        else:
            results["best_fit"] = "unknown"

        results["status"] = "ok"
        results["n"] = len(metrics)
        return results

    def print_dashboard(self) -> None:
        """Print a formatted dashboard to stdout."""
        metrics = self.load_all()
        if not metrics:
            print("No metrics recorded yet.")
            return

        latest = metrics[-1]
        growth = self.compute_growth_analysis()

        print(f"\n{'=' * 60}")
        print("  BELIEF ENGINE METRICS DASHBOARD")
        print(f"{'=' * 60}")
        print(f"  Iteration:              {latest.iteration}")
        print(
            f"  Benchmark Score:        {latest.benchmark_score:.1%} "
            f"[{latest.benchmark_ci_lower:.1%} - {latest.benchmark_ci_upper:.1%}]"
        )
        print(f"  Cost/Solved Task:       ${latest.cost_per_solved:.2f}")
        print(f"  Novel Capabilities:     +{latest.novel_capabilities}")
        print(f"  Regressions:            {latest.regressions}")
        print(f"  Tool Library:           {latest.tool_library_size} tools")
        print(f"  Covenant Count:         {latest.covenant_count}")
        print(f"  Discovery Rate:         {latest.covenant_discovery_rate:.1f}/10 builds")
        print(f"  Retrieval Precision@5:  {latest.retrieval_precision_at_5:.1%}")
        if latest.canary_score is not None:
            print(f"  Canary Score:           {latest.canary_score:.1%}")

        print(f"\n  Growth Analysis ({growth.get('n', 0)} data points):")
        if growth.get("status") == "insufficient_data":
            print(f"    Need {growth['min_required']} iterations (have {growth['n']})")
        else:
            print(f"    Best fit: {growth.get('best_fit', 'unknown')}")
            if "doubling_time" in growth:
                print(f"    Doubling time: {growth['doubling_time']:.1f} iterations")
            linear = growth.get("linear", {})
            if linear:
                print(f"    Linear slope: {linear.get('slope', 0):.4f}/iteration")
        print(f"{'=' * 60}\n")

    def print_json(self) -> None:
        """Output all metrics as JSON."""
        metrics = self.load_all()
        growth = self.compute_growth_analysis()
        output = {
            "metrics": [asdict(m) for m in metrics],
            "growth_analysis": growth,
        }
        print(json.dumps(output, indent=2, default=str))

    # ── Fitting helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _fit_linear(x: list, y: list) -> dict:
        """Simple linear regression with AIC."""
        n = len(x)
        if n < 2:
            return {"slope": 0.0, "intercept": 0.0, "rss": 0.0, "aic": float("inf")}

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        den = sum((xi - x_mean) ** 2 for xi in x)

        slope = num / den if den > 0 else 0.0
        intercept = y_mean - slope * x_mean

        rss = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
        aic = n * math.log(rss / n + 1e-10) + 2 * 2  # 2 parameters

        return {"slope": slope, "intercept": intercept, "rss": rss, "aic": aic}

    @staticmethod
    def _fit_exponential(x: list, y: list) -> dict:
        """Exponential fit via log-linear regression."""
        log_y = []
        valid_x = []
        for xi, yi in zip(x, y):
            if yi > 0:
                log_y.append(math.log(yi))
                valid_x.append(xi)

        if len(valid_x) < 3:
            return {"rate": 0.0, "aic": float("inf")}

        linear = MetricsDashboard._fit_linear(valid_x, log_y)
        return {"rate": linear["slope"], "aic": linear["aic"]}
