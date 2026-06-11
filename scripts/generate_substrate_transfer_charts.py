"""Generate the three substrate-transfer experiment charts as SVG.

Reads results from ~/.belief-engine/experiments.db for experiment id
``subxfer-20260528-210730`` (the full 140-build run) and produces three
publication-ready SVGs in docs/experiments/charts/:

  - chart1_learning_curve.svg   — data_pipeline avg score vs build_seq
  - chart2_component_attribution.svg — soil retrieval vs covenants+FSRS lift
  - chart3_per_domain.svg       — substrate lift by domain (headline chart)

Usage:
    python3 scripts/generate_substrate_transfer_charts.py
    python3 scripts/generate_substrate_transfer_charts.py --exp-id <id>

Renders with matplotlib (install with: pip3 install matplotlib).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_EXP_ID = "subxfer-20260528-210730"
DEFAULT_DB = Path.home() / ".belief-engine" / "experiments.db"


def _classify_domain(challenge_id: str) -> str:
    if challenge_id in (
        "t2-health-api",
        "t3-url-shortener",
        "t3-bookmark-api",
        "t3-notes-api",
        "t3-contact-api",
    ):
        return "Microservices"
    if challenge_id in (
        "t1-wordcount",
        "t2-todo-cli",
        "t2-calculator-cli",
        "t2-csv-stats",
        "t3-expense-tracker",
    ):
        return "CLI"
    if challenge_id in (
        "t3-data-pipeline",
        "t6-data-pipeline",
        "t5-workflow-engine",
        "t3-task-queue",
        "t3-schema-validator",
    ):
        return "Data pipeline"
    if challenge_id.startswith("novel-"):
        return "Novel artifact"
    return "Other"


def _load_results(db_path: Path, exp_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT condition, build_seq, challenge_id, passed, weighted_score "
            "FROM results WHERE experiment_id = ? "
            "ORDER BY condition, build_seq, challenge_id",
            (exp_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "condition": r[0],
            "build_seq": r[1],
            "challenge_id": r[2],
            "passed": bool(r[3]),
            "score": float(r[4]),
            "domain": _classify_domain(r[2]),
        }
        for r in rows
    ]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def chart1_learning_curve(results: list[dict], out_path: Path) -> None:
    """Data-pipeline avg score vs build_seq, one line per condition."""
    import matplotlib.pyplot as plt

    dp = [r for r in results if r["domain"] == "Data pipeline"]
    build_seqs = sorted({r["build_seq"] for r in dp if r["build_seq"] > 0})

    raw_score = _avg([r["score"] for r in dp if r["condition"] == "raw_local"])

    soil_scores = []
    full_scores = []
    for b in build_seqs:
        soil_scores.append(
            _avg([r["score"] for r in dp if r["condition"] == "soil_only" and r["build_seq"] == b])
        )
        full_scores.append(
            _avg([r["score"] for r in dp if r["condition"] == "full" and r["build_seq"] == b])
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    # Raw baseline as horizontal line
    ax.axhline(
        raw_score,
        color="#888",
        linestyle="--",
        linewidth=1.5,
        label=f"raw_local baseline ({raw_score:.2f})",
    )
    ax.plot(
        build_seqs,
        soil_scores,
        marker="o",
        linewidth=2.5,
        color="#FFA500",
        label="soil_only",
    )
    ax.plot(
        build_seqs,
        full_scores,
        marker="s",
        linewidth=2.5,
        color="#1E88E5",
        label="full engine",
    )

    ax.set_xlabel("Build sequence number (soil size)")
    ax.set_ylabel("Avg weighted score (data pipeline domain)")
    ax.set_title(
        "Learning curve — data pipeline domain only\n"
        "Soil accumulation correlates with engine improvement on data pipelines"
    )
    ax.set_xticks(build_seqs)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path}")


def chart2_component_attribution(results: list[dict], out_path: Path) -> None:
    """2-bar chart: soil retrieval contribution, covenants+FSRS contribution."""
    import matplotlib.pyplot as plt

    # Pool all build_seq for the substrate conditions
    raw_avg = _avg([r["score"] for r in results if r["condition"] == "raw_local"])
    soil_avg = _avg([r["score"] for r in results if r["condition"] == "soil_only"])
    full_avg = _avg([r["score"] for r in results if r["condition"] == "full"])

    soil_contribution = soil_avg - raw_avg
    covenants_contribution = full_avg - soil_avg

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Soil retrieval", "Covenants + FSRS"],
        [soil_contribution, covenants_contribution],
        color=["#FFA500", "#1E88E5"],
        width=0.6,
    )
    for bar, val in zip(bars, [soil_contribution, covenants_contribution]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"+{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("Average weighted score lift (vs raw_local)")
    ax.set_title(
        f"Component attribution — substrate contribution (n=120 substrate cells)\n"
        f"Raw baseline: {raw_avg:.3f}    |    Soil-only: {soil_avg:.3f}    |    Full: {full_avg:.3f}"
    )
    ax.set_ylim(0, max(soil_contribution, covenants_contribution) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path}")


def chart3_per_domain(results: list[dict], out_path: Path) -> None:
    """Grouped bar chart: per-domain avg score, one group per domain,
    3 bars (raw_local, soil_only b15, full b15) per group."""
    import matplotlib.pyplot as plt
    import numpy as np

    domains = ["Microservices", "CLI", "Data pipeline", "Novel artifact"]

    def domain_score(condition: str, build_seq: int | None, domain: str) -> float:
        return _avg(
            [
                r["score"]
                for r in results
                if r["condition"] == condition
                and r["domain"] == domain
                and (build_seq is None or r["build_seq"] == build_seq)
            ]
        )

    raw_scores = [domain_score("raw_local", None, d) for d in domains]
    soil_scores = [domain_score("soil_only", 15, d) for d in domains]
    full_scores = [domain_score("full", 15, d) for d in domains]

    x = np.arange(len(domains))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_raw = ax.bar(x - width, raw_scores, width, color="#888", label="raw_local")
    bars_soil = ax.bar(x, soil_scores, width, color="#FFA500", label="soil_only @ b15")
    bars_full = ax.bar(x + width, full_scores, width, color="#1E88E5", label="full @ b15")

    for bar_group in [bars_raw, bars_soil, bars_full]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Avg weighted score (per domain)")
    ax.set_title(
        "Substrate-transfer headline — per-domain performance at build_seq=15\n"
        "Substrate transferred across Python subdomains but not to novel-artifact paradigms"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-id",
        default=DEFAULT_EXP_ID,
        help=f"Experiment id (default: {DEFAULT_EXP_ID})",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help=f"DB path (default: {DEFAULT_DB})")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/charts",
        help="Output directory (default: docs/experiments/charts)",
    )
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "ERROR: matplotlib not installed. Run: pip3 install matplotlib",
            file=sys.stderr,
        )
        return 1

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 1

    results = _load_results(db_path, args.exp_id)
    if not results:
        print(
            f"ERROR: no results for experiment_id={args.exp_id!r} in {db_path}",
            file=sys.stderr,
        )
        return 1
    print(f"Loaded {len(results)} cells from {args.exp_id}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chart1_learning_curve(results, out_dir / "chart1_learning_curve.svg")
    chart2_component_attribution(results, out_dir / "chart2_component_attribution.svg")
    chart3_per_domain(results, out_dir / "chart3_per_domain.svg")

    print("\nAll three charts generated. View with:")
    print(f"  open {out_dir}/chart1_learning_curve.svg")
    print(f"  open {out_dir}/chart2_component_attribution.svg")
    print(f"  open {out_dir}/chart3_per_domain.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
