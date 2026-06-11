"""Generate a LinkedIn-optimized PNG card for the substrate-transfer experiment.

Single 1200x1500 PNG (4:5 portrait — LinkedIn's optimal feed ratio) with:
  - Top: headline number ("4× improvement")
  - Middle: per-domain comparison chart (the headline visual)
  - Bottom: the dual-finding pull quote
  - Footer: meta (run id, model, cells)

Output:
    docs/experiments/substrate_transfer_linkedin_card.png

Usage:
    python3 scripts/generate_substrate_transfer_linkedin_card.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_EXP_ID = "subxfer-20260528-210730"
DEFAULT_DB = Path.home() / ".belief-engine" / "experiments.db"
DEFAULT_OUT = Path("docs/experiments/substrate_transfer_linkedin_card.png")


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
            "FROM results WHERE experiment_id = ?",
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


def render_card(results: list[dict], out_path: Path, exp_id: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # Compute the headline numbers
    raw_avg = _avg([r["score"] for r in results if r["condition"] == "raw_local"])
    full_b5_avg = _avg(
        [r["score"] for r in results if r["condition"] == "full" and r["build_seq"] == 5]
    )
    multiplier = full_b5_avg / raw_avg if raw_avg > 0 else 0

    # Per-domain numbers for the chart
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
    full_scores = [domain_score("full", 15, d) for d in domains]

    # Build the figure — 1200×1500 px at 150 DPI = 8×10 inches
    fig = plt.figure(figsize=(8, 10), dpi=150, facecolor="white")
    gs = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.3, 0.4, 3.3, 1.8],
        hspace=0.45,
        left=0.07,
        right=0.93,
        top=0.95,
        bottom=0.05,
    )

    # ── Section 1: Headline number ──
    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")
    ax_top.text(
        0.5,
        0.85,
        f"{multiplier:.1f}×",
        ha="center",
        va="top",
        fontsize=110,
        fontweight="bold",
        color="#1E88E5",
        transform=ax_top.transAxes,
    )
    ax_top.text(
        0.5,
        0.05,
        "improvement on Python software-engineering tasks",
        ha="center",
        va="bottom",
        fontsize=18,
        color="#333",
        transform=ax_top.transAxes,
    )

    # ── Section 2: Subtitle / context ──
    ax_sub = fig.add_subplot(gs[1])
    ax_sub.axis("off")
    ax_sub.text(
        0.5,
        0.7,
        "140 controlled builds. Local 14B-parameter code model.",
        ha="center",
        va="center",
        fontsize=13,
        color="#666",
        transform=ax_sub.transAxes,
    )
    ax_sub.text(
        0.5,
        0.25,
        "Three conditions × four problem domains × three accumulated-soil sizes.",
        ha="center",
        va="center",
        fontsize=13,
        color="#666",
        transform=ax_sub.transAxes,
    )

    # ── Section 3: Per-domain chart ──
    ax_chart = fig.add_subplot(gs[2])
    x = np.arange(len(domains))
    width = 0.38
    bars_raw = ax_chart.bar(x - width / 2, raw_scores, width, color="#888", label="Bare model")
    bars_full = ax_chart.bar(
        x + width / 2,
        full_scores,
        width,
        color="#1E88E5",
        label="With substrate (full engine)",
    )

    for bar_group in [bars_raw, bars_full]:
        for bar in bar_group:
            h = bar.get_height()
            ax_chart.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.03,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#333",
            )

    ax_chart.set_ylabel("Avg weighted score", fontsize=13)
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(domains, fontsize=12)
    ax_chart.set_ylim(0, 1.20)
    ax_chart.legend(loc="upper right", fontsize=11, frameon=False)
    ax_chart.grid(True, alpha=0.3, axis="y")
    ax_chart.spines["top"].set_visible(False)
    ax_chart.spines["right"].set_visible(False)
    ax_chart.set_title(
        "Per-domain performance — bare model vs. substrate",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # ── Section 4: Pull quote + footer ──
    ax_quote = fig.add_subplot(gs[3])
    ax_quote.axis("off")

    quote = (
        "The substrate transferred across Python software-engineering subdomains\n"
        "the engine wasn't trained on, but did not transfer to non-Python\n"
        "artifact paradigms (Sokoban, SMT-LIB, TLA+, crossword)."
    )
    ax_quote.text(
        0.5,
        0.85,
        quote,
        ha="center",
        va="top",
        fontsize=12,
        color="#222",
        style="italic",
        transform=ax_quote.transAxes,
        linespacing=1.5,
    )

    ax_quote.text(
        0.5,
        0.08,
        f"Belief Engine v3.3   |   qwen2.5-coder:14b   |   {len(results)} cells   |   run {exp_id}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#888",
        transform=ax_quote.transAxes,
    )

    # Background tint behind the headline (decorative)
    fig.patches.append(
        plt.Rectangle(
            (0, 0.78),
            1,
            0.22,
            transform=fig.transFigure,
            color="#F4F8FD",
            zorder=-10,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
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
        print(f"ERROR: no results for {args.exp_id!r}", file=sys.stderr)
        return 1

    render_card(results, Path(args.out), args.exp_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
