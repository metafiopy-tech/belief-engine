"""Generate a single self-contained HTML brief of the substrate-transfer experiment.

Reads results from ~/.belief-engine/experiments.db, generates the 3 charts as
inline SVG (no separate files needed), and embeds them into an HTML template
along with the writeup prose. Output is one .html file you can open in a
browser, share as a link, or copy-paste into a blog post.

Usage:
    python3 scripts/generate_substrate_transfer_brief.py

Output:
    docs/experiments/substrate_transfer_brief.html

Dependencies:
    matplotlib  (pip3 install matplotlib)
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
from pathlib import Path

DEFAULT_EXP_ID = "subxfer-20260528-210730"
DEFAULT_DB = Path.home() / ".belief-engine" / "experiments.db"
DEFAULT_OUT = Path("docs/experiments/substrate_transfer_brief.html")


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


def _fig_to_inline_svg(fig) -> str:
    """Render a matplotlib Figure to an inline SVG string suitable for HTML embedding."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    # Strip XML declaration and DOCTYPE — they're invalid inside HTML body.
    lines = svg.splitlines()
    out_lines = []
    skipping = True
    for line in lines:
        if skipping and line.lstrip().startswith("<svg"):
            skipping = False
        if not skipping:
            out_lines.append(line)
    return "\n".join(out_lines)


def _chart_learning_curve(results: list[dict]) -> str:
    import matplotlib.pyplot as plt

    dp = [r for r in results if r["domain"] == "Data pipeline"]
    build_seqs = sorted({r["build_seq"] for r in dp if r["build_seq"] > 0})
    raw_score = _avg([r["score"] for r in dp if r["condition"] == "raw_local"])
    soil_scores = [
        _avg([r["score"] for r in dp if r["condition"] == "soil_only" and r["build_seq"] == b])
        for b in build_seqs
    ]
    full_scores = [
        _avg([r["score"] for r in dp if r["condition"] == "full" and r["build_seq"] == b])
        for b in build_seqs
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(
        raw_score,
        color="#888",
        linestyle="--",
        linewidth=1.5,
        label=f"raw_local baseline ({raw_score:.2f})",
    )
    ax.plot(build_seqs, soil_scores, marker="o", linewidth=2.5, color="#E89B23", label="soil_only")
    ax.plot(
        build_seqs, full_scores, marker="s", linewidth=2.5, color="#1E88E5", label="full engine"
    )
    ax.set_xlabel("Build sequence number (soil size)")
    ax.set_ylabel("Avg weighted score")
    ax.set_title("Learning curve — data pipeline domain only")
    ax.set_xticks(build_seqs)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    svg = _fig_to_inline_svg(fig)
    plt.close(fig)
    return svg


def _chart_component_attribution(results: list[dict]) -> tuple[str, float, float, float]:
    import matplotlib.pyplot as plt

    raw_avg = _avg([r["score"] for r in results if r["condition"] == "raw_local"])
    soil_avg = _avg([r["score"] for r in results if r["condition"] == "soil_only"])
    full_avg = _avg([r["score"] for r in results if r["condition"] == "full"])
    soil_contribution = soil_avg - raw_avg
    covenants_contribution = full_avg - soil_avg

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ["Soil retrieval", "Covenants + FSRS"],
        [soil_contribution, covenants_contribution],
        color=["#E89B23", "#1E88E5"],
        width=0.5,
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
    ax.set_ylabel("Avg weighted score lift (vs raw_local)")
    ax.set_title("Component attribution — substrate contribution")
    ax.set_ylim(0, max(soil_contribution, covenants_contribution) * 1.35)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    svg = _fig_to_inline_svg(fig)
    plt.close(fig)
    return svg, raw_avg, soil_avg, full_avg


def _chart_per_domain(results: list[dict]) -> str:
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
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_raw = ax.bar(x - width, raw_scores, width, color="#888", label="raw_local")
    bars_soil = ax.bar(x, soil_scores, width, color="#E89B23", label="soil_only @ b15")
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
    ax.set_title("Per-domain performance at build_seq=15")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    svg = _fig_to_inline_svg(fig)
    plt.close(fig)
    return svg


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Substrate-Transfer Experiment — Findings</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    max-width: 760px;
    margin: 2em auto;
    padding: 0 1.2em;
    color: #222;
    line-height: 1.55;
    font-size: 16px;
  }}
  h1 {{ font-size: 1.9em; margin-top: 0; }}
  h2 {{ font-size: 1.35em; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  h3 {{ font-size: 1.1em; margin-top: 1.5em; }}
  .meta {{ color: #666; font-size: 0.9em; margin-top: -0.5em; margin-bottom: 1.5em; }}
  table {{
    border-collapse: collapse;
    margin: 1em 0;
    font-size: 0.95em;
  }}
  th, td {{ padding: 0.4em 0.9em; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f5f5f5; }}
  .chart {{ margin: 2em 0; text-align: center; }}
  .chart svg {{ max-width: 100%; height: auto; }}
  .caption {{ color: #555; font-size: 0.9em; margin-top: 0.5em; font-style: italic; }}
  .pullquote {{
    border-left: 4px solid #1E88E5;
    background: #f8f9fc;
    padding: 1em 1.2em;
    margin: 1.5em 0;
    font-size: 1.05em;
    font-style: italic;
  }}
  code {{ background: #f3f3f3; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.92em; }}
  .lead {{ font-size: 1.1em; color: #333; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 2.5em 0; }}
  .pos {{ color: #2E7D32; font-weight: 600; }}
  .neg {{ color: #C62828; font-weight: 600; }}
</style>
</head>
<body>

<h1>Did the engineering loop transfer?</h1>
<p class="meta">
  Substrate-transfer experiment — 140 controlled builds on local qwen2.5-coder:14b<br>
  Run <code>subxfer-20260528-210730</code> | 2026-05-28 → 2026-05-29 | ~25 hours wall-clock | $7.62 cost
</p>

<p class="lead">
The Belief Engine — an iterative substrate built on top of a 14B-parameter local code model — was tested against the bare model across 140 controlled builds spanning four problem domains. The headline result: <span class="pos">a 4× improvement on Python software-engineering tasks</span> across three subdomains the engine wasn't trained on. The headline limit: <span class="neg">the substrate did not transfer to non-Python artifact paradigms</span>. The engineering loop is paradigm-internal, not paradigm-general.
</p>

<h2>Experimental design</h2>
<p>
Three conditions: <code>raw_local</code> (bare qwen2.5-coder:14b, no engine), <code>soil_only</code> (engine with soil retrieval but no covenant enforcement or FSRS decay), and <code>full</code> (the complete engine). Soil-using conditions measured at three accumulated-soil levels (build_seq 1, 5, 15) drawn from pre-built snapshots. Twenty challenges: five Python microservices, five CLI scripts, five data pipelines, and five novel-artifact challenges (Sokoban level design, SMT-LIB constraint encoding, crossword construction, TLA+ specification, IPv4 regex synthesis). All scoring mechanical — no LLM-as-judge.
</p>

<h2>Headline: substrate works on Python, fails on non-Python paradigms</h2>

<div class="chart">
{chart3_per_domain}
<div class="caption">Per-domain weighted score at build_seq=15. Bars in each group: raw_local (gray), soil_only (orange), full engine (blue). Higher is better.</div>
</div>

<p>
Across all three Python subdomains the engine wasn't specifically tuned on, the substrate produced near-perfect performance versus a near-zero bare model. Microservices and CLI both jumped from ~0.2 to 1.00 under either substrate condition. Data pipelines — the intermediate-complexity bucket — moved from 0.35 raw to 0.80 under full.
</p>

<p>
The novel-artifact bucket tells the opposite story. The bare model produced 0/5 passes; both substrate conditions also produced 0/5 at build_seq=15. The substrate failed to transfer to paradigms it had no training signal for. More striking: only 1/5 novel-artifact challenges passed at build_seq 1 and 5 (and only IPv4 regex), and even that dropped to 0/5 at build_seq=15. <strong>Accumulated soil appears to actively hurt novel-artifact performance at scale</strong> — the substrate seems to overfit to its training paradigm.
</p>

<h2>What does the substrate actually contribute?</h2>

<div class="chart">
{chart2_component_attribution}
<div class="caption">Marginal contribution of each substrate component, pooled across all 120 substrate cells. Raw baseline {raw_avg:.2f}, soil-only {soil_avg:.2f}, full {full_avg:.2f}.</div>
</div>

<p>
Soil retrieval (semantic search over past build patterns) explains roughly 95% of the substrate's measured benefit. Covenant enforcement and FSRS decay together contribute about 5%. A simpler engine that retrieves from soil but skips covenant rewriting would capture most of the headline win.
</p>

<p>
This is worth saying out loud because the engine's complexity has been growing for months. The data here suggests the marginal complexity is buying small marginal performance. If you're going to invest in engine surface area, soil retrieval is the high-leverage component; covenants and FSRS are real but incremental.
</p>

<h2>The one clean learning curve</h2>

<div class="chart">
{chart1_learning_curve}
<div class="caption">Data-pipeline domain only. Microservices and CLI saturate at 1.00 immediately, so no learning curve to plot. Novel artifacts stay near 0.</div>
</div>

<p>
Only one domain shows the textbook "more soil makes the engine measurably better" curve: data pipelines. There the bare model scores 0.35, the substrate at build_seq=1 sits at ~0.50, and accumulated soil pushes performance up monotonically to 0.80 by build_seq=15. This is the small arc-reactor proof: the engine genuinely learns from its prior builds within paradigms it has signal for.
</p>

<p>
The other two Python domains don't show a learning curve because they saturate at the first sample — the substrate already gets them right at build_seq=1. The novel-artifact domain doesn't show one because the engine has no signal to learn from. If you want to see learning in future experiments, focus on intermediate-complexity challenges that aren't yet solved.
</p>

<h2>Why did novel-artifact challenges fail?</h2>
<p>
Spot-checked four novel-artifact build outputs to confirm the failures weren't a validator bug. They weren't — the engine wrote files at the expected paths and names. The content was wrong in characteristic ways:
</p>

<ul>
<li><strong>Sokoban level:</strong> 6×6-ish grid with two targets, a wall mid-row, and inconsistent row widths. Structurally close, semantically invalid.</li>
<li><strong>SMT-LIB encoding:</strong> The file starts with a bare <code>2</code> on line 1 before any actual SMT directive. Z3 parse error.</li>
<li><strong>IPv4 regex:</strong> The canonical near-correct regex <code>(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)</code>. The <code>[01]?</code> allows leading-zero octets like <code>01.1.1.1</code>. F1=0.93, just below the 0.95 threshold.</li>
<li><strong>TLA+ specification:</strong> Module header, EXTENDS, VARIABLES all present, but the program counter has 2 states instead of Peterson's required 4 (<code>ncs</code>/<code>trying</code>/<code>cs</code>/<code>exit</code>). Peterson's algorithm isn't actually implemented.</li>
</ul>

<p>
Pattern: the engine produces artifacts of the <em>right shape</em> but with <em>semantic errors strict validators catch</em>. It knows what TLA+ syntax looks like; it doesn't know what Peterson's algorithm requires. It knows IPv4 regex grammar; it doesn't know the leading-zero edge case. It knows the Sokoban grid layout; it doesn't reason about move counts.
</p>

<p>
This is informative. The substrate's "iterate, validate, refine" loop only refines on signals it knows how to interpret. <code>pytest</code> failure messages drive Python self-improvement. TLC error messages do not drive TLA+ self-improvement. The validator says "wrong" but the engine has no mechanism to translate that signal into a structural fix for a paradigm it doesn't deeply understand.
</p>

<h2>What this experiment supports and doesn't</h2>

<p><strong>Supported by the data:</strong></p>
<ul>
<li>The Belief Engine substrate provides a 4× improvement over a bare local model on Python software-engineering tasks.</li>
<li>This generalizes across three Python subdomains the engine wasn't specifically trained on.</li>
<li>Soil retrieval is the dominant component of that improvement; covenants and FSRS add small marginal value.</li>
<li>A clean learning curve is visible in data pipelines (0.44 → 0.80 across build_seq 1 → 15 under full).</li>
</ul>

<p><strong>Not supported by the data:</strong></p>
<ul>
<li>Any "general intelligence layer" framing. The substrate is paradigm-internal.</li>
<li>Cross-paradigm transfer to formal-methods, puzzle-construction, or constraint-solving artifacts.</li>
<li>The originally locked thesis that "the engineering loop transferred to artifacts the soil never saw" — for non-Python artifacts, it did not.</li>
</ul>

<div class="pullquote">
The Belief Engine substrate generalizes across Python software-engineering subdomains but does not transfer to non-Python artifact paradigms. The engineering discipline is paradigm-internal: it iterates and refines within a known problem class, but cannot bootstrap correctness in a paradigm it has no signal for.
</div>

<h2>What's next</h2>
<p>
Three follow-up experiments come directly out of this run:
</p>
<ol>
<li><strong>Disentangle soil retrieval from soil composition.</strong> Soil at build_seq=15 has 14 Python-heavy nutrients. Does running soil_only with a curated cross-paradigm soil (mix of formal-methods and software-engineering builds) help novel-artifact performance? If yes, the limitation is data, not the loop. If no, the limitation is the loop itself.</li>
<li><strong>Test paradigm-specific validators driving paradigm-specific iteration.</strong> The engine's debugger currently interprets pytest errors. If it also interpreted Z3 unsat-core output or TLC counterexamples as fix signals, would it self-improve on those paradigms?</li>
<li><strong>Saturate the data-pipeline learning curve.</strong> The 0.44 → 0.80 trajectory at build_seq 1 → 15 suggests there's more room. Does build_seq 30 reach 0.90? 50? Where does it plateau?</li>
</ol>

<hr>
<p class="meta">
  All raw data lives in <code>~/.belief-engine/experiments.db</code>. Reproduce: <code>git log</code> on metafiopy-tech/belief-engine, baseline snapshots in <code>~/.belief-engine/snapshots/2026-05-27T*_substrate-baseline-*</code>, runner driver at <code>scripts/run_full_substrate_transfer.py</code>.
</p>

</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip3 install matplotlib", file=sys.stderr)
        return 1

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 1
    results = _load_results(db_path, args.exp_id)
    if not results:
        print(f"ERROR: no results for {args.exp_id!r}", file=sys.stderr)
        return 1
    print(f"Loaded {len(results)} cells")

    chart1_svg = _chart_learning_curve(results)
    chart2_svg, raw_avg, soil_avg, full_avg = _chart_component_attribution(results)
    chart3_svg = _chart_per_domain(results)

    html = HTML_TEMPLATE.format(
        chart1_learning_curve=chart1_svg,
        chart2_component_attribution=chart2_svg,
        chart3_per_domain=chart3_svg,
        raw_avg=raw_avg,
        soil_avg=soil_avg,
        full_avg=full_avg,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote brief: {out_path}")
    print(f"Open with: open {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
