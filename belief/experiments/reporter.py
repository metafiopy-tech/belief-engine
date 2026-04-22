"""Generate comparison reports from A/B experiment results.

All functions accept an optional db_path override so tests can
point at a fixture database without touching the real one.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

from belief.experiments.ab_runner import DEFAULT_DB_PATH


def comparison_table(
    experiment_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> str:
    """Return a comparison table across conditions for one experiment.

    If experiment_id is None, uses the most recent completed experiment.
    Returns a plain-text message when no data is found.
    """
    if not db_path.exists():
        return "No experiments database found. Run `belief experiment run` first."

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if experiment_id is None:
            row = conn.execute(
                "SELECT experiment_id FROM experiment_meta "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return "No experiments found."
            experiment_id = row["experiment_id"]

        rows = conn.execute(
            "SELECT * FROM results "
            "WHERE experiment_id = ? "
            "ORDER BY challenge_id, condition",
            (experiment_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"No results for experiment {experiment_id}."

    # Group by challenge
    by_challenge: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_challenge[r["challenge_id"]][r["condition"]] = dict(r)

    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  EXPERIMENT: {experiment_id}")
    lines.append(f"{'=' * 80}\n")

    col_w = 16
    lines.append(
        f"{'Challenge':<25} {'Engine+Cloud':>{col_w}} {'Engine+Local':>{col_w}} {'Raw Local':>{col_w}}"
    )
    lines.append("-" * 75)

    totals: dict[str, dict] = defaultdict(lambda: {
        "passed": 0, "total": 0, "cost": 0.0, "time": 0.0
    })

    for cid in sorted(by_challenge):
        conditions = by_challenge[cid]
        row_parts = [f"{cid:<25}"]

        for cond in ("engine_cloud", "engine_local", "raw_local"):
            if cond in conditions:
                r = conditions[cond]
                check = "✓" if r["passed"] else "✗"
                cell = f"{check} {r['weighted_score']:.2f}  ${r['cost_usd']:>5.3f}"
                row_parts.append(f"{cell:>{col_w}}")
                totals[cond]["total"] += 1
                if r["passed"]:
                    totals[cond]["passed"] += 1
                totals[cond]["cost"] += r["cost_usd"]
                totals[cond]["time"] += r["time_seconds"]
            else:
                row_parts.append(f"{'—':>{col_w}}")

        lines.append("  ".join(row_parts))

    lines.append("-" * 75)

    # Summary row
    sum_parts = [f"{'TOTAL':<25}"]
    for cond in ("engine_cloud", "engine_local", "raw_local"):
        t = totals[cond]
        if t["total"] > 0:
            rate = t["passed"] / t["total"]
            cell = f"{t['passed']}/{t['total']} ({rate:.0%})  ${t['cost']:.2f}"
            sum_parts.append(f"{cell:>{col_w}}")
        else:
            sum_parts.append(f"{'—':>{col_w}}")
    lines.append("  ".join(sum_parts))

    # Soil lift
    el = totals["engine_local"]
    rl = totals["raw_local"]
    lines.append(f"\n{'─' * 75}")
    if el["total"] > 0 and rl["total"] > 0:
        el_rate = el["passed"] / el["total"]
        rl_rate = rl["passed"] / rl["total"]
        lift = el_rate - rl_rate
        direction = "adds value" if lift > 0.0 else ("neutral" if lift == 0.0 else "subtracts value")
        lines.append(
            f"  Soil Lift: {lift:+.1%}  — Engine+Local vs Raw Local"
        )
        lines.append(
            f"  Interpretation: {direction}."
        )
        if lift <= 0.0:
            lines.append(
                "  Note: run more experiments to distinguish signal from noise."
            )
    lines.append("")

    return "\n".join(lines)


def longitudinal_report(db_path: Path = DEFAULT_DB_PATH) -> str:
    """Show how performance changes across experiments as the soil grows.

    Requires at least 2 experiments. The key metric is Lift
    (Engine+Local pass-rate minus Raw Local pass-rate) — if it
    increases as soil grows, compound learning is real.
    """
    if not db_path.exists():
        return "No experiments database found. Run `belief experiment run` first."

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        experiments = conn.execute(
            "SELECT experiment_id, created_at FROM experiment_meta "
            "ORDER BY created_at"
        ).fetchall()

        if not experiments:
            return "No experiments found."

        if len(experiments) < 2:
            return (
                "Need at least 2 experiments for longitudinal analysis.\n"
                "Run `belief experiment run` again after building more."
            )

        lines = []
        lines.append(f"\n{'=' * 80}")
        lines.append("  LONGITUDINAL: Compound Learning Over Time")
        lines.append(f"{'=' * 80}\n")
        lines.append(
            f"{'Experiment':<28} {'Date':<12} {'Soil':>6}  "
            f"{'Eng+Local':>12}  {'Raw Local':>12}  {'Lift':>7}"
        )
        lines.append("-" * 80)

        for exp in experiments:
            eid = exp["experiment_id"]
            date = exp["created_at"][:10]

            rows = conn.execute(
                "SELECT condition, passed, soil_size "
                "FROM results WHERE experiment_id=?",
                (eid,),
            ).fetchall()

            el_pass = sum(1 for r in rows if r["condition"] == "engine_local" and r["passed"])
            el_total = sum(1 for r in rows if r["condition"] == "engine_local")
            rl_pass = sum(1 for r in rows if r["condition"] == "raw_local" and r["passed"])
            rl_total = sum(1 for r in rows if r["condition"] == "raw_local")
            soil = max((r["soil_size"] for r in rows), default=0)

            el_rate = el_pass / max(el_total, 1)
            rl_rate = rl_pass / max(rl_total, 1)
            lift = el_rate - rl_rate

            el_cell = f"{el_pass}/{el_total} ({el_rate:.0%})" if el_total else "—"
            rl_cell = f"{rl_pass}/{rl_total} ({rl_rate:.0%})" if rl_total else "—"

            lines.append(
                f"{eid:<28} {date:<12} {soil:>6}  "
                f"{el_cell:>12}  {rl_cell:>12}  {lift:>+7.1%}"
            )

    finally:
        conn.close()

    lines.append("")
    lines.append("  Rising Lift → compound learning is real.")
    lines.append("  Flat/falling Lift → the engine is overhead, not intelligence.")
    lines.append("")
    return "\n".join(lines)
