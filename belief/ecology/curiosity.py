"""Curiosity Gate — propose what the engine should build next (v3.3 §3.5).

Curiosity uses idle cycles to self-direct the engine toward the most
informative builds. v3.3 ships ``suggest()`` only — ``auto_build()``
chains to the production pipeline via subprocess and is deferred to
Session 5b so the budget-isolation testing can land separately.

**Algorithm v1:**

    gaps      = identify_gaps(soil)                       # info_gain helpers
    goals     = await _propose_goals(gaps, llm)            # Haiku call
    candidates = [GoalCandidate(g, info_gain, cost, ...)]  # rank
    selected  = top-1 by (info_gain / max(cost, ε))

Plug into Economist contract at $0.05 per ``curiosity.propose`` action
(spec §3.4 price table). LLM-stub mode (``llm=None``) returns
deterministic synthesized goals so tests run hermetically.

Audit JSONL: ``~/.belief-engine/audit/ecology_curiosity.jsonl``.
State:      ``~/.belief-engine/curiosity_state.json``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from belief.ecology._information_gain import (
    Gap,
    estimate_info_gain,
    gaps_summary,
    identify_gaps,
)
from belief.ecology.economist import Economist

logger = logging.getLogger("belief.ecology.curiosity")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_SUGGEST_N = 5
DEFAULT_BUDGET_USD = 1.0
CURIOSITY_PROPOSE_USD = 0.05  # per spec §3.4 price table
CURIOSITY_ACTION_KEY = "curiosity.propose"

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_STATE_PATH = _BELIEF_HOME / "curiosity_state.json"
_DEFAULT_AUDIT_PATH = _BELIEF_HOME / "audit" / "ecology_curiosity.jsonl"


# ── Data ───────────────────────────────────────────────────────────────────


@dataclass
class GoalCandidate:
    goal: str
    estimated_info_gain: float  # 0..1
    estimated_cost_usd: float
    rationale: str
    coverage_gaps_addressed: list[str] = field(default_factory=list)
    bang_per_buck: float = 0.0


@dataclass
class CuriosityResult:
    candidates: list[GoalCandidate] = field(default_factory=list)
    selected: GoalCandidate | None = None
    gaps_identified: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    dry_run: bool = False
    economist_approved: bool = True
    economist_reason: str = ""
    llm_stub: bool = True  # True when no real LLM was used


# ── State + audit (mirror prior organs' patterns) ─────────────────────────


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
    os.replace(tmp, path)


def _audit_append(audit_path: Path, record: dict) -> None:
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:  # pragma: no cover
        logger.warning("Curiosity audit write failed: %s", e)


# ── Goal proposer (LLM-stub mode + real-LLM hook) ─────────────────────────


async def _propose_goals(
    gaps: list[Gap],
    llm: Any | None,
    n: int,
) -> tuple[list[str], bool]:
    """Return (proposed_goal_strings, used_real_llm).

    With ``llm=None`` (default in tests + dev), we synthesize one goal
    per gap deterministically — covers each gap's name in plain text so
    the info-gain estimator can match it. With a real LLM client (an
    object exposing ``async generate(prompt) -> str``), we ask for a
    JSON list of goal strings and parse it.
    """
    if not gaps:
        return [], llm is not None

    if llm is None:
        # Deterministic stub — cheap, audit-friendly, test-friendly.
        stub_goals: list[str] = []
        for gap in gaps[:n]:
            stub_goals.append(_synthetic_goal_for_gap(gap))
        return stub_goals[:n], False

    try:
        prompt = _build_proposer_prompt(gaps, n)
        response = await llm.generate(prompt)  # type: ignore[attr-defined]
        goals = _parse_goal_response(response, n)
        if not goals:
            # LLM returned something unparseable — fall back to stubs.
            logger.warning("Curiosity: LLM proposer returned no parseable goals; falling back")
            return [_synthetic_goal_for_gap(g) for g in gaps[:n]], False
        return goals, True
    except Exception as e:
        logger.warning("Curiosity: LLM proposer failed (%s); falling back to stubs", e)
        return [_synthetic_goal_for_gap(g) for g in gaps[:n]], False


def _synthetic_goal_for_gap(gap: Gap) -> str:
    """Generate a one-line goal string that addresses ``gap``.

    The synthesized text intentionally includes the gap's ``name`` so
    ``estimate_info_gain`` can match it back when ranking. Wording is
    plain enough that a human can read it and decide to ship.
    """
    if gap.category == "file_ext":
        return f"Build a small parser/validator service that handles {gap.name} config files"
    if gap.category == "framework":
        return f"Build a hello-world {gap.name} service with a healthcheck endpoint"
    if gap.category == "covenant_sparse":
        return (
            "Build a small project that exercises diverse failure modes so the "
            "crystallizer can extract more covenants"
        )
    return f"Explore the unexplored area: {gap.name}"


def _build_proposer_prompt(gaps: list[Gap], n: int) -> str:
    gap_lines = "\n".join(
        f"  - [{g.category}] {g.name} (signal={g.signal_strength:.2f}): {g.rationale}"
        for g in gaps[:20]
    )
    return (
        f"You are proposing build goals for an autonomous codegen engine. "
        f"The engine has identified the following gaps in its accumulated "
        f"knowledge substrate:\n\n{gap_lines}\n\n"
        f"Propose exactly {n} build goals (one per line, no numbering, no markdown) "
        f"that would each address one or more of these gaps. Each goal should be "
        f"a single concrete sentence describing what to build. Mention the gap "
        f"name(s) explicitly so the info-gain estimator can match them."
    )


def _parse_goal_response(response: str, n: int) -> list[str]:
    """Parse the LLM's response into a list of goal strings."""
    if not response:
        return []
    lines = [line.strip().lstrip("-•*0123456789. ") for line in response.splitlines()]
    return [line for line in lines if line and len(line) > 8][:n]


# ── The organ ──────────────────────────────────────────────────────────────


async def suggest(
    n: int = DEFAULT_SUGGEST_N,
    *,
    soil: Any | None = None,
    economist: Economist | None = None,
    llm: Any | None = None,
    state_path: Path | None = None,
    audit_path: Path | None = None,
    dry_run: bool = False,
    budget_usd: float = DEFAULT_BUDGET_USD,
) -> CuriosityResult:
    """Identify gaps, propose ``n`` goals, rank by bang-per-buck.

    ``soil`` / ``economist`` / ``llm`` are dependency-injection slots
    for tests. Production callers leave them None: real Soil constructed
    lazily, Economist with default daily budget, llm None ⇒ stub mode.

    ``dry_run=True`` skips the Economist commit (and state write) but
    still issues the quote so the contract trail is intact.
    """
    started = time.monotonic()
    state_path = state_path or _DEFAULT_STATE_PATH
    audit_path = audit_path or _DEFAULT_AUDIT_PATH

    if soil is None:
        from belief.memory.soil import Soil  # noqa: PLC0415

        soil = Soil()
    if economist is None:
        economist = Economist()

    # Per-call estimated cost: one LLM call total (Haiku). Stubbed runs
    # are still quoted at the same rate so the contract surface is
    # consistent — Sleep does the same.
    quote = economist.quote(CURIOSITY_ACTION_KEY, estimated_usd=CURIOSITY_PROPOSE_USD)
    if not quote.approved:
        result = CuriosityResult(
            economist_approved=False,
            economist_reason=quote.reason,
            duration_seconds=time.monotonic() - started,
            dry_run=dry_run,
            llm_stub=(llm is None),
        )
        _audit_append(audit_path, {"event": "rejected_by_economist", "reason": quote.reason})
        return result

    # Local budget guard (in addition to Economist's daily ceiling).
    if budget_usd < CURIOSITY_PROPOSE_USD:
        result = CuriosityResult(
            economist_approved=True,
            economist_reason=quote.reason,
            duration_seconds=time.monotonic() - started,
            dry_run=dry_run,
            llm_stub=(llm is None),
        )
        _audit_append(
            audit_path,
            {
                "event": "skipped_local_budget",
                "budget_usd": budget_usd,
                "needed_usd": CURIOSITY_PROPOSE_USD,
            },
        )
        return result

    # ── Identify gaps ──────────────────────────────────────────────────
    gaps = identify_gaps(soil)
    _audit_append(
        audit_path,
        {
            "event": "gaps_identified",
            "count": len(gaps),
            "summary": gaps_summary(gaps),
        },
    )

    # ── Propose goals ──────────────────────────────────────────────────
    raw_goals, used_real_llm = await _propose_goals(gaps, llm, n)
    cost_so_far = CURIOSITY_PROPOSE_USD if used_real_llm else 0.0

    # ── Score + rank ───────────────────────────────────────────────────
    candidates: list[GoalCandidate] = []
    for goal_text in raw_goals:
        info_gain, addressed = estimate_info_gain(goal_text, gaps)
        # Cost estimate per future build is a placeholder — Economist
        # self-tuning (Session 5) will refine this from real history.
        # For now, every proposed build estimates at $1.00.
        cost = 1.00
        bang_per_buck = info_gain / max(cost, 0.01)
        candidates.append(
            GoalCandidate(
                goal=goal_text,
                estimated_info_gain=info_gain,
                estimated_cost_usd=cost,
                rationale=(
                    f"addresses {len(addressed)}/{len(gaps)} gaps: "
                    + ", ".join(f"{g.category}:{g.name}" for g in addressed[:5])
                    + (" …" if len(addressed) > 5 else "")
                )
                if addressed
                else "no clear gap match — exploratory only",
                coverage_gaps_addressed=[f"{g.category}:{g.name}" for g in addressed],
                bang_per_buck=bang_per_buck,
            )
        )

    candidates.sort(key=lambda c: c.bang_per_buck, reverse=True)
    selected = candidates[0] if candidates else None

    # ── Commit + persist ────────────────────────────────────────────────
    result = CuriosityResult(
        candidates=candidates,
        selected=selected,
        gaps_identified=len(gaps),
        cost_usd=cost_so_far,
        duration_seconds=time.monotonic() - started,
        dry_run=dry_run,
        economist_approved=True,
        economist_reason=quote.reason,
        llm_stub=not used_real_llm,
    )

    if not dry_run:
        try:
            _atomic_write(
                state_path,
                {
                    "last_run_iso": datetime.now(timezone.utc).isoformat(),
                    "gaps_identified": len(gaps),
                    "candidates_proposed": len(candidates),
                    "selected_goal": selected.goal if selected else None,
                    "cost_usd": cost_so_far,
                },
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Curiosity: state write failed: %s", e)
        economist.commit(CURIOSITY_ACTION_KEY, actual_usd=cost_so_far)

    for cand in candidates:
        _audit_append(
            audit_path,
            {
                "event": "candidate",
                "goal": cand.goal,
                "info_gain": round(cand.estimated_info_gain, 6),
                "cost_usd": cand.estimated_cost_usd,
                "bang_per_buck": round(cand.bang_per_buck, 6),
                "addressed": cand.coverage_gaps_addressed,
            },
        )
    _audit_append(
        audit_path,
        {
            "event": "run_summary",
            "gaps": len(gaps),
            "candidates": len(candidates),
            "selected": selected.goal if selected else None,
            "cost_usd": cost_so_far,
            "llm_stub": not used_real_llm,
            "dry_run": dry_run,
            "duration_seconds": round(result.duration_seconds, 4),
        },
    )
    return result


async def auto_build(*_args, **_kw) -> dict:
    """Deferred to v3.3 Session 5b. See spec §3.5 + Session 5 plan.

    Raises NotImplementedError immediately so the deferral is loud and
    visible. The CLI wraps this so users see a helpful message rather
    than a stack trace.
    """
    raise NotImplementedError(
        "Curiosity auto_build is deferred to v3.3 Session 5b. Use "
        "`belief curiosity --suggest N` then run the chosen goal "
        "manually via `belief --goal '...'` for now."
    )


# ── CLI helpers ────────────────────────────────────────────────────────────


def cli_format_result(result: CuriosityResult) -> str:
    header = "Curiosity (DRY RUN)" if result.dry_run else "Curiosity"
    if result.llm_stub:
        header += " [stub LLM]"
    lines = [
        f"{header} — {result.gaps_identified} gaps, {len(result.candidates)} candidates "
        f"in {result.duration_seconds:.2f}s, ${result.cost_usd:.4f} spent",
        f"  Economist: approved={result.economist_approved} ({result.economist_reason})",
    ]
    if not result.candidates:
        lines.append("  no candidates — soil may be empty or too small to identify gaps")
        return "\n".join(lines)
    if result.selected:
        lines.append(
            f"  → selected: {result.selected.goal}\n"
            f"      info_gain={result.selected.estimated_info_gain:.3f}  "
            f"bang/buck={result.selected.bang_per_buck:.3f}"
        )
    lines.append("  top candidates:")
    for cand in result.candidates[:5]:
        lines.append(
            f"    [{cand.estimated_info_gain:.2f}] {cand.goal[:90]}"
            f"{'…' if len(cand.goal) > 90 else ''}"
        )
    return "\n".join(lines)


def result_to_dict(result: CuriosityResult) -> dict:
    return asdict(result)
