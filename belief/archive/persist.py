"""Post-build persistence — Session 6 (v3.2).

The CLI calls :func:`persist_build_outcome` after printing BUILD
COMPLETE.  It constructs a :class:`BuildOutcome` from the final
LangGraph state and writes it to the default
:class:`~belief.archive.AgentArchive`.

Never raises — an archive write failure is logged and swallowed.  The
archive is a learning aid; a single failed write doesn't block the
user, and a corrupted row won't corrupt the build that just finished.
"""

from __future__ import annotations

import logging
from typing import Any

from belief.archive.config import AgentConfiguration
from belief.archive.outcome import BuildOutcome
from belief.archive.store import AgentArchive

logger = logging.getLogger("belief.archive.persist")


def persist_build_outcome(
    final_state: dict[str, Any],
    *,
    archive: AgentArchive | None = None,
) -> None:
    """Build a BuildOutcome from final_state and persist it."""
    try:
        outcome = _build_outcome_from_state(final_state)
    except Exception as e:
        logger.debug("persist_build_outcome: skipped (cannot build outcome: %s)", e)
        return
    if outcome is None:
        return
    try:
        archive = archive or AgentArchive()
        archive.persist(outcome)
    except Exception as e:
        logger.warning("agent archive persist failed (non-fatal): %s", e)


def _build_outcome_from_state(final_state: dict[str, Any]) -> BuildOutcome | None:
    run_id = str(final_state.get("run_id") or "").strip()
    if not run_id:
        logger.debug("no run_id in state; skipping archive persist")
        return None

    # Goal — try requirement_spec first, fall back to user_goal.
    spec = final_state.get("requirement_spec") or {}
    if isinstance(spec, dict):
        goal = str(spec.get("goal") or final_state.get("user_goal") or "")
    else:
        goal = str(getattr(spec, "goal", None) or final_state.get("user_goal") or "")

    validation = final_state.get("validation_result") or {}
    if isinstance(validation, dict):
        verdict = str(validation.get("verdict") or "unknown")
        weighted = float(validation.get("weighted_score") or 0.0)
    else:
        verdict = str(getattr(validation, "verdict", "unknown"))
        weighted = float(getattr(validation, "weighted_score", 0.0))
    # verdict might be an enum value
    if hasattr(verdict, "value"):
        verdict = verdict.value  # type: ignore[attr-defined]

    exec_r = final_state.get("execution_result") or {}
    if isinstance(exec_r, dict):
        tests_passed = int(exec_r.get("tests_passed") or 0)
        tests_total = int(exec_r.get("tests_total") or 0)
    else:
        tests_passed = int(getattr(exec_r, "tests_passed", 0) or 0)
        tests_total = int(getattr(exec_r, "tests_total", 0) or 0)

    # Wall clock — sum of per-agent timings.
    timings = final_state.get("agent_timings") or {}
    wall = 0.0
    if isinstance(timings, dict):
        for v in timings.values():
            try:
                wall += float(v or 0.0)
            except (TypeError, ValueError):
                pass

    # Cost — token_usage.total_cost_usd if present.
    cost = 0.0
    tu = final_state.get("token_usage") or {}
    if isinstance(tu, dict):
        try:
            cost = float(tu.get("total_cost_usd") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0

    # Debug iterations.
    debug_iter = int(final_state.get("iteration") or 0)

    # Agent configurations — we don't have per-agent snapshots in state
    # (that's Session 6 follow-up work to wire BaseAgent to record them
    # on each call).  For now we synthesise a minimal one for the
    # planner so the archive has a searchable document.
    configs: dict[str, AgentConfiguration] = {}
    try:
        from belief.prompts import PLANNER_SYSTEM

        configs["planner"] = AgentConfiguration(
            agent_name="planner",
            system_prompt=PLANNER_SYSTEM,
            user_prompt_template="",
            model="qwen2.5-coder:14b",  # default; future work attaches actual model used
        )
    except Exception as e:
        logger.debug("cannot load PLANNER_SYSTEM (%s); using stub", e)
        configs["planner"] = AgentConfiguration(agent_name="planner", system_prompt="")

    covenant_violations = _extract_covenant_violations(final_state)

    # Snapshot per-build local-call ledger (Gate 4 instrumentation).
    prompt_tokens = 0
    completion_tokens = 0
    n_llm_calls = 0
    tokens_by_role: dict[str, int] = {}
    try:
        from belief.llm import LOCAL_TRACKER

        n_llm_calls = LOCAL_TRACKER.total_calls()
        by_role = LOCAL_TRACKER.by_role()
        for role, bucket in by_role.items():
            prompt_tokens += int(bucket.get("prompt_tokens", 0))
            completion_tokens += int(bucket.get("completion_tokens", 0))
            tokens_by_role[role] = int(
                bucket.get("prompt_tokens", 0) + bucket.get("completion_tokens", 0)
            )
    except Exception as e:
        logger.debug("LOCAL_TRACKER snapshot failed (%s); zeros recorded", e)

    outcome = BuildOutcome(
        run_id=run_id,
        goal=goal,
        verdict=verdict,
        tests_passed=tests_passed,
        tests_total=tests_total,
        weighted_score=weighted,
        wallclock_s=wall,
        estimated_cost_usd=cost,
        covenant_violations=covenant_violations,
        debug_iterations=debug_iter,
        agent_configurations=configs,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        n_llm_calls=n_llm_calls,
        tokens_by_role=tokens_by_role,
    )
    # Trajectory signature — stable hash of agent_timings keys + verdict.
    agent_sequence = list(timings.keys()) if isinstance(timings, dict) else []
    outcome.trajectory_signature = outcome.compute_trajectory_signature(agent_sequence)
    return outcome


def _extract_covenant_violations(final_state: dict[str, Any]) -> list[str]:
    warns = final_state.get("warnings") or []
    out: list[str] = []
    for w in warns:
        if isinstance(w, str) and "Covenant" in w:
            out.append(w[:200])
    return out


__all__ = ["persist_build_outcome"]
