"""Sleep — offline consolidation pass for the soil (v3.3 §3.3).

Sleep runs while no live build is happening. Two phases ship in this
session; the spec's third phase (synthetic challenges) is deferred per
Joe's Session-2 notes — it overlaps heavily with Curiosity (Sessions
6–7) and the spec already gates it behind a feature flag.

**Phase A — anomaly-weighted episode replay → crystallizer.**
Sleep samples recent build episodes weighted toward *anomalies* (failed
builds, low scores, high cost) on the theory that anomalies are where
new invariants live. Calls the existing crystallizer pipeline functions
(``sweep_templates`` → ``propose_invariants`` → ``filter_candidates``
→ ``promote_to_covenant``) directly so we control the sample. The
wrapper ``run_crystallization`` does naive limit-based fetch — Sleep
needs the anomaly weighting.

**Phase B — FSRS schedule housekeeping.**
For every active nutrient whose ``fsrs_next_review`` is overdue (or
never scheduled), recompute it from the current ``stability`` so
``get_due_reviews`` returns sensible items. This is *not* a re-review:
no grade is applied, no stability/difficulty/reps changes happen.
Pure schedule refresh.

**Phase C — synth challenges.** Deferred. If ``synth_challenges=True``
the run logs a warning and skips Phase C; defaults to False.

Budget integration: each cycle quotes Economist at $0.30
(``sleep.replay_cycle``) per spec §3.4 price table. The current
cycle's actual is committed at the end. Future Economist self-tuning
(Session 5) will refine the estimate from observed cost history.

Audit JSONL: ``~/.belief-engine/audit/ecology_sleep.jsonl``.
State:      ``~/.belief-engine/sleep_state.json``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from belief.ecology.economist import Economist

logger = logging.getLogger("belief.ecology.sleep")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_CYCLES = 3
DEFAULT_MAX_MINUTES = 60
DEFAULT_BUDGET_USD = 1.0
DEFAULT_EPISODES_PER_CYCLE = 20
SLEEP_REPLAY_CYCLE_USD = 0.30
SLEEP_ACTION_KEY = "sleep.replay_cycle"

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_STATE_PATH = _BELIEF_HOME / "sleep_state.json"
_DEFAULT_AUDIT_PATH = _BELIEF_HOME / "audit" / "ecology_sleep.jsonl"


# ── Config + result types ──────────────────────────────────────────────────


@dataclass
class SleepConfig:
    cycles: int = DEFAULT_CYCLES
    max_minutes: int = DEFAULT_MAX_MINUTES
    budget_usd: float = DEFAULT_BUDGET_USD
    episodes_per_cycle: int = DEFAULT_EPISODES_PER_CYCLE
    crystallize: bool = True
    recompute_fsrs: bool = True
    synth_challenges: bool = False  # deferred to Curiosity, see module docstring
    dry_run: bool = False
    state_path: Path | None = None
    audit_path: Path | None = None


@dataclass
class SleepResult:
    cycles_completed: int = 0
    new_covenant_ids: list[str] = field(default_factory=list)
    fsrs_schedules_refreshed: int = 0
    episodes_replayed: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    dry_run: bool = False
    truncated_reason: str = ""  # "" on clean finish, else "timeout" / "budget" / "economist"
    economist_approved: bool = True


# ── Concurrency lock ───────────────────────────────────────────────────────


@contextmanager
def _sleep_lock(state_path: Path) -> Iterator[bool]:
    """Best-effort exclusive lock so two ``belief sleep`` runs serialize.

    Yields ``True`` if we hold the lock, ``False`` if another process
    has it. Mirrors the Economist's fcntl pattern but non-blocking
    here: a second concurrent invocation should bail out quickly with
    a diagnostic, not queue up.
    """
    try:
        import fcntl  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        yield True
        return

    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    f = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
        except BlockingIOError:
            held = False
        try:
            yield held
        finally:
            if held:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


# ── State + audit (mirror Predator/Economist patterns) ────────────────────


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
        logger.warning("Sleep audit write failed: %s", e)


# ── Anomaly-weighted episode sampling ─────────────────────────────────────


def _episode_anomaly_score(ep: dict) -> float:
    """Score an episode 0..3 by how anomalous it is.

    Components:
      * passed=False contributes 1.0 (failed builds are the obvious anomaly)
      * 1.0 - score (so a score of 0 contributes 1.0; a score of 1.0 contributes 0)
      * cost_usd / 5.0 capped at 1.0 (expensive builds are noteworthy)

    Sum is in [0.0, 3.0]. Used by ``_sample_anomaly_weighted`` for
    deterministic top-N ranking — no random sampling, so tests can
    assert ordering.
    """
    score = 0.0
    if not ep.get("passed", True):
        score += 1.0
    raw_score = float(ep.get("score", 1.0) or 0.0)
    score += max(0.0, 1.0 - min(1.0, raw_score))
    cost = float(ep.get("cost_usd", 0.0) or 0.0)
    score += min(1.0, max(0.0, cost / 5.0))
    return score


def _fetch_episodes(soil: Any, limit: int) -> list[dict]:
    """Pull every episode metadata row from the belief_episodes collection.

    Returns trace dicts in the format crystallizer's ``sweep_templates``
    expects (flat metadata + ``trace_id`` + optional ``description``).
    Caps at ``limit * 5`` so we have a wider candidate pool to rank
    before slicing.
    """
    try:
        episodes_col = soil._collections.get("belief_episodes")
    except AttributeError:
        return []
    if episodes_col is None:
        return []
    try:
        count = episodes_col.count()
    except Exception:
        return []
    if count == 0:
        return []
    fetch_n = min(count, max(limit * 5, limit))
    try:
        results = episodes_col.get(
            include=["documents", "metadatas"],
            limit=fetch_n,
        )
    except Exception as e:
        logger.debug("Sleep: episodes fetch failed: %s", e)
        return []
    out: list[dict] = []
    ids = results.get("ids") or []
    metas = results.get("metadatas") or []
    docs = results.get("documents") or []
    for i, doc_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else None
        if not isinstance(meta, dict):
            continue
        trace = dict(meta)
        trace["trace_id"] = doc_id
        if i < len(docs) and docs[i]:
            trace["description"] = docs[i]
        out.append(trace)
    return out


def _sample_anomaly_weighted(soil: Any, n: int) -> list[dict]:
    """Top-N episodes by anomaly score (deterministic for testability)."""
    pool = _fetch_episodes(soil, limit=n)
    if not pool:
        return []
    pool.sort(key=_episode_anomaly_score, reverse=True)
    return pool[:n]


# ── Phase B: FSRS schedule refresh ────────────────────────────────────────


def _schedule_days_for_retention(stability: float, desired_retention: float = 0.9) -> float:
    """Inverse of FSRS retrievability — days until R drops to *desired_retention*.

    Inlined from ``belief.memory.fsrs.schedule_next_review`` so this
    module doesn't transitively import the (pydantic-dependent)
    ``belief.memory`` package during sandbox tests. The two
    implementations are pinned to agree by
    ``tests/test_sleep.py::test_inline_schedule_matches_fsrs`` (Mac-only).

        interval = S * (R_target^(-1/0.5) - 1) * 81/19
    """
    if stability <= 0:
        return 0.0
    if desired_retention <= 0 or desired_retention >= 1.0:
        return stability
    return stability * (desired_retention ** (-1.0 / 0.5) - 1.0) * (81.0 / 19.0)


def _refresh_fsrs_schedules(soil: Any, *, now_ts: float | None = None) -> int:
    """Recompute ``fsrs_next_review`` for active nutrients with stale schedules.

    "Stale" means: never scheduled (``fsrs_next_review == 0``) OR
    overdue (``fsrs_next_review < now``). For each, we recompute
    ``next_review = now + schedule_days_for_retention(stability) * 86400``
    and write the metadata back. No stability/difficulty/reps/lapses
    changes.

    Returns the number of nutrients whose schedule was refreshed.
    Intentionally tolerant of soils that don't have the typed
    collections layer — returns 0 if the schema is unfamiliar.
    """
    now = now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp()
    refreshed = 0

    collections = getattr(soil, "_collections", None)
    if not collections:
        return 0

    for col in collections.values():
        try:
            count = col.count()
        except Exception:
            continue
        if count == 0:
            continue
        try:
            data = col.get(include=["metadatas"], limit=count)
        except Exception:
            continue
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else None
            if not isinstance(meta, dict):
                continue
            # Skip invalidated.
            valid_until = float(meta.get("valid_until", 0.0) or 0.0)
            if valid_until > 0 and valid_until <= now:
                continue
            next_review = float(meta.get("fsrs_next_review", 0.0) or 0.0)
            if next_review > now:
                continue  # not stale
            stability = float(meta.get("fsrs_stability", meta.get("stability", 1.0)) or 1.0)
            interval_days = _schedule_days_for_retention(stability)
            new_next = now + interval_days * 86400.0
            updated = dict(meta)
            updated["fsrs_next_review"] = new_next
            try:
                col.update(ids=[doc_id], metadatas=[updated])
                refreshed += 1
            except Exception as e:
                logger.debug("Sleep: fsrs refresh skipped %s (%s)", doc_id, e)
    return refreshed


# ── The organ ──────────────────────────────────────────────────────────────


async def run(
    config: SleepConfig | None = None,
    *,
    soil: Any | None = None,
    economist: Economist | None = None,
    registry: Any | None = None,
    crystallizer: Any | None = None,
) -> SleepResult:
    """Execute up to ``cycles`` consolidation cycles.

    ``crystallizer`` is an injection slot — pass a module-like object
    exposing ``sweep_templates``, ``propose_invariants``,
    ``filter_candidates``, ``promote_to_covenant`` to unit-test Sleep
    without importing the real crystallizer (which drags pydantic +
    LLM client). Production callers leave it None.
    """
    config = config or SleepConfig()
    started = time.monotonic()
    state_path = config.state_path or _DEFAULT_STATE_PATH
    audit_path = config.audit_path or _DEFAULT_AUDIT_PATH

    # Lazy injections
    if soil is None:
        from belief.memory.soil import Soil  # noqa: PLC0415

        soil = Soil()
    if economist is None:
        economist = Economist()
    if registry is None and config.crystallize:
        try:
            from belief.validators.covenant_registry import CovenantRegistry  # noqa: PLC0415

            registry = CovenantRegistry(soil)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Sleep: CovenantRegistry unavailable, disabling crystallize: %s", e)
            config.crystallize = False
    if crystallizer is None and config.crystallize:
        try:
            from belief.evolution import crystallizer as _real_crystallizer  # noqa: PLC0415

            crystallizer = _real_crystallizer
        except Exception as e:  # pragma: no cover
            logger.warning("Sleep: crystallizer unavailable, disabling Phase A: %s", e)
            config.crystallize = False

    result = SleepResult(dry_run=config.dry_run)

    with _sleep_lock(state_path) as held:
        if not held:
            result.truncated_reason = "concurrent_run"
            _audit_append(audit_path, {"event": "concurrent_run_skipped"})
            result.duration_seconds = time.monotonic() - started
            return result

        for cycle_idx in range(max(0, config.cycles)):
            elapsed = time.monotonic() - started
            if elapsed > config.max_minutes * 60:
                result.truncated_reason = "timeout"
                _audit_append(
                    audit_path,
                    {
                        "event": "cycle_skipped",
                        "cycle": cycle_idx,
                        "reason": "max_minutes_exceeded",
                        "elapsed_seconds": round(elapsed, 2),
                    },
                )
                break

            if result.cost_usd + SLEEP_REPLAY_CYCLE_USD > config.budget_usd:
                result.truncated_reason = "budget"
                _audit_append(
                    audit_path,
                    {
                        "event": "cycle_skipped",
                        "cycle": cycle_idx,
                        "reason": "local_budget_exhausted",
                        "spent_usd": round(result.cost_usd, 6),
                    },
                )
                break

            quote = economist.quote(SLEEP_ACTION_KEY, SLEEP_REPLAY_CYCLE_USD)
            if not quote.approved:
                result.truncated_reason = "economist"
                result.economist_approved = False
                _audit_append(
                    audit_path,
                    {
                        "event": "cycle_skipped",
                        "cycle": cycle_idx,
                        "reason": "economist_rejected",
                        "economist_reason": quote.reason,
                    },
                )
                break

            cycle_cost = await _run_one_cycle(
                cycle_idx,
                config,
                soil=soil,
                registry=registry,
                crystallizer=crystallizer,
                audit_path=audit_path,
                result=result,
            )

            if not config.dry_run:
                economist.commit(SLEEP_ACTION_KEY, cycle_cost)
            result.cost_usd += cycle_cost
            result.cycles_completed += 1

            if config.synth_challenges:
                _audit_append(
                    audit_path,
                    {
                        "event": "phase_c_deferred",
                        "cycle": cycle_idx,
                        "note": "synth_challenges deferred to Curiosity (Sessions 6-7)",
                    },
                )
                logger.warning(
                    "Sleep Phase C (synth_challenges) is deferred; skipping cycle %d.",
                    cycle_idx,
                )

        if not config.dry_run:
            _atomic_write(
                state_path,
                {
                    "last_run_iso": datetime.now(timezone.utc).isoformat(),
                    "cycles_completed": result.cycles_completed,
                    "new_covenants": list(result.new_covenant_ids),
                    "fsrs_refreshed": result.fsrs_schedules_refreshed,
                    "cost_usd": round(result.cost_usd, 6),
                },
            )

    result.duration_seconds = time.monotonic() - started
    _audit_append(
        audit_path,
        {
            "event": "run_summary",
            "cycles_completed": result.cycles_completed,
            "new_covenants": len(result.new_covenant_ids),
            "fsrs_refreshed": result.fsrs_schedules_refreshed,
            "cost_usd": round(result.cost_usd, 6),
            "duration_seconds": round(result.duration_seconds, 4),
            "truncated_reason": result.truncated_reason,
            "dry_run": config.dry_run,
        },
    )
    return result


async def _run_one_cycle(
    cycle_idx: int,
    config: SleepConfig,
    *,
    soil: Any,
    registry: Any,
    crystallizer: Any,
    audit_path: Path,
    result: SleepResult,
) -> float:
    """Execute one Phase-A + Phase-B cycle. Returns the actual cost in USD.

    Phase A failures (e.g., LLM unreachable) downgrade gracefully —
    template sweep still runs, the cycle still counts. Phase B always
    runs unless ``config.recompute_fsrs=False`` or ``dry_run=True``.
    """
    cycle_cost = 0.0
    new_covenant_ids: list[str] = []

    # Phase A — anomaly-weighted replay → crystallizer
    if config.crystallize and registry is not None and crystallizer is not None:
        episodes = _sample_anomaly_weighted(soil, n=config.episodes_per_cycle)
        result.episodes_replayed += len(episodes)
        if episodes:
            template_cands = crystallizer.sweep_templates(episodes)
            try:
                existing = registry.get_all_covenant_descriptions()
            except Exception:
                existing = []
            try:
                claude_cands = await crystallizer.propose_invariants(episodes, existing)
                # Conservative cost estimate; future: read actual from llm client.
                cycle_cost += SLEEP_REPLAY_CYCLE_USD
            except Exception as e:
                logger.warning("Sleep: claude proposer failed in cycle %d: %s", cycle_idx, e)
                claude_cands = []
            all_cands = list(template_cands) + list(claude_cands)
            try:
                filtered = crystallizer.filter_candidates(all_cands, episodes)
            except Exception as e:
                logger.warning("Sleep: filter_candidates failed in cycle %d: %s", cycle_idx, e)
                filtered = []
            for cand in filtered:
                if not getattr(cand, "qualified", False):
                    continue
                if config.dry_run:
                    new_covenant_ids.append(f"dry-run:{getattr(cand, 'name', '?')}")
                    continue
                try:
                    cid = crystallizer.promote_to_covenant(cand, soil)
                    new_covenant_ids.append(cid)
                except Exception as e:
                    logger.warning("Sleep: promotion failed in cycle %d: %s", cycle_idx, e)
                    _audit_append(
                        audit_path,
                        {
                            "event": "promotion_failed",
                            "cycle": cycle_idx,
                            "candidate": getattr(cand, "name", "?"),
                            "error": str(e),
                        },
                    )
            if not config.dry_run and new_covenant_ids:
                try:
                    registry.load_dynamic_covenants()
                except Exception as e:
                    logger.warning("Sleep: registry reload failed: %s", e)
        _audit_append(
            audit_path,
            {
                "event": "phase_a_complete",
                "cycle": cycle_idx,
                "episodes_sampled": len(episodes),
                "candidates_promoted": len(new_covenant_ids),
                "dry_run": config.dry_run,
            },
        )

    # Phase B — FSRS housekeeping
    fsrs_refreshed = 0
    if config.recompute_fsrs and not config.dry_run:
        try:
            fsrs_refreshed = _refresh_fsrs_schedules(soil)
        except Exception as e:
            logger.warning("Sleep: FSRS refresh failed in cycle %d: %s", cycle_idx, e)
        _audit_append(
            audit_path,
            {
                "event": "phase_b_complete",
                "cycle": cycle_idx,
                "schedules_refreshed": fsrs_refreshed,
            },
        )

    result.new_covenant_ids.extend(new_covenant_ids)
    result.fsrs_schedules_refreshed += fsrs_refreshed
    return cycle_cost


# ── CLI helpers ────────────────────────────────────────────────────────────


def cli_format_result(result: SleepResult) -> str:
    header = "Sleep (DRY RUN)" if result.dry_run else "Sleep"
    lines = [
        f"{header} — {result.cycles_completed} cycle(s) in {result.duration_seconds:.2f}s, "
        f"${result.cost_usd:.4f} spent",
        f"  episodes replayed:      {result.episodes_replayed}",
        f"  new covenants:          {len(result.new_covenant_ids)}",
        f"  fsrs schedules refreshed: {result.fsrs_schedules_refreshed}",
    ]
    if result.truncated_reason:
        lines.append(f"  truncated:              {result.truncated_reason}")
    if result.new_covenant_ids:
        lines.append("  covenant ids:")
        for cid in result.new_covenant_ids[:5]:
            lines.append(f"    - {cid}")
        if len(result.new_covenant_ids) > 5:
            lines.append(f"    ... and {len(result.new_covenant_ids) - 5} more")
    return "\n".join(lines)


def result_to_dict(result: SleepResult) -> dict:
    from dataclasses import asdict

    return asdict(result)
