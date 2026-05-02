"""Predator — utility-driven culling of low-value soil nutrients (v3.3 §3.1).

Predator is the first selection-pressure organ. It walks the active soil,
scores each nutrient with the shared utility function (see
``belief.ecology._utility``), and soft-tombstones the bottom slice via
``Soil.invalidate_nutrient``. Nothing is hard-deleted — Sleep can later
restore false-positive prunes via ``Soil.revalidate_nutrient``.

Safety rails (per spec §3.1 + Joe's accepted §6 pushback):
    * ``min_age_days`` — never touch nutrients younger than 7 days. Avoids
      pruning a pattern before it has had a chance to be reused.
    * ``max_delete_per_run`` — soft cap; defaults to 50.
    * **First-run safety cap** — if no ``predator_state.json`` exists, the
      effective cap is hard-clamped to 10 deletions regardless of config,
      unless ``--confirm-first-run`` is passed. The current 356-build soil
      could otherwise see hundreds of tombstones on the first run.
    * ``dry_run`` — produces a report with no soil writes.
    * Covenants are skipped categorically. They are immutable per CLAUDE.md.

Budget integration: Predator is LLM-free in the v3.3 shell, so the
Economist quote is for $0.00 — the call still happens so the contract is
exercised and the audit trail is consistent with future organs that do
spend.

Audit JSONL: ``~/.belief-engine/audit/ecology_predator.jsonl``.
State:      ``~/.belief-engine/predator_state.json``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from belief.ecology._utility import (
    UtilityBreakdown,
    compute_utility,
    load_weights,
)
from belief.ecology.economist import Economist

logger = logging.getLogger("belief.ecology.predator")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_UTILITY_THRESHOLD = 0.15
DEFAULT_MIN_AGE_DAYS = 7
DEFAULT_MAX_DELETE_PER_RUN = 50
FIRST_RUN_HARD_CAP = 10
PREDATOR_ACTION_KEY = "predator.run"

# By default Predator considers PATTERN / ANTIPATTERN / SKELETON nutrients.
# COVENANT is excluded — covenants are immutable per CLAUDE.md, and even
# soft-tombstoning them would short-circuit the validator's covenant
# enforcement on subsequent builds.
DEFAULT_NUTRIENT_TYPES: tuple[str, ...] = ("pattern", "antipattern", "skeleton")
NEVER_PRUNE_TYPES: frozenset[str] = frozenset({"covenant"})

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_STATE_PATH = _BELIEF_HOME / "predator_state.json"
_DEFAULT_AUDIT_PATH = _BELIEF_HOME / "audit" / "ecology_predator.jsonl"


# ── Config + result types ──────────────────────────────────────────────────


@dataclass
class PredatorConfig:
    """Inputs to a Predator run.

    ``collections`` is a list of ``NutrientType`` string values
    (e.g. ``"pattern"``, ``"antipattern"``). Defaults to everything
    except covenants. Pass an explicit list to scope (e.g. only prune
    failed antipatterns).
    """

    collections: tuple[str, ...] = DEFAULT_NUTRIENT_TYPES
    utility_threshold: float = DEFAULT_UTILITY_THRESHOLD
    min_age_days: int = DEFAULT_MIN_AGE_DAYS
    max_delete_per_run: int = DEFAULT_MAX_DELETE_PER_RUN
    dry_run: bool = False
    confirm_first_run: bool = False
    weights_path: Path | None = None  # override for tests
    state_path: Path | None = None
    audit_path: Path | None = None


@dataclass
class PredatorResult:
    """Outcome of a Predator run — safe to round-trip through JSON."""

    examined: int = 0
    eligible: int = 0  # past min_age, in scoped collections, not covenant
    tombstoned: int = 0  # actually invalidated this run
    skipped_first_run_cap: int = 0  # candidates dropped due to safety cap
    skipped_per_run_cap: int = 0  # candidates dropped due to max_delete cap
    items: list[dict] = field(default_factory=list)  # per-tombstone audit rows
    duration_seconds: float = 0.0
    dry_run: bool = False
    economist_approved: bool = True
    economist_reason: str = ""
    first_run: bool = False
    effective_cap: int = 0


# ── First-run state ────────────────────────────────────────────────────────


def _is_first_run(state_path: Path) -> bool:
    """First run iff no state file (i.e., Predator has never committed before)."""
    return not state_path.exists()


def _record_run_state(
    state_path: Path,
    summary: dict[str, Any],
) -> None:
    """Atomically persist a small marker so the next call isn't first-run.

    Mirrors the Economist's atomic-write pattern. Crash-mid-write leaves
    the prior valid state file or no file (both are recoverable).
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.flush()
    import os

    os.replace(tmp, state_path)


# ── Audit ──────────────────────────────────────────────────────────────────


def _audit_append(audit_path: Path, record: dict) -> None:
    """Append one JSONL line. Never raises — audit is best-effort."""
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:  # pragma: no cover
        logger.warning("Predator audit write failed: %s", e)


# ── The organ ──────────────────────────────────────────────────────────────


async def run(
    config: PredatorConfig | None = None,
    *,
    soil: Any | None = None,
    economist: Economist | None = None,
) -> PredatorResult:
    """Score the active soil and soft-tombstone the bottom slice.

    ``soil`` and ``economist`` are dependency-injection slots for tests
    and for future grinder use. In normal operation, both are
    constructed lazily inside the function so the import graph stays
    clean (importing this module does not pull in ChromaDB).
    """
    config = config or PredatorConfig()
    started = time.monotonic()

    state_path = config.state_path or _DEFAULT_STATE_PATH
    audit_path = config.audit_path or _DEFAULT_AUDIT_PATH

    # Lazy soil import so test + module load doesn't require ChromaDB.
    if soil is None:
        from belief.memory.soil import Soil  # noqa: PLC0415

        soil = Soil()
    if economist is None:
        economist = Economist()

    # ── Economist gate (LLM-free, but contract still exercised) ─────────
    quote = economist.quote(PREDATOR_ACTION_KEY, estimated_usd=0.0)
    if not quote.approved:
        result = PredatorResult(
            economist_approved=False,
            economist_reason=quote.reason,
            duration_seconds=time.monotonic() - started,
            dry_run=config.dry_run,
        )
        _audit_append(
            audit_path,
            {
                "event": "rejected_by_economist",
                "reason": quote.reason,
                "config": _config_to_audit_dict(config),
            },
        )
        return result

    # ── First-run cap ───────────────────────────────────────────────────
    first_run = _is_first_run(state_path)
    if first_run and not config.confirm_first_run:
        effective_cap = min(config.max_delete_per_run, FIRST_RUN_HARD_CAP)
    else:
        effective_cap = config.max_delete_per_run

    weights = load_weights(config.weights_path)
    now_ts = datetime.now(timezone.utc).timestamp()
    min_age_seconds = float(config.min_age_days) * 86400.0
    allowed_types = frozenset(config.collections) - NEVER_PRUNE_TYPES

    # ── Score every active nutrient ──────────────────────────────────────
    scored: list[tuple[float, UtilityBreakdown, Any]] = []  # (utility, breakdown, nutrient)
    examined = 0
    eligible = 0
    for nutrient in soil.iter_all_nutrients(include_invalidated=False):
        examined += 1
        ntype = _nutrient_type_str(nutrient)
        if ntype in NEVER_PRUNE_TYPES:
            continue
        if allowed_types and ntype not in allowed_types:
            continue
        created_at = float(getattr(nutrient, "created_at", 0.0))
        if created_at <= 0 or (now_ts - created_at) < min_age_seconds:
            continue
        eligible += 1
        breakdown = compute_utility(nutrient, weights=weights, now_ts=now_ts)
        if breakdown.total < config.utility_threshold:
            scored.append((breakdown.total, breakdown, nutrient))

    # ── Pick the bottom slice ───────────────────────────────────────────
    scored.sort(key=lambda t: t[0])  # ascending — worst first
    candidates = scored[:effective_cap]
    skipped_per_run = max(0, len(scored) - effective_cap) if not first_run else 0
    skipped_first_run = (
        max(0, len(scored) - effective_cap) if first_run and not config.confirm_first_run else 0
    )

    # ── Apply (or report) ───────────────────────────────────────────────
    items: list[dict] = []
    tombstoned = 0
    for utility, breakdown, nutrient in candidates:
        nid = breakdown.nutrient_id
        ntype = _nutrient_type_str(nutrient)
        reason = (
            f"predator: utility={utility:.4f} below threshold {config.utility_threshold:.4f} "
            f"(usage={breakdown.usage:.2f} retr={breakdown.retrievability:.2f} "
            f"recency={breakdown.recency:.2f} failure={breakdown.failure:.2f})"
        )
        record = {
            "nutrient_id": nid,
            "nutrient_type": ntype,
            "utility": round(utility, 6),
            "usage": round(breakdown.usage, 6),
            "retrievability": round(breakdown.retrievability, 6),
            "recency": round(breakdown.recency, 6),
            "failure": round(breakdown.failure, 6),
            "reason": reason,
            "applied": False,
        }
        if not config.dry_run:
            try:
                ok = soil.invalidate_nutrient(nid, reason=reason, now=now_ts)
            except Exception as e:
                logger.warning("Predator: invalidate_nutrient %s failed: %s", nid, e)
                ok = False
            record["applied"] = bool(ok)
            if ok:
                tombstoned += 1
        items.append(record)
        _audit_append(audit_path, {"event": "tombstone_candidate", **record})

    # ── Persist run summary + Economist commit ──────────────────────────
    duration = time.monotonic() - started
    result = PredatorResult(
        examined=examined,
        eligible=eligible,
        tombstoned=tombstoned,
        skipped_first_run_cap=skipped_first_run,
        skipped_per_run_cap=skipped_per_run,
        items=items,
        duration_seconds=duration,
        dry_run=config.dry_run,
        economist_approved=True,
        economist_reason=quote.reason,
        first_run=first_run,
        effective_cap=effective_cap,
    )

    if not config.dry_run:
        _record_run_state(
            state_path,
            {
                "last_run_ts": now_ts,
                "last_run_iso": datetime.now(timezone.utc).isoformat(),
                "last_examined": examined,
                "last_tombstoned": tombstoned,
                "first_run_was": first_run,
            },
        )
        economist.commit(PREDATOR_ACTION_KEY, actual_usd=0.0)

    _audit_append(
        audit_path,
        {
            "event": "run_summary",
            "examined": examined,
            "eligible": eligible,
            "tombstoned": tombstoned,
            "first_run": first_run,
            "dry_run": config.dry_run,
            "duration_seconds": round(duration, 4),
        },
    )
    return result


# ── Helpers ────────────────────────────────────────────────────────────────


def _nutrient_type_str(nutrient: Any) -> str:
    """Extract the lowercase ``nutrient_type`` string regardless of Enum vs str."""
    nt = getattr(nutrient, "nutrient_type", None)
    if nt is None:
        return ""
    val = getattr(nt, "value", nt)
    return str(val).lower()


def _config_to_audit_dict(config: PredatorConfig) -> dict:
    return {
        "collections": list(config.collections),
        "utility_threshold": config.utility_threshold,
        "min_age_days": config.min_age_days,
        "max_delete_per_run": config.max_delete_per_run,
        "dry_run": config.dry_run,
        "confirm_first_run": config.confirm_first_run,
    }


def result_to_dict(result: PredatorResult) -> dict:
    """JSON-friendly representation of a PredatorResult."""
    from dataclasses import asdict

    return asdict(result)


# ── CLI helpers (called from belief.cli) ───────────────────────────────────


def cli_format_result(result: PredatorResult) -> str:
    """One-screen summary for ``belief predator`` stdout."""
    header = "Predator (DRY RUN)" if result.dry_run else "Predator"
    lines = [
        f"{header} — examined {result.examined}, eligible {result.eligible}, "
        f"tombstoned {result.tombstoned} in {result.duration_seconds:.2f}s",
        f"  first run:        {result.first_run}",
        f"  effective cap:    {result.effective_cap}",
        f"  skipped (first-run cap): {result.skipped_first_run_cap}",
        f"  skipped (per-run cap):   {result.skipped_per_run_cap}",
        f"  Economist:        approved={result.economist_approved} ({result.economist_reason})",
    ]
    if result.items:
        lines.append("  worst items:")
        for item in result.items[:10]:
            applied = "✓" if item["applied"] else ("·" if result.dry_run else "✗")
            lines.append(
                f"    {applied} {item['nutrient_id'][:24]:24}  u={item['utility']:.3f}  "
                f"({item['nutrient_type']})"
            )
    return "\n".join(lines)
