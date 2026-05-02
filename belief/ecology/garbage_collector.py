"""Garbage Collector — remove broken / invalid / duplicate items from soil.

Per v3.3 spec §3.2, GC is a different decision rule from Predator:
Predator removes *low-utility* items, GC removes *actively wrong* items.

**Detection rules implemented:**

1. **broken_tool** — entries in ``belief_tools`` whose ``code`` field
   fails ``ast.parse``. Soft-deleted via ``soil.invalidate_nutrient``
   with reason explaining the syntax error. Importlib resolution of
   declared dependencies is intentionally NOT attempted — actually
   importing a tool's deps has side effects and may install network
   calls, which violates "best-effort cleanup, never break the world."
2. **invalid_covenant** — entries in ``belief_covenants`` whose
   ``code_sample`` fails ``ast.parse``. Same soft-delete path. Static
   covenants (the 6 hand-written ones in covenant_registry) are NOT
   touched — only crystallized dynamic covenants are in soil.
3. **duplicate_tool** — pairs of entries in ``belief_tools`` with
   identical normalized source (whitespace + comment stripped via
   ``ast.parse → ast.unparse``). Keeps the entry with the higher
   ``reinforcement_count`` (ties broken by older ``created_at``);
   invalidates the duplicate. Returns ``(kept_id, removed_id)`` tuples.
4. **orphan_episode** — spec'd as "episodes that reference deleted
   file paths." The current ``episode_recorder`` stores ``code_files``
   inline as metadata rather than as external paths, so the spec's
   premise doesn't fit this codebase. We retain the field in
   ``GCReport`` for spec-API parity but the detector currently returns
   ``[]``. Document this explicitly so future-Joe doesn't think it
   silently broke.

iCloud-fork awareness (per Joe's `project_icloud_dupes.md` memory): the
duplicate detector compares normalized SOURCE, so iCloud-spawned
``foo 2.py`` copies with identical bodies are caught as duplicates,
which is the correct behaviour. No realpath dance needed.

Plays nicely with Predator's tombstones: walks
``iter_all_nutrients(include_invalidated=False)``, so already-invalidated
items are skipped automatically.

Budget integration: GC is LLM-free in v3.3 — Economist quote at $0.00
exercises the contract for audit consistency.

Audit JSONL: ``~/.belief-engine/audit/ecology_gc.jsonl``.
State:      ``~/.belief-engine/gc_state.json``.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from belief.ecology.economist import Economist

logger = logging.getLogger("belief.ecology.gc")

# ── Defaults ────────────────────────────────────────────────────────────────

GC_ACTION_KEY = "gc.run"

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_STATE_PATH = _BELIEF_HOME / "gc_state.json"
_DEFAULT_AUDIT_PATH = _BELIEF_HOME / "audit" / "ecology_gc.jsonl"

# Reasons (used as prefixes in invalidate_nutrient(reason=...))
REASON_BROKEN_TOOL = "gc: broken_tool: ast.parse failed"
REASON_INVALID_COVENANT = "gc: invalid_covenant: ast.parse failed"
REASON_DUPLICATE_TOOL = "gc: duplicate_tool: identical normalized source"


# ── Data types ─────────────────────────────────────────────────────────────


@dataclass
class GCReport:
    """Per-spec §3.2 result shape. Safe to round-trip through JSON."""

    broken_tools: list[str] = field(default_factory=list)
    orphan_episodes: list[str] = field(default_factory=list)
    invalid_covenants: list[str] = field(default_factory=list)
    duplicate_tools: list[tuple[str, str]] = field(default_factory=list)  # (kept, removed)
    examined: int = 0
    cleaned: int = 0  # number of nutrients soft-tombstoned this run
    dry_run: bool = False
    check_only: bool = False
    duration_seconds: float = 0.0
    economist_approved: bool = True
    economist_reason: str = ""


# ── State + audit (mirror Predator/Sleep patterns) ────────────────────────


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
        logger.warning("GC audit write failed: %s", e)


# ── Source normalization (for duplicate detection) ────────────────────────


def _normalized_source(code: str) -> str | None:
    """Return canonical form of ``code`` for equality comparison.

    Uses ``ast.parse → ast.unparse`` which strips comments and
    normalizes whitespace, so two functionally identical files with
    different formatting compare equal. Returns ``None`` if the code
    doesn't parse — broken code can't be deduplicated against, since
    broken_tool detection handles it separately.
    """
    if not code:
        return ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _is_parseable(code: str) -> tuple[bool, str]:
    """Return (parseable, error_message). Empty string code is parseable."""
    if not code:
        return True, ""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"{type(e).__name__}: {e.msg} (line {e.lineno})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ── The organ ──────────────────────────────────────────────────────────────


async def run(
    check_only: bool = False,
    dry_run: bool = False,
    *,
    soil: Any | None = None,
    economist: Economist | None = None,
    state_path: Path | None = None,
    audit_path: Path | None = None,
) -> GCReport:
    """Walk soil, identify broken/invalid/duplicate items, soft-tombstone them.

    ``check_only=True`` short-circuits like ``dry_run`` but is logged
    distinctly in the audit trail so Joe can tell "I'm just looking" from
    "I was going to act but bailed."

    ``soil`` and ``economist`` are dependency-injection slots; production
    callers leave them None and the real instances are constructed lazily
    so module load doesn't drag ChromaDB.
    """
    started = time.monotonic()
    state_path = state_path or _DEFAULT_STATE_PATH
    audit_path = audit_path or _DEFAULT_AUDIT_PATH
    effective_dry = bool(dry_run) or bool(check_only)

    if soil is None:
        from belief.memory.soil import Soil  # noqa: PLC0415

        soil = Soil()
    if economist is None:
        economist = Economist()

    quote = economist.quote(GC_ACTION_KEY, estimated_usd=0.0)
    if not quote.approved:
        result = GCReport(
            economist_approved=False,
            economist_reason=quote.reason,
            duration_seconds=time.monotonic() - started,
            dry_run=bool(dry_run),
            check_only=bool(check_only),
        )
        _audit_append(
            audit_path,
            {
                "event": "rejected_by_economist",
                "reason": quote.reason,
            },
        )
        return result

    # Bucket nutrients by type as we iterate (one pass, no double-fetch).
    tools: list[Any] = []
    covenants: list[Any] = []
    examined = 0
    for nutrient in soil.iter_all_nutrients(include_invalidated=False):
        examined += 1
        ntype = _nutrient_type_str(nutrient)
        if ntype in {"pattern", "skeleton", "antipattern"}:
            # Skeletons and patterns may also have code, but we only GC
            # tools and covenants per spec scope. Patterns stored as
            # nutrients are not "broken executable code" in the same
            # sense — they are reference snippets.
            if _has_tool_shape(nutrient):
                tools.append(nutrient)
        elif ntype == "covenant":
            covenants.append(nutrient)
        # Everything else (episodes if they ever land in iter_all): skip.

    # ── Detection passes ────────────────────────────────────────────────
    broken_tools = _find_broken_tools(tools)
    invalid_covenants = _find_invalid_covenants(covenants)
    duplicate_tools = _find_duplicate_tools(tools)

    report = GCReport(
        broken_tools=[nid for nid, _ in broken_tools],
        invalid_covenants=[nid for nid, _ in invalid_covenants],
        duplicate_tools=duplicate_tools,
        orphan_episodes=[],  # see module docstring
        examined=examined,
        dry_run=bool(dry_run),
        check_only=bool(check_only),
    )

    # ── Action pass ─────────────────────────────────────────────────────
    cleaned = 0
    if not effective_dry:
        for nid, reason in broken_tools:
            if _safe_invalidate(soil, nid, reason):
                cleaned += 1
        for nid, reason in invalid_covenants:
            if _safe_invalidate(soil, nid, reason):
                cleaned += 1
        for kept_id, removed_id in duplicate_tools:
            reason = f"{REASON_DUPLICATE_TOOL} (kept={kept_id})"
            if _safe_invalidate(soil, removed_id, reason):
                cleaned += 1

    report.cleaned = cleaned

    # Per-finding audit lines for forensics.
    for nid, reason in broken_tools:
        _audit_append(
            audit_path,
            {
                "event": "finding",
                "category": "broken_tool",
                "nutrient_id": nid,
                "reason": reason,
                "applied": (not effective_dry),
            },
        )
    for nid, reason in invalid_covenants:
        _audit_append(
            audit_path,
            {
                "event": "finding",
                "category": "invalid_covenant",
                "nutrient_id": nid,
                "reason": reason,
                "applied": (not effective_dry),
            },
        )
    for kept_id, removed_id in duplicate_tools:
        _audit_append(
            audit_path,
            {
                "event": "finding",
                "category": "duplicate_tool",
                "kept": kept_id,
                "removed": removed_id,
                "applied": (not effective_dry),
            },
        )

    duration = time.monotonic() - started
    report.duration_seconds = duration

    if not effective_dry:
        try:
            _atomic_write(
                state_path,
                {
                    "last_run_iso": datetime.now(timezone.utc).isoformat(),
                    "examined": examined,
                    "cleaned": cleaned,
                    "broken_tools": len(report.broken_tools),
                    "invalid_covenants": len(report.invalid_covenants),
                    "duplicate_tools": len(report.duplicate_tools),
                },
            )
        except Exception as e:  # pragma: no cover
            logger.warning("GC: state write failed: %s", e)
        economist.commit(GC_ACTION_KEY, actual_usd=0.0)

    _audit_append(
        audit_path,
        {
            "event": "run_summary",
            "examined": examined,
            "cleaned": cleaned,
            "broken_tools": len(report.broken_tools),
            "invalid_covenants": len(report.invalid_covenants),
            "duplicate_tools": len(report.duplicate_tools),
            "duration_seconds": round(duration, 4),
            "dry_run": bool(dry_run),
            "check_only": bool(check_only),
        },
    )
    return report


# ── Detection helpers ──────────────────────────────────────────────────────


def _find_broken_tools(tools: list[Any]) -> list[tuple[str, str]]:
    """Return [(nutrient_id, reason)] for tools whose code fails to parse."""
    out: list[tuple[str, str]] = []
    for tool in tools:
        code = _tool_code(tool)
        ok, msg = _is_parseable(code)
        if not ok:
            out.append((str(_nutrient_id(tool)), f"{REASON_BROKEN_TOOL}: {msg}"))
    return out


def _find_invalid_covenants(covenants: list[Any]) -> list[tuple[str, str]]:
    """Return [(nutrient_id, reason)] for covenants whose code_sample fails to parse."""
    out: list[tuple[str, str]] = []
    for cov in covenants:
        code = _covenant_code(cov)
        if not code:
            # Covenant with no code field — not "invalid", just sparse. Skip.
            continue
        ok, msg = _is_parseable(code)
        if not ok:
            out.append((str(_nutrient_id(cov)), f"{REASON_INVALID_COVENANT}: {msg}"))
    return out


def _find_duplicate_tools(tools: list[Any]) -> list[tuple[str, str]]:
    """Return [(kept_id, removed_id)] for tools with identical normalized source.

    Tie-breaker: keep the tool with higher ``reinforcement_count``;
    on equal reinforcements, keep the OLDER ``created_at`` (more
    established). Returns empty list if all tools are unique.
    """
    by_norm: dict[str, list[Any]] = {}
    for tool in tools:
        code = _tool_code(tool)
        norm = _normalized_source(code)
        if norm is None:
            continue  # broken — handled by broken_tool category
        by_norm.setdefault(norm, []).append(tool)

    pairs: list[tuple[str, str]] = []
    for group in by_norm.values():
        if len(group) < 2:
            continue
        group.sort(
            key=lambda t: (
                -int(getattr(t, "reinforcement_count", 0) or 0),
                float(getattr(t, "created_at", 0.0) or 0.0),
            )
        )
        kept = group[0]
        for dup in group[1:]:
            pairs.append((str(_nutrient_id(kept)), str(_nutrient_id(dup))))
    return pairs


def _safe_invalidate(soil: Any, nutrient_id: str, reason: str) -> bool:
    try:
        return bool(soil.invalidate_nutrient(nutrient_id, reason=reason))
    except Exception as e:
        logger.warning("GC: invalidate_nutrient %s failed: %s", nutrient_id, e)
        return False


# ── Field accessors (duck-typed against Nutrient + SelfAuthoredTool) ──────


def _nutrient_type_str(nutrient: Any) -> str:
    nt = getattr(nutrient, "nutrient_type", None)
    if nt is None:
        return ""
    val = getattr(nt, "value", nt)
    return str(val).lower()


def _nutrient_id(nutrient: Any) -> str:
    return str(getattr(nutrient, "nutrient_id", "?"))


def _has_tool_shape(nutrient: Any) -> bool:
    """Tool-like nutrients carry a ``code`` field (or ``code_sample``)."""
    return bool(getattr(nutrient, "code", None) or getattr(nutrient, "code_sample", None))


def _tool_code(nutrient: Any) -> str:
    """Tools store source in ``code``; some legacy entries may use ``code_sample``."""
    return str(getattr(nutrient, "code", None) or getattr(nutrient, "code_sample", "") or "")


def _covenant_code(nutrient: Any) -> str:
    """Crystallized covenants store their checker in ``code_sample``."""
    return str(getattr(nutrient, "code_sample", "") or "")


# ── CLI helpers ────────────────────────────────────────────────────────────


def cli_format_result(report: GCReport) -> str:
    if report.check_only:
        header = "GC (CHECK ONLY)"
    elif report.dry_run:
        header = "GC (DRY RUN)"
    else:
        header = "GC"
    lines = [
        f"{header} — examined {report.examined}, cleaned {report.cleaned} "
        f"in {report.duration_seconds:.2f}s",
        f"  broken tools:      {len(report.broken_tools)}",
        f"  invalid covenants: {len(report.invalid_covenants)}",
        f"  duplicate tools:   {len(report.duplicate_tools)}",
        f"  orphan episodes:   {len(report.orphan_episodes)} (detector inactive — see module docstring)",
        f"  Economist:         approved={report.economist_approved} ({report.economist_reason})",
    ]
    if report.broken_tools:
        lines.append("  broken tool ids:")
        for nid in report.broken_tools[:5]:
            lines.append(f"    - {nid}")
    if report.duplicate_tools:
        lines.append("  duplicate (kept → removed):")
        for kept, removed in report.duplicate_tools[:5]:
            lines.append(f"    - {kept[:24]}{'…':>4} → {removed[:24]}")
    return "\n".join(lines)


def report_to_dict(report: GCReport) -> dict:
    return asdict(report)
