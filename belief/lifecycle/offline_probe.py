"""Weekly engine-offline probe (mycorrhizal Stage 8, Area 12).

Modern arbuscular-mycorrhizal fungi cannot complete their life cycle without
a plant host — they lost their own lipid biosynthesis and import fatty acids
from the plant (Jiang et al. 2017). The coupling became *obligate* over
evolutionary time. Unlike biology, a software developer benefits from knowing
exactly how obligate the coupling has become: which operations genuinely
cannot run without the engine, versus which only happen to route through it.

This probe answers that question. It runs the engine's read-side and
durability paths with the LLM-spending machinery conceptually "offline" and
records which operations still work. Operations that fail are the ones that
have become structurally dependent on the engine — information for the
operator, not a failure condition.

Scoping note (consistent with Stage 3 cold-start): the Belief Engine is a
one-shot LangGraph per ``belief build``, not a long-running daemon with a
live agent registry. So "run registered agents against a sandboxed engine
with the cross-domain synthesizer disabled" becomes: run a set of registered
*offline checks* — pure-read / durability operations that the institutional
memory must support with no engine process and no LLM calls. Each check is a
callable returning ``(ok, detail)``. The default suite probes the soil layer,
the three SQLite ledgers, the signal store, and a snapshot round-trip — the
"can a developer use the institutional memory with nothing else running?"
surface from Area 11's resilience principle.

Output: a markdown report at ``~/.belief-engine/probes/YYYY-MM-DD.md`` for
the operator to review. The weekly cadence hooks into the photosynthesis
daemon; ``belief probe offline`` runs it on demand.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("belief.lifecycle.offline_probe")

_BELIEF_HOME = Path.home() / ".belief-engine"
_DEFAULT_PROBES_DIR = _BELIEF_HOME / "probes"

# A check returns (ok, detail). ok=True means the operation succeeded with
# the engine offline (no obligate coupling); ok=False means it failed —
# i.e. that operation has become engine-dependent.
OfflineCheck = Callable[[], tuple[bool, str]]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    error: Optional[str] = None


@dataclass
class OfflineProbeReport:
    ran_at: datetime
    results: list[CheckResult] = field(default_factory=list)
    report_path: Optional[Path] = None

    @property
    def all_passed(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def obligate_operations(self) -> list[str]:
        """Names of checks that FAILED — operations that can't survive
        engine-offline, i.e. have become structurally coupled."""
        return [r.name for r in self.results if not r.ok]

    def to_markdown(self) -> str:
        lines = [
            "# Engine-offline probe report",
            "",
            f"- **ran at (UTC):** {self.ran_at.isoformat()}",
            f"- **checks:** {len(self.results)}",
            f"- **passed:** {sum(1 for r in self.results if r.ok)}",
            f"- **obligate (failed offline):** {len(self.obligate_operations)}",
            "",
            "## Results",
            "",
            "| check | offline-ok | detail |",
            "|-------|-----------|--------|",
        ]
        for r in self.results:
            mark = "yes" if r.ok else "**NO**"
            detail = (r.detail or r.error or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.name} | {mark} | {detail[:160]} |")
        lines.append("")
        if self.obligate_operations:
            lines.append(
                "## Obligate coupling detected\n\n"
                "The following operations failed with the engine offline. "
                "They have become structurally dependent on the engine — "
                "this is information for the operator, not necessarily a bug. "
                "If a dependency was unintended, decouple it; if it's "
                "expected, document it.\n"
            )
            for name in self.obligate_operations:
                lines.append(f"- `{name}`")
        else:
            lines.append(
                "## No obligate coupling\n\n"
                "Every probed operation works with the engine offline — the "
                "institutional memory is usable independently of the engine "
                "process, as Area 11's resilience principle requires."
            )
        lines.append("")
        return "\n".join(lines)


class WeeklyOfflineProbe:
    """Runs offline checks and writes a dated markdown report.

    Default checks probe the durable institutional memory. Callers can
    register additional checks (e.g. an agent's own obligate-coupling
    probe) via ``register_check``.
    """

    def __init__(
        self,
        probes_dir: Path = _DEFAULT_PROBES_DIR,
        checks: Optional[dict[str, OfflineCheck]] = None,
        register_defaults: bool = True,
    ) -> None:
        self.probes_dir = Path(probes_dir).expanduser()
        self._checks: dict[str, OfflineCheck] = {}
        if register_defaults:
            self._register_default_checks()
        if checks:
            for name, fn in checks.items():
                self.register_check(name, fn)

    def register_check(self, name: str, fn: OfflineCheck) -> None:
        if not name:
            raise ValueError("check name must be non-empty")
        self._checks[name] = fn

    def check_names(self) -> list[str]:
        return sorted(self._checks.keys())

    # ── Default check suite ─────────────────────────────────────────────

    def _register_default_checks(self) -> None:
        self.register_check("soil_readable", _check_soil_readable)
        self.register_check("reciprocity_ledger_readable", _check_reciprocity)
        self.register_check("niche_ledger_readable", _check_niche)
        self.register_check("signal_store_readable", _check_signal_store)
        self.register_check("snapshot_round_trip", _check_snapshot_round_trip)

    # ── Run ─────────────────────────────────────────────────────────────

    def run(self, write_report: bool = True, now: Optional[datetime] = None) -> OfflineProbeReport:
        """Execute every registered check and (optionally) write the report.

        A check that raises is recorded as a failure with the traceback
        summary — a raising check is itself an obligate-coupling signal.
        """
        now = now or datetime.now(timezone.utc)
        report = OfflineProbeReport(ran_at=now)
        for name in sorted(self._checks.keys()):
            fn = self._checks[name]
            try:
                ok, detail = fn()
                report.results.append(CheckResult(name=name, ok=bool(ok), detail=str(detail)))
            except Exception as e:  # check itself blew up → treat as failure
                report.results.append(
                    CheckResult(
                        name=name,
                        ok=False,
                        detail="",
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                logger.debug("offline check %r raised: %s", name, traceback.format_exc())
        if write_report:
            report.report_path = self._write_report(report, now)
        return report

    def _write_report(self, report: OfflineProbeReport, now: datetime) -> Path:
        self.probes_dir.mkdir(parents=True, exist_ok=True)
        path = self.probes_dir / f"{now.strftime('%Y-%m-%d')}.md"
        path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info("offline probe report written: %s", path)
        return path


# ── Default checks (module-level so they're picklable / overridable) ────────


def _check_soil_readable() -> tuple[bool, str]:
    """ChromaDB soil must be openable + countable with no engine running."""
    try:
        import chromadb  # noqa: PLC0415

        soil_dir = _BELIEF_HOME / "soil"
        if not soil_dir.exists():
            return True, "no soil dir yet (fresh install) — vacuously offline-ok"
        client = chromadb.PersistentClient(path=str(soil_dir))
        cols = client.list_collections()
        total = 0
        for c in cols:
            try:
                total += c.count()
            except Exception:
                pass
        return True, f"{len(cols)} collections, {total} records readable offline"
    except Exception as e:
        return False, f"soil unreadable offline: {type(e).__name__}: {e}"


def _check_reciprocity() -> tuple[bool, str]:
    from belief.memory.reciprocity import ReciprocityLedger

    db = _BELIEF_HOME / "reciprocity.db"
    if not db.exists():
        return True, "no reciprocity.db yet — vacuously offline-ok"
    ledg = ReciprocityLedger(db_path=db)
    try:
        n = len(ledg.all_agent_ids())
        return True, f"{n} agents readable offline"
    finally:
        ledg.close()


def _check_niche() -> tuple[bool, str]:
    from belief.memory.niche_ledger import NicheLedger

    db = _BELIEF_HOME / "niches.db"
    if not db.exists():
        return True, "no niches.db yet — vacuously offline-ok"
    nl = NicheLedger(db_path=db)
    try:
        n = nl.count_niches()
        return True, f"{n} niches readable offline"
    finally:
        nl.close()


def _check_signal_store() -> tuple[bool, str]:
    from belief.signal.store import SignalStore

    db = _BELIEF_HOME / "signals.db"
    if not db.exists():
        return True, "no signals.db yet — vacuously offline-ok"
    st = SignalStore(db_path=db)
    try:
        n = len(st.known_agents())
        return True, f"{n} signalling agents readable offline"
    finally:
        st.close()


def _check_snapshot_round_trip() -> tuple[bool, str]:
    """Take a snapshot and verify it — the resilience floor must work with
    no engine running. We take + verify but do NOT restore (restoring would
    mutate live state during a read-only probe)."""
    import tempfile

    from belief.memory.snapshot import SoilSnapshot

    with tempfile.TemporaryDirectory() as td:
        snap = SoilSnapshot(
            snapshots_dir=Path(td) / "snaps",
            audit_path=Path(td) / "audit.jsonl",
        )
        dest = snap.take_snapshot(label="offline-probe")
        ok = snap.verify_snapshot(dest)
        return ok, ("snapshot taken + verified offline" if ok else "snapshot verify failed")


# ── CLI entry point ─────────────────────────────────────────────────────────


def cli_run(write_report: bool = True) -> str:
    """``belief probe offline`` — run the probe and render a summary."""
    probe = WeeklyOfflineProbe()
    report = probe.run(write_report=write_report)
    lines = [
        f"Engine-offline probe — {len(report.results)} checks, "
        f"{sum(1 for r in report.results if r.ok)} passed",
    ]
    for r in report.results:
        mark = "ok " if r.ok else "OBLIGATE"
        lines.append(f"  [{mark}] {r.name}: {r.detail or r.error or ''}")
    if report.report_path:
        lines.append(f"\nReport written to: {report.report_path}")
    if report.obligate_operations:
        lines.append(
            f"\n⚠ {len(report.obligate_operations)} operation(s) failed offline "
            "— see report for obligate-coupling detail."
        )
    return "\n".join(lines)
