"""Covenant precision gate — Session 8 (v3.2).

Shadow-applies a proposed covenant against a corpus of past builds
and measures whether the rule would have (a) prevented past failures
or (b) broken past passing builds.

Gate verdict is binary: :data:`GatePolicy` thresholds (all four must
pass) → ``auto_pass``; anything else → ``auto_fail`` with metrics.
Never auto-merges — the human reviews via ``belief covenants review``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from belief.covenants.policy import DEFAULT_POLICY, GatePolicy
from belief.covenants.proposer import CovenantProposal

logger = logging.getLogger("belief.covenants.precision_gate")


# ---------------------------------------------------------------------------
# Archived-build corpus abstraction
# ---------------------------------------------------------------------------


@dataclass
class ArchivedBuild:
    """A past build's materials, as the gate sees them."""

    run_id: str
    goal: str
    verdict: str  # pass / fail_fixable / fail_hard
    code_files: dict[str, str]  # filename → source


# ---------------------------------------------------------------------------
# Shadow applier
# ---------------------------------------------------------------------------


Applier = Callable[[CovenantProposal, str], tuple[str, bool]]
"""Given (proposal, source), return (rewritten_source, did_rewrite)."""


def default_regex_applier(proposal: CovenantProposal, source: str) -> tuple[str, bool]:
    """Treat proposed_pattern as a regex.  If proposed_replacement is
    empty, the rule is "forbid this pattern" — the applier returns
    ``(source_without_matches, True)`` on match.  Otherwise the
    pattern is substituted with the replacement.
    """
    pat = proposal.proposed_pattern or ""
    if not pat:
        return source, False
    try:
        regex = re.compile(pat)
    except re.error:
        return source, False

    if proposal.proposed_replacement:
        new, n_subs = regex.subn(proposal.proposed_replacement, source)
        return new, n_subs > 0

    # Forbidden-pattern rule: strip matching lines.
    any_match = bool(regex.search(source))
    if not any_match:
        return source, False
    kept = [ln for ln in source.splitlines() if not regex.search(ln)]
    # Preserve trailing newline so a no-match file round-trips
    # identically (the bug was `"\n".join` dropping the final \n).
    new = "\n".join(kept)
    if source.endswith("\n") and kept:
        new += "\n"
    return new, True


# ---------------------------------------------------------------------------
# Metrics + gate evaluation
# ---------------------------------------------------------------------------


@dataclass
class PrecisionMetrics:
    would_have_prevented: int = 0
    would_have_broken: int = 0
    n_applicable: int = 0

    @property
    def precision(self) -> float:
        total = self.would_have_prevented + self.would_have_broken
        if total == 0:
            return 0.0
        return self.would_have_prevented / total


def measure_precision(
    proposal: CovenantProposal,
    archive: Iterable[ArchivedBuild],
    *,
    applier: Applier = default_regex_applier,
) -> PrecisionMetrics:
    """Shadow-apply ``proposal`` against every build in ``archive``.

    Counts:
      * a ``fail_fixable`` build whose failing source the rule would
        have rewritten → ``would_have_prevented``
      * a ``pass`` build whose source the rule also rewrites →
        ``would_have_broken`` (because the rewrite might have
        silently changed working code)

    Builds with ``verdict='fail_hard'`` are skipped — we don't learn
    from unrecoverable failures.
    """
    m = PrecisionMetrics()
    for build in archive:
        if build.verdict == "fail_hard":
            continue
        hit = False
        for _fname, source in (build.code_files or {}).items():
            _, changed = applier(proposal, source)
            if changed:
                hit = True
                break
        if not hit:
            continue
        m.n_applicable += 1
        if build.verdict == "pass":
            m.would_have_broken += 1
        elif build.verdict == "fail_fixable":
            m.would_have_prevented += 1
    return m


def evaluate_gate(
    proposal: CovenantProposal,
    archive: Iterable[ArchivedBuild],
    *,
    policy: GatePolicy = DEFAULT_POLICY,
    applier: Applier = default_regex_applier,
) -> CovenantProposal:
    """Measure and annotate a proposal with its gate verdict.

    Mutates and returns the input proposal (in-place) for
    pipeline-friendly chaining.  Sets ``proposal.status`` to
    ``auto_pass`` or ``auto_fail`` and populates ``proposal.metrics``.
    """
    # materialise archive once — tests often pass a generator
    builds = list(archive)
    metrics = measure_precision(proposal, builds, applier=applier)
    proposal.metrics = {
        "would_have_prevented": metrics.would_have_prevented,
        "would_have_broken": metrics.would_have_broken,
        "n_applicable": metrics.n_applicable,
        "precision": metrics.precision,
    }
    if (
        metrics.would_have_prevented >= policy.min_would_have_prevented
        and metrics.would_have_broken <= policy.max_would_have_broken
        and metrics.precision >= policy.min_precision
        and proposal.cluster_size >= policy.min_cluster_size
    ):
        proposal.status = "auto_pass"
    else:
        proposal.status = "auto_fail"
    return proposal


__all__ = [
    "ArchivedBuild",
    "Applier",
    "PrecisionMetrics",
    "default_regex_applier",
    "evaluate_gate",
    "measure_precision",
]
