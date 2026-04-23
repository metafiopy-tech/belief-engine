"""Covenant review CLI — Session 8 (v3.2).

Invoked via ``belief covenants review|approve|reject``.  Thin
wrapper around :mod:`belief.covenants.proposer`'s persistence
helpers; the main cli.py dispatches here based on sub-action.

No auto-merging.  Approving a proposal moves its generated rule
code into ``belief/covenants/auto_generated/<proposal_id>.py`` and
adds an entry to a manifest, but a human must have explicitly
typed ``approve`` — that's the safety boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path

from belief.covenants.proposer import (
    CovenantProposal,
    load_proposals,
    save_proposals,
)

logger = logging.getLogger("belief.covenants.review_cli")


_AUTO_GEN_DIR = Path(__file__).resolve().parent / "auto_generated"
_MANIFEST_FILE = _AUTO_GEN_DIR / "MANIFEST.txt"


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_review(status_filter: str | None = None) -> None:
    """Print a table of proposals, optionally filtered by status."""
    proposals = load_proposals()
    if status_filter and status_filter != "all":
        proposals = [p for p in proposals if p.status == status_filter]
    if not proposals:
        print("(no proposals)")
        return
    print(f"{'id':<18} {'status':<14} {'cluster':<8} {'prevented':<10} {'broken':<8} signature")
    for p in proposals:
        prev = p.metrics.get("would_have_prevented", "—")
        broken = p.metrics.get("would_have_broken", "—")
        sig_short = (p.error_signature or "")[:60]
        print(
            f"{p.proposal_id:<18} {p.status:<14} {p.cluster_size:<8} "
            f"{prev!s:<10} {broken!s:<8} {sig_short}"
        )


def cmd_approve(proposal_id: str) -> None:
    """Move the proposal's generated code into auto_generated/ and
    update the manifest.  Status set to ``approved``.
    """
    proposals = load_proposals()
    target = _find(proposals, proposal_id)
    if target is None:
        print(f"No proposal with id={proposal_id!r}")
        return
    if target.status == "auto_fail":
        print(
            f"Warning: proposal {proposal_id} has status=auto_fail "
            f"(metrics={target.metrics}). Approving anyway as the human override."
        )
    _AUTO_GEN_DIR.mkdir(parents=True, exist_ok=True)
    code_path = _AUTO_GEN_DIR / f"{proposal_id}.py"
    code_path.write_text(
        f'''"""Auto-generated covenant — {proposal_id}.

Error signature: {target.error_signature}
Cluster size: {target.cluster_size}
Rationale: {target.rationale}
"""

import re

_PATTERN = re.compile({target.proposed_pattern!r})
_REPLACEMENT = {target.proposed_replacement!r}

def apply(source: str) -> tuple[str, bool]:
    if _REPLACEMENT:
        new = _PATTERN.sub(_REPLACEMENT, source)
    else:
        new = "\\n".join(l for l in source.splitlines() if not _PATTERN.search(l))
    return new, new != source
'''
    )
    _append_manifest(target)
    target.status = "approved"
    save_proposals(proposals)
    print(f"Approved {proposal_id} → {code_path}")


def cmd_reject(proposal_id: str, reason: str = "") -> None:
    proposals = load_proposals()
    target = _find(proposals, proposal_id)
    if target is None:
        print(f"No proposal with id={proposal_id!r}")
        return
    target.status = "rejected"
    if reason:
        target.rationale = f"{target.rationale}\n---\nREJECTED: {reason}"
    save_proposals(proposals)
    print(f"Rejected {proposal_id}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find(proposals: list[CovenantProposal], pid: str) -> CovenantProposal | None:
    for p in proposals:
        if p.proposal_id == pid:
            return p
    return None


def _append_manifest(p: CovenantProposal) -> None:
    _MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{p.proposal_id}\tcluster={p.cluster_size}\tsig={p.error_signature[:120]}\n"
    with _MANIFEST_FILE.open("a", encoding="utf-8") as f:
        f.write(line)


__all__ = ["cmd_approve", "cmd_reject", "cmd_review"]
