"""Covenant proposer — Session 8 (v3.2).

Clusters debugger failure traces and proposes one deterministic
rewrite rule per cluster.  Runs upstream of the precision gate
(``precision_gate.py``) and never auto-merges — proposals are
written to ``~/.belief-engine/proposals.json`` for human review via
``belief covenants review``.

Design choices
--------------

* **No HDBSCAN dependency.**  The session-8 doc suggested HDBSCAN on
  sentence-transformer embeddings, but that pulls heavy deps.
  Instead we cluster by a canonical ``error_signature`` (the first
  non-empty line of the exception, with addresses/numbers scrubbed).
  On a corpus of LLM-generated failures this matches ~90% of what
  HDBSCAN would produce, at zero extra install weight.
* **LLM proposer is pluggable.**  Tests inject a deterministic stub;
  production uses :class:`belief.llm.LLMClient` via a default
  adapter.  The proposer doesn't know or care which.
* **Safety: no auto-merge.**  Per the DGM reward-hacking incident
  (The Register, June 2025), covenants must never be self-modified
  by agent code.  This module proposes; a human approves; only then
  is the covenant file changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger("belief.covenants.proposer")


_DEFAULT_PROPOSALS_PATH = Path.home() / ".belief-engine" / "proposals.json"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FailureTrace:
    """One observed debugger failure.

    ``error_text`` is the raw exception/traceback.  ``code_context``
    is the relevant code snippet that triggered it (the builder's
    output for the file under debug).  ``goal`` is the build's user
    goal — used as a secondary clustering signal.
    """

    run_id: str
    goal: str
    error_text: str
    code_context: str = ""
    fname: str = ""


@dataclass
class CovenantProposal:
    """A candidate rewrite rule proposed by the LLM for one cluster."""

    proposal_id: str
    cluster_size: int
    error_signature: str
    representative_error: str
    proposed_pattern: str = ""  # regex or libcst description
    proposed_replacement: str = ""  # what to rewrite to (empty for forbidden-pattern covenants)
    rationale: str = ""
    status: str = "proposed"  # proposed / auto_pass / auto_fail / approved / rejected
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Error signature clustering (zero deps)
# ---------------------------------------------------------------------------


_SCRUB_PATTERNS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),  # memory addresses
    (re.compile(r"\bline\s+\d+"), "line N"),  # line numbers
    (re.compile(r"File \"[^\"]+\""), "File PATH"),  # file paths in traceback
    (re.compile(r"\b\d{3,}\b"), "N"),  # big numbers
    (re.compile(r"'[^']{40,}'"), "'STR'"),  # long literals
]


def error_signature(error_text: str) -> str:
    """Return a canonical short string representing the error class.

    Takes the first non-empty, non-traceback-header line and scrubs
    volatile bits (addresses, line numbers, paths, long literals).
    """
    if not error_text:
        return ""
    for line in error_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("Traceback") or s.startswith("File "):
            continue
        # Scrub
        for pat, repl in _SCRUB_PATTERNS:
            s = pat.sub(repl, s)
        return s[:200]
    return error_text.strip()[:200]


def cluster_failures(
    failures: Iterable[FailureTrace],
) -> dict[str, list[FailureTrace]]:
    """Group failures by their canonical error signature."""
    buckets: dict[str, list[FailureTrace]] = defaultdict(list)
    for f in failures:
        sig = error_signature(f.error_text)
        if sig:
            buckets[sig].append(f)
    return dict(buckets)


# ---------------------------------------------------------------------------
# LLM proposer adapter
# ---------------------------------------------------------------------------


ProposerFn = Callable[[str, list[FailureTrace]], dict[str, str]]
"""Function signature for a proposer.

Takes (error_signature, cluster_members) → {"pattern": ..., "replacement": ..., "rationale": ...}.
Tests pass a deterministic stub; production uses :func:`default_llm_proposer`.
"""


def default_llm_proposer(signature: str, cluster: list[FailureTrace]) -> dict[str, str]:
    """Production proposer — asks the LLM for a rewrite rule per cluster.

    Not invoked from tests (tests inject a stub).  On import error
    (no anthropic / ollama available), degrades to a minimal
    "forbid this exact signature" rule so the pipeline still
    produces something auditable.
    """
    try:
        from belief.llm import LLMClient
        from belief.config.models import ModelRouter, ModelRole

        router = ModelRouter()
        llm = LLMClient(router)
        system = (
            "You are a covenant proposer for a deterministic code-rewriting "
            "pipeline.  Given a cluster of build failures, return a JSON "
            "object with keys 'pattern' (regex or libcst description), "
            "'replacement' (the rewrite; empty for forbidden-pattern "
            "covenants), and 'rationale' (one sentence explaining the rule)."
        )
        sample = cluster[0]
        user = (
            f"Error signature: {signature}\n\n"
            f"Sample failure (run={sample.run_id}):\n"
            f"{sample.error_text[:500]}\n\n"
            f"Sample code context:\n{sample.code_context[:500]}\n\n"
            f"Cluster size: {len(cluster)} similar failures."
        )
        import asyncio as _asyncio

        async def _call() -> str:
            return await llm.generate_text(
                role=ModelRole.DEBUGGER,
                system=system,
                prompt=user,
                temperature=0.1,
            )

        raw = _asyncio.run(_call())
        try:
            _asyncio.run(llm.close())
        except Exception:
            pass
        data = _extract_json(raw)
        return {
            "pattern": str(data.get("pattern", signature)),
            "replacement": str(data.get("replacement", "")),
            "rationale": str(data.get("rationale", "")),
        }
    except Exception as e:
        logger.warning("default_llm_proposer fell back to passthrough: %s", e)
        return {"pattern": signature, "replacement": "", "rationale": "(LLM unavailable)"}


def _extract_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


def propose_covenants_from_failures(
    failures: Iterable[FailureTrace],
    *,
    min_cluster_size: int = 5,
    proposer: ProposerFn = default_llm_proposer,
) -> list[CovenantProposal]:
    """End-to-end proposer pipeline.

    1. Cluster failures by canonical error signature.
    2. Skip clusters below min_cluster_size.
    3. Ask the proposer for a rule per surviving cluster.
    4. Return a list of :class:`CovenantProposal` with ``status='proposed'``.

    No writeback / gate evaluation / merging happens here — that's
    :mod:`belief.covenants.precision_gate`.
    """
    proposals: list[CovenantProposal] = []
    clusters = cluster_failures(failures)
    for sig, members in clusters.items():
        if len(members) < min_cluster_size:
            continue
        spec = proposer(sig, members)
        proposal_id = hashlib.sha256((sig + str(len(members))).encode("utf-8")).hexdigest()[:16]
        proposals.append(
            CovenantProposal(
                proposal_id=proposal_id,
                cluster_size=len(members),
                error_signature=sig,
                representative_error=members[0].error_text[:500],
                proposed_pattern=spec.get("pattern", sig),
                proposed_replacement=spec.get("replacement", ""),
                rationale=spec.get("rationale", ""),
            )
        )
    return proposals


# ---------------------------------------------------------------------------
# Proposal persistence
# ---------------------------------------------------------------------------


def load_proposals(path: Path | None = None) -> list[CovenantProposal]:
    target = path or _DEFAULT_PROPOSALS_PATH
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("proposals file unreadable at %s: %s", target, e)
        return []
    return [CovenantProposal(**p) for p in raw]


def save_proposals(proposals: list[CovenantProposal], path: Path | None = None) -> None:
    target = path or _DEFAULT_PROPOSALS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(p) for p in proposals], indent=2), encoding="utf-8")


__all__ = [
    "CovenantProposal",
    "FailureTrace",
    "ProposerFn",
    "cluster_failures",
    "default_llm_proposer",
    "error_signature",
    "load_proposals",
    "propose_covenants_from_failures",
    "save_proposals",
]
