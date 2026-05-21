"""Recalcitrant decomposition — the "laccase/peroxidase" path (Stage 7, Area 4).

For builds that failed opaquely or systemically — the agent got stuck in a
reasoning loop, hallucinated an API, the build crashed for an external
reason — most of the substrate is unrecoverable (like lignin). What remains
is *negative evidence*: a failure signature that says "do not do this." This
is the slowest, lowest-yield enzyme class, and it must NOT block the
easy/structural paths.

Pure function: builds a signature, no soil writes. The dispatcher persists
signatures to a dedicated anti-pattern collection.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("belief.memory.decomposers.recalcitrant")

# Heuristic markers of a systemic/opaque failure mode in error text.
_SYSTEMIC_MARKERS = (
    "recursion",
    "maximum recursion",
    "timeout",
    "timed out",
    "rate limit",
    "context length",
    "context window",
    "hallucinat",
    "no module named",  # often a hallucinated import
    "connection",
    "loop",
)


@dataclass
class FailureSignature:
    """A compact signature of an opaque/systemic build failure."""

    signature_id: str
    error_type: str
    stack_pattern: str  # normalized stack/error pattern
    last_messages: list[str] = field(default_factory=list)
    source_build_id: str = ""
    systemic_markers: list[str] = field(default_factory=list)


def _normalize_trace(text: str) -> str:
    """Strip volatile bits (line numbers, hex addresses, tmp paths) so two
    instances of the same failure mode hash to the same signature."""
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"line \d+", "line N", text)
    # Collapse whole filesystem paths (incl. the trailing filename) — a
    # per-run temp path like /tmp/x/foo.py is volatile end-to-end, so two
    # instances of the same failure mode in different temp files normalize
    # to the same signature.
    text = re.sub(r"/\S+", "/PATH", text)
    text = re.sub(r"\d+", "N", text)
    return text.strip()


def extract_failure_signature(
    errors: list[str],
    exec_error: str = "",
    last_agent_messages: list[str] | None = None,
    source_build_id: str = "",
) -> FailureSignature | None:
    """Build a failure signature from a build's error trail.

    Returns ``None`` if there's no error material to work with (a clean
    build has nothing recalcitrant to extract). The signature_id is a hash
    of the normalized pattern so repeat failures of the same mode collapse.
    """
    blob = "\n".join([*(errors or []), exec_error or ""]).strip()
    if not blob:
        return None
    normalized = _normalize_trace(blob)
    # Error type: first token that looks like an ExceptionName, else 'unknown'.
    m = re.search(r"([A-Z][A-Za-z0-9]*(?:Error|Exception|Warning))", blob)
    error_type = m.group(1) if m else "unknown"
    markers = [mk for mk in _SYSTEMIC_MARKERS if mk in blob.lower()]
    sig_id = hashlib.sha256(f"{error_type}:{normalized}".encode("utf-8")).hexdigest()[:24]
    tail = (last_agent_messages or [])[-3:]
    return FailureSignature(
        signature_id=sig_id,
        error_type=error_type,
        stack_pattern=normalized[:500],
        last_messages=[m[:200] for m in tail],
        source_build_id=source_build_id,
        systemic_markers=markers,
    )
