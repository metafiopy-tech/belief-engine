"""Protocol version surface (mycorrhizal Stage 8, Area 12).

The arbuscular-mycorrhizal symbiosis bootstrapped from facultative to
obligate over ~400 million years (Strullu-Derrien et al. 2018), and the
thing that made deep coevolution possible was a *stable shared interface* —
the conserved common-symbiosis signaling pathway (CSSP/SYM: DMI1, DMI2/SYMRK,
DMI3/CCaMK) present across land plants. Partners can only specialize against
an interface that doesn't shift under them.

The architectural translation: the Belief Engine commits to a small,
versioned protocol surface — the signal alphabet (Stage 4), the reciprocity
contract (Stage 1), the niche-ledger schema (Stage 2), the warning protocol
(Stage 6), and the onboarding contract (Stage 6). The canonical spec lives in
``docs/PROTOCOL_v1.md``. This module is the machine-readable version tag plus
a compatibility check.

Compatibility policy: a version mismatch *warns*, it does not refuse.
Backward compatibility is the rule within a deprecation window — breaking
changes go to a new major version (v2) with a labelled deprecation phase of
many builds first. This mirrors the biological constraint: mutating the SYM
pathway randomly would strand every partner that specialized against it, so
the interface changes slowly and additively.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("belief.protocol")

#: The current protocol version. Bump the minor for additive changes
#: (new optional fields, new tokens via migration); bump the major only
#: for breaking changes, and only after a deprecation window.
PROTOCOL_VERSION = "1.0"

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")


@dataclass(frozen=True)
class CompatibilityResult:
    """Outcome of a ``compatibility_check``.

    ``compatible`` is True when the client can talk to this engine without
    a breaking mismatch. ``warn`` is True when the versions differ but the
    difference is within the backward-compatible (same-major) window —
    callers should log, not refuse.
    """

    client_version: str
    engine_version: str
    compatible: bool
    warn: bool
    reason: str


def _parse(version: str) -> tuple[int, int]:
    m = _VERSION_RE.match(version.strip())
    if not m:
        raise ValueError(f"unparseable protocol version: {version!r} (expected 'MAJOR.MINOR')")
    return int(m.group(1)), int(m.group(2))


def compatibility_check(client_version: str) -> CompatibilityResult:
    """Compare a client's protocol version against the engine's.

    Rules:
      * Exact match → compatible, no warning.
      * Same major, different minor → compatible *with* a warning
        (additive drift; backward compatibility holds within the major).
      * Different major → incompatible (a breaking change; the caller
        should upgrade). Still does not raise — entry points log and
        proceed best-effort, because refusing outright would be more
        disruptive than a degraded interaction during a deprecation
        window.
      * Unparseable client version → incompatible with a clear reason.
    """
    try:
        c_major, c_minor = _parse(client_version)
    except ValueError as e:
        return CompatibilityResult(
            client_version=client_version,
            engine_version=PROTOCOL_VERSION,
            compatible=False,
            warn=True,
            reason=str(e),
        )
    e_major, e_minor = _parse(PROTOCOL_VERSION)
    if (c_major, c_minor) == (e_major, e_minor):
        return CompatibilityResult(client_version, PROTOCOL_VERSION, True, False, "exact match")
    if c_major == e_major:
        return CompatibilityResult(
            client_version,
            PROTOCOL_VERSION,
            True,
            True,
            f"minor-version drift ({client_version} vs {PROTOCOL_VERSION}); "
            "backward compatible within the major",
        )
    return CompatibilityResult(
        client_version,
        PROTOCOL_VERSION,
        False,
        True,
        f"major-version mismatch ({client_version} vs {PROTOCOL_VERSION}); upgrade recommended",
    )


def assert_compatible(client_version: str) -> CompatibilityResult:
    """Like ``compatibility_check`` but emits the warning/log side effect.

    Entry points (router, onboarding gate, signal emitter) call this so
    mismatches surface in logs without crashing the interaction. Never
    raises on a version mismatch — only on a structurally invalid call.
    """
    result = compatibility_check(client_version)
    if not result.compatible:
        logger.warning("protocol: %s", result.reason)
    elif result.warn:
        logger.info("protocol: %s", result.reason)
    return result


__all__ = [
    "PROTOCOL_VERSION",
    "CompatibilityResult",
    "compatibility_check",
    "assert_compatible",
]
