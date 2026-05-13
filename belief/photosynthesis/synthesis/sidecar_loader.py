"""Sidecar loader (Synthesis Engine S7.5).

Closes the disk -> state plumbing gap. Photosynthesis's renderer
writes ``pending_sessions/{goal_id}.json`` containing the full
``GoalSpec`` model_dump, which optionally carries a
``structural_mechanism`` field. The Grinder picks the markdown +
sidecar pair off disk and dispatches a build, but until S7.5 the
sidecar's ``structural_mechanism`` was never re-hydrated -- it sat
on disk while the build pipeline ran without it.

This module is the bridge: pure, no I/O of its own. The caller
(daemon._default_build_runner, or anything else hydrating
UnifiedState from a sidecar) passes the already-loaded sidecar dict
in; we return either a validated :class:`StructuralMechanism` or
``None`` and log if the field was present but malformed.

Permissive on errors. The build must NOT die because the synthesis
side wrote a sidecar with a schema-broken mechanism block; we log
and degrade to no-mechanism behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from belief.photosynthesis.synthesis.structural_mechanism import StructuralMechanism


logger = logging.getLogger("belief.photosynthesis.synthesis.sidecar_loader")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_structural_mechanism(sidecar: dict[str, Any] | None) -> Optional[StructuralMechanism]:
    """Hydrate a StructuralMechanism from a sidecar dict.

    Returns:
      - None if ``sidecar`` is None, missing the key, or the value is
        falsy / not a dict.
      - A validated :class:`StructuralMechanism` if the value passes
        Pydantic validation.
      - None (with a warning logged) if the value is a dict but fails
        validation. We never raise -- the build pipeline must keep going.
    """
    if not sidecar:
        return None
    raw = sidecar.get("structural_mechanism")
    if raw is None:
        return None
    if isinstance(raw, StructuralMechanism):
        # Already a model (e.g. caller passed a partially-typed dict).
        return raw
    if not isinstance(raw, dict):
        logger.warning(
            "structural_mechanism in sidecar is not a dict (got %s); ignoring",
            type(raw).__name__,
        )
        return None
    try:
        return StructuralMechanism.model_validate(raw)
    except Exception as exc:  # ValidationError + anything else
        logger.warning(
            "structural_mechanism in sidecar failed validation; ignoring: %s",
            exc,
        )
        return None


def load_sidecar_from_path(json_path: Path) -> dict[str, Any] | None:
    """Read a JSON sidecar from disk into a dict.

    Convenience wrapper for callers that have a path rather than an
    already-parsed dict. Returns None on read / parse error and logs.
    Use :func:`extract_structural_mechanism` on the returned dict.
    """
    import json

    try:
        text = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read sidecar %s: %s", json_path, exc)
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("sidecar %s is not valid JSON: %s", json_path, exc)
        return None
    if not isinstance(loaded, dict):
        logger.warning("sidecar %s did not parse to a dict", json_path)
        return None
    return loaded


def hydrate_initial_state(
    initial_state: dict,
    sidecar_path: "str | Path | None",
) -> dict:
    """Add `structural_mechanism` to ``initial_state`` from a sidecar JSON.

    Pure function. Returns a NEW dict (does not mutate the input).
    Permissive on errors: missing path / unreadable file / invalid JSON /
    sidecar without a mechanism / mechanism that fails validation all
    return ``initial_state`` unchanged. Logs WARNING via the loader's
    helpers; does not raise.

    Used by both call sites that build initial state for the pipeline
    out-of-band of the daemon's GoalEnvelope path:
      - ``belief.cli.run`` (the CLI ``belief build --sidecar PATH`` flag).
      - any future programmatic builder that has a sidecar in hand.

    The Grinder daemon path uses ``extract_structural_mechanism`` directly
    on ``GoalEnvelope.sidecar`` (the dict has already been parsed by
    ``GoalQueue._load_envelope``), not this helper.
    """
    if sidecar_path is None:
        return initial_state
    sidecar = load_sidecar_from_path(Path(sidecar_path))
    mechanism = extract_structural_mechanism(sidecar)
    if mechanism is None:
        return initial_state
    out = dict(initial_state)
    out["structural_mechanism"] = mechanism
    logger.info(
        "hydrated structural_mechanism from sidecar %s (%d open probes)",
        sidecar_path,
        len(mechanism.incompleteness_probes_open),
    )
    return out


__all__ = [
    "extract_structural_mechanism",
    "hydrate_initial_state",
    "load_sidecar_from_path",
]
