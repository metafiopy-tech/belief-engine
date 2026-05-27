"""Experimental-condition flag for the substrate-transfer experiment.

The reduced 3-condition experiment runs builds under one of:

- ``full``       (default) — current engine behavior; soil, covenants, FSRS decay all ON.
- ``soil_only``  — soil retrieval ON, but covenants stripped from profile,
                   covenant_enforce node short-circuits, FSRS decay treated as 1.0.
- ``raw_local``  — never reaches the engine; raw_runner.py handles this condition
                   independently. Included here so callers can check it for completeness.

Set via the environment variable ``BELIEF_EXPERIMENT_CONDITION``. Unset, empty,
or unrecognized values resolve to ``full`` so existing builds and the hard gate
are completely unaffected when no experiment is running.

Why an env var rather than a kwarg: the toggle has to propagate to deep callers
(soil.retrieve, covenant_enforce LangGraph node) that don't naturally receive
build configuration. The runner sets the env var before invoking ``build``;
the env var stays in scope for the whole build; nothing in the call graph has
to be re-plumbed.

This module is in ``belief.experiments`` rather than ``belief.config`` because
the flag exists *only* for the controlled experiment and has no production
meaning — it should not appear in the main ``Settings`` surface.
"""

from __future__ import annotations

import os
from typing import Literal

Condition = Literal["full", "soil_only", "raw_local"]

_VALID_CONDITIONS: frozenset[str] = frozenset({"full", "soil_only", "raw_local"})

ENV_VAR = "BELIEF_EXPERIMENT_CONDITION"


def current_condition() -> Condition:
    """Read the active experiment condition from the environment.

    Returns ``"full"`` whenever the env var is unset, empty, or holds an
    unrecognized value. This means the hard gate (which never sets the env
    var) sees the same behavior as production.
    """
    raw = os.environ.get(ENV_VAR, "").strip().lower()
    if raw in _VALID_CONDITIONS:
        return raw  # type: ignore[return-value]
    return "full"


def covenants_enabled() -> bool:
    """True if the covenant-enforce node and covenant-soil injection should run.

    False in ``soil_only`` (covenants explicitly OFF) and ``raw_local`` (the
    engine isn't running at all, but defensive).
    """
    return current_condition() == "full"


def fsrs_decay_enabled() -> bool:
    """True if FSRS retrievability decay should affect soil retrieval ranking.

    False in ``soil_only`` (we want raw retrieval without temporal decay).
    """
    return current_condition() == "full"


def soil_retrieval_enabled() -> bool:
    """True if the engine should retrieve nutrients from soil at all.

    True for both ``full`` and ``soil_only``. False for ``raw_local`` — but
    raw_local never enters the engine pipeline in practice; this exists so
    code paths that *might* be reached from a misconfigured run still behave
    correctly.
    """
    return current_condition() in ("full", "soil_only")
