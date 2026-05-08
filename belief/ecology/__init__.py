"""Ecosystem organs for the Belief Engine (v3.3).

Per docs/v33_ecosystem_organs_spec.md, this package houses the nine organs
that turn the engine from a single-shot producer into a self-sustaining
ecology. Organs ship one per session, each with its own ``run()``-style
entry point and ``belief <organ>`` CLI subcommand.

Session 1 (this file): Economist budget-contract shell. Subsequent organs
(Predator, GC, Sleep, Curiosity, Speciator, Storyteller, Red-team, Body)
plug into the contract instead of taking ad-hoc ``budget_usd`` env-vars.
"""

from __future__ import annotations

from belief.ecology.economist import (
    Economist,
    PriceQuote,
    QuoteRejected,
)
from belief.ecology.predator import (
    PredatorConfig,
    PredatorResult,
    run as run_predator,
)
from belief.ecology.curiosity import (
    CuriosityResult,
    GoalCandidate,
    suggest as curiosity_suggest,
)
from belief.ecology.garbage_collector import (
    GCReport,
    run as run_gc,
)
from belief.ecology.sleep import (
    SleepConfig,
    SleepResult,
    run as run_sleep,
)

__all__ = [
    "CuriosityResult",
    "Economist",
    "GCReport",
    "GoalCandidate",
    "PredatorConfig",
    "PredatorResult",
    "PriceQuote",
    "QuoteRejected",
    "SleepConfig",
    "SleepResult",
    "curiosity_suggest",
    "run_gc",
    "run_predator",
    "run_sleep",
]
