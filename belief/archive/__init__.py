"""Session 6 (v3.2) — DGM-style agent archive.

Public API::

    from belief.archive import (
        AgentArchive,           # ChromaDB wrapper
        AgentConfiguration,     # per-agent snapshot
        BuildOutcome,           # per-build snapshot
        parent_sample,          # Boltzmann-weighted retrieval
        utility,                # SICA-style scalar fitness
    )

Internal layout::

    config.py    — AgentConfiguration dataclass
    outcome.py   — BuildOutcome dataclass
    fitness.py   — utility(outcome) + covenant_rate
    store.py     — AgentArchive (ChromaDB)
    sampler.py   — parent_sample (Boltzmann)
    priors.py    — planner prompt injection helper
"""

from belief.archive.config import AgentConfiguration
from belief.archive.fitness import covenant_rate, utility
from belief.archive.outcome import BuildOutcome
from belief.archive.sampler import parent_sample
from belief.archive.store import AgentArchive

__all__ = [
    "AgentArchive",
    "AgentConfiguration",
    "BuildOutcome",
    "covenant_rate",
    "parent_sample",
    "utility",
]
