"""BuildOutcome — Session 6 (v3.2).

One row in the agent archive.  Captures the full per-build state that
a future retrieval can learn from: the goal, the verdict, the per-
agent configurations, and a stable ``trajectory_signature`` for
clustering similar trajectories.

This is distinct from :class:`belief.archive.AgentConfiguration`
(which describes ONE agent on ONE build); a BuildOutcome bundles a
full dict of those plus per-build metrics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from belief.archive.config import AgentConfiguration


@dataclass
class BuildOutcome:
    """Persisted result of one build, archived for future retrieval.

    The ``agent_configurations`` dict maps agent_name to the exact
    AgentConfiguration that agent ran with.  The planner's config is
    the one the archive embeds as its searchable document (it's the
    most diagnostic signature of "what kind of build was this"), but
    every agent's config is stored in the metadata so a future
    retrieval can reconstruct the full pipeline state.
    """

    run_id: str
    goal: str
    verdict: str  # pass / fail_fixable / fail_hard
    tests_passed: int = 0
    tests_total: int = 0
    weighted_score: float = 0.0
    wallclock_s: float = 0.0
    estimated_cost_usd: float = 0.0
    covenant_violations: list[str] = field(default_factory=list)
    debug_iterations: int = 0
    agent_configurations: dict[str, AgentConfiguration] = field(default_factory=dict)
    trajectory_signature: str = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict recurses into AgentConfiguration fields already.  We
        # just want to make sure the nested type survives round-trip.
        d["agent_configurations"] = {
            name: cfg if isinstance(cfg, dict) else cfg.to_dict()
            for name, cfg in self.agent_configurations.items()
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildOutcome":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        ac = filtered.get("agent_configurations", {})
        filtered["agent_configurations"] = {
            name: (
                cfg if isinstance(cfg, AgentConfiguration) else AgentConfiguration.from_dict(cfg)
            )
            for name, cfg in ac.items()
        }
        return cls(**filtered)

    @classmethod
    def from_json(cls, raw: str) -> "BuildOutcome":
        return cls.from_dict(json.loads(raw))

    # ------------------------------------------------------------------
    # Helpers for the archive
    # ------------------------------------------------------------------

    def planner_config_json(self) -> str:
        """JSON of the planner's AgentConfiguration — the canonical
        embedding document for the archive.  Falls back to the goal
        if the planner config isn't present (shouldn't happen in
        normal builds, but defensive against partial BuildOutcomes).
        """
        planner = self.agent_configurations.get("planner")
        if isinstance(planner, AgentConfiguration):
            return planner.to_json()
        if isinstance(planner, dict):
            return json.dumps(planner, sort_keys=True, default=str)
        return json.dumps({"goal": self.goal})

    def embedding_text(self) -> str:
        """Text the archive's embedding model ingests.  Goal + planner
        config is the right granularity — goal alone clusters by
        topic, planner config alone clusters by configuration, both
        together give a useful mix for retrieval.
        """
        planner_str = self.planner_config_json()
        return f"GOAL: {self.goal}\nPLANNER_CONFIG: {planner_str}"

    def compute_trajectory_signature(self, agent_sequence: list[str]) -> str:
        """Stable hash of the sequence of agent calls (plus each one's
        verdict marker) — lets the crystallizer / proposer cluster
        similar trajectories without re-running the pipeline.
        """
        marker = "->".join(agent_sequence) + f"::{self.verdict}"
        return hashlib.sha256(marker.encode("utf-8")).hexdigest()


__all__ = ["BuildOutcome"]
