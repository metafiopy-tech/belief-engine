"""AgentConfiguration — Session 6 (v3.2).

The DGM pattern (Zhang et al., arXiv:2505.22954) persists every agent
variant's full configuration to an archive so later builds can sample
high-utility priors as context.  Belief v3.0 already had a
lineage-tracking :class:`belief.evolution.archive.AgentVersion`; this
module adds a RETRIEVAL-layer complement keyed by build context
rather than by parent-pointer DAG.

AgentConfiguration captures everything needed to reproduce a single
agent's behaviour on a single build — prompts, model, options,
covenant set, tool schemas, and a SHA-256 of the Python source file
that implemented the agent.  Paired with :class:`BuildOutcome` in
``outcome.py``, this is the unit the archive indexes and retrieves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentConfiguration:
    """Full snapshot of how an agent was configured on one build.

    Serialisable to JSON via :meth:`to_dict` / :meth:`from_dict` so the
    archive can embed the JSON string as ChromaDB's document text and
    round-trip it back to a dataclass on retrieval.

    ``code_hash`` is a SHA-256 of the agent's Python source.  When the
    source changes (new agent version, new prompt file), the hash
    changes; future retrievals can filter to only "priors whose code
    matches the code we're running now" if strict reproducibility is
    needed, or ignore the field for broader retrieval.
    """

    agent_name: str
    system_prompt: str
    user_prompt_template: str = ""
    model: str = ""
    model_options: dict[str, Any] = field(default_factory=dict)
    covenant_set: list[str] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    code_hash: str = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """JSON-serialise deterministically (sorted keys) so identical
        configs produce byte-identical JSON — useful for hashing and
        for ChromaDB deduplication.
        """
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfiguration":
        # Defensive: tolerate missing / extra keys across schema drift.
        known_fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known_fields})

    @classmethod
    def from_json(cls, raw: str) -> "AgentConfiguration":
        return cls.from_dict(json.loads(raw))

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @staticmethod
    def compute_code_hash(source_text: str) -> str:
        """SHA-256 of a source string, hex-encoded.  Used to tag a
        configuration with the exact agent code that produced it.
        """
        return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


__all__ = ["AgentConfiguration"]
