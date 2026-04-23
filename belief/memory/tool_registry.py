"""
Tool Registry — manages self-authored tools in the belief_tools collection.

The autocatalytic core: the engine uses its own pipeline to build tools
for itself.  Each tool is a self-contained Python module that performs
a specific validation, extraction, or transformation task.

Tools are stored in ChromaDB's belief_tools collection with full FSRS
lifecycle tracking (stability, difficulty, decay state).

Usage:
    from belief.memory.tool_registry import ToolRegistry, SelfAuthoredTool
    from belief.memory.soil import Soil

    soil = Soil()
    registry = ToolRegistry(soil)

    tool = SelfAuthoredTool(
        id="tool-001",
        name="fastapi_route_validator",
        description="Validates FastAPI route definitions",
        code='def validate(code): ...',
        input_description="Python source code string",
        output_description="List of validation error strings",
    )
    registry.register_tool(tool)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("belief.memory.tool_registry")


@dataclass
class SelfAuthoredTool:
    """A tool authored by the engine for its own use."""

    id: str
    name: str  # e.g. "fastapi_route_validator"
    description: str  # What it does, when to use it
    code: str  # Full Python source
    input_description: str = ""  # What inputs it expects
    output_description: str = ""  # What it returns
    dependencies: list[str] = field(default_factory=list)
    version: int = 1
    parent_id: Optional[str] = None  # If evolved from another tool
    created_by: str = "sica"  # "human" | "sica" | "jitterbug"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_score: float = 0.5  # Updated after each use
    success_rate: float = 0.0
    use_count: int = 0
    last_used: Optional[datetime] = None
    # FSRS fields
    fsrs_stability: float = 1.0
    fsrs_difficulty: float = 5.0
    fsrs_decay_state: str = "new"

    def to_metadata(self) -> dict:
        """Serialize to flat dict for ChromaDB metadata."""
        return {
            "tool_name": self.name,
            "description": self.description,
            "input_description": self.input_description,
            "output_description": self.output_description,
            "code": self.code,
            "dependencies": ",".join(self.dependencies) if self.dependencies else "",
            "version": self.version,
            "parent_id": self.parent_id or "",
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "quality_score": self.quality_score,
            "success_rate": self.success_rate,
            "use_count": self.use_count,
            "last_used": self.last_used.isoformat() if self.last_used else "",
            "fsrs_stability": self.fsrs_stability,
            "fsrs_difficulty": self.fsrs_difficulty,
            "fsrs_decay_state": self.fsrs_decay_state,
            "record_type": "self_authored_tool",
        }

    @classmethod
    def from_metadata(cls, doc_id: str, document: str, metadata: dict) -> SelfAuthoredTool:
        """Reconstruct from ChromaDB query result."""
        deps_str = metadata.get("dependencies", "")
        deps = [d.strip() for d in deps_str.split(",") if d.strip()] if deps_str else []

        last_used_str = metadata.get("last_used", "")
        last_used = None
        if last_used_str:
            try:
                last_used = datetime.fromisoformat(last_used_str)
            except (ValueError, TypeError):
                pass

        created_at_str = metadata.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except (ValueError, TypeError):
            created_at = datetime.now(timezone.utc)

        return cls(
            id=doc_id,
            name=metadata.get("tool_name", doc_id),
            description=metadata.get("description", document),
            code=metadata.get("code", ""),
            input_description=metadata.get("input_description", ""),
            output_description=metadata.get("output_description", ""),
            dependencies=deps,
            version=int(metadata.get("version", 1)),
            parent_id=metadata.get("parent_id") or None,
            created_by=metadata.get("created_by", "sica"),
            created_at=created_at,
            quality_score=float(metadata.get("quality_score", 0.5)),
            success_rate=float(metadata.get("success_rate", 0.0)),
            use_count=int(metadata.get("use_count", 0)),
            last_used=last_used,
            fsrs_stability=float(metadata.get("fsrs_stability", 1.0)),
            fsrs_difficulty=float(metadata.get("fsrs_difficulty", 5.0)),
            fsrs_decay_state=metadata.get("fsrs_decay_state", "new"),
        )


class ToolRegistry:
    """Manages self-authored tools in the belief_tools ChromaDB collection."""

    def __init__(self, soil) -> None:
        self.soil = soil
        self._col = soil._collections.get("belief_tools")

    def register_tool(self, tool: SelfAuthoredTool) -> str:
        """Store a tool in ChromaDB.  Returns the tool ID."""
        if self._col is None:
            raise RuntimeError("belief_tools collection not available")

        embedding_text = (
            f"tool: {tool.name} — {tool.description}. "
            f"Input: {tool.input_description}. Output: {tool.output_description}."
        )

        self._col.upsert(
            ids=[tool.id],
            documents=[embedding_text],
            metadatas=[tool.to_metadata()],
        )

        logger.info(f"Registered tool: {tool.name} (id={tool.id})")
        return tool.id

    def get_tool(self, tool_id: str) -> SelfAuthoredTool:
        """Retrieve a single tool by ID."""
        result = self._col.get(
            ids=[tool_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            raise KeyError(f"Tool {tool_id} not found")

        return SelfAuthoredTool.from_metadata(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
        )

    def get_active_tools(self) -> list[SelfAuthoredTool]:
        """Get all non-lapsed tools."""
        if self._col is None or self._col.count() == 0:
            return []

        results = self._col.get(
            include=["documents", "metadatas"],
            limit=self._col.count(),
        )

        tools: list[SelfAuthoredTool] = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            # Filter: only self-authored tools (not skeletons etc.)
            if meta.get("record_type") != "self_authored_tool":
                continue
            if meta.get("fsrs_decay_state") == "lapsed":
                continue
            tools.append(
                SelfAuthoredTool.from_metadata(
                    doc_id,
                    results["documents"][i],
                    meta,
                )
            )

        return tools

    def record_usage(self, tool_id: str, success: bool) -> None:
        """Update use_count, success_rate, last_used, and FSRS state."""
        try:
            tool = self.get_tool(tool_id)
        except KeyError:
            logger.warning(f"record_usage: tool {tool_id} not found")
            return

        tool.use_count += 1
        tool.last_used = datetime.now(timezone.utc)

        # Update success rate (running average)
        old_total = tool.use_count - 1
        if old_total > 0:
            tool.success_rate = (
                tool.success_rate * old_total + (1.0 if success else 0.0)
            ) / tool.use_count
        else:
            tool.success_rate = 1.0 if success else 0.0

        # Update quality score (weighted by success rate and use count)
        tool.quality_score = min(1.0, 0.3 + 0.7 * tool.success_rate)

        # Update FSRS state
        if success:
            tool.fsrs_stability = min(tool.fsrs_stability * 1.5, 365.0)
            if tool.fsrs_decay_state == "new":
                tool.fsrs_decay_state = "learning"
            elif tool.use_count >= 5 and tool.success_rate >= 0.7:
                tool.fsrs_decay_state = "stable"
        else:
            tool.fsrs_stability = max(0.5, tool.fsrs_stability * 0.5)
            if tool.fsrs_decay_state == "stable":
                tool.fsrs_decay_state = "lapsed"

        # Re-save
        self._col.update(
            ids=[tool_id],
            metadatas=[tool.to_metadata()],
        )

    def find_tools_for_goal(self, goal: str, k: int = 5) -> list[SelfAuthoredTool]:
        """Semantic search for tools relevant to a build goal."""
        if self._col is None or self._col.count() == 0:
            return []

        n = min(k * 2, self._col.count())
        if n == 0:
            return []

        try:
            results = self._col.query(
                query_texts=[goal],
                n_results=n,
                where={"record_type": "self_authored_tool"},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Fallback without where filter (older chromadb or empty)
            try:
                results = self._col.query(
                    query_texts=[goal],
                    n_results=n,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return []

        if not results["ids"] or not results["ids"][0]:
            return []

        tools: list[SelfAuthoredTool] = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            if meta.get("record_type") != "self_authored_tool":
                continue
            if meta.get("fsrs_decay_state") == "lapsed":
                continue
            tools.append(
                SelfAuthoredTool.from_metadata(
                    doc_id,
                    results["documents"][0][i],
                    meta,
                )
            )

        return tools[:k]

    def get_tool_health(self) -> dict:
        """Tool library stats."""
        tools = self.get_active_tools()
        if not tools:
            return {
                "count": 0,
                "avg_quality": 0.0,
                "avg_success_rate": 0.0,
                "total_uses": 0,
            }

        return {
            "count": len(tools),
            "avg_quality": sum(t.quality_score for t in tools) / len(tools),
            "avg_success_rate": sum(t.success_rate for t in tools) / len(tools),
            "total_uses": sum(t.use_count for t in tools),
        }
