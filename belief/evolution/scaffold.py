"""FunSearch Scaffold Decomposition — Fixed Scaffold + Evolvable Priority Functions.

Decomposes the Belief Engine codebase into:
1. FIXED SCAFFOLD: core pipeline, models, hardening — never auto-modified
2. EVOLVABLE PRIORITY FUNCTIONS: prompts, thresholds, classification rules —
   safe for SEED to evolve via benchmark-gated mutations

Research basis:
- FunSearch (Nature 2023): decompose problem into fixed scaffold + small evolvable
  priority function. LLM evolves only the critical logic.
- SICA (arXiv 2504.15228): self-modifying agent, 17%→53% in 15 iterations.
  Key safety: Docker sandbox, async overseer, version archive.

The scaffold ensures structural integrity while priority functions are the
"DNA" that selection pressure optimizes.

Usage:
    from belief.evolution.scaffold import ScaffoldDecomposition
    decomp = ScaffoldDecomposition.from_project("/path/to/belief-engine")
    evolvable = decomp.get_evolvable_files()
    safe = decomp.is_safe_to_modify("belief/prompts/__init__.py")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("belief.evolution.scaffold")


# Files that form the FIXED SCAFFOLD — never auto-modified
FIXED_SCAFFOLD = frozenset(
    {
        # Core pipeline
        "belief/graph.py",
        "belief/graph_multi.py",
        "belief/llm.py",
        "belief/cli.py",
        "belief/benchmark.py",
        "belief/benchmark_generator.py",
        # Security and safety
        "belief/hardening.py",
        "belief/validators/__init__.py",
        # Models (structural contracts)
        "belief/models/state.py",
        "belief/models/artifacts.py",
        "belief/models/skeleton.py",
        "belief/models/service_architecture.py",
        "belief/models/dependency_dag.py",
        "belief/models/symbol_registry.py",
        "belief/models/project_manifest.py",
        "belief/models/patch.py",
        "belief/models/openapi.py",
        # Agent base class
        "belief/agents/base.py",
        # Package inits
        "belief/__init__.py",
        "belief/config/__init__.py",
        "belief/models/__init__.py",
        "belief/memory/__init__.py",
        # Evolution infrastructure (the system that does the evolving)
        "belief/evolution/__init__.py",
        "belief/evolution/self_improvement.py",
        "belief/evolution/scaffold.py",
        "belief/evolution/evolutionary_search.py",
    }
)


# Files that are EVOLVABLE PRIORITY FUNCTIONS — safe for SEED to modify
EVOLVABLE_PRIORITY = frozenset(
    {
        # Agent prompts (highest-value target for optimization)
        "belief/prompts/__init__.py",
        "belief/prompts/skeleton_prompts.py",
        # Model routing (which model handles which task)
        "belief/config/models.py",
        "belief/config/settings.py",
        # Validator scoring thresholds and test classification
        "belief/agents/validator.py",
        # Tester configuration (test counts, fixture patterns)
        "belief/agents/tester.py",
        # Refinement analyzer (failure classification rules)
        "belief/refinement/analyzer.py",
        # Memory retrieval parameters
        "belief/memory/recomposer.py",
    }
)


@dataclass
class EvolvableFunction:
    """A specific evolvable parameter within a file."""

    file_path: str
    name: str
    description: str
    current_value: str = ""
    value_type: str = "string"  # string, float, int, list


@dataclass
class ScaffoldDecomposition:
    """Maps the codebase into fixed scaffold and evolvable priority functions."""

    project_root: Path
    fixed_files: set[str] = field(default_factory=lambda: set(FIXED_SCAFFOLD))
    evolvable_files: set[str] = field(default_factory=lambda: set(EVOLVABLE_PRIORITY))
    evolvable_params: list[EvolvableFunction] = field(default_factory=list)

    @classmethod
    def from_project(cls, project_root: str | Path) -> ScaffoldDecomposition:
        """Build decomposition from a project directory."""
        root = Path(project_root)
        decomp = cls(project_root=root)
        decomp._discover_evolvable_params()
        return decomp

    def is_safe_to_modify(self, file_path: str) -> bool:
        """Check if a file is safe for autonomous modification."""
        # Normalize path
        normalized = file_path.replace(str(self.project_root) + "/", "")
        if normalized in self.fixed_files:
            return False
        if normalized in self.evolvable_files:
            return True
        # Unknown files — not safe by default
        return False

    def get_evolvable_files(self) -> list[str]:
        """List all files safe for SEED to modify."""
        return sorted(self.evolvable_files)

    def get_fixed_files(self) -> list[str]:
        """List all files in the fixed scaffold."""
        return sorted(self.fixed_files)

    def _discover_evolvable_params(self) -> None:
        """Discover specific evolvable parameters in evolvable files.

        These are the concrete values SEED can propose changes to:
        - Scoring thresholds (0.75, 0.80, etc.)
        - Test count caps (10, 14, 20)
        - Model routing assignments (Haiku vs Sonnet)
        - Prompt text modifications
        """
        self.evolvable_params = [
            EvolvableFunction(
                file_path="belief/agents/validator.py",
                name="pass_threshold",
                description="Weighted score threshold for PASS verdict with smoke_pass=True",
                value_type="float",
            ),
            EvolvableFunction(
                file_path="belief/agents/validator.py",
                name="high_score_override",
                description="Weighted score threshold for PASS verdict regardless of smoke_pass",
                value_type="float",
            ),
            EvolvableFunction(
                file_path="belief/agents/tester.py",
                name="test_cap_simple",
                description="Max test functions for simple projects (complexity 1-3)",
                value_type="int",
            ),
            EvolvableFunction(
                file_path="belief/agents/tester.py",
                name="test_cap_medium",
                description="Max test functions for medium projects (complexity 4-6)",
                value_type="int",
            ),
            EvolvableFunction(
                file_path="belief/agents/tester.py",
                name="test_cap_complex",
                description="Max test functions for complex projects (complexity 7+)",
                value_type="int",
            ),
            EvolvableFunction(
                file_path="belief/config/models.py",
                name="model_routing",
                description="Which LLM model handles which agent role",
                value_type="string",
            ),
            EvolvableFunction(
                file_path="belief/prompts/__init__.py",
                name="tester_system_prompt",
                description="System prompt for the tester agent",
                value_type="string",
            ),
            EvolvableFunction(
                file_path="belief/prompts/__init__.py",
                name="builder_system_prompt",
                description="System prompt for the builder agent",
                value_type="string",
            ),
        ]

        logger.info(
            f"Scaffold: {len(self.fixed_files)} fixed files, "
            f"{len(self.evolvable_files)} evolvable files, "
            f"{len(self.evolvable_params)} evolvable parameters"
        )
