"""
Dependency DAG — Milestone 2

Implements Kahn's algorithm for topological sorting of the file
dependency graph. Files are grouped into execution levels:

  Level 0: files with no dependencies (models, configs)
  Level 1: files that depend only on Level 0
  Level 2: files that depend on Level 0 and/or Level 1
  ...

Within each level, files can be generated in parallel since they
have no inter-dependencies.

Based on:
- Microsoft CodePlan (FSE 2024): DAG-based change planning
- SWE-AF: Kahn's algorithm for parallel issue execution
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from belief.models.skeleton import SkeletonArtifact


# ---------------------------------------------------------------------------
# DAG Node
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """A node in the dependency DAG."""
    path: str
    in_degree: int = 0
    level: int = -1  # Assigned by topological sort
    dependents: list[str] = field(default_factory=list)  # files that depend on this
    dependencies: list[str] = field(default_factory=list)  # files this depends on


# ---------------------------------------------------------------------------
# Cycle Detection Error
# ---------------------------------------------------------------------------

class DependencyCycleError(Exception):
    """Raised when the dependency graph contains a cycle."""
    def __init__(self, remaining_nodes: list[str]):
        self.remaining_nodes = remaining_nodes
        super().__init__(
            f"Dependency cycle detected among {len(remaining_nodes)} files: "
            f"{', '.join(remaining_nodes[:5])}"
            + ("..." if len(remaining_nodes) > 5 else "")
        )


# ---------------------------------------------------------------------------
# Kahn's Algorithm
# ---------------------------------------------------------------------------

@dataclass
class TopologicalResult:
    """Result of topological sorting."""
    sorted_files: list[str]           # All files in topological order
    levels: list[list[str]]           # Files grouped by execution level
    node_levels: dict[str, int]       # file_path → level number
    total_levels: int

    def files_at_level(self, level: int) -> list[str]:
        """Get files at a specific level."""
        if 0 <= level < len(self.levels):
            return self.levels[level]
        return []

    def max_parallelism(self) -> int:
        """Maximum number of files that can be generated in parallel."""
        return max(len(level) for level in self.levels) if self.levels else 0


def topological_sort(skeleton: SkeletonArtifact) -> TopologicalResult:
    """
    Topologically sort files using Kahn's algorithm.

    Groups files into execution levels where all files within a level
    can be generated in parallel (no inter-dependencies).

    Args:
        skeleton: The SkeletonArtifact with file_tree and dependency_edges.

    Returns:
        TopologicalResult with sorted files and level groupings.

    Raises:
        DependencyCycleError: If the dependency graph contains a cycle.
    """
    # Build adjacency structures
    all_paths = {f.path for f in skeleton.file_tree}
    nodes: dict[str, DAGNode] = {path: DAGNode(path=path) for path in all_paths}

    # Process edges
    for edge in skeleton.dependency_edges:
        if edge.source in nodes and edge.target in nodes:
            source_node = nodes[edge.source]
            target_node = nodes[edge.target]

            if edge.target not in source_node.dependencies:
                source_node.dependencies.append(edge.target)
                source_node.in_degree += 1

            if edge.source not in target_node.dependents:
                target_node.dependents.append(edge.source)

    # Kahn's algorithm
    queue: deque[str] = deque()
    for path, node in nodes.items():
        if node.in_degree == 0:
            node.level = 0
            queue.append(path)

    sorted_files: list[str] = []
    levels: dict[int, list[str]] = defaultdict(list)

    while queue:
        current_path = queue.popleft()
        current_node = nodes[current_path]
        sorted_files.append(current_path)
        levels[current_node.level].append(current_path)

        # Process dependents
        for dependent_path in current_node.dependents:
            dependent_node = nodes[dependent_path]
            dependent_node.in_degree -= 1

            # Level = max(dependency levels) + 1
            candidate_level = current_node.level + 1
            if candidate_level > dependent_node.level:
                dependent_node.level = candidate_level

            if dependent_node.in_degree == 0:
                queue.append(dependent_path)

    # Check for cycles
    if len(sorted_files) != len(all_paths):
        remaining = [p for p in all_paths if p not in set(sorted_files)]
        raise DependencyCycleError(remaining)

    # Build level list
    max_level = max(levels.keys()) if levels else 0
    level_list = [levels.get(i, []) for i in range(max_level + 1)]

    node_levels = {path: nodes[path].level for path in all_paths}

    return TopologicalResult(
        sorted_files=sorted_files,
        levels=level_list,
        node_levels=node_levels,
        total_levels=max_level + 1,
    )


# ---------------------------------------------------------------------------
# DAG from SkeletonArtifact (with skeleton/impl split awareness)
# ---------------------------------------------------------------------------

@dataclass
class BuildPlan:
    """
    A build plan that respects both topological order and skeleton/impl split.

    Phase 1 (Pass 1): Generate skeleton files in topological order
    Phase 2 (Pass 2): Generate implementation files in topological order
    """
    skeleton_order: list[list[str]]   # Skeleton files grouped by level
    impl_order: list[list[str]]       # Implementation files grouped by level
    full_order: list[list[str]]       # All files in combined level order
    topo_result: TopologicalResult    # Raw topological sort result

    @property
    def total_skeleton_files(self) -> int:
        return sum(len(level) for level in self.skeleton_order)

    @property
    def total_impl_files(self) -> int:
        return sum(len(level) for level in self.impl_order)


def create_build_plan(skeleton: SkeletonArtifact) -> BuildPlan:
    """
    Create a build plan that combines topological ordering with
    the skeleton/implementation split.

    Skeleton files are always generated first (they have no deps
    on impl files). Then implementation files are generated in
    topological order.

    Args:
        skeleton: The SkeletonArtifact.

    Returns:
        BuildPlan with ordered generation levels.
    """
    topo = topological_sort(skeleton)

    # Categorize files
    skeleton_paths = {f.path for f in skeleton.file_tree if f.skeleton}
    impl_paths = {f.path for f in skeleton.file_tree if not f.skeleton}

    # Split levels into skeleton and impl
    skeleton_levels: list[list[str]] = []
    impl_levels: list[list[str]] = []

    for level_files in topo.levels:
        skel_in_level = [f for f in level_files if f in skeleton_paths]
        impl_in_level = [f for f in level_files if f in impl_paths]

        if skel_in_level:
            skeleton_levels.append(skel_in_level)
        if impl_in_level:
            impl_levels.append(impl_in_level)

    return BuildPlan(
        skeleton_order=skeleton_levels,
        impl_order=impl_levels,
        full_order=topo.levels,
        topo_result=topo,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dependency_depth(skeleton: SkeletonArtifact, file_path: str) -> int:
    """
    Calculate the maximum dependency depth for a file.
    Useful for estimating build complexity.
    """
    topo = topological_sort(skeleton)
    return topo.node_levels.get(file_path, 0)


def critical_path(skeleton: SkeletonArtifact) -> list[str]:
    """
    Find the critical path — the longest chain of dependencies.
    This determines the minimum number of sequential build steps.
    """
    topo = topological_sort(skeleton)

    # The critical path goes through the file with the highest level
    max_level = max(topo.node_levels.values()) if topo.node_levels else 0

    # Find a file at the max level and trace back
    path = []
    for file_path, level in topo.node_levels.items():
        if level == max_level:
            # Trace back through dependencies
            current = file_path
            while current:
                path.append(current)
                deps = [
                    e.target for e in skeleton.dependency_edges
                    if e.source == current
                ]
                # Pick the dependency with the highest level
                if deps:
                    current = max(deps, key=lambda d: topo.node_levels.get(d, 0))
                    if current in path:  # Safety: prevent infinite loop
                        break
                else:
                    current = None
            break

    path.reverse()
    return path
