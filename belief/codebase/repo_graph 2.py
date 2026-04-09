"""Repo Graph — Dependency Graph with Personalized PageRank.

Builds a NetworkX graph from a Codebase's import edges, then uses
Personalized PageRank seeded from issue-relevant files to rank code
by structural importance relative to the bug.

Research basis:
- PRFL (Zhang et al., TSE 2019): PageRank boosts Top-1 fault localization by 39%
- RepoGraph (arXiv 2410.14684): line-level deps give 32.8% relative improvement
- Hybrid BM25+PPR outperforms either alone (0.6 × BM25 + 0.4 × PPR)

Usage:
    from belief.codebase.repo_graph import RepoGraph
    graph = RepoGraph.from_codebase(codebase)
    ranked = graph.localize("fix the login endpoint", max_files=10)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger("belief.codebase.repo_graph")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.debug("networkx not installed — RepoGraph will use fallback ranking")


@dataclass
class RankedFile:
    """A file with its relevance score and ranking components."""
    path: str
    combined_score: float
    bm25_score: float = 0.0
    pagerank_score: float = 0.0
    dependents_count: int = 0


class RepoGraph:
    """Dependency graph with PageRank-based code localization."""

    def __init__(self):
        self.graph: nx.DiGraph | None = None
        self._file_contents: dict[str, str] = {}
        self._file_exports: dict[str, list[str]] = {}
        self._test_files: set[str] = set()

    @classmethod
    def from_codebase(cls, codebase) -> RepoGraph:
        """Build a RepoGraph from an existing Codebase object."""
        rg = cls()

        if not HAS_NETWORKX:
            # Store enough for fallback BM25-only ranking
            for fpath, info in codebase.files.items():
                rg._file_exports[fpath] = [e.name for e in info.exports]
                if info.is_test:
                    rg._test_files.add(fpath)
            return rg

        rg.graph = nx.DiGraph()

        # Add nodes (files)
        for fpath, info in codebase.files.items():
            if info.is_test:
                rg._test_files.add(fpath)
            rg.graph.add_node(fpath, **{
                "language": info.language.value,
                "lines": info.line_count,
                "is_test": info.is_test,
                "exports": [e.name for e in info.exports],
            })
            rg._file_exports[fpath] = [e.name for e in info.exports]

        # Add edges (imports)
        for edge in codebase.edges:
            if rg.graph.has_node(edge.source) and rg.graph.has_node(edge.target):
                rg.graph.add_edge(edge.source, edge.target, symbols=edge.imported_names)

        # Cache file contents for BM25
        for fpath in codebase.files:
            rg._file_contents[fpath] = codebase.get_file_content(fpath)

        logger.info(
            f"RepoGraph: {rg.graph.number_of_nodes()} nodes, "
            f"{rg.graph.number_of_edges()} edges"
        )
        return rg

    def localize(
        self,
        query: str,
        max_files: int = 10,
        bm25_weight: float = 0.6,
        ppr_weight: float = 0.4,
    ) -> list[RankedFile]:
        """Hybrid BM25 + Personalized PageRank file ranking.

        1. BM25 keyword scoring against file contents and exports
        2. Personalized PageRank seeded from top BM25 results
        3. Combined: bm25_weight × BM25 + ppr_weight × PPR (both z-normalized)

        Returns ranked list of non-test files.
        """
        # Step 1: BM25 scoring
        bm25_scores = self._bm25_score(query)

        # Filter out test files
        source_scores = {
            f: s for f, s in bm25_scores.items()
            if f not in self._test_files
        }

        if not source_scores:
            return []

        # Step 2: PageRank (if networkx available)
        ppr_scores = {}
        if HAS_NETWORKX and self.graph and self.graph.number_of_nodes() > 0:
            # Seed PPR from top BM25 results
            top_bm25 = sorted(source_scores, key=source_scores.get, reverse=True)[:5]
            ppr_scores = self._personalized_pagerank(top_bm25)

        # Step 3: Z-normalize and combine
        bm25_z = _z_normalize(source_scores)
        ppr_z = _z_normalize(ppr_scores) if ppr_scores else {}

        combined = {}
        for fpath in source_scores:
            b = bm25_z.get(fpath, 0.0)
            p = ppr_z.get(fpath, 0.0)
            combined[fpath] = bm25_weight * b + ppr_weight * p

        # Build ranked results
        ranked = sorted(combined.keys(), key=lambda f: combined[f], reverse=True)
        results = []
        for fpath in ranked[:max_files]:
            dependents = len(list(self.graph.predecessors(fpath))) if self.graph else 0
            results.append(RankedFile(
                path=fpath,
                combined_score=combined[fpath],
                bm25_score=source_scores.get(fpath, 0.0),
                pagerank_score=ppr_scores.get(fpath, 0.0),
                dependents_count=dependents,
            ))

        return results

    def get_context_files(self, target_file: str, hops: int = 2) -> list[str]:
        """Get k-hop ego-graph around a file (RepoGraph pattern).

        Returns files within `hops` distance in the dependency graph,
        useful for providing context to the LLM when editing target_file.
        """
        if not self.graph or not self.graph.has_node(target_file):
            return [target_file]

        # BFS to find neighbors within k hops
        visited = {target_file}
        frontier = {target_file}

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                # Both predecessors (files importing this) and successors (files this imports)
                for neighbor in list(self.graph.predecessors(node)) + list(self.graph.successors(node)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # Sort by dependency count (most important first)
        return sorted(visited, key=lambda f: self.graph.degree(f), reverse=True)[:10]

    def _bm25_score(self, query: str) -> dict[str, float]:
        """Simple BM25 scoring of files against a query.

        Scores based on:
        - Filename word overlap (5× weight)
        - Export name overlap (3× weight)
        - File content term frequency (1× weight, with IDF)
        """
        query_terms = set(re.findall(r'\w+', query.lower()))
        if not query_terms:
            return {}

        scores = {}
        # Compute IDF across all files
        doc_count = len(self._file_contents) or 1
        term_doc_freq = {}
        for content in self._file_contents.values():
            content_terms = set(re.findall(r'\w+', content.lower()))
            for term in query_terms:
                if term in content_terms:
                    term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

        for fpath in self._file_exports:
            score = 0.0

            # Filename match
            fname_terms = set(re.findall(r'\w+', fpath.lower()))
            score += len(query_terms & fname_terms) * 5.0

            # Export name match
            for export_name in self._file_exports.get(fpath, []):
                export_terms = set(re.findall(r'\w+', export_name.lower()))
                score += len(query_terms & export_terms) * 3.0

            # Content TF-IDF
            content = self._file_contents.get(fpath, "")
            if content:
                content_lower = content.lower()
                for term in query_terms:
                    tf = content_lower.count(term)
                    if tf > 0:
                        df = term_doc_freq.get(term, 1)
                        idf = math.log(doc_count / df)
                        score += (tf * idf) * 0.1  # Dampen content signal

            if score > 0:
                scores[fpath] = score

        return scores

    def _personalized_pagerank(
        self, seed_files: list[str], alpha: float = 0.85
    ) -> dict[str, float]:
        """Run Personalized PageRank seeded from specific files.

        Args:
            seed_files: files to bias random walk restarts toward
            alpha: damping factor (0.85 = 15% chance of restart)
        """
        if not self.graph or not seed_files:
            return {}

        # Build personalization vector
        personalization = {}
        seed_weight = 1.0 / len(seed_files)
        for node in self.graph.nodes():
            personalization[node] = seed_weight if node in seed_files else 0.0

        try:
            scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                weight=None,  # Unweighted edges
                max_iter=100,
                tol=1e-6,
            )
            return scores
        except Exception as e:
            logger.debug(f"PageRank failed: {e}")
            return {}


def _z_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Z-score normalize a dict of scores."""
    if not scores:
        return {}

    values = list(scores.values())
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    if std < 1e-10:
        return {k: 0.0 for k in scores}

    return {k: (v - mean) / std for k, v in scores.items()}
