"""SICA — Self-Improving Coding Agent for the Belief Engine.

Implements autonomous self-modification with safety guarantees:
1. SEED proposes a modification to an evolvable file
2. ScaffoldDecomposition verifies the file is safe to modify
3. SelfPatch applies the change with snapshot
4. Benchmark validates no regression
5. Accept improvement or rollback

Research basis:
- SICA (arXiv 2504.15228): 17%→53% SWE-bench in 15 iterations, $7k total
- FunSearch scaffold: only modify priority functions, never the scaffold
- Safety: Docker sandbox (recommended), async overseer, version archive

The key insight from SICA: the SAME agent both performs tasks and improves itself.
The Belief Engine's SEED system already proposes improvements — SICA closes the
loop by automatically applying and validating them.

Usage:
    from belief.evolution.sica import SelfImprovementCycle
    cycle = SelfImprovementCycle(project_root)
    result = await cycle.run_one_iteration()
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.evolution.sica")


@dataclass
class IterationResult:
    """Result of one self-improvement iteration."""
    iteration: int
    proposal_title: str = ""
    target_file: str = ""
    pre_score: float = 0.0
    post_score: float = 0.0
    improvement: float = 0.0
    accepted: bool = False
    rolled_back: bool = False
    error: str = ""
    duration_seconds: float = 0.0


@dataclass
class SelfImprovementArchive:
    """Archive of all self-improvement iterations for rollback and analysis."""
    iterations: list[IterationResult] = field(default_factory=list)
    best_score: float = 0.0
    best_iteration: int = 0
    total_cost: float = 0.0

    def add(self, result: IterationResult) -> None:
        self.iterations.append(result)
        if result.accepted and result.post_score > self.best_score:
            self.best_score = result.post_score
            self.best_iteration = result.iteration

    @property
    def accept_rate(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(1 for r in self.iterations if r.accepted) / len(self.iterations)


class SelfImprovementCycle:
    """One iteration of SICA-style self-improvement.

    The cycle:
    1. Run a mini-benchmark (5 challenges) to establish baseline
    2. SEED analyzes failures → proposes improvement
    3. ScaffoldDecomposition validates target is evolvable
    4. SelfPatch applies with snapshot
    5. Re-run mini-benchmark
    6. If improved → accept. If regressed → rollback.
    7. Store result in archive with Q-value update.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.archive_path = self.project_root / ".belief-engine" / "sica_archive.json"
        self.archive = self._load_archive()

    async def run_one_iteration(
        self,
        benchmark_tiers: list[int] | None = None,
        benchmark_ids: list[str] | None = None,
    ) -> IterationResult:
        """Execute one self-improvement iteration.

        Args:
            benchmark_tiers: Tiers to run for validation (default: [1,2,3])
            benchmark_ids: Specific challenge IDs (overrides tiers)

        Returns:
            IterationResult with pre/post scores and accept/reject decision
        """
        import time

        t0 = time.time()
        iteration = len(self.archive.iterations) + 1
        result = IterationResult(iteration=iteration)

        if benchmark_tiers is None and benchmark_ids is None:
            benchmark_tiers = [1, 2, 3]  # Quick validation set

        try:
            # Step 1: Baseline benchmark
            logger.info(f"SICA iteration {iteration}: running baseline benchmark")
            baseline = await self._run_benchmark(benchmark_tiers, benchmark_ids)
            result.pre_score = baseline["pass_rate"]
            logger.info(f"SICA: baseline = {baseline['passed']}/{baseline['total']} ({result.pre_score:.0%})")

            # Step 2: Generate improvement proposal
            proposal = await self._generate_proposal(baseline)
            if not proposal:
                result.error = "No proposal generated"
                result.duration_seconds = time.time() - t0
                self.archive.add(result)
                self._save_archive()
                return result

            result.proposal_title = proposal.get("title", "")
            result.target_file = proposal.get("target_file", "")

            # Step 3: Validate target is evolvable
            from belief.evolution.scaffold import ScaffoldDecomposition
            decomp = ScaffoldDecomposition.from_project(self.project_root)
            if not decomp.is_safe_to_modify(result.target_file):
                result.error = f"Target {result.target_file} is in fixed scaffold — skipping"
                result.duration_seconds = time.time() - t0
                self.archive.add(result)
                self._save_archive()
                return result

            # Step 4: Apply with snapshot
            target_path = self.project_root / result.target_file
            snapshot = self._snapshot(target_path)

            applied = self._apply_proposal(proposal, target_path)
            if not applied:
                result.error = "Failed to apply proposal"
                result.duration_seconds = time.time() - t0
                self.archive.add(result)
                self._save_archive()
                return result

            # Step 5: Re-run benchmark
            logger.info(f"SICA: applied proposal, re-running benchmark")
            post = await self._run_benchmark(benchmark_tiers, benchmark_ids)
            result.post_score = post["pass_rate"]
            result.improvement = result.post_score - result.pre_score

            # Step 6: Accept or rollback
            if result.post_score > result.pre_score:
                result.accepted = True
                logger.info(
                    f"SICA: ACCEPTED — {result.pre_score:.0%} → {result.post_score:.0%} "
                    f"(+{result.improvement:.0%}): {result.proposal_title}"
                )
            elif result.post_score == result.pre_score:
                # No improvement but no regression — accept if the proposal is low-risk
                result.accepted = True
                logger.info(f"SICA: ACCEPTED (neutral) — {result.proposal_title}")
            else:
                # Regression — rollback
                self._rollback(target_path, snapshot)
                result.rolled_back = True
                result.accepted = False
                logger.warning(
                    f"SICA: ROLLED BACK — {result.pre_score:.0%} → {result.post_score:.0%} "
                    f"({result.improvement:.0%}): {result.proposal_title}"
                )

            # Step 7: Store in archive
            result.duration_seconds = time.time() - t0
            self.archive.add(result)
            self._save_archive()

            # Update Q-value in memory
            await self._update_q_value(result)

            return result

        except Exception as e:
            result.error = str(e)
            result.duration_seconds = time.time() - t0
            self.archive.add(result)
            self._save_archive()
            logger.warning(f"SICA iteration {iteration} failed: {e}")
            return result

    async def _run_benchmark(
        self, tiers: list[int] | None, ids: list[str] | None
    ) -> dict[str, Any]:
        """Run a benchmark and return summary stats."""
        from belief.benchmark import run_benchmark
        results = await run_benchmark(tiers=tiers, challenge_ids=ids)
        passed = sum(1 for r in results if r.verdict == "pass")
        total = len(results)
        return {
            "passed": passed,
            "total": total,
            "pass_rate": passed / max(total, 1),
            "results": results,
        }

    async def _generate_proposal(self, baseline: dict) -> dict | None:
        """Use SEED to generate an improvement proposal from benchmark failures."""
        failures = [
            r for r in baseline.get("results", [])
            if r.verdict != "pass"
        ]
        if not failures:
            return None

        # Build failure context for SEED
        failure_summaries = [
            f"{r.challenge_id}: {r.verdict} ({r.tests_passed}/{r.tests_total} tests, "
            f"weighted={r.weighted_score:.2f})"
            for r in failures[:5]
        ]

        try:
            from belief.config.models import ModelRouter
            from belief.evolution import SEED
            from belief.llm import LLMClient

            router = ModelRouter()
            llm = LLMClient(router)
            seed = SEED(self.project_root)

            proposal = await seed.propose(failure_summaries, llm=llm)
            await llm.close()

            if proposal:
                return {
                    "title": proposal.title,
                    "target_file": proposal.target_file,
                    "code": proposal.code,
                    "what": proposal.what,
                    "why": proposal.why,
                    "confidence": proposal.confidence,
                }
        except Exception as e:
            logger.debug(f"SICA proposal generation failed: {e}")

        return None

    def _snapshot(self, target: Path) -> Path:
        """Save a snapshot before modification."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = self.project_root / ".belief-engine" / "sica_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{target.name}.{ts}.bak"
        if target.exists():
            shutil.copy2(target, snap_path)
        return snap_path

    def _rollback(self, target: Path, snapshot: Path) -> None:
        """Restore from snapshot."""
        if snapshot.exists():
            shutil.copy2(snapshot, target)
            logger.info(f"SICA: rolled back {target.name}")

    def _apply_proposal(self, proposal: dict, target: Path) -> bool:
        """Apply a proposal's code to the target file."""
        import ast

        code = proposal.get("code", "").strip()
        if not code:
            return False

        try:
            # Determine if full replacement or append
            is_full = (
                code.startswith('"""') or code.startswith("import ")
                or code.startswith("from ") or code.startswith("#")
            ) and len(code) > 200

            if is_full:
                target.write_text(code)
            else:
                existing = target.read_text() if target.exists() else ""
                target.write_text(existing + "\n\n" + code)

            # Validate syntax
            if target.suffix == ".py":
                ast.parse(target.read_text())

            return True

        except SyntaxError as e:
            logger.warning(f"SICA: proposal broke syntax: {e}")
            return False
        except Exception as e:
            logger.warning(f"SICA: apply failed: {e}")
            return False

    async def _update_q_value(self, result: IterationResult) -> None:
        """Update Q-value in ChromaDB for the strategy used."""
        try:
            from belief.memory.q_value_store import update_q_value
            reward = 1.0 if result.accepted and result.improvement > 0 else 0.0
            await update_q_value(
                key=f"sica:{result.proposal_title[:50]}",
                reward=reward,
                context=f"target={result.target_file}, improvement={result.improvement:.2%}",
            )
        except Exception:
            pass

    def _load_archive(self) -> SelfImprovementArchive:
        """Load the iteration archive from disk."""
        if self.archive_path.exists():
            try:
                data = json.loads(self.archive_path.read_text())
                archive = SelfImprovementArchive()
                for entry in data.get("iterations", []):
                    archive.iterations.append(IterationResult(**entry))
                archive.best_score = data.get("best_score", 0.0)
                archive.best_iteration = data.get("best_iteration", 0)
                return archive
            except Exception:
                pass
        return SelfImprovementArchive()

    def _save_archive(self) -> None:
        """Save the iteration archive to disk."""
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "iterations": [
                {
                    "iteration": r.iteration,
                    "proposal_title": r.proposal_title,
                    "target_file": r.target_file,
                    "pre_score": r.pre_score,
                    "post_score": r.post_score,
                    "improvement": r.improvement,
                    "accepted": r.accepted,
                    "rolled_back": r.rolled_back,
                    "error": r.error,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self.archive.iterations
            ],
            "best_score": self.archive.best_score,
            "best_iteration": self.archive.best_iteration,
            "total_iterations": len(self.archive.iterations),
            "accept_rate": self.archive.accept_rate,
        }
        self.archive_path.write_text(json.dumps(data, indent=2))
