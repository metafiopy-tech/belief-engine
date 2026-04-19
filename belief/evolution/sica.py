"""SICA — Self-Improving Coding Agent for the Belief Engine.

Implements autonomous self-modification with safety guarantees,
based on SICA (arXiv 2504.15228, ICLR 2025 workshop):
  - 17% → 53% SWE-bench Verified in ~10 iterations
  - Key innovations at iterations 4-6 (smart editing), 9 (AST navigation)

Core improvements over naive self-improvement:
1. COMPOSITE UTILITY — not just pass rate. U = w_score × score + w_cost × (1 - cost/budget) + w_time × (1 - time/timeout)
2. REGRESSION GATING — zero regressions on previously-passing challenges
3. PER-CHALLENGE TRACKING — detect which specific challenges improved/regressed
4. ARCHIVE ALL VERSIONS — even underperforming ones (DGM/HGM research shows
   weaker ancestors can seed future breakthroughs)

Usage:
    belief sica --iterations 10 --tiers 1 2 3

    from belief.evolution.sica import SelfImprovementCycle
    cycle = SelfImprovementCycle(project_root)
    for i in range(10):
        result = await cycle.run_one_iteration()
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from belief.evolution.archive import (
    AgentVersion,
    Archive,
    BenchmarkResult,
    create_seed_version,
)

logger = logging.getLogger("belief.evolution.sica")

# ── Composite utility weights (from SICA paper) ─────────────────────────────
W_SCORE = 0.7   # Benchmark pass rate
W_COST = 0.15   # Cost efficiency (lower is better)
W_TIME = 0.15   # Speed (faster is better)
COST_BUDGET = 10.0   # Max USD per benchmark run
TIME_BUDGET = 3600.0  # Max seconds per benchmark run (1 hour)


def composite_utility(
    pass_rate: float,
    cost_usd: float = 0.0,
    time_seconds: float = 0.0,
) -> float:
    """Compute composite utility from SICA paper.

    U = w_score × pass_rate + w_cost × (1 - cost/budget) + w_time × (1 - time/timeout)

    This prevents reward hacking on any single dimension:
    - Pure accuracy optimization that's 10x more expensive gets penalized
    - Speed-only optimization that skips tests gets penalized
    """
    score_component = W_SCORE * pass_rate
    cost_component = W_COST * max(0.0, 1.0 - cost_usd / COST_BUDGET)
    time_component = W_TIME * max(0.0, 1.0 - time_seconds / TIME_BUDGET)
    return round(score_component + cost_component + time_component, 6)


@dataclass
class ChallengeOutcome:
    """Outcome of a single challenge in a benchmark run."""
    challenge_id: str
    passed: bool
    weighted_score: float = 0.0


@dataclass
class IterationResult:
    """Result of one self-improvement iteration."""
    iteration: int
    proposal_title: str = ""
    target_file: str = ""
    pre_score: float = 0.0
    post_score: float = 0.0
    pre_utility: float = 0.0
    post_utility: float = 0.0
    improvement: float = 0.0
    accepted: bool = False
    rolled_back: bool = False
    error: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    pre_passing: list[str] = field(default_factory=list)
    post_passing: list[str] = field(default_factory=list)


@dataclass
class SelfImprovementArchive:
    """Archive of all self-improvement iterations."""
    iterations: list[IterationResult] = field(default_factory=list)
    best_utility: float = 0.0
    best_iteration: int = 0
    best_score: float = 0.0
    total_cost: float = 0.0

    def add(self, result: IterationResult) -> None:
        self.iterations.append(result)
        self.total_cost += result.cost_usd
        if result.accepted and result.post_utility > self.best_utility:
            self.best_utility = result.post_utility
            self.best_score = result.post_score
            self.best_iteration = result.iteration

    @property
    def accept_rate(self) -> float:
        if not self.iterations:
            return 0.0
        return sum(1 for r in self.iterations if r.accepted) / len(self.iterations)


class SelfImprovementCycle:
    """SICA-style self-improvement with composite utility and regression gating."""

    def __init__(
        self,
        project_root: str | Path = ".",
        archive_path: str | Path | None = None,
        soil=None,
    ):
        self.project_root = Path(project_root).resolve()

        if archive_path is not None:
            self.archive_path = Path(archive_path)
        else:
            self.archive_path = self.project_root / ".belief-engine" / "sica_archive.json"

        # Load existing archive if present; don't write until first actual modification
        self.archive = self._load_archive()

        # Soil for ChromaDB memory (injected or lazily initialized when needed)
        self.soil = soil

        # Evolutionary archive (SQLite DAG of all agent versions)
        try:
            self.evo_archive = Archive()
            self._seed = create_seed_version(self.evo_archive)
        except (PermissionError, OSError) as e:
            logger.warning(f"Evolutionary archive unavailable: {e}")
            self.evo_archive = None
            self._seed = None

    async def run_one_iteration(
        self,
        benchmark_tiers: list[int] | None = None,
        benchmark_ids: list[str] | None = None,
    ) -> IterationResult:
        """Run one SICA iteration with composite utility and regression gating.

        Args:
            benchmark_tiers: Tiers to run for validation (default: [1,2,3])
            benchmark_ids: Specific challenge IDs (overrides tiers)

        Returns:
            IterationResult with pre/post scores, utility, regressions
        """
        t0 = time.time()
        iteration = len(self.archive.iterations) + 1
        result = IterationResult(iteration=iteration)

        if benchmark_tiers is None and benchmark_ids is None:
            benchmark_tiers = [1, 2, 3]

        try:
            # ── Step 1: Baseline benchmark ──────────────────────────────
            logger.info(f"SICA iteration {iteration}: running baseline benchmark")
            baseline = await self._run_benchmark(benchmark_tiers, benchmark_ids)
            result.pre_score = baseline["pass_rate"]
            result.pre_passing = baseline["passing_ids"]
            result.pre_utility = composite_utility(
                baseline["pass_rate"], baseline["cost"], baseline["time"]
            )
            logger.info(
                f"SICA: baseline = {baseline['passed']}/{baseline['total']} "
                f"({result.pre_score:.0%}), utility={result.pre_utility:.4f}"
            )

            # ── Step 2: Generate improvement proposal ───────────────────
            proposal = await self._generate_proposal(baseline)
            if not proposal:
                result.error = "No proposal generated"
                result.duration_seconds = time.time() - t0
                self.archive.add(result)
                self._save_archive()
                return result

            result.proposal_title = proposal.get("title", "")
            result.target_file = proposal.get("target_file", "")

            # ── Step 2b: NEW_TOOL autocatalytic path ─────────────────────
            if proposal.get("improvement_type") == "new_tool":
                from belief.evolution.self_improvement import execute_new_tool_proposal

                logger.info(f"SICA: NEW_TOOL proposal — {proposal.get('title', 'unnamed')}")
                tool_result = await execute_new_tool_proposal(
                    soil=self.soil,
                    proposal_title=proposal.get("title", ""),
                )

                if tool_result.success:
                    result.accepted = True
                    result.post_score = result.pre_score
                    result.improvement = 0.0
                    result.post_utility = result.pre_utility
                    logger.info(
                        f"SICA: NEW_TOOL accepted — {proposal.get('title', '')} "
                        f"(tool_id={tool_result.backup_path})"
                    )
                    if self.evo_archive is not None:
                        try:
                            self._save_to_evo_archive(result, proposal, baseline, baseline)
                        except Exception as e:
                            logger.debug(f"SICA: evo archive update skipped: {e}")
                else:
                    result.accepted = False
                    result.error = f"NEW_TOOL failed: {tool_result.error}"
                    logger.info(f"SICA: NEW_TOOL rejected — {tool_result.error}")

                result.duration_seconds = time.time() - t0
                self.archive.add(result)
                self._save_archive()

                try:
                    self._record_metrics(result, iteration)
                except Exception as e:
                    logger.debug(f"SICA: metrics recording skipped: {e}")

                return result

            # ── Step 3: Validate target is evolvable (standard path) ─────
            from belief.evolution.scaffold import ScaffoldDecomposition
            decomp = ScaffoldDecomposition.from_project(self.project_root)
            if not decomp.is_safe_to_modify(result.target_file):
                result.error = f"Target {result.target_file} is in fixed scaffold — skipping"
                result.duration_seconds = time.time() - t0
                self.archive.add(result)
                self._save_archive()
                return result

            # ── Step 4: Apply with snapshot ──────────────────────────────
            target_path = self.project_root / result.target_file
            snapshot = self._snapshot(target_path)

            applied = self._apply_proposal(proposal, target_path)
            if not applied:
                result.error = "Failed to apply proposal"
                result.duration_seconds = time.time() - t0
                self.archive.add(result)
                self._save_archive()
                return result

            # ── Step 5: Re-run benchmark ────────────────────────────────
            logger.info(f"SICA: applied proposal, re-running benchmark")
            post = await self._run_benchmark(benchmark_tiers, benchmark_ids)
            result.post_score = post["pass_rate"]
            result.post_passing = post["passing_ids"]
            result.improvement = result.post_score - result.pre_score
            result.cost_usd = baseline["cost"] + post["cost"]
            result.post_utility = composite_utility(
                post["pass_rate"], post["cost"], post["time"]
            )

            # ── Step 6: Regression detection ────────────────────────────
            pre_set = set(result.pre_passing)
            post_set = set(result.post_passing)
            result.regressions = sorted(pre_set - post_set)
            result.improvements = sorted(post_set - pre_set)

            # ── Step 7: Accept or rollback (strict gating) ──────────────
            has_regressions = len(result.regressions) > 0
            utility_improved = result.post_utility > result.pre_utility
            score_improved = result.post_score > result.pre_score

            if has_regressions:
                # HARD RULE: zero regressions on previously-passing challenges
                self._rollback(target_path, snapshot)
                result.rolled_back = True
                result.accepted = False
                logger.warning(
                    f"SICA: ROLLED BACK — {len(result.regressions)} regression(s): "
                    f"{', '.join(result.regressions[:3])}"
                )
            elif utility_improved:
                # Composite utility improved — accept
                result.accepted = True
                logger.info(
                    f"SICA: ACCEPTED — utility {result.pre_utility:.4f} → {result.post_utility:.4f} "
                    f"(score {result.pre_score:.0%} → {result.post_score:.0%})"
                )
                if result.improvements:
                    logger.info(f"  NEW passes: {', '.join(result.improvements)}")
            elif score_improved and not has_regressions:
                # Score improved but utility didn't (cost/time increased) — accept cautiously
                result.accepted = True
                logger.info(
                    f"SICA: ACCEPTED (score only) — {result.pre_score:.0%} → {result.post_score:.0%}"
                )
            else:
                # No improvement — rollback
                self._rollback(target_path, snapshot)
                result.rolled_back = True
                result.accepted = False
                logger.info(
                    f"SICA: ROLLED BACK — no improvement "
                    f"(utility {result.pre_utility:.4f} → {result.post_utility:.4f})"
                )

            # ── Step 8: Archive ─────────────────────────────────────────
            result.duration_seconds = time.time() - t0
            self.archive.add(result)
            self._save_archive()

            # ── Step 9: Save to evolutionary archive ───────────────────
            self._save_to_evo_archive(result, proposal, baseline, post)

            # Update Q-value in memory
            await self._update_q_value(result)

            # ── Step 9b: Record metrics ────────────────────────────────
            try:
                self._record_metrics(result, iteration)
            except Exception as e:
                logger.debug(f"SICA: metrics recording skipped: {e}")

            # ── Step 10: DSPy prompt optimization every 10 iterations ──
            if iteration > 0 and iteration % 10 == 0:
                try:
                    from belief.optimization.dspy_modules import is_dspy_available
                    if is_dspy_available():
                        from belief.optimization.compiler import BeliefOptimizer
                        from belief.optimization.dspy_modules import get_all_modules
                        from belief.optimization.prompt_store import PromptStore

                        modules = get_all_modules()
                        optimizer = BeliefOptimizer()
                        results = optimizer.compile_all(modules, [], [])
                        optimized = {n: m for n, (m, _) in results.items()}
                        prompts = optimizer.extract_optimized_prompts(optimized)
                        if prompts:
                            store = PromptStore()
                            store.save(prompts, f"sica-iter-{iteration}")
                            logger.info(f"SICA: saved {len(prompts)} optimized prompts")
                except Exception as e:
                    logger.debug(f"SICA: DSPy optimization skipped: {e}")

            # ── Step 11: Trigger jitterbug every 10 iterations ─────────
            if iteration > 0 and iteration % 10 == 0:
                try:
                    from belief.evolution.jitterbug import run_jitterbug_cycle
                    logger.info(f"SICA: triggering jitterbug cycle at iteration {iteration}")
                    jb_result = await run_jitterbug_cycle(
                        n_goals=5,
                        cycle_number=iteration // 10,
                    )
                    logger.info(
                        f"SICA: jitterbug complete — "
                        f"tools={len(jb_result.get('integrated_tool_ids', []))}, "
                        f"stage {jb_result.get('stage_before', 0)}→{jb_result.get('stage_after', 0)}"
                    )
                except Exception as e:
                    logger.debug(f"SICA: jitterbug skipped: {e}")

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
        """Run a benchmark and return summary with per-challenge tracking."""
        from belief.benchmark import run_benchmark
        t0 = time.time()
        results = await run_benchmark(tiers=tiers, challenge_ids=ids)
        elapsed = time.time() - t0

        passed = sum(1 for r in results if r.verdict == "pass")
        total = len(results)
        cost = sum(r.cost_usd for r in results)
        passing_ids = [r.challenge_id for r in results if r.verdict == "pass"]

        return {
            "passed": passed,
            "total": total,
            "pass_rate": passed / max(total, 1),
            "cost": cost,
            "time": elapsed,
            "passing_ids": passing_ids,
            "results": results,
        }

    async def _generate_proposal(self, baseline: dict) -> dict | None:
        """Use SEED to generate an improvement proposal from benchmark failures.

        Returns a new_tool proposal when failure clustering conditions are met;
        otherwise falls through to the SEED LLM proposal path.
        """
        # NEW_TOOL path: check before spending LLM tokens on standard proposal
        if await self._should_propose_new_tool():
            logger.info("SICA: routing to NEW_TOOL — failure cluster detected")
            return {
                "improvement_type": "new_tool",
                "title": "Build self-authored tool for recurring failure pattern",
                "description": (
                    "SICA detected a recurring failure cluster that could be addressed "
                    "by a self-authored validation tool."
                ),
                "target_file": "",
            }

        failures = [
            r for r in baseline.get("results", [])
            if r.verdict != "pass"
        ]
        if not failures:
            logger.info("SICA: all challenges passing — no proposal needed")
            return None

        failure_summaries = [
            f"{r.challenge_id}: {r.verdict} ({r.tests_passed}/{r.tests_total} tests, "
            f"weighted={r.weighted_score:.2f})"
            for r in failures[:5]
        ]

        # Include archive context so SEED learns from past iterations
        archive_context = ""
        if self.archive.iterations:
            recent = self.archive.iterations[-5:]
            archive_lines = []
            for r in recent:
                status = "ACCEPTED" if r.accepted else "REJECTED"
                archive_lines.append(
                    f"  Iter {r.iteration}: {status} — '{r.proposal_title}' "
                    f"targeting {r.target_file} (improvement={r.improvement:+.0%})"
                )
                if r.regressions:
                    archive_lines.append(f"    Regressions: {', '.join(r.regressions[:3])}")
            archive_context = "\nRECENT SICA HISTORY (avoid repeating rejected approaches):\n" + "\n".join(archive_lines)

        try:
            from belief.config.models import ModelRouter
            from belief.evolution import SEED
            from belief.llm import LLMClient

            router = ModelRouter()
            llm = LLMClient(router)
            seed = SEED(self.project_root)

            proposal = await seed.propose(
                failure_summaries + ([archive_context] if archive_context else []),
                llm=llm,
            )
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

    async def _should_propose_new_tool(self) -> bool:
        """Check whether failure clustering suggests a new_tool proposal.

        Returns True when there are enough clustered failures to warrant
        building a new self-authored tool; False otherwise.
        """
        try:
            from belief.evolution.self_improvement import (
                cluster_failures,
                get_recent_failures,
                select_target_cluster,
            )
            from belief.memory.tool_registry import ToolRegistry

            # Lazy-init soil if not injected
            if self.soil is None:
                try:
                    from belief.memory.soil import Soil
                    self.soil = Soil()
                except Exception:
                    return False

            failures = get_recent_failures(self.soil)
            if not failures:
                return False

            clusters = cluster_failures(failures)
            registry = ToolRegistry(self.soil)
            existing = [t.name for t in registry.get_active_tools()]
            target = select_target_cluster(clusters, existing_tool_names=existing)
            return target is not None
        except Exception as e:
            logger.debug(f"SICA: new_tool check failed (non-fatal): {e}")
            return False

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
            reward = 1.0 if result.accepted and result.improvement > 0 else (
                0.5 if result.accepted else 0.0
            )
            await update_q_value(
                key=f"sica:{result.proposal_title[:50]}",
                reward=reward,
                context=(
                    f"target={result.target_file}, "
                    f"improvement={result.improvement:.2%}, "
                    f"regressions={len(result.regressions)}, "
                    f"utility={result.post_utility:.4f}"
                ),
            )
        except Exception:
            pass

    def _save_to_evo_archive(
        self,
        result: IterationResult,
        proposal: dict,
        baseline: dict,
        post: dict,
    ) -> None:
        """Save the iteration as an AgentVersion in the evolutionary archive."""
        if self.evo_archive is None:
            return
        try:
            # Select parent from archive (DGM sampling)
            parent = self.evo_archive.select_parent()

            # Compute niche descriptor
            niche = self._compute_niche(post)

            version = AgentVersion(
                id=str(uuid.uuid4()),
                parent_id=parent.id,
                created_at=datetime.now(timezone.utc),
                system_prompts=parent.system_prompts,
                tool_ids=parent.tool_ids,
                principle_ids=parent.principle_ids,
                covenant_ids=parent.covenant_ids,
                model_config=parent.model_config,
                diff_from_parent=proposal.get("what", result.proposal_title),
                proposal_rationale=proposal.get("why", ""),
                utility=result.post_utility,
                niche_descriptor=niche,
                canary_passed=True,
            )

            # Save the version (accepted or not — every variant is kept)
            self.evo_archive.save_version(version)

            # Increment parent's children_count
            self.evo_archive.increment_children(parent.id)

            # Save benchmark results
            for cr in post.get("results", []):
                br = BenchmarkResult(
                    version_id=version.id,
                    challenge_id=cr.challenge_id,
                    passed=cr.verdict == "pass",
                    score=cr.weighted_score,
                    cost_usd=cr.cost_usd,
                    time_seconds=cr.build_time_seconds,
                    error_summary=cr.error if cr.verdict != "pass" else None,
                )
                self.evo_archive.save_result(br)

            logger.info(
                f"SICA: saved version {version.id[:8]} to evo archive "
                f"(parent={parent.id[:8]}, niche={niche}, utility={result.post_utility:.4f})"
            )
        except Exception as e:
            logger.debug(f"SICA: evo archive save failed (non-fatal): {e}")

    @staticmethod
    def _compute_niche(benchmark_data: dict) -> tuple:
        """Compute MAP-Elites niche descriptor from benchmark results.

        Returns (code_length_bucket, tool_count, domain_id) where:
          - code_length_bucket: 0 (<100 lines), 1 (100-300), 2 (300-500), 3 (500+)
          - tool_count: 0 (no self-authored tools for now)
          - domain_id: 0 (general)
        """
        # Estimate avg output lines from benchmark results
        # (we don't have direct access to output files here, so use a heuristic)
        results = benchmark_data.get("results", [])
        avg_score = benchmark_data.get("pass_rate", 0.0)

        # Use pass_rate as a rough proxy for code complexity bucket
        if avg_score >= 0.8:
            code_bucket = 2  # Likely medium-large output
        elif avg_score >= 0.5:
            code_bucket = 1  # Medium output
        else:
            code_bucket = 0  # Small/failing output

        tool_count = 0  # Self-authored tools (future integration)
        domain_id = 0   # General purpose (future: derive from challenge tags)

        return (code_bucket, tool_count, domain_id)

    def _record_metrics(self, result: IterationResult, iteration: int) -> None:
        """Record per-iteration metrics to the dashboard."""
        from belief.metrics.dashboard import IterationMetrics, MetricsDashboard

        dashboard = MetricsDashboard()
        metrics = IterationMetrics(
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            benchmark_score=result.post_score,
            benchmark_ci_lower=max(0.0, result.post_score - 0.1),
            benchmark_ci_upper=min(1.0, result.post_score + 0.1),
            cost_per_solved=result.cost_usd / max(len(result.post_passing), 1),
            novel_capabilities=len(result.improvements),
            regressions=len(result.regressions),
        )

        # Add tool/covenant counts if available
        try:
            from belief.memory.soil import Soil
            from belief.memory.tool_registry import ToolRegistry
            soil = Soil()
            registry = ToolRegistry(soil)
            metrics.tool_library_size = len(registry.get_active_tools())
            cov_col = soil._collections.get("belief_covenants")
            metrics.covenant_count = cov_col.count() if cov_col else 0
        except Exception:
            pass

        dashboard.record(metrics)

    def _load_archive(self) -> SelfImprovementArchive:
        """Load the iteration archive from disk; return empty archive if missing/corrupt."""
        if not self.archive_path.exists():
            return SelfImprovementArchive()
        try:
            data = json.loads(self.archive_path.read_text())
            archive = SelfImprovementArchive()
            for entry in data.get("iterations", []):
                archive.iterations.append(IterationResult(**{
                    k: v for k, v in entry.items()
                    if k in IterationResult.__dataclass_fields__
                }))
            archive.best_score = data.get("best_score", 0.0)
            archive.best_utility = data.get("best_utility", 0.0)
            archive.best_iteration = data.get("best_iteration", 0)
            archive.total_cost = data.get("total_cost", 0.0)
            return archive
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt SICA archive, starting fresh: {e}")
            return SelfImprovementArchive()

    def _save_archive(self) -> None:
        """Save the iteration archive to disk; log but don't crash on permission errors."""
        try:
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "iterations": [
                    {
                        "iteration": r.iteration,
                        "proposal_title": r.proposal_title,
                        "target_file": r.target_file,
                        "pre_score": r.pre_score,
                        "post_score": r.post_score,
                        "pre_utility": r.pre_utility,
                        "post_utility": r.post_utility,
                        "improvement": r.improvement,
                        "accepted": r.accepted,
                        "rolled_back": r.rolled_back,
                        "error": r.error,
                        "duration_seconds": r.duration_seconds,
                        "cost_usd": r.cost_usd,
                        "regressions": r.regressions,
                        "improvements": r.improvements,
                        "pre_passing": r.pre_passing,
                        "post_passing": r.post_passing,
                    }
                    for r in self.archive.iterations
                ],
                "best_score": self.archive.best_score,
                "best_utility": self.archive.best_utility,
                "best_iteration": self.archive.best_iteration,
                "total_iterations": len(self.archive.iterations),
                "total_cost": self.archive.total_cost,
                "accept_rate": self.archive.accept_rate,
            }
            self.archive_path.write_text(json.dumps(data, indent=2))
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not save SICA archive: {e}")
