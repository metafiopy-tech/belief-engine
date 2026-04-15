"""Belief Engine CLI — the entry point.

Wires together:
  - The LangGraph pipeline (graph.py)
  - Build memory (store completed builds, search for similar priors)
  - SEED (propose improvements every N builds)
  - Health daemon (background monitoring)

Usage:
    python -m belief.cli --goal "build a script that fetches HN top stories"
    python -m belief.cli --goal "..." --max-cost 5.0
    python -m belief.cli --goal "..." --max-iterations 5
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

from belief.config.models import ModelRouter
from belief.config.settings import settings
from belief.graph import build_pipeline
from belief.models.state import Phase


def _configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "\033[90m%(asctime)s\033[0m \033[36m%(levelname)-8s\033[0m \033[90m%(name)-28s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("belief")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    for noisy in ("httpx", "httpcore", "hpack", "h2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _get_project_root() -> Path:
    """Find the project root (where CLAUDE.md lives)."""
    # Try current directory, then walk up
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "CLAUDE.md").exists() or (p / "belief").is_dir():
            return p
    return cwd


async def run(goal: str, max_cost: float = 10.0, max_iterations: int = 3) -> dict:
    """Run the full pipeline on a goal. Returns final state dict."""
    _configure_logging()
    logger = logging.getLogger("belief.cli")

    if not settings.anthropic_api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY not set.")
        print("  Copy .env.template to .env and add your key.\n")
        sys.exit(1)

    project_root = _get_project_root()
    run_id = f"belief-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting build: {goal[:80]}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Budget: ${max_cost:.2f} | Max iterations: {max_iterations}")

    # ── Initialize memory ─────────────────────────────────────────────────
    from belief.memory.store import BuildStore, BuildRecord, SessionState

    db_path = Path(settings.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    build_store = BuildStore(db_path)
    session = SessionState(project_root / ".belief-engine" / "sessions" / run_id)

    build_count = build_store.count()
    logger.info(f"Build memory: {build_count} prior build(s)")

    # Search for similar past builds
    similar = build_store.find_similar(goal, n=3)
    similar_context = ""
    if similar:
        logger.info(f"Found {len(similar)} similar prior build(s):")
        for s in similar:
            logger.info(f"  - {s.goal[:60]} (verdict={s.verdict})")
        similar_context = "\n".join(
            f"  - {s.goal} (verdict={s.verdict}, files: {', '.join(s.file_summaries.keys())})"
            for s in similar
        )

    # ── Initialize SEED ───────────────────────────────────────────────────
    from belief.evolution import SEED

    seed = SEED(project_root)

    # ── Start health daemon ───────────────────────────────────────────────
    from belief.daemons.health import HealthDaemon

    health = HealthDaemon(project_root, check_interval=600)
    health.start()

    # ── Build the pipeline and run ────────────────────────────────────────
    router = ModelRouter()

    # Detect multi-service goals — LLM classification with keyword fallback
    is_multi_service = False
    try:
        from belief.tools.multi_service import classify_goal
        from belief.llm import LLMClient

        classifier_llm = LLMClient(router)
        classification = await classify_goal(goal, classifier_llm)
        await classifier_llm.close()
        is_multi_service = classification.is_multi_service
    except Exception as e:
        # Fallback to keyword detection if LLM fails
        from belief.tools.multi_service import _classify_by_keywords
        classification = _classify_by_keywords(goal)
        is_multi_service = classification.is_multi_service
        logger.debug(f"LLM classification failed ({e}), used keywords: {is_multi_service}")

    if is_multi_service:
        try:
            from belief.graph_multi import build_multi_pipeline
            pipeline = build_multi_pipeline(router)
            logger.info("CLI: multi-service goal detected — using graph_multi pipeline")
        except Exception as e:
            logger.debug(f"Multi-service pipeline failed to load: {e}")
            pipeline = build_pipeline(router)
    else:
        pipeline = build_pipeline(router)

    initial_state = {
        "run_id": run_id,
        "user_goal": goal,
        "phase": Phase.INTAKE.value,
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_cost_usd": max_cost,
        "code_files": {},
        "test_files": {},
        "errors": [],
        "warnings": [],
        "agent_timings": {},
        "previous_gap_summaries": [],
        "similar_builds_context": similar_context,
        "polarity": {
            "latios_coherence": 0.5,
            "latias_coherence": 0.5,
            "world_state": "dormant",
            "emergence_events": 0,
            "dissonance_events": 0,
            "current_remainder": None,
            "current_covenant": None,
            "accumulated_remainders": [],
            "accumulated_covenants": [],
        },
    }

    start = time.time()
    final_state = await pipeline.ainvoke(initial_state)
    elapsed = time.time() - start

    # ── Store the completed build in memory ───────────────────────────────
    code_files = final_state.get("code_files", {})
    validation = final_state.get("validation_result")
    usage = final_state.get("token_usage")

    verdict_str = ""
    if validation:
        v = validation.get("verdict") if isinstance(validation, dict) else getattr(validation, "verdict", "")
        verdict_str = v.value if hasattr(v, "value") else str(v)

    cost = 0.0
    if usage:
        cost = usage.get("total_cost_usd", 0) if isinstance(usage, dict) else getattr(usage, "total_cost_usd", 0)

    if code_files:
        out_dir = settings.output_path / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in code_files.items():
            fpath = out_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)
            if fname.endswith(".sh"):
                fpath.chmod(0o755)

        # Store in build memory
        spec = final_state.get("requirement_spec")
        goal_refined = ""
        if spec:
            goal_refined = spec.get("goal_refined", "") if isinstance(spec, dict) else getattr(spec, "goal_refined", "")

        file_summaries = {}
        manifest = final_state.get("file_manifest")
        if manifest:
            files_list = manifest.get("files", []) if isinstance(manifest, dict) else getattr(manifest, "files", [])
            for f in files_list:
                fname_key = f.get("filename", "") if isinstance(f, dict) else getattr(f, "filename", "")
                purpose = f.get("purpose", "") if isinstance(f, dict) else getattr(f, "purpose", "")
                if fname_key:
                    file_summaries[fname_key] = purpose

        quality_scores = {}
        if validation:
            for attr in ("correctness_score", "completeness_score", "code_quality_score", "security_score"):
                val = validation.get(attr, 0) if isinstance(validation, dict) else getattr(validation, attr, 0)
                quality_scores[attr] = float(val)

        build_record = BuildRecord(
            run_id=run_id,
            goal=goal,
            goal_refined=goal_refined,
            file_summaries=file_summaries,
            quality_scores=quality_scores,
            output_path=str(out_dir),
            cost_usd=cost,
            duration_seconds=elapsed,
            verdict=verdict_str,
            tags=_extract_tags(goal),
        )
        build_store.store(build_record)
        logger.info(f"Build stored in memory (total: {build_store.count()})")

        # Store remainders from polarity
        polarity = final_state.get("polarity", {})
        remainders = polarity.get("accumulated_remainders", []) if isinstance(polarity, dict) else getattr(polarity, "accumulated_remainders", [])
        for r in remainders:
            session.add_remainder(r)

        # ── Reflexion — verbal self-critique on failed builds ──────────
        if verdict_str and verdict_str != "pass":
            try:
                from belief.memory.reflexion import generate_reflexion, store_reflexion
                from belief.llm import LLMClient

                reflexion_llm = LLMClient(router)
                validation = final_state.get("validation_result", {})
                errors = final_state.get("errors", [])
                reflection = await generate_reflexion(
                    goal=goal, validation_result=validation,
                    code_files=code_files, errors=errors, llm=reflexion_llm,
                )
                await reflexion_llm.close()

                if reflection:
                    vr = validation if isinstance(validation, dict) else {}
                    await store_reflexion(
                        goal=goal, reflection=reflection, verdict=verdict_str,
                        tests_passed=vr.get("tests_passed", 0),
                        tests_total=vr.get("tests_total", 0),
                    )
                    logger.info(f"Reflexion stored: {reflection[:80]}...")
            except Exception as e:
                logger.debug(f"Reflexion skipped: {e}")

        # ── SEED tick — check if it's time to propose an improvement ──────
        if seed.tick():
            # Gather remainders from current session + soil antipatterns
            all_remainders = session.get_remainders(10)
            try:
                from belief.memory.soil import Soil
                from belief.memory.nutrients import NutrientType
                soil_path = Path("~/.belief-engine/soil").expanduser()
                soil = Soil(soil_path)
                antipatterns = soil.retrieve("build failure", nutrient_type=NutrientType.ANTIPATTERN, n=5)
                all_remainders.extend([a.content for a in antipatterns])
            except Exception:
                pass

            if all_remainders:
                logger.info(f"SEED: triggered — analyzing {len(all_remainders)} remainders/antipatterns")
                try:
                    from belief.llm import LLMClient
                    llm = LLMClient(router)
                    proposal = await seed.propose(all_remainders, llm=llm)
                    await llm.close()
                    if proposal:
                        logger.info(f"SEED proposal: {proposal.title}")
                        print(f"\n  🌱 SEED Proposal: {proposal.title}")
                        print(f"     What: {proposal.what}")
                        print(f"     Why: {proposal.why}")
                        print(f"     Target: {proposal.target_file}")
                        print(f"     Confidence: {proposal.confidence}")

                        # Check if auto-apply is allowed
                        from belief.hardening import seed_requires_approval
                        needs_approval = seed_requires_approval(
                            proposal.title, proposal.target_file, proposal.confidence
                        )

                        if needs_approval:
                            print(f"     Status: propose-only (human approval required)")
                            print(f"     Review: cat ~/.belief-engine/proposals.json\n")
                        else:
                            # Auto-apply: HIGH confidence + evolvable file
                            from belief.evolution import SelfPatch
                            patcher = SelfPatch(project_root)
                            success, msg = patcher.apply(proposal)
                            if success:
                                print(f"     Status: AUTO-APPLIED ✓ ({msg})")
                                logger.info(f"SEED auto-applied: {proposal.title}")
                            else:
                                print(f"     Status: auto-apply FAILED, rolled back ({msg})")
                                logger.warning(f"SEED auto-apply failed: {msg}")
                except Exception as e:
                    logger.debug(f"SEED proposal failed: {e}")

        # ── Print results ─────────────────────────────────────────────────
        print(f"\n{'═' * 60}")
        print(f"  BUILD COMPLETE — {elapsed:.1f}s")
        print(f"{'═' * 60}")
        print(f"\n  Output: {out_dir}/")
        print(f"  Files:")
        for fname in sorted(code_files):
            size = len(code_files[fname])
            print(f"    {fname}  ({size} chars)")

        if cost:
            print(f"\n  Cost: ${cost:.4f}")
        if verdict_str:
            print(f"  Verdict: {verdict_str}")

        # Memory stats
        print(f"  Build memory: {build_store.count()} total builds stored")
        if similar:
            print(f"  Similar prior builds: {len(similar)}")

        errors = final_state.get("errors", [])
        warnings = final_state.get("warnings", [])
        if errors:
            print(f"\n  Errors: {len(errors)}")
            for e in errors[:3]:
                print(f"    ! {e[:100]}")
        if warnings:
            print(f"  Warnings: {len(warnings)}")

        print()

        # ── Telegram notification ──────────────────────────────────────
        try:
            from belief.tools.notify import notify_build_complete
            notify_build_complete(run_id, goal, verdict_str, cost, elapsed, len(code_files))
        except Exception:
            pass  # Never block on notification failure

    else:
        print(f"\n  BUILD FAILED — no code files produced after {elapsed:.1f}s")
        for e in final_state.get("errors", []):
            print(f"    ! {e[:150]}")
        print()

        try:
            from belief.tools.notify import notify_build_complete
            notify_build_complete(run_id, goal, "failed", 0, elapsed, 0)
        except Exception:
            pass

    # Cleanup
    health.stop()
    build_store.close()

    return final_state


def _extract_tags(goal: str) -> list[str]:
    """Extract simple tags from the goal for indexing."""
    keywords = {"mcp", "api", "server", "script", "bot", "telegram", "discord",
                "web", "scraper", "monitor", "dashboard", "cli", "flask", "fastapi",
                "database", "sqlite", "postgres", "redis", "docker", "aws", "gcp"}
    words = set(goal.lower().split())
    return sorted(words & keywords)


async def _run_benchmark_cmd(args):
    """Run benchmark from CLI."""
    _configure_logging()
    from belief.benchmark import run_benchmark

    tiers = args.tiers if not args.ids else None
    results = await run_benchmark(tiers=tiers, challenge_ids=args.ids)

    passed = sum(1 for r in results if r.verdict == "pass")
    total = len(results)
    avg_score = sum(r.weighted_score for r in results) / max(total, 1)

    print(f"\n  Pass rate: {passed}/{total} ({passed/max(total,1):.0%})")
    print(f"  Avg score: {avg_score:.2f}")
    for r in results:
        marker = "✓" if r.verdict == "pass" else "✗"
        print(f"  {marker} {r.challenge_id}: {r.tests_passed}/{r.tests_total} tests, score={r.weighted_score:.2f}")


async def _run_sica_cmd(args):
    """Run SICA self-improvement from CLI."""
    _configure_logging()
    logger = logging.getLogger("belief.cli")

    from belief.evolution.sica import SelfImprovementCycle, composite_utility
    project_root = _get_project_root()
    cycle = SelfImprovementCycle(project_root)

    print(f"\n{'═' * 60}")
    print(f"  SICA Self-Improvement — {args.iterations} iteration(s)")
    print(f"  Benchmark tiers: {args.tiers}")
    print(f"  Project: {project_root}")
    print(f"{'═' * 60}\n")

    for i in range(args.iterations):
        print(f"\n  ── Iteration {i+1}/{args.iterations} {'─' * 40}")
        try:
            if hasattr(args, 'dry_run') and args.dry_run:
                # Dry run: show what WOULD happen without modifying files
                from belief.evolution.sica import SelfImprovementCycle as _SIC
                # Run benchmark only
                benchmark_data = await cycle._run_benchmark(args.tiers, None)
                print(f"  Baseline: {benchmark_data['passed']}/{benchmark_data['total']} "
                      f"({benchmark_data['pass_rate']:.0%})")

                # Early stop: all passing = nothing to improve
                if benchmark_data['pass_rate'] >= 1.0:
                    print(f"  All challenges passing — nothing to improve. Stopping.")
                    break

                # Generate proposal only
                proposal = await cycle._generate_proposal(benchmark_data)
                if proposal:
                    print(f"  Proposal: {proposal.get('title', 'untitled')}")
                    print(f"  Target: {proposal.get('target_file', '?')}")
                    print(f"  Why: {proposal.get('why', '?')[:200]}")
                    print(f"  (dry run — not applied)")
                else:
                    print(f"  No proposal generated — stopping.")
                    break
                continue

            result = await cycle.run_one_iteration(benchmark_tiers=args.tiers)

            # Early stop: no proposal generated (nothing to improve)
            if result.error == "No proposal generated":
                print(f"  ○ All challenges passing — stopping early.")
                break

            if result.accepted:
                print(f"  ✓ ACCEPTED: {result.proposal_title}")
                print(f"    Score: {result.pre_score:.0%} → {result.post_score:.0%}")
                print(f"    Utility: {result.pre_utility:.4f} → {result.post_utility:.4f}")
                if result.improvements:
                    print(f"    New passes: {', '.join(result.improvements)}")
                print(f"    File: {result.target_file}")
                print(f"    Cost: ${result.cost_usd:.4f}, Time: {result.duration_seconds:.0f}s")
            elif result.rolled_back:
                print(f"  ✗ ROLLED BACK: {result.proposal_title}")
                print(f"    Score: {result.pre_score:.0%} → {result.post_score:.0%}")
                if result.regressions:
                    print(f"    Regressions: {', '.join(result.regressions[:5])}")
                print(f"    File: {result.target_file}")
            elif result.error:
                print(f"  ⚠ ERROR: {result.error[:200]}")
            else:
                print(f"  ○ NO CHANGE: {result.proposal_title}")

        except Exception as e:
            print(f"  ⚠ Iteration {i+1} crashed: {e}")
            logger.error(f"SICA iteration {i+1} error: {e}", exc_info=True)

    # Summary
    archive = cycle.archive
    print(f"\n{'═' * 60}")
    print(f"  SICA SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Iterations: {len(archive.iterations)}")
    print(f"  Accepted:   {sum(1 for r in archive.iterations if r.accepted)}")
    print(f"  Rejected:   {sum(1 for r in archive.iterations if r.rolled_back)}")
    print(f"  Errors:     {sum(1 for r in archive.iterations if r.error)}")
    print(f"  Accept rate: {archive.accept_rate:.0%}")
    print(f"  Best score:  {archive.best_score:.0%} (iteration {archive.best_iteration})")
    print(f"  Best utility: {archive.best_utility:.4f}")
    print(f"  Total cost:  ${archive.total_cost:.2f}")
    print(f"  Archive: {cycle.archive_path}")

    # Held-out validation — run on tiers NOT used during optimization
    if hasattr(args, 'validate_tiers') and args.validate_tiers:
        print(f"\n  ── Held-out Validation (Tiers {args.validate_tiers}) ──")
        from belief.benchmark import run_benchmark
        val_results = await run_benchmark(tiers=args.validate_tiers)
        val_passed = sum(1 for r in val_results if r.verdict == "pass")
        val_total = len(val_results)
        print(f"  Validation: {val_passed}/{val_total} ({val_passed/max(val_total,1)*100:.0f}%)")
        for r in val_results:
            icon = "✓" if r.verdict == "pass" else "✗"
            print(f"    {icon} {r.challenge_id}: {r.tests_passed}/{r.tests_total} tests")

    print()


async def _run_fix_cmd(args):
    """Run brownfield issue fixing from CLI."""
    _configure_logging()
    logger = logging.getLogger("belief.cli")

    repo = args.repo
    repo_path = None

    # Handle GitHub URLs — clone to temp directory
    if repo.startswith("http://") or repo.startswith("https://") or repo.startswith("git@"):
        import subprocess
        import tempfile

        clone_dir = tempfile.mkdtemp(prefix="belief_fix_")
        clone_cmd = ["git", "clone", "--depth", "1", repo, clone_dir]
        if hasattr(args, "commit") and args.commit:
            clone_cmd = ["git", "clone", repo, clone_dir]

        print(f"  Cloning {repo}...")
        proc = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"  ✗ Clone failed: {proc.stderr[:200]}")
            sys.exit(1)

        # Checkout specific commit if provided
        if hasattr(args, "commit") and args.commit:
            subprocess.run(
                ["git", "checkout", args.commit],
                capture_output=True, cwd=clone_dir, timeout=30,
            )

        repo_path = clone_dir
        print(f"  ✓ Cloned to {clone_dir}")
    else:
        repo_path = repo

    # Run the brownfield pipeline
    from belief.agents.brownfield_agent import fix_issue

    print(f"\n  Fixing: {args.issue[:80]}...")
    print(f"  Repo: {repo_path}")
    print(f"  Patches: {args.patches}\n")

    result = await fix_issue(
        repo_path=repo_path,
        issue=args.issue,
        n_patches=args.patches,
    )

    if result.success:
        print(f"\n  ✓ FIXED ({result.method})")
        print(f"  File: {result.patch_file}")
        print(f"  Tests: {result.tests_passed}/{result.tests_total} passed, {result.regressions} regressions")
        print(f"  Time: {result.duration_seconds:.1f}s")
        if result.patch_explanation:
            print(f"  Explanation: {result.patch_explanation[:200]}")

        # Show the diff
        if result.patch_old and result.patch_new:
            print(f"\n  --- a/{result.patch_file}")
            print(f"  +++ b/{result.patch_file}")
            for line in result.patch_old.split("\n")[:10]:
                print(f"  - {line}")
            print(f"  ...")
            for line in result.patch_new.split("\n")[:10]:
                print(f"  + {line}")
    else:
        print(f"\n  ✗ FAILED")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Time: {result.duration_seconds:.1f}s")


async def _run_recombine_cmd(args) -> None:
    """Run N recombinations to enrich the soil between builds."""
    from pathlib import Path
    from belief.memory.soil import Soil
    from belief.memory.recombination import RecombinationEngine

    soil_dir = Path.home() / ".belief-engine" / "soil"
    soil = Soil(soil_dir)
    engine = RecombinationEngine(soil)

    count = getattr(args, "n", 5)
    print(f"  Recombining {count} nutrient pairs...\n")
    results = await engine.run(n=count)
    print(f"\n  Created {len(results)} new recombination nutrients.")
    for r in results:
        parents = ", ".join(r.lineage_parent_ids)
        print(f"    {r.nutrient_id} (from {parents})")


def app():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Belief Engine — build from a goal")
    subparsers = parser.add_subparsers(dest="command")

    # Default: build
    build_parser = subparsers.add_parser("build", help="Build from a goal")
    build_parser.add_argument("--goal", required=True, help="What to build (natural language)")
    build_parser.add_argument("--max-cost", type=float, default=10.0, help="Max USD budget")
    build_parser.add_argument("--max-iterations", type=int, default=3, help="Max build loop iterations")
    build_parser.add_argument("--deploy", choices=["railway", "docker_local"],
                              help="Deploy after build")
    build_parser.add_argument("--deploy-name", help="Project name for deployment")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark challenges")
    bench_parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                              help="Tiers to run (default: 1-5)")
    bench_parser.add_argument("--ids", nargs="+", help="Specific challenge IDs")

    # SICA self-improvement
    sica_parser = subparsers.add_parser("sica", help="Run self-improvement cycle")
    sica_parser.add_argument("--iterations", type=int, default=1,
                             help="Number of improvement iterations (default: 1)")
    sica_parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3],
                             help="Benchmark tiers for validation (default: 1-3)")
    sica_parser.add_argument("--dry-run", action="store_true",
                             help="Run benchmark and generate proposals without applying them")
    sica_parser.add_argument("--validate-tiers", type=int, nargs="+",
                             help="Held-out tiers for post-run validation (e.g., 4 5)")

    # Bittensor miner
    mine_parser = subparsers.add_parser("mine", help="Run as Bittensor subnet miner")
    mine_parser.add_argument("--netuid", type=int, default=62, help="Subnet ID (default: 62)")
    mine_parser.add_argument("--wallet-name", default="miner", help="Wallet name")
    mine_parser.add_argument("--hotkey", default="default", help="Hotkey name")
    mine_parser.add_argument("--network", default="finney", help="Network (finney/test/local)")
    mine_parser.add_argument("--port", type=int, default=8091, help="Axon port")
    mine_parser.add_argument("--max-cost", type=float, default=0.50,
                             help="Max USD per challenge")

    # Recombine
    recombine_parser = subparsers.add_parser("recombine", help="Cross-pollinate soil nutrients")
    recombine_parser.add_argument("--n", type=int, default=5,
                                  help="Number of recombinations to run (default: 5)")

    # Brownfield fix
    fix_parser = subparsers.add_parser("fix", help="Fix an issue in an existing codebase")
    fix_parser.add_argument("--repo", required=True,
                            help="Path to repo or GitHub URL (https://github.com/user/repo)")
    fix_parser.add_argument("--issue", required=True, help="Issue description (natural language)")
    fix_parser.add_argument("--patches", type=int, default=3, help="Candidate patches (default: 3)")
    fix_parser.add_argument("--commit", help="Specific commit to check out")

    # Backward compat: --goal without subcommand
    parser.add_argument("--goal", help="What to build (shortcut for: build --goal)")
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--deploy", choices=["railway", "docker_local"])
    parser.add_argument("--deploy-name")

    args = parser.parse_args()

    # Route to correct handler
    if args.command == "benchmark":
        asyncio.run(_run_benchmark_cmd(args))
        sys.exit(0)
    elif args.command == "sica":
        asyncio.run(_run_sica_cmd(args))
        sys.exit(0)
    elif args.command == "mine":
        from belief.bittensor.miner import BeliefMiner
        miner = BeliefMiner(
            netuid=args.netuid,
            wallet_name=args.wallet_name,
            hotkey=args.hotkey,
            network=args.network,
            axon_port=args.port,
            max_cost_per_challenge=args.max_cost,
        )
        miner.start()
        sys.exit(0)
    elif args.command == "fix":
        asyncio.run(_run_fix_cmd(args))
        sys.exit(0)
    elif args.command == "recombine":
        asyncio.run(_run_recombine_cmd(args))
        sys.exit(0)
    elif args.command == "build" or args.goal:
        goal = getattr(args, "goal", None)
        if not goal:
            parser.error("--goal is required")
        result = asyncio.run(run(goal, args.max_cost, args.max_iterations))

        # Deploy if requested
        if args.deploy and result.get("code_files"):
            code_files = result["code_files"]
            deploy_name = args.deploy_name or goal.split()[-1].lower()[:20].replace(" ", "-")

            async def _do_deploy():
                from belief.deploy import deploy, DeployConfig, DeployTarget, DeployStatus
                config = DeployConfig(
                    target=DeployTarget(args.deploy),
                    project_name=deploy_name,
                )
                dr = await deploy(code_files, config)
                if dr.status == DeployStatus.LIVE:
                    print(f"\n  \033[32m✓ DEPLOYED\033[0m → {dr.url} ({dr.duration_seconds:.1f}s)")
                else:
                    print(f"\n  \033[31m✗ DEPLOY FAILED\033[0m: {dr.error}")

            asyncio.run(_do_deploy())

        phase = result.get("phase", "unknown")
        if isinstance(phase, str):
            success = phase == "complete"
        else:
            success = phase == Phase.COMPLETE
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    app()
