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
import logging
import os
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
    handler.setFormatter(
        logging.Formatter(
            "\033[90m%(asctime)s\033[0m \033[36m%(levelname)-8s\033[0m \033[90m%(name)-28s\033[0m %(message)s",
            datefmt="%H:%M:%S",
        )
    )
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


async def run(
    goal: str,
    max_cost: float = 10.0,
    max_iterations: int = 3,
    json_output: bool = False,
) -> dict:
    """Run the full pipeline on a goal. Returns final state dict."""
    _configure_logging()
    logger = logging.getLogger("belief.cli")

    _active_mode = os.environ.get("BELIEF_MODEL_MODE", "cloud").strip().lower()
    if not settings.anthropic_api_key and _active_mode != "local":
        print("\n  ERROR: ANTHROPIC_API_KEY not set.")
        print("  Copy .env.template to .env and add your key.\n")
        print("  (To run fully locally without an API key, set BELIEF_MODEL_MODE=local)\n")
        sys.exit(1)

    if _active_mode == "local":
        logger.info("Model mode: LOCAL (Ollama) — no Anthropic API calls will be made")

    project_root = _get_project_root()
    run_id = f"belief-{uuid.uuid4().hex[:8]}"

    # Reset per-build token tracker (Gate 4 instrumentation).
    from belief.llm import LOCAL_TRACKER

    LOCAL_TRACKER.records.clear()
    LOCAL_TRACKER.fallback_count = 0

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

    # Detect multi-service goals.
    #
    # In cloud mode: an LLM classifier (Haiku) makes the call; the
    # keyword classifier is a fallback on error.
    #
    # In local mode: skip the LLM call entirely — it was burning ~25s on
    # a 14B model for what is almost always a single-service build.
    # The keyword classifier catches the real multi-service cases
    # (compose/docker/microservices keywords) at near-zero cost.
    # If a local user genuinely needs LLM classification they can
    # override via BELIEF_FORCE_LLM_CLASSIFY=1 (escape hatch; undocumented
    # intentionally — use only if the keyword classifier gets a case
    # wrong in practice).
    from belief.config.models import RouteMode as _RouteMode

    _force_llm_classify = os.environ.get("BELIEF_FORCE_LLM_CLASSIFY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    _skip_llm_classify = router.mode is _RouteMode.LOCAL and not _force_llm_classify

    is_multi_service = False
    if _skip_llm_classify:
        from belief.tools.multi_service import _classify_by_keywords

        classification = _classify_by_keywords(goal)
        is_multi_service = classification.is_multi_service
        logger.info(
            f"CLI: local mode — using keyword classifier "
            f"(multi_service={is_multi_service}, 0 LLM cost)"
        )
    else:
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

    # Pipeline selection precedence:
    #   multi-service goal    → graph_multi   (handles service orchestration)
    #   local mode + single   → graph_local   (collapsed 4-stage pipeline)
    #   default               → graph         (full 9-agent cloud pipeline)
    # Cloud + single-service is unchanged from v3.1.0.
    #
    # When the graph_local path is taken, the decomposer node is NOT
    # part of the graph — it runs as a deferred post-print task so the
    # user sees BUILD COMPLETE ~60s sooner on a 14B local model.  We
    # track this in `used_local_pipeline` so the finalisation code
    # below knows to invoke the decomposer explicitly.
    RouteMode = _RouteMode  # reuse the import aliased above
    used_local_pipeline = False
    if is_multi_service:
        try:
            from belief.graph_multi import build_multi_pipeline

            pipeline = build_multi_pipeline(router)
            logger.info("CLI: multi-service goal detected — using graph_multi pipeline")
        except Exception as e:
            logger.debug(f"Multi-service pipeline failed to load: {e}")
            pipeline = build_pipeline(router)
    elif router.mode is RouteMode.LOCAL:
        try:
            from belief.graph_local import build_local_pipeline

            pipeline = build_local_pipeline(router)
            used_local_pipeline = True
            logger.info("CLI: local mode + single-service — using collapsed graph_local pipeline")
        except Exception as e:
            logger.warning(f"graph_local failed to load ({e}); falling back to full pipeline")
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
        v = (
            validation.get("verdict")
            if isinstance(validation, dict)
            else getattr(validation, "verdict", "")
        )
        verdict_str = v.value if hasattr(v, "value") else str(v)

    cost = 0.0
    if usage:
        cost = (
            usage.get("total_cost_usd", 0)
            if isinstance(usage, dict)
            else getattr(usage, "total_cost_usd", 0)
        )

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
            goal_refined = (
                spec.get("goal_refined", "")
                if isinstance(spec, dict)
                else getattr(spec, "goal_refined", "")
            )

        file_summaries = {}
        manifest = final_state.get("file_manifest")
        if manifest:
            files_list = (
                manifest.get("files", [])
                if isinstance(manifest, dict)
                else getattr(manifest, "files", [])
            )
            for f in files_list:
                fname_key = (
                    f.get("filename", "") if isinstance(f, dict) else getattr(f, "filename", "")
                )
                purpose = f.get("purpose", "") if isinstance(f, dict) else getattr(f, "purpose", "")
                if fname_key:
                    file_summaries[fname_key] = purpose

        quality_scores = {}
        if validation:
            for attr in (
                "correctness_score",
                "completeness_score",
                "code_quality_score",
                "security_score",
            ):
                val = (
                    validation.get(attr, 0)
                    if isinstance(validation, dict)
                    else getattr(validation, attr, 0)
                )
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
        remainders = (
            polarity.get("accumulated_remainders", [])
            if isinstance(polarity, dict)
            else getattr(polarity, "accumulated_remainders", [])
        )
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
                    goal=goal,
                    validation_result=validation,
                    code_files=code_files,
                    errors=errors,
                    llm=reflexion_llm,
                )
                await reflexion_llm.close()

                if reflection:
                    vr = validation if isinstance(validation, dict) else {}
                    await store_reflexion(
                        goal=goal,
                        reflection=reflection,
                        verdict=verdict_str,
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
                antipatterns = soil.retrieve(
                    "build failure", nutrient_type=NutrientType.ANTIPATTERN, n=5
                )
                all_remainders.extend([a.content for a in antipatterns])
            except Exception as e:
                logger.debug(f"Soil antipattern retrieval failed: {e}")

            if all_remainders:
                logger.info(
                    f"SEED: triggered — analyzing {len(all_remainders)} remainders/antipatterns"
                )
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
                            print("     Status: propose-only (human approval required)")
                            print("     Review: cat ~/.belief-engine/proposals.json\n")
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
        print("  Files:")
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
        except Exception as e:
            logger.debug(f"Notification failed (non-blocking): {e}")

        # ── Deferred decomposer (local-pipeline only) ─────────────────
        # The graph_local pipeline omits the decomposer so BUILD COMPLETE
        # prints 60+s earlier on a 14B local model.  We invoke it here,
        # after the user has already seen their build, so soil gets the
        # same nutrients it would have in cloud mode.  Failures here
        # must never retroactively fail the build — log + move on.
        if used_local_pipeline:
            try:
                from belief.memory.decomposer import decomposer_node as _decomposer

                logger.info("Decomposer: deferred post-print execution starting")
                _d_start = time.time()
                await _decomposer(final_state)
                logger.info(
                    f"Decomposer: deferred execution finished in {time.time() - _d_start:.1f}s"
                )
            except Exception as _decomposer_err:
                logger.warning(f"Deferred decomposer failed (non-fatal): {_decomposer_err}")

        # ── Session 6 (v3.2): persist a BuildOutcome to the agent archive ─
        # Fires for both successful and fail_fixable builds; hard failures
        # are still recorded but filtered out on retrieval.  Never blocks
        # the user — the call is wrapped in its own try/except internally.
        try:
            from belief.archive.persist import persist_build_outcome

            persist_build_outcome(final_state)
        except Exception as _archive_err:
            logger.debug("archive persist top-level error: %s", _archive_err)

    else:
        print(f"\n  BUILD FAILED — no code files produced after {elapsed:.1f}s")
        for e in final_state.get("errors", []):
            print(f"    ! {e[:150]}")
        print()

        try:
            from belief.tools.notify import notify_build_complete

            notify_build_complete(run_id, goal, "failed", 0, elapsed, 0)
        except Exception as e:
            logger.debug(f"Notification failed (non-blocking): {e}")

        # Even on a failed build, soil should capture the antipattern.
        # Run the deferred decomposer after the failure print too.
        if used_local_pipeline:
            try:
                from belief.memory.decomposer import decomposer_node as _decomposer

                logger.info("Decomposer: deferred post-print execution starting (failed build)")
                await _decomposer(final_state)
                logger.info("Decomposer: deferred execution finished")
            except Exception as _decomposer_err:
                logger.warning(f"Deferred decomposer failed (non-fatal): {_decomposer_err}")

    # Machine-readable summary (for A/B runner and scripting)
    if json_output:
        import json as _json

        validation_result = final_state.get("validation_result") or {}
        usage_result = final_state.get("token_usage") or {}

        def _gv(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        v = _gv(validation_result, "verdict", "")
        if hasattr(v, "value"):
            v = v.value
        summary = {
            "verdict": str(v) if v else ("pass" if verdict_str == "pass" else "fail"),
            "tests_passed": _gv(validation_result, "tests_passed", 0),
            "tests_total": _gv(validation_result, "tests_total", 0),
            "weighted_score": _gv(validation_result, "weighted_score", 0.0),
            "cost_usd": _gv(usage_result, "total_cost_usd", cost),
            "time_seconds": elapsed,
            "run_id": run_id,
        }
        print(_json.dumps(summary))

    # Cleanup
    health.stop()
    build_store.close()

    return final_state


def _extract_tags(goal: str) -> list[str]:
    """Extract simple tags from the goal for indexing."""
    keywords = {
        "mcp",
        "api",
        "server",
        "script",
        "bot",
        "telegram",
        "discord",
        "web",
        "scraper",
        "monitor",
        "dashboard",
        "cli",
        "flask",
        "fastapi",
        "database",
        "sqlite",
        "postgres",
        "redis",
        "docker",
        "aws",
        "gcp",
    }
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

    print(f"\n  Pass rate: {passed}/{total} ({passed / max(total, 1):.0%})")
    print(f"  Avg score: {avg_score:.2f}")
    for r in results:
        marker = "✓" if r.verdict == "pass" else "✗"
        print(
            f"  {marker} {r.challenge_id}: {r.tests_passed}/{r.tests_total} tests, score={r.weighted_score:.2f}"
        )


async def _run_sica_cmd(args):
    """Run SICA self-improvement from CLI."""
    _configure_logging()
    logger = logging.getLogger("belief.cli")

    from belief.evolution.sica import SelfImprovementCycle

    project_root = _get_project_root()
    cycle = SelfImprovementCycle(project_root)

    print(f"\n{'═' * 60}")
    print(f"  SICA Self-Improvement — {args.iterations} iteration(s)")
    print(f"  Benchmark tiers: {args.tiers}")
    print(f"  Project: {project_root}")
    print(f"{'═' * 60}\n")

    for i in range(args.iterations):
        print(f"\n  ── Iteration {i + 1}/{args.iterations} {'─' * 40}")
        try:
            if hasattr(args, "dry_run") and args.dry_run:
                # Dry run: show what WOULD happen without modifying files
                # Run benchmark only
                benchmark_data = await cycle._run_benchmark(args.tiers, None)
                print(
                    f"  Baseline: {benchmark_data['passed']}/{benchmark_data['total']} "
                    f"({benchmark_data['pass_rate']:.0%})"
                )

                # Early stop: all passing = nothing to improve
                if benchmark_data["pass_rate"] >= 1.0:
                    print("  All challenges passing — nothing to improve. Stopping.")
                    break

                # Generate proposal only
                proposal = await cycle._generate_proposal(benchmark_data)
                if proposal:
                    print(f"  Proposal: {proposal.get('title', 'untitled')}")
                    print(f"  Target: {proposal.get('target_file', '?')}")
                    print(f"  Why: {proposal.get('why', '?')[:200]}")
                    print("  (dry run — not applied)")
                else:
                    print("  No proposal generated — stopping.")
                    break
                continue

            result = await cycle.run_one_iteration(benchmark_tiers=args.tiers)

            # Early stop: no proposal generated (nothing to improve)
            if result.error == "No proposal generated":
                print("  ○ All challenges passing — stopping early.")
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
            print(f"  ⚠ Iteration {i + 1} crashed: {e}")
            logger.error(f"SICA iteration {i + 1} error: {e}", exc_info=True)

    # Summary
    archive = cycle.archive
    print(f"\n{'═' * 60}")
    print("  SICA SUMMARY")
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
    if hasattr(args, "validate_tiers") and args.validate_tiers:
        print(f"\n  ── Held-out Validation (Tiers {args.validate_tiers}) ──")
        from belief.benchmark import run_benchmark

        val_results = await run_benchmark(tiers=args.validate_tiers)
        val_passed = sum(1 for r in val_results if r.verdict == "pass")
        val_total = len(val_results)
        print(
            f"  Validation: {val_passed}/{val_total} ({val_passed / max(val_total, 1) * 100:.0f}%)"
        )
        for r in val_results:
            icon = "✓" if r.verdict == "pass" else "✗"
            print(f"    {icon} {r.challenge_id}: {r.tests_passed}/{r.tests_total} tests")

    print()


async def _run_fix_cmd(args):
    """Run brownfield issue fixing from CLI."""
    _configure_logging()
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
                capture_output=True,
                cwd=clone_dir,
                timeout=30,
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
        print(
            f"  Tests: {result.tests_passed}/{result.tests_total} passed, {result.regressions} regressions"
        )
        print(f"  Time: {result.duration_seconds:.1f}s")
        if result.patch_explanation:
            print(f"  Explanation: {result.patch_explanation[:200]}")

        # Show the diff
        if result.patch_old and result.patch_new:
            print(f"\n  --- a/{result.patch_file}")
            print(f"  +++ b/{result.patch_file}")
            for line in result.patch_old.split("\n")[:10]:
                print(f"  - {line}")
            print("  ...")
            for line in result.patch_new.split("\n")[:10]:
                print(f"  + {line}")
    else:
        print("\n  ✗ FAILED")
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


async def _run_jitterbug_cmd(args):
    """Run one jitterbug cycle."""
    from belief.evolution.jitterbug import run_jitterbug_cycle

    print(
        f"\n  Jitterbug cycle: {args.goals} expansion builds{' (dry run)' if args.dry_run else ''}"
    )

    result = await run_jitterbug_cycle(
        n_goals=args.goals,
        dry_run=args.dry_run,
    )

    print(f"\n  Cost: ${result.get('total_cost', 0):.2f}")
    print(f"  Expansion: {len(result.get('expansion_traces', []))} builds")
    if result.get("compression_summary"):
        print(
            f"  Compression:\n    {result['compression_summary'].replace(chr(10), chr(10) + '    ')}"
        )
    if not args.dry_run:
        print(f"  Tools built: {len(result.get('new_tools_built', []))}")
        print(f"  Covenants: {len(result.get('new_covenants', []))}")
        print(f"  Validation: {'PASSED' if result.get('validation_passed') else 'FAILED'}")
        print(f"  Stage: {result.get('stage_before', 0)} -> {result.get('stage_after', 0)}")


def _run_dashboard_cmd(args):
    """Display metrics dashboard."""
    from belief.metrics.dashboard import MetricsDashboard

    dashboard = MetricsDashboard()
    if hasattr(args, "json") and args.json:
        dashboard.print_json()
    else:
        dashboard.print_dashboard()


def _run_optimize_cmd(args):
    """Run DSPy/GEPA prompt optimization."""
    try:
        from belief.optimization.dspy_modules import AGENT_MODULES, is_dspy_available
    except ImportError:
        print("  Error: optimization module not found")
        return

    if not is_dspy_available():
        print("  Error: dspy is not installed. Install with: pip install 'belief-engine[optimize]'")
        return

    agents_to_optimize = []
    if args.all:
        agents_to_optimize = list(AGENT_MODULES.keys())
    elif args.agent:
        if args.agent not in AGENT_MODULES:
            print(f"  Unknown agent: {args.agent}")
            print(f"  Available: {', '.join(AGENT_MODULES.keys())}")
            return
        agents_to_optimize = [args.agent]
    else:
        print("  Specify an agent name or use --all")
        print(f"  Available: {', '.join(AGENT_MODULES.keys())}")
        return

    if args.dry_run:
        print(f"\n  Would optimize: {', '.join(agents_to_optimize)}")
        print("  (dry-run mode, no optimization performed)")
        return

    print(f"\n  Optimizing: {', '.join(agents_to_optimize)}")
    from belief.optimization.compiler import BeliefOptimizer
    from belief.optimization.prompt_store import PromptStore

    optimizer = BeliefOptimizer()
    store = PromptStore()

    # Build modules for selected agents
    modules = {}
    for name in agents_to_optimize:
        modules[name] = AGENT_MODULES[name]()

    # Use empty trainset/valset — real data would come from benchmark results
    trainset: list[dict] = []
    valset: list[dict] = []

    results = optimizer.compile_all(modules, trainset, valset)

    # Extract and save prompts
    optimized_modules = {name: mod for name, (mod, _) in results.items()}
    prompts = optimizer.extract_optimized_prompts(optimized_modules)

    if prompts:
        import uuid

        version_id = f"opt-{uuid.uuid4().hex[:8]}"
        store.save(prompts, version_id)
        print(f"  Saved {len(prompts)} optimized prompts (version: {version_id})")

    for name, (_, metrics) in results.items():
        print(
            f"  {name}: avg_score={metrics.get('avg_score', 0):.2f} ({metrics.get('optimizer', 'none')})"
        )

    print("  Optimization complete.")


def _run_library_cmd(args) -> None:
    """Session 12: list tools that were promoted from apex-predator nutrients.

    Reads the belief_tools ChromaDB collection via ToolRegistry and
    filters for tools whose `created_by` is 'jitterbug' (the marker
    Session 12's library_inductor sets). Prints a compact table.
    """
    from belief.memory.soil import Soil
    from belief.memory.tool_registry import ToolRegistry

    soil = Soil()
    registry = ToolRegistry(soil)
    tools = []
    try:
        tools = registry.get_active_tools()
    except Exception as exc:
        print(f"library: could not load tool registry: {exc}")
        return

    promoted = [t for t in tools if getattr(t, "created_by", "") == "jitterbug"]
    if not promoted:
        print("library: no promoted tools yet")
        return

    print(f"{'name':<32} {'uses':<5} {'quality':<8} parent")
    print("-" * 72)
    for tool in promoted:
        parent = (getattr(tool, "parent_id", "") or "-")[:24]
        print(
            f"{tool.name[:32]:<32} {int(tool.use_count):<5} "
            f"{float(tool.quality_score):<8.2f} {parent}"
        )


def _run_probe_cmd(args) -> None:
    """Session 10: belief probe train|test."""
    from belief.metrics.trace_collector import TraceCollector
    from belief.safety.confidence_probe import (
        ConfidenceProbe,
        DEFAULT_PROBE_PATH,
    )

    action = getattr(args, "probe_action", None) or "test"
    traces_path = getattr(args, "traces", None)
    probe_path = Path(getattr(args, "probe", None) or DEFAULT_PROBE_PATH)

    if action == "train":
        out_path = Path(getattr(args, "out", None) or DEFAULT_PROBE_PATH)
        min_samples = int(getattr(args, "min_samples", 200))
        tc_path = traces_path or str(Path("~/.belief-engine/traces.db").expanduser())
        tc = TraceCollector(tc_path, start_writer=False)
        rows = tc.get_training_data(min_builds=0)
        probe = ConfidenceProbe(out_path)
        meta = probe.train(rows, min_samples=min_samples)
        if probe.model is None:
            print(
                f"probe.train: refused (n_samples={meta.n_samples}, "
                f"min_samples={meta.min_samples_required})"
            )
            return
        print(
            f"probe trained: n={meta.n_samples} positives={meta.n_positive} "
            f"calibrated={meta.calibrated} feature_dim={meta.feature_dim}"
        )
        print(f"saved to: {out_path}")
        return

    if action == "test":
        tc_path = traces_path or str(Path("~/.belief-engine/traces.db").expanduser())
        tc = TraceCollector(tc_path, start_writer=False)
        rows = tc.get_training_data(min_builds=0)
        probe = ConfidenceProbe(probe_path)
        report = probe.evaluate(rows)
        if not report.get("trained"):
            print(f"probe.test: probe not trained (reason: {report.get('reason', 'unknown')})")
            return
        if "error" in report:
            print(f"probe.test: error {report['error']}")
            return
        print(
            f"n={report.get('n', 0)} accuracy={report.get('accuracy', 0):.3f} "
            f"brier={report.get('brier', 0):.3f}"
        )
        cm = report.get("confusion_matrix", [])
        if cm:
            print("confusion_matrix (rows=actual, cols=predicted):")
            for row in cm:
                print("  " + " ".join(f"{int(v):5d}" for v in row))
        return

    print("unknown probe action; try: train / test")


def _run_grinder_cmd(args) -> None:
    """Session 8: belief grinder start|status|pause|resume."""
    action = getattr(args, "grinder_action", None) or "status"
    from belief.grinder.daemon import GrinderDaemon, DEFAULT_PENDING_DIR
    from belief.grinder.status import format_status, read_status
    from belief.photosynthesis.safety.kill_switch import (
        ControlStatus,
        get_default_state,
    )

    if action == "start":
        pending = getattr(args, "pending_dir", None) or DEFAULT_PENDING_DIR
        daemon = GrinderDaemon(pending_dir=Path(pending))
        daemon.install_signal_handlers()
        asyncio.run(daemon.run_forever(max_builds=getattr(args, "max_builds", None)))
        return
    if action == "status":
        print(format_status(read_status()))
        return
    if action in {"pause", "resume"}:
        target = ControlStatus.PAUSED if action == "pause" else ControlStatus.RUNNING
        state = get_default_state()
        state.set_status(target, reason=f"cli:{action}")
        print(f"grinder control: {state.current_status().value}")
        return
    print("unknown grinder action; try: start / status / pause / resume")


async def _run_benchmark_compare_cmd(args) -> None:
    """Run cloud + local benchmark and print the comparison table."""
    from belief.benchmark_compare import format_report, run_benchmark_compare

    report = await run_benchmark_compare(
        challenge_ids=getattr(args, "ids", None),
        tiers=getattr(args, "tiers", None),
    )
    print(format_report(report))


def _run_models_cmd() -> None:
    """Print the active model routing table.

    Respects BELIEF_MODEL_MODE / BELIEF_LOCAL_MODEL / BELIEF_OLLAMA_URL
    env vars and the --mode / --local-model / --ollama-url CLI flags.
    Format is a three-column table: role / backend / model.
    """
    from belief.config.models import Backend, ModelRouter

    router = ModelRouter()
    rows = router.routing_table()

    print(f"mode:          {router.mode.value}")
    print(f"local_model:   {router.local_model}")
    print(f"ollama_url:    {router.ollama_base_url}")
    print()
    print(f"{'role':<15} {'backend':<8} {'model':<40}")
    print("-" * 64)
    for role, backend, model in rows:
        b = backend.value if isinstance(backend, Backend) else str(backend)
        print(f"{role:<15} {b:<8} {model:<40}")


def _run_manifold_cmd(args):
    """Show the domain-manifold summary — see belief/memory/manifold.py.

    ``belief manifold``         plain-text render (default)
    ``belief manifold --json``  machine-readable JSON
    ``belief manifold --gap-threshold 10``  custom sparse-soil cutoff
    """
    from belief.memory.manifold import build_manifold, format_report
    from belief.memory.soil import Soil

    soil = Soil()
    gap_threshold = getattr(args, "gap_threshold", None) or 5
    report = build_manifold(soil, gap_threshold=gap_threshold)
    if getattr(args, "json", False):
        print(report.to_json())
        return
    print(format_report(report, gap_threshold=gap_threshold))


def _run_progression_cmd():
    """Display current generative chain stage (per-domain + global)."""
    from belief.evolution.progression import (
        compute_all_domains,
        compute_progression,
        format_all_domains_report,
        format_progression_report,
    )
    from belief.memory.soil import Soil
    from belief.memory.tool_registry import ToolRegistry

    soil = Soil()
    registry = ToolRegistry(soil)

    # Session 7: per-domain view first, then the existing global report
    by_domain = compute_all_domains(soil, registry, [])
    print("\n" + format_all_domains_report(by_domain))
    print()
    metrics = compute_progression(soil, registry, [])
    print(format_progression_report(metrics))


def _run_experiment_cmd(args) -> None:
    """Dispatch experiment sub-commands."""
    action = getattr(args, "exp_action", None)

    if action in (None, "report"):
        from belief.experiments.reporter import comparison_table

        exp_id = getattr(args, "experiment_id", None)
        print(comparison_table(exp_id))
        return

    if action == "longitudinal":
        from belief.experiments.reporter import longitudinal_report

        print(longitudinal_report())
        return

    if action in ("run", "quick"):
        from belief.benchmark import CHALLENGES
        from belief.experiments.ab_runner import run_experiment

        tiers = getattr(args, "tiers", [1, 2])
        model = getattr(args, "model", "qwen2.5-coder:14b")

        if action == "quick":
            n = getattr(args, "n", 5)
            conditions = ["engine_local", "raw_local"]
        else:
            n = getattr(args, "challenges", 5)
            conditions = list(
                getattr(args, "conditions", ["engine_cloud", "engine_local", "raw_local"])
            )

        # Sample N challenges from the selected tiers
        import random

        pool = [c for c in CHALLENGES if c.tier in tiers]
        if not pool:
            pool = list(CHALLENGES)
        selected = random.sample(pool, min(n, len(pool)))
        challenge_list = [{"id": c.id, "goal": c.goal} for c in selected]

        print(
            f"\n  Experiment: {len(challenge_list)} challenge(s), "
            f"conditions: {', '.join(conditions)}"
        )
        print(f"  Model: {model}")
        print(f"  Tiers: {tiers}\n")

        experiment_id = asyncio.run(
            run_experiment(challenge_list, conditions=conditions, model=model)
        )

        print(f"\n  Experiment {experiment_id} complete.")
        print("  Results:\n")

        from belief.experiments.reporter import comparison_table

        print(comparison_table(experiment_id))
        return

    if action == "ablation-synth":
        # Session 4: delegate to scripts/synthesizer_ablation.py via
        # subprocess so the harness's env-var manipulation of the
        # build pipeline is cleanly isolated from this CLI process.
        import subprocess as _sp
        import sys as _sys
        from pathlib import Path as _Path

        script = _Path(__file__).resolve().parent.parent / "scripts" / "synthesizer_ablation.py"
        cmd: list[str] = [_sys.executable, str(script)]
        if getattr(args, "report", False):
            cmd.append("--report")
        else:
            cmd.extend(["--n", str(getattr(args, "n", 3))])
            chs = getattr(args, "challenges", None)
            if chs:
                cmd.extend(["--challenges", *chs])
        _sp.run(cmd, check=False)
        return

    print(f"Unknown experiment action: {action!r}")
    print("Try: belief experiment run | quick | report | longitudinal | ablation-synth")


def _run_validator_cmd(args) -> None:
    """Dispatch ``belief validator …`` sub-commands (Session 3/4)."""
    action = getattr(args, "validator_action", None)
    if action == "add-hallucination":
        from belief.validators.package_validator import PackageValidator

        validator = PackageValidator()
        name = args.name
        validator.add_hallucination(name)
        print(f"Added '{name}' to the hallucination blocklist.")
        return
    print("Usage: belief validator add-hallucination <name>")


def _run_archive_cmd(args) -> None:
    """Dispatch ``belief archive …`` sub-commands (Session 6)."""
    action = getattr(args, "archive_action", None)
    if action == "inspect":
        from belief.archive import AgentArchive

        archive = AgentArchive()
        size = archive.size()
        print(f"Agent archive size: {size} builds")
        if size == 0:
            return
        goal = getattr(args, "goal", None) or ""
        top = int(getattr(args, "top", 5))
        hits = archive.query_by_goal(goal or "build", k=top, verdicts=None)
        print(
            f"Top {len(hits)} results"
            + (f" for goal={goal!r}" if goal else " (ordered by semantic match to 'build')")
        )
        for i, hit in enumerate(hits, start=1):
            meta = hit.get("metadata") or {}
            print(
                f"  {i}. {hit['id']}  "
                f"verdict={meta.get('verdict')}  "
                f"score={float(meta.get('weighted_score') or 0):.2f}  "
                f"U={float(meta.get('utility_score') or 0):.3f}"
            )
            g = (meta.get("goal") or "")[:120]
            if g:
                print(f"       goal: {g}")
        return
    print("Usage: belief archive inspect [--goal <goal>] [--top N]")


def _run_synth_cmd(args) -> None:
    """Dispatch ``belief synth ...`` -- Synthesis Engine S2 word-set entry.

    Inserts user-submitted words into the photosynthesis raw_signals
    table as a synthetic source so the existing cascade filter / novelty
    gate / ranker / heap treat them uniformly with harvested signals.
    Optionally runs a synthesis cycle so the user sees end-to-end
    promotion in one command (degrades gracefully without LLM keys).
    """
    import asyncio

    action = getattr(args, "synth_action", None)
    if action != "words":
        print('Usage: belief synth words "x,y" [--db-path PATH] [--no-cycle]')
        return

    from belief.photosynthesis.config import PhotoConfig
    from belief.photosynthesis.sources.word_set import emit, parse_words
    from belief.photosynthesis.state import PhotosynthesisState

    try:
        words = parse_words(args.words)
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(2)

    # PhotoConfig defaults state_dir to /var/lib/photosynthesis -- the
    # daemon's launchd-root path. An interactive CLI doesn't have those
    # permissions, so root the user-mode state under ~/.belief-engine/
    # to mirror the rest of belief's user-state layout (soil, archive,
    # audit logs all live there). --db-path still overrides the
    # signals.sqlite location for hermetic testing if supplied.
    user_state_dir = Path.home() / ".belief-engine" / "photosynthesis"
    user_state_dir.mkdir(parents=True, exist_ok=True)
    config = PhotoConfig(
        state_dir=user_state_dir,
        log_dir=user_state_dir / "logs",
        config_dir=user_state_dir / "cfg",
    )
    db_path = args.db_path or str(config.signals_db)
    state = PhotosynthesisState(db_path=db_path)

    inserted = asyncio.run(emit(state, config, words=words, bundle_id=args.bundle_id))
    print(
        f"word_set: bundle={args.bundle_id or '(auto)'} words={len(words)} inserted={len(inserted)}"
    )
    for seed in inserted:
        print(f"  - {seed.source_id}  title={seed.title!r}")

    if args.no_cycle:
        print("--no-cycle: synthesis cycle skipped")
        return

    # Best-effort cycle. Without a generator_client/archive (the live
    # path requires API keys + ChromaDB), the cycle survey-and-tracks but
    # doesn't promote anything. Either way we report what happened.
    try:
        from belief.photosynthesis.synthesis.cycle import run_synthesis_cycle

        summary = asyncio.run(run_synthesis_cycle(state, config))
        print(
            f"cycle: surveyed={summary.surveyed} pushed={summary.pushed_to_heap} "
            f"promoted={summary.promoted} rejected={summary.rejected} "
            f"saturated={summary.saturated} errors={len(summary.errors)}"
        )
        if summary.errors:
            for err in summary.errors[:5]:
                print(f"  err: {err}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"cycle: skipped ({exc})")


def app():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Belief Engine — build from a goal")
    subparsers = parser.add_subparsers(dest="command")

    # Default: build
    build_parser = subparsers.add_parser("build", help="Build from a goal")
    build_parser.add_argument("--goal", required=True, help="What to build (natural language)")
    build_parser.add_argument("--max-cost", type=float, default=10.0, help="Max USD budget")
    build_parser.add_argument(
        "--max-iterations", type=int, default=3, help="Max build loop iterations"
    )
    build_parser.add_argument(
        "--deploy", choices=["railway", "docker_local"], help="Deploy after build"
    )
    build_parser.add_argument("--deploy-name", help="Project name for deployment")
    build_parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print a JSON summary line at the end (for scripting/A/B tests)",
    )

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark challenges")
    bench_parser.add_argument(
        "--tiers", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Tiers to run (default: 1-5)"
    )
    bench_parser.add_argument("--ids", nargs="+", help="Specific challenge IDs")

    # SICA self-improvement
    sica_parser = subparsers.add_parser("sica", help="Run self-improvement cycle")
    sica_parser.add_argument(
        "--iterations", type=int, default=1, help="Number of improvement iterations (default: 1)"
    )
    sica_parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Benchmark tiers for validation (default: 1-3)",
    )
    sica_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run benchmark and generate proposals without applying them",
    )
    sica_parser.add_argument(
        "--validate-tiers",
        type=int,
        nargs="+",
        help="Held-out tiers for post-run validation (e.g., 4 5)",
    )

    # Bittensor miner
    mine_parser = subparsers.add_parser("mine", help="Run as Bittensor subnet miner")
    mine_parser.add_argument("--netuid", type=int, default=62, help="Subnet ID (default: 62)")
    mine_parser.add_argument("--wallet-name", default="miner", help="Wallet name")
    mine_parser.add_argument("--hotkey", default="default", help="Hotkey name")
    mine_parser.add_argument("--network", default="finney", help="Network (finney/test/local)")
    mine_parser.add_argument("--port", type=int, default=8091, help="Axon port")
    mine_parser.add_argument("--max-cost", type=float, default=0.50, help="Max USD per challenge")

    # Recombine
    recombine_parser = subparsers.add_parser("recombine", help="Cross-pollinate soil nutrients")
    recombine_parser.add_argument(
        "--n", type=int, default=5, help="Number of recombinations to run (default: 5)"
    )

    # Brownfield fix
    fix_parser = subparsers.add_parser("fix", help="Fix an issue in an existing codebase")
    fix_parser.add_argument(
        "--repo", required=True, help="Path to repo or GitHub URL (https://github.com/user/repo)"
    )
    fix_parser.add_argument("--issue", required=True, help="Issue description (natural language)")
    fix_parser.add_argument("--patches", type=int, default=3, help="Candidate patches (default: 3)")
    fix_parser.add_argument("--commit", help="Specific commit to check out")

    # Jitterbug cycle
    jitter_parser = subparsers.add_parser(
        "jitterbug", help="Run one jitterbug compression-reconstruction cycle"
    )
    jitter_parser.add_argument(
        "--goals", type=int, default=5, help="Number of expansion builds (default: 5)"
    )
    jitter_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run expansion + compression only, don't build tools or integrate",
    )

    # Progression
    subparsers.add_parser("progression", help="Display current generative chain stage and metrics")

    # Session 14: domain manifold — knowledge-topology summary
    manifold_parser = subparsers.add_parser(
        "manifold",
        help="Show domain clusters, cross-domain links, and coverage gaps",
    )
    manifold_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the text renderer",
    )
    manifold_parser.add_argument(
        "--gap-threshold",
        type=int,
        default=5,
        help="Active-nutrient count below which a domain is flagged as a coverage gap (default: 5)",
    )

    # Dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Display metrics dashboard")
    dash_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # DSPy optimization
    opt_parser = subparsers.add_parser("optimize", help="Run DSPy/GEPA prompt optimization")
    opt_parser.add_argument(
        "agent",
        nargs="?",
        default=None,
        help="Agent to optimize (planner, architect, builder, tester, debugger)",
    )
    opt_parser.add_argument("--all", action="store_true", help="Optimize all 5 agents")
    opt_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be optimized without running"
    )

    # Backward compat: --goal without subcommand
    parser.add_argument("--goal", help="What to build (shortcut for: build --goal)")
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--deploy", choices=["railway", "docker_local"])
    parser.add_argument("--deploy-name")
    parser.add_argument(
        "--json-output", action="store_true", help="Print a JSON summary line at the end"
    )

    # Experiment A/B harness
    exp_parser = subparsers.add_parser(
        "experiment",
        help="Run controlled A/B experiments (engine vs raw model)",
    )
    exp_sub = exp_parser.add_subparsers(dest="exp_action")

    # experiment run
    exp_run = exp_sub.add_parser("run", help="Run an A/B experiment")
    exp_run.add_argument(
        "--challenges",
        type=int,
        default=5,
        help="Number of random challenges to test (default: 5)",
    )
    exp_run.add_argument(
        "--conditions",
        nargs="+",
        default=["engine_cloud", "engine_local", "raw_local"],
        choices=["engine_cloud", "engine_local", "raw_local"],
        help="Which conditions to run",
    )
    exp_run.add_argument(
        "--model", default="qwen2.5-coder:14b", help="Ollama model for local conditions"
    )
    exp_run.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Benchmark tiers to sample challenges from (default: 1 2)",
    )

    # experiment quick
    exp_quick = exp_sub.add_parser(
        "quick",
        help="Quick local-only comparison on N challenges (no cloud cost)",
    )
    exp_quick.add_argument("--n", type=int, default=5, help="Number of challenges (default: 5)")
    exp_quick.add_argument("--model", default="qwen2.5-coder:14b", help="Ollama model")
    exp_quick.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Benchmark tiers to sample from (default: 1 2)",
    )

    # experiment report
    exp_report = exp_sub.add_parser("report", help="Show results from an experiment")
    exp_report.add_argument(
        "--id", dest="experiment_id", default=None, help="Experiment ID (default: most recent)"
    )

    # experiment longitudinal
    exp_sub.add_parser(
        "longitudinal",
        help="Show how performance changes over time as soil grows",
    )

    # Session 4 (v3.2): synthesizer ablation harness
    exp_ablate = exp_sub.add_parser(
        "ablation-synth",
        help="Run synthesizer A/B ablation (builder_only vs builder_plus_synth vs router)",
    )
    exp_ablate.add_argument(
        "--n",
        type=int,
        default=3,
        help="Runs per (challenge, condition) (default: 3)",
    )
    exp_ablate.add_argument(
        "--challenges",
        nargs="*",
        help="Challenge IDs (default: 10 tier-1/2 challenges)",
    )
    exp_ablate.add_argument(
        "--report",
        action="store_true",
        help="Print summary of existing runs and exit (no new builds)",
    )

    # Session 3 (v3.2) follow-up: validator CLI
    validator_parser = subparsers.add_parser(
        "validator",
        help="Package validator utilities (hallucination blocklist, etc.)",
    )
    validator_sub = validator_parser.add_subparsers(dest="validator_action")
    validator_add = validator_sub.add_parser(
        "add-hallucination",
        help="Append a package name to the known-hallucination blocklist",
    )
    validator_add.add_argument("name", help="Package name (will be canonicalised)")

    # Session 6 (v3.2): agent archive inspection CLI
    archive_parser = subparsers.add_parser(
        "archive",
        help="Inspect the DGM-style agent archive of past build outcomes",
    )
    archive_sub = archive_parser.add_subparsers(dest="archive_action")
    archive_inspect = archive_sub.add_parser(
        "inspect",
        help="Show top-N highest-utility prior configurations for a goal",
    )
    archive_inspect.add_argument(
        "--goal",
        help="Goal to query the archive for (omit for global top-N)",
    )
    archive_inspect.add_argument("--top", type=int, default=5)

    # Session 7 (v3.2): tree-sitter + PageRank repo-map CLI
    repomap_parser = subparsers.add_parser(
        "repomap",
        help="Print a PageRank-ranked symbol map of the project",
    )
    repomap_parser.add_argument(
        "--root",
        default=".",
        help="Directory to walk (default: current dir)",
    )
    repomap_parser.add_argument(
        "--query",
        help="Boost symbols containing this identifier in the ranking",
    )
    repomap_parser.add_argument(
        "--top", type=int, default=2000, help="Max tokens to emit (default: 2000)"
    )

    # Session 8 (v3.2): covenant proposer review CLI
    cov_parser = subparsers.add_parser(
        "covenants",
        help="Review / approve / reject auto-proposed covenant rules",
    )
    cov_sub = cov_parser.add_subparsers(dest="covenants_action")
    cov_review = cov_sub.add_parser("review", help="List pending proposals")
    cov_review.add_argument(
        "--status",
        choices=["auto_pass", "auto_fail", "approved", "rejected", "all"],
        default="all",
    )
    cov_approve = cov_sub.add_parser(
        "approve", help="Promote a proposal to an auto-generated covenant"
    )
    cov_approve.add_argument("proposal_id")
    cov_reject = cov_sub.add_parser("reject", help="Mark a proposal as rejected")
    cov_reject.add_argument("proposal_id")
    cov_reject.add_argument("--reason", default="")
    cov_sub.add_parser("run-proposer", help="Run the proposer on the current archive")

    # Session 6: model-routing flags (set env before ModelRouter loads).
    # Applied to every subcommand; equivalent to exporting the env vars.
    parser.add_argument(
        "--mode",
        choices=["cloud", "hybrid", "local"],
        help="Backend routing: cloud (Anthropic only), hybrid (mechanical->local), local (all local)",
    )
    parser.add_argument(
        "--local-model",
        help="Ollama model name when --mode is hybrid/local (default: qwen2.5-coder:14b)",
    )
    parser.add_argument(
        "--ollama-url",
        help="Ollama base URL (default: http://localhost:11434)",
    )

    # `belief models` — print the active routing table
    subparsers.add_parser("models", help="Show active model routing table")

    # `belief benchmark-compare` — Session 7: cloud vs local side-by-side
    bc_parser = subparsers.add_parser(
        "benchmark-compare",
        help="Run benchmark in cloud and local modes; print a comparison table",
    )
    bc_parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Tiers to run (default: 1-3)",
    )
    bc_parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific challenge IDs",
    )

    # `belief library` — Session 12: list promoted library tools
    subparsers.add_parser(
        "library",
        help="List promoted library functions (apex predators)",
    )

    # `belief probe` — Session 10: confidence probe training + eval
    probe_parser = subparsers.add_parser("probe", help="Confidence probe commands")
    probe_sub = probe_parser.add_subparsers(dest="probe_action")
    pt = probe_sub.add_parser("train", help="Train the probe on collected traces")
    pt.add_argument(
        "--traces",
        default=None,
        help="Path to traces.db (default: ~/.belief-engine/traces.db)",
    )
    pt.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Refuse to train on fewer than N labeled step rows (default: 200)",
    )
    pt.add_argument(
        "--out",
        default=None,
        help="Where to save the trained probe (default: ~/.belief-engine/probe.pkl)",
    )
    pe = probe_sub.add_parser("test", help="Evaluate the trained probe on traces")
    pe.add_argument(
        "--traces",
        default=None,
        help="Path to traces.db to evaluate against",
    )
    pe.add_argument(
        "--probe",
        default=None,
        help="Path to a saved probe (default: ~/.belief-engine/probe.pkl)",
    )

    # `belief grinder` — Session 8: autonomous build loop
    grinder_parser = subparsers.add_parser("grinder", help="Grinder daemon commands")
    grinder_sub = grinder_parser.add_subparsers(dest="grinder_action")
    gs_start = grinder_sub.add_parser("start", help="Run the grinder in foreground")
    gs_start.add_argument(
        "--max-builds",
        type=int,
        default=None,
        help="Stop after N completed builds (default: run forever)",
    )
    gs_start.add_argument(
        "--pending-dir",
        default=None,
        help="Override pending_sessions directory",
    )
    grinder_sub.add_parser("status", help="Show the last-persisted status")
    grinder_sub.add_parser("pause", help="Pause the grinder (control table)")
    grinder_sub.add_parser("resume", help="Resume the grinder (control table)")

    # v3.3 Session 1 — Economist daily-budget contract.
    econ_parser = subparsers.add_parser(
        "economy",
        help="Show or reset the Economist's daily budget tracker (v3.3)",
    )
    econ_parser.add_argument(
        "--show",
        action="store_true",
        help="Print today's spend, remaining headroom, and storage paths",
    )
    econ_parser.add_argument(
        "--reset",
        action="store_true",
        help="Zero today's spend (audit history preserved)",
    )
    econ_parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Override the daily ceiling (USD). Default: $5.00",
    )

    # v3.3 Session 2 — Predator (utility-driven soft-tombstone of low-value soil).
    pred_parser = subparsers.add_parser(
        "predator",
        help="Soft-tombstone low-utility soil nutrients (v3.3)",
    )
    pred_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and report candidates but do not invalidate anything",
    )
    pred_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Utility threshold; nutrients below this are pruning candidates (default: 0.15)",
    )
    pred_parser.add_argument(
        "--min-age-days",
        type=int,
        default=None,
        help="Skip nutrients younger than this (default: 7)",
    )
    pred_parser.add_argument(
        "--max-delete",
        type=int,
        default=None,
        help="Soft cap on tombstones per run (default: 50; first-run hard cap: 10)",
    )
    pred_parser.add_argument(
        "--confirm-first-run",
        action="store_true",
        help="Bypass the 10-deletion first-run safety cap (use after a dry-run inspection)",
    )
    pred_parser.add_argument(
        "--collections",
        nargs="+",
        default=None,
        help="Restrict to nutrient types (default: pattern antipattern skeleton; never covenant)",
    )

    # v3.3 Session 3 — Sleep (offline consolidation: replay → crystallizer + FSRS housekeeping).
    sleep_parser = subparsers.add_parser(
        "sleep",
        help="Run offline soil consolidation cycles (v3.3)",
    )
    sleep_parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="How many replay → crystallize cycles to run (default: 3)",
    )
    sleep_parser.add_argument(
        "--max-minutes",
        type=int,
        default=None,
        help="Wall-clock cap for the whole run (default: 60)",
    )
    sleep_parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Local budget cap in USD; Economist daily ceiling still applies (default: 1.00)",
    )
    sleep_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run replay + scoring but do not promote covenants or refresh FSRS",
    )
    sleep_parser.add_argument(
        "--no-crystallize",
        action="store_true",
        help="Skip Phase A (episode replay → covenants); FSRS housekeeping only",
    )
    sleep_parser.add_argument(
        "--no-fsrs-recompute",
        action="store_true",
        help="Skip Phase B (FSRS schedule refresh)",
    )

    # v3.3 Session 4 — Garbage Collector (broken/invalid/duplicate cleanup).
    gc_parser = subparsers.add_parser(
        "gc",
        help="Soft-tombstone broken tools, invalid covenants, and duplicate tool sources (v3.3)",
    )
    gc_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify findings but do not invalidate anything",
    )
    gc_parser.add_argument(
        "--check-only",
        action="store_true",
        help='Same as --dry-run but logged distinctly in audit ("just looking")',
    )

    # v3.3 Session 5 — Curiosity (gap-driven goal proposal).
    cur_parser = subparsers.add_parser(
        "curiosity",
        help="Suggest build goals that fill gaps in the soil (v3.3)",
    )
    cur_parser.add_argument(
        "--suggest",
        type=int,
        default=None,
        help="How many candidate goals to surface (default: 5)",
    )
    cur_parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Local budget cap in USD (Economist daily ceiling still applies; default: 1.00)",
    )
    cur_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and rank but do not commit Economist or write state",
    )
    cur_parser.add_argument(
        "--auto-build",
        action="store_true",
        help="DEFERRED to v3.3 Session 5b (subprocess pipeline chain). Errors with diagnostic.",
    )

    # Synthesis Engine S2: word-set input adapter.
    # ``belief synth words "x,y"`` inserts synthetic raw_signals into the
    # photosynthesis pipeline so cross-domain word inputs flow through the
    # same cascade filter / novelty gate / ranker as harvested signals.
    synth_parser = subparsers.add_parser(
        "synth",
        help="Synthesis Engine commands (cross-domain word-set input)",
    )
    synth_sub = synth_parser.add_subparsers(dest="synth_action")
    synth_words = synth_sub.add_parser(
        "words",
        help='Submit a comma-separated word set, e.g. "mantis_shrimp,camera"',
    )
    synth_words.add_argument(
        "words",
        type=str,
        help='Comma-separated concept words (e.g. "mantis_shrimp,camera")',
    )
    synth_words.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override the photosynthesis signals.sqlite path (defaults to PhotoConfig)",
    )
    synth_words.add_argument(
        "--no-cycle",
        action="store_true",
        help="Insert signals only -- skip the synthesis cycle (useful for hermetic testing)",
    )
    synth_words.add_argument(
        "--bundle-id",
        type=str,
        default=None,
        help="Override the auto-generated bundle id (default: random 12-char hex)",
    )

    args = parser.parse_args()

    # Session 6: apply routing CLI flags to env so ModelRouter picks them up
    if getattr(args, "mode", None):
        os.environ["BELIEF_MODEL_MODE"] = args.mode
    if getattr(args, "local_model", None):
        os.environ["BELIEF_LOCAL_MODEL"] = args.local_model
    if getattr(args, "ollama_url", None):
        os.environ["BELIEF_OLLAMA_URL"] = args.ollama_url

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
    elif args.command == "dashboard":
        _run_dashboard_cmd(args)
        sys.exit(0)
    elif args.command == "optimize":
        _run_optimize_cmd(args)
        sys.exit(0)
    elif args.command == "jitterbug":
        asyncio.run(_run_jitterbug_cmd(args))
        sys.exit(0)
    elif args.command == "progression":
        _run_progression_cmd()
        sys.exit(0)
    elif args.command == "manifold":
        _run_manifold_cmd(args)
        sys.exit(0)
    elif args.command == "models":
        _run_models_cmd()
        sys.exit(0)
    elif args.command == "benchmark-compare":
        asyncio.run(_run_benchmark_compare_cmd(args))
        sys.exit(0)
    elif args.command == "grinder":
        _run_grinder_cmd(args)
        sys.exit(0)
    elif args.command == "probe":
        _run_probe_cmd(args)
        sys.exit(0)
    elif args.command == "library":
        _run_library_cmd(args)
        sys.exit(0)
    elif args.command == "experiment":
        _run_experiment_cmd(args)
        sys.exit(0)
    elif args.command == "validator":
        _run_validator_cmd(args)
        sys.exit(0)
    elif args.command == "archive":
        _run_archive_cmd(args)
        sys.exit(0)
    elif args.command == "repomap":
        from belief.repomap import RepoMap

        mentioned_idents = [args.query] if getattr(args, "query", None) else []
        rm = RepoMap(root=args.root)
        print(
            rm.get_ranked_tags_map(
                mentioned_idents=mentioned_idents,
                max_tokens=args.top,
            )
        )
        sys.exit(0)
    elif args.command == "economy":
        from belief.ecology.economist import (
            DEFAULT_DAILY_BUDGET_USD,
            cli_reset,
            cli_show,
        )

        budget = args.budget if args.budget is not None else DEFAULT_DAILY_BUDGET_USD
        if args.reset:
            print(cli_reset(daily_budget_usd=budget))
        else:
            # --show is the default when no flag is given.
            print(cli_show(daily_budget_usd=budget))
        sys.exit(0)
    elif args.command == "predator":
        from belief.ecology.predator import (
            DEFAULT_MAX_DELETE_PER_RUN,
            DEFAULT_MIN_AGE_DAYS,
            DEFAULT_NUTRIENT_TYPES,
            DEFAULT_UTILITY_THRESHOLD,
            PredatorConfig,
            cli_format_result,
            run as run_predator_cli,
        )

        cfg = PredatorConfig(
            collections=tuple(args.collections) if args.collections else DEFAULT_NUTRIENT_TYPES,
            utility_threshold=(
                args.threshold if args.threshold is not None else DEFAULT_UTILITY_THRESHOLD
            ),
            min_age_days=(
                args.min_age_days if args.min_age_days is not None else DEFAULT_MIN_AGE_DAYS
            ),
            max_delete_per_run=(
                args.max_delete if args.max_delete is not None else DEFAULT_MAX_DELETE_PER_RUN
            ),
            dry_run=bool(args.dry_run),
            confirm_first_run=bool(args.confirm_first_run),
        )
        result = asyncio.run(run_predator_cli(cfg))
        print(cli_format_result(result))
        sys.exit(0)
    elif args.command == "sleep":
        from belief.ecology.sleep import (
            DEFAULT_BUDGET_USD,
            DEFAULT_CYCLES,
            DEFAULT_MAX_MINUTES,
            SleepConfig,
            cli_format_result as sleep_format,
            run as run_sleep_cli,
        )

        cfg = SleepConfig(
            cycles=args.cycles if args.cycles is not None else DEFAULT_CYCLES,
            max_minutes=(args.max_minutes if args.max_minutes is not None else DEFAULT_MAX_MINUTES),
            budget_usd=args.budget if args.budget is not None else DEFAULT_BUDGET_USD,
            crystallize=not args.no_crystallize,
            recompute_fsrs=not args.no_fsrs_recompute,
            dry_run=bool(args.dry_run),
        )
        result = asyncio.run(run_sleep_cli(cfg))
        print(sleep_format(result))
        sys.exit(0)
    elif args.command == "gc":
        from belief.ecology.garbage_collector import (
            cli_format_result as gc_format,
            run as run_gc_cli,
        )

        result = asyncio.run(
            run_gc_cli(check_only=bool(args.check_only), dry_run=bool(args.dry_run))
        )
        print(gc_format(result))
        sys.exit(0)
    elif args.command == "curiosity":
        if args.auto_build:
            print(
                "belief curiosity --auto-build is deferred to v3.3 Session 5b "
                "(subprocess pipeline chain not yet implemented).\n"
                "Use --suggest N to see candidate goals, then run the chosen goal "
                "manually via 'belief --goal \"...\"'."
            )
            sys.exit(2)

        from belief.ecology.curiosity import (
            DEFAULT_BUDGET_USD as CURIOSITY_BUDGET_USD,
            DEFAULT_SUGGEST_N,
            cli_format_result as cur_format,
            suggest as run_curiosity_cli,
        )

        result = asyncio.run(
            run_curiosity_cli(
                n=args.suggest if args.suggest is not None else DEFAULT_SUGGEST_N,
                budget_usd=args.budget if args.budget is not None else CURIOSITY_BUDGET_USD,
                dry_run=bool(args.dry_run),
            )
        )
        print(cur_format(result))
        sys.exit(0)
    elif args.command == "synth":
        _run_synth_cmd(args)
        sys.exit(0)
    elif args.command == "covenants":
        from belief.covenants.review_cli import cmd_approve, cmd_reject, cmd_review

        action = getattr(args, "covenants_action", None)
        if action == "review":
            cmd_review(status_filter=args.status)
        elif action == "approve":
            cmd_approve(args.proposal_id)
        elif action == "reject":
            cmd_reject(args.proposal_id, reason=args.reason)
        elif action == "run-proposer":
            print("run-proposer: not yet wired — run on Mac via scripts/run_proposer.py")
        else:
            print("Usage: belief covenants {review|approve|reject|run-proposer}")
        sys.exit(0)
    elif args.command == "build" or args.goal:
        goal = getattr(args, "goal", None)
        if not goal:
            parser.error("--goal is required")
        json_out = getattr(args, "json_output", False)
        result = asyncio.run(run(goal, args.max_cost, args.max_iterations, json_output=json_out))

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
