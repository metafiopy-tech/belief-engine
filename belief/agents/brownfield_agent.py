"""Brownfield Agent — Modify Existing Codebases.

Orchestrates the full Tier 7 pipeline for brownfield code modification:
1. Ingest codebase (Codebase.from_directory)
2. Build repo graph (RepoGraph with BM25 + PageRank)
3. Localize fault (3-phase Agentless: file → symbol → line)
4. Self-play patch (Kimi-Dev: 3 patches × 3 tests, CodeT ranking)
5. Validate (run affected tests, check for regressions)
6. Escalate to agentic mode if Agentless fails after 3 iterations

Default: Agentless pipeline ($0.40-0.70/issue)
Escalation: Full agentic with tool access ($5 cap)

Usage:
    from belief.agents.brownfield_agent import fix_issue
    result = await fix_issue(
        repo_path="/path/to/repo",
        issue="Fix the login endpoint validation for nested nullable fields",
    )
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.agents.brownfield")


@dataclass
class BrownfieldResult:
    """Result of a brownfield modification attempt."""

    success: bool = False
    patch_file: str = ""
    patch_old: str = ""
    patch_new: str = ""
    patch_explanation: str = ""
    tests_passed: int = 0
    tests_total: int = 0
    regressions: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    method: str = "agentless"  # "agentless" or "agentic"
    iterations: int = 0
    error: str = ""


async def fix_issue(
    repo_path: str | Path,
    issue: str,
    max_agentless_iterations: int = 3,
    escalate_to_agentic: bool = True,
    agentic_budget_usd: float = 5.0,
    n_patches: int = 3,
    n_tests: int = 3,
) -> BrownfieldResult:
    """Fix an issue in an existing codebase.

    The pipeline:
    1. Ingest the codebase
    2. Localize the fault (BM25 + PageRank + LLM narrowing)
    3. Self-play: generate patches + reproduction tests, CodeT rank
    4. Validate: run affected tests
    5. If failed and escalate=True: switch to agentic mode

    Args:
        repo_path: Path to the git repository
        issue: Natural language description of the issue/bug/feature
        max_agentless_iterations: Max localize→patch→validate cycles
        escalate_to_agentic: Whether to escalate on Agentless failure
        agentic_budget_usd: Max cost for agentic mode
        n_patches: Number of candidate patches (Kimi-Dev)
        n_tests: Number of reproduction tests (Kimi-Dev)

    Returns:
        BrownfieldResult with the patch and validation results
    """
    from belief.codebase import Codebase
    from belief.codebase.localizer import HierarchicalLocalizer
    from belief.codebase.patch_sampler import PatchSampler
    from belief.config.models import ModelRouter
    from belief.llm import LLMClient

    t0 = time.time()
    result = BrownfieldResult()

    # Step 1: Ingest
    repo_path = Path(repo_path).resolve()
    if not repo_path.is_dir():
        result.error = f"Not a directory: {repo_path}"
        return result

    logger.info(f"Brownfield: ingesting {repo_path}")
    codebase = Codebase.from_directory(repo_path)
    logger.info(f"Brownfield: {codebase.summary()}")

    router = ModelRouter()
    llm = LLMClient(router)

    try:
        # Step 2-4: Agentless iterations
        localizer = HierarchicalLocalizer()
        sampler = PatchSampler()

        for iteration in range(max_agentless_iterations):
            result.iterations = iteration + 1
            logger.info(
                f"Brownfield: Agentless iteration {iteration + 1}/{max_agentless_iterations}"
            )

            # Step 2: Localize
            locations = await localizer.localize(codebase, issue, max_locations=3, llm=llm)
            if not locations:
                logger.warning("Brownfield: localization found no edit locations")
                continue

            # Step 3: Self-play patch + rank
            best_result = None
            for loc in locations:
                sample_result = await sampler.sample_and_rank(
                    codebase,
                    loc,
                    issue,
                    llm,
                    n_patches=n_patches,
                    n_tests=n_tests,
                )
                if sample_result.best_patch and sample_result.best_patch.syntax_valid:
                    if not best_result or len(sample_result.best_patch.tests_passed) > len(
                        best_result.best_patch.tests_passed
                    ):
                        best_result = sample_result

            if not best_result or not best_result.best_patch:
                logger.warning(f"Brownfield: no valid patch in iteration {iteration + 1}")
                continue

            patch = best_result.best_patch

            # Step 4: Validate — apply patch and run affected tests
            validation = await _validate_patch(codebase, patch, issue)

            result.patch_file = patch.file_path
            result.patch_old = patch.old_code
            result.patch_new = patch.new_code
            result.patch_explanation = patch.explanation
            result.tests_passed = validation.get("passed", 0)
            result.tests_total = validation.get("total", 0)
            result.regressions = validation.get("regressions", 0)

            if validation.get("success"):
                result.success = True
                result.method = "agentless"
                logger.info(
                    f"Brownfield: FIXED in iteration {iteration + 1} — "
                    f"{result.tests_passed}/{result.tests_total} tests, "
                    f"{result.regressions} regressions"
                )
                break

            logger.info(
                f"Brownfield: iteration {iteration + 1} patch failed validation "
                f"({result.tests_passed}/{result.tests_total} tests)"
            )

        # Step 5: Escalation (if Agentless failed).  Session 8.5b:
        # agentic mode is deferred work.  Gate it behind the explicit
        # env var BELIEF_BROWNFIELD_AGENTIC so we never silently enter
        # a half-implemented code path in production.  Default: off.
        # When on, the current implementation still just logs + marks
        # skipped — the env-var gate's job is to make "not-implemented"
        # loud rather than accidentally-reachable.
        if not result.success and escalate_to_agentic:
            agentic_enabled = os.environ.get("BELIEF_BROWNFIELD_AGENTIC", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not agentic_enabled:
                logger.info(
                    "Brownfield: Agentless exhausted; agentic escalation disabled "
                    "(set BELIEF_BROWNFIELD_AGENTIC=1 to opt in once implemented)"
                )
                result.method = "agentic_disabled"
                result.error = "Agentic mode gated by BELIEF_BROWNFIELD_AGENTIC env var; not set"
            else:
                logger.info("Brownfield: Agentless exhausted — escalating to agentic mode")
                # Agentic mode implementation is still pending (tracked
                # as a follow-up, not a silent stub — the env-var
                # contract above makes that visible to the operator).
                result.method = "agentic_unimplemented"
                result.error = (
                    "BELIEF_BROWNFIELD_AGENTIC=1 requested but agentic mode "
                    "is not yet implemented in this version."
                )

    except Exception as e:
        result.error = str(e)
        logger.warning(f"Brownfield failed: {e}")

    finally:
        await llm.close()
        result.duration_seconds = time.time() - t0

    return result


async def _validate_patch(
    codebase,
    patch,
    issue: str,
) -> dict[str, Any]:
    """Apply patch to a temp copy and run affected tests."""
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from belief.codebase.change_impact import select_affected_tests

    current_code = codebase.get_file_content(patch.file_path)
    if not current_code or patch.old_code not in current_code:
        return {
            "success": False,
            "error": "Patch old_code not found in file",
            "passed": 0,
            "total": 0,
            "regressions": 0,
        }

    patched_code = current_code.replace(patch.old_code, patch.new_code, 1)

    # Select tests to run
    affected_tests = select_affected_tests(codebase, [patch.file_path])

    with tempfile.TemporaryDirectory(prefix="belief_brownfield_") as tmp:
        tmp_path = Path(tmp)

        # Copy the entire codebase
        for fpath in codebase.files:
            content = codebase.get_file_content(fpath)
            if content:
                out = tmp_path / fpath
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content)

        # Apply the patch
        patched_path = tmp_path / patch.file_path
        patched_path.write_text(patched_code)

        # Run affected tests
        if not affected_tests:
            return {"success": True, "passed": 0, "total": 0, "regressions": 0}

        test_args = [str(tmp_path / t) for t in affected_tests if (tmp_path / t).exists()]
        if not test_args:
            return {"success": True, "passed": 0, "total": 0, "regressions": 0}

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", "--tb=short", "-q"] + test_args,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(tmp_path),
                env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)},
            )

            import re

            passed = 0
            failed = 0
            match = re.search(r"(\d+) passed", proc.stdout)
            if match:
                passed = int(match.group(1))
            match = re.search(r"(\d+) failed", proc.stdout)
            if match:
                failed = int(match.group(1))

            return {
                "success": failed == 0 and passed > 0,
                "passed": passed,
                "total": passed + failed,
                "regressions": failed,
                "output": proc.stdout[-2000:],
            }

        except Exception as e:
            return {"success": False, "error": str(e), "passed": 0, "total": 0, "regressions": 0}
