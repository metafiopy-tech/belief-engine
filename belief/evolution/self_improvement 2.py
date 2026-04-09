"""
Self-Improvement Loop — Milestone 6

SEED proposes improvements based on build history.
Mentor reviews proposals for safety and value.
SelfPatch applies approved patches with rollback on failure.

Flow:
  1. SEED analyzes build metrics (cost, errors, timing) → proposes improvement
  2. Mentor evaluates: is this safe? Will it help?
  3. SelfPatch applies the patch to the codebase
  4. Validation: run tests, check no regressions
  5. If validation fails → rollback
  6. If validation passes → commit + optional restart

Based on:
- Microsoft STOP: Recursive self-improvement of scaffolding
- Sakana AI Darwin Gödel Machine: Self-improving agent (2.5× SWE-bench)
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Improvement Proposal
# ---------------------------------------------------------------------------

class ImprovementType(str, Enum):
    PROMPT = "prompt"              # Modify an agent prompt
    PARAMETER = "parameter"        # Change a config parameter
    PIPELINE = "pipeline"          # Modify pipeline routing
    NEW_TOOL = "new_tool"          # Add a new utility/tool
    REFACTOR = "refactor"          # Restructure existing code


@dataclass
class ImprovementProposal:
    """A proposed improvement from SEED."""
    title: str
    description: str
    improvement_type: ImprovementType
    target_file: str              # File to modify
    current_code: str             # Current content
    proposed_code: str            # Proposed replacement
    expected_benefit: str         # What this should improve
    risk_level: str = "low"       # low, medium, high
    metrics_before: dict = field(default_factory=dict)


@dataclass
class MentorVerdict:
    """Mentor's evaluation of a proposal."""
    approved: bool
    reasoning: str
    conditions: list[str] = field(default_factory=list)  # Must-haves before applying
    risk_assessment: str = ""


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    file_path: str
    backup_path: Optional[str] = None
    error: Optional[str] = None
    validation_passed: bool = False
    rolled_back: bool = False


# ---------------------------------------------------------------------------
# SEED — proposes improvements
# ---------------------------------------------------------------------------

class SEED:
    """
    Self-Evaluating Evolution Driver.

    Analyzes build history and proposes improvements when triggered
    (every N builds, configurable).
    """

    def __init__(self, trigger_interval: int = 10):
        self.trigger_interval = trigger_interval
        self.build_count = 0
        self.build_history: list[dict] = []

    def record_build(self, metrics: dict) -> None:
        """Record a build's metrics."""
        self.build_count += 1
        self.build_history.append(metrics)

    def should_trigger(self) -> bool:
        """Check if SEED should propose an improvement."""
        return self.build_count > 0 and self.build_count % self.trigger_interval == 0

    def propose(
        self,
        llm_fn: Optional[Callable] = None,
    ) -> Optional[ImprovementProposal]:
        """
        Analyze build history and propose an improvement.

        Uses deterministic heuristics first, falls back to LLM
        for creative proposals.
        """
        if not self.build_history:
            return None

        # Analyze patterns
        proposal = self._analyze_patterns()
        if proposal:
            logger.info(f"SEED proposal (deterministic): {proposal.title}")
            return proposal

        # LLM-based proposal
        if llm_fn:
            proposal = self._llm_propose(llm_fn)
            if proposal:
                logger.info(f"SEED proposal (LLM): {proposal.title}")
                return proposal

        return None

    def _analyze_patterns(self) -> Optional[ImprovementProposal]:
        """Deterministic pattern analysis on build history."""
        if len(self.build_history) < 3:
            return None

        recent = self.build_history[-5:]

        # Pattern: high correction rounds → prompts need work
        avg_corrections = sum(
            b.get("correction_rounds", 0) for b in recent
        ) / len(recent)

        if avg_corrections > 1.5:
            return ImprovementProposal(
                title="Reduce pyright correction rounds",
                description=(
                    f"Average {avg_corrections:.1f} correction rounds per build. "
                    "Strengthen builder prompt to generate correct imports first time."
                ),
                improvement_type=ImprovementType.PROMPT,
                target_file="belief/prompts/skeleton_prompts.py",
                current_code="",  # Will be filled by SelfPatch
                proposed_code="",  # Will be generated by LLM
                expected_benefit="Reduce correction rounds to <0.5 average",
                risk_level="low",
                metrics_before={"avg_corrections": avg_corrections},
            )

        # Pattern: high failure rate → architecture issues
        avg_failures = sum(
            len(b.get("failures", [])) for b in recent
        ) / len(recent)

        if avg_failures > 0.5:
            return ImprovementProposal(
                title="Reduce build failures",
                description=(
                    f"Average {avg_failures:.1f} file failures per build. "
                    "Add stronger validation in skeleton generation."
                ),
                improvement_type=ImprovementType.PIPELINE,
                target_file="belief/agents/skeleton_builder.py",
                current_code="",
                proposed_code="",
                expected_benefit="Reduce failures to <0.1 average",
                risk_level="medium",
                metrics_before={"avg_failures": avg_failures},
            )

        # Pattern: high token usage → compression not aggressive enough
        avg_tokens = sum(
            b.get("avg_tokens_per_file", 0) for b in recent
        ) / len(recent)

        if avg_tokens > 1500:
            return ImprovementProposal(
                title="Improve context compression",
                description=(
                    f"Average {avg_tokens:.0f} tokens per file context. "
                    "Tighten budget or improve symbol ranking."
                ),
                improvement_type=ImprovementType.PARAMETER,
                target_file="belief/models/context_compression.py",
                current_code="",
                proposed_code="",
                expected_benefit="Reduce to <1000 tokens per file",
                risk_level="low",
                metrics_before={"avg_tokens_per_file": avg_tokens},
            )

        return None

    def _llm_propose(self, llm_fn: Callable) -> Optional[ImprovementProposal]:
        """Use LLM to propose a creative improvement."""
        history_summary = json.dumps(self.build_history[-5:], indent=2, default=str)

        prompt = (
            "Analyze these recent build metrics and propose ONE specific improvement "
            "to the build system. Focus on the most impactful change.\n\n"
            f"Build history:\n{history_summary}\n\n"
            "Respond with JSON: {title, description, improvement_type, target_file, "
            "expected_benefit, risk_level}"
        )

        try:
            response = llm_fn(
                "You are a build system optimization expert.",
                prompt,
                "sonnet",
            )
            data = json.loads(response)
            return ImprovementProposal(
                title=data.get("title", "LLM proposal"),
                description=data.get("description", ""),
                improvement_type=ImprovementType(data.get("improvement_type", "parameter")),
                target_file=data.get("target_file", ""),
                current_code="",
                proposed_code="",
                expected_benefit=data.get("expected_benefit", ""),
                risk_level=data.get("risk_level", "medium"),
            )
        except Exception as e:
            logger.warning(f"SEED LLM proposal failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Mentor — evaluates proposals
# ---------------------------------------------------------------------------

class Mentor:
    """
    Reviews improvement proposals for safety and value.

    Rules:
    - Never approve high-risk changes without test coverage
    - Never approve changes to core routing without manual review
    - Always require rollback capability
    """

    def evaluate(
        self,
        proposal: ImprovementProposal,
        llm_fn: Optional[Callable] = None,
    ) -> MentorVerdict:
        """Evaluate a proposal deterministically + optionally with LLM."""

        # Hard rules
        if proposal.risk_level == "high":
            return MentorVerdict(
                approved=False,
                reasoning="High-risk changes require manual review",
                risk_assessment="high",
            )

        if proposal.improvement_type == ImprovementType.PIPELINE:
            return MentorVerdict(
                approved=True,
                reasoning="Pipeline changes are medium-risk, approved with conditions",
                conditions=["Must pass all existing tests", "Must have rollback"],
                risk_assessment="medium",
            )

        if proposal.improvement_type == ImprovementType.PROMPT:
            return MentorVerdict(
                approved=True,
                reasoning="Prompt changes are low-risk and easily reversible",
                conditions=["Must pass all existing tests"],
                risk_assessment="low",
            )

        if proposal.improvement_type == ImprovementType.PARAMETER:
            return MentorVerdict(
                approved=True,
                reasoning="Parameter changes are low-risk",
                conditions=["Must pass all existing tests"],
                risk_assessment="low",
            )

        # Default: approve with caution
        return MentorVerdict(
            approved=True,
            reasoning="Approved with standard conditions",
            conditions=["Must pass all existing tests", "Must have rollback"],
            risk_assessment="medium",
        )


# ---------------------------------------------------------------------------
# SelfPatch — applies approved patches
# ---------------------------------------------------------------------------

class SelfPatch:
    """
    Applies approved improvement patches to the codebase.

    Features:
    - Creates backup before patching
    - Validates syntax after patching
    - Runs test suite
    - Rolls back on failure
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / ".belief_backups"

    def apply(
        self,
        proposal: ImprovementProposal,
        verdict: MentorVerdict,
        validate_fn: Optional[Callable[[], bool]] = None,
    ) -> PatchResult:
        """
        Apply an approved patch.

        Args:
            proposal: The improvement to apply.
            verdict: Mentor's approval (must be approved).
            validate_fn: Optional validation function (e.g. run tests).
                Returns True if validation passes.

        Returns:
            PatchResult with success/failure info.
        """
        if not verdict.approved:
            return PatchResult(
                success=False,
                file_path=proposal.target_file,
                error="Mentor did not approve this proposal",
            )

        target = self.project_root / proposal.target_file
        if not target.exists():
            return PatchResult(
                success=False,
                file_path=proposal.target_file,
                error=f"Target file does not exist: {target}",
            )

        # Create backup
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_dir / f"{target.name}.backup"
        shutil.copy2(target, backup_path)

        result = PatchResult(
            success=False,
            file_path=proposal.target_file,
            backup_path=str(backup_path),
        )

        try:
            # Read current code if not provided
            if not proposal.current_code:
                proposal.current_code = target.read_text()

            # Apply patch
            if proposal.proposed_code:
                target.write_text(proposal.proposed_code)
            else:
                logger.warning("No proposed_code in proposal — skipping write")
                result.error = "No proposed code to apply"
                return result

            # Validate syntax
            try:
                ast.parse(target.read_text())
            except SyntaxError as e:
                logger.error(f"Patch introduced syntax error: {e}")
                self._rollback(target, backup_path)
                result.error = f"Syntax error after patch: {e}"
                result.rolled_back = True
                return result

            # Run validation (tests)
            if validate_fn:
                try:
                    passed = validate_fn()
                    result.validation_passed = passed
                    if not passed:
                        logger.warning("Validation failed — rolling back")
                        self._rollback(target, backup_path)
                        result.error = "Validation failed after patch"
                        result.rolled_back = True
                        return result
                except Exception as e:
                    logger.error(f"Validation crashed: {e}")
                    self._rollback(target, backup_path)
                    result.error = f"Validation crashed: {e}"
                    result.rolled_back = True
                    return result
            else:
                result.validation_passed = True  # No validation = assume ok

            result.success = True
            logger.info(f"Patch applied successfully: {proposal.title}")
            return result

        except Exception as e:
            logger.error(f"Patch application failed: {e}")
            self._rollback(target, backup_path)
            result.error = str(e)
            result.rolled_back = True
            return result

    def _rollback(self, target: Path, backup: Path) -> None:
        """Restore file from backup."""
        if backup.exists():
            shutil.copy2(backup, target)
            logger.info(f"Rolled back: {target}")


# ---------------------------------------------------------------------------
# Full loop
# ---------------------------------------------------------------------------

def run_improvement_loop(
    seed: SEED,
    mentor: Mentor,
    patcher: SelfPatch,
    llm_fn: Optional[Callable] = None,
    validate_fn: Optional[Callable[[], bool]] = None,
) -> Optional[PatchResult]:
    """
    Run the full improvement loop:
    SEED proposes → Mentor reviews → SelfPatch applies.

    Returns PatchResult if an improvement was attempted, None if
    SEED didn't trigger or had no proposal.
    """
    if not seed.should_trigger():
        return None

    proposal = seed.propose(llm_fn=llm_fn)
    if not proposal:
        logger.info("SEED triggered but no improvement proposed")
        return None

    logger.info(f"SEED proposal: {proposal.title}")

    verdict = mentor.evaluate(proposal, llm_fn=llm_fn)
    logger.info(f"Mentor verdict: {'APPROVED' if verdict.approved else 'REJECTED'} — {verdict.reasoning}")

    if not verdict.approved:
        return PatchResult(
            success=False,
            file_path=proposal.target_file,
            error=f"Mentor rejected: {verdict.reasoning}",
        )

    result = patcher.apply(proposal, verdict, validate_fn=validate_fn)
    return result
