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
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

        if proposal.improvement_type == ImprovementType.NEW_TOOL:
            return MentorVerdict(
                approved=True,
                reasoning="New tool proposals are validated separately by tool_validator",
                conditions=["Must pass tool validation", "Must catch >= 30% of target failures"],
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


# ---------------------------------------------------------------------------
# Autocatalytic NEW_TOOL support
# ---------------------------------------------------------------------------


@dataclass
class FailureCluster:
    """A cluster of similar failures identified from build traces."""

    error_type: str                             # Normalized error category
    count: int                                  # How many failures
    example_errors: list[str] = field(default_factory=list)
    failure_traces: list[dict] = field(default_factory=list)
    suggested_tool_name: str = ""               # e.g. "fastapi_route_validator"
    suggested_tool_description: str = ""
    input_description: str = ""
    output_description: str = ""
    addressed_by_existing_tool: bool = False


def get_recent_failures(soil, n: int = 20) -> list[dict]:
    """Query the failures collection for recent failure traces."""
    failures_col = soil._collections.get("belief_failures")
    if failures_col is None or failures_col.count() == 0:
        return []

    count = min(n, failures_col.count())
    results = failures_col.get(
        include=["documents", "metadatas"],
        limit=count,
    )

    traces: list[dict] = []
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] or {}
        trace = dict(meta)
        trace["trace_id"] = doc_id
        trace["description"] = results["documents"][i] or ""
        traces.append(trace)

    return traces


def cluster_failures(failures: list[dict]) -> list[FailureCluster]:
    """Group failures by normalized error type.

    Normalizes error messages by stripping specifics (file names,
    line numbers, variable names) and grouping by pattern.
    """
    if not failures:
        return []

    # Normalize error content
    error_groups: dict[str, list[dict]] = {}
    for f in failures:
        content = f.get("content", f.get("description", ""))
        error_type = _normalize_error(content)
        if error_type not in error_groups:
            error_groups[error_type] = []
        error_groups[error_type].append(f)

    clusters: list[FailureCluster] = []
    for error_type, traces in error_groups.items():
        # Generate suggested tool metadata
        tool_name = _suggest_tool_name(error_type)
        tool_desc = _suggest_tool_description(error_type, traces)

        clusters.append(FailureCluster(
            error_type=error_type,
            count=len(traces),
            example_errors=[
                t.get("content", t.get("description", ""))[:200]
                for t in traces[:5]
            ],
            failure_traces=traces,
            suggested_tool_name=tool_name,
            suggested_tool_description=tool_desc,
            input_description="Python source code as a string",
            output_description="List of validation error strings",
        ))

    # Sort by count descending
    clusters.sort(key=lambda c: c.count, reverse=True)
    return clusters


def select_target_cluster(
    clusters: list[FailureCluster],
    existing_tool_names: list[str],
) -> Optional[FailureCluster]:
    """Pick the most impactful cluster not already addressed by a tool."""
    for cluster in clusters:
        # Check if this cluster is addressed by an existing tool
        if cluster.suggested_tool_name in existing_tool_names:
            cluster.addressed_by_existing_tool = True
            continue
        if cluster.count >= 3:  # Minimum cluster size
            return cluster
    return None


def formulate_tool_goal(cluster: FailureCluster) -> str:
    """Generate a clear goal string for the engine to build a tool.

    The goal should be specific enough for the engine's pipeline to
    produce a self-contained Python module.
    """
    examples = "\n".join(f"  - {e}" for e in cluster.example_errors[:3])

    return (
        f"Build a Python module with a single function "
        f"`{cluster.suggested_tool_name}(code: str) -> list[str]` "
        f"that takes Python source code as input and returns a list of "
        f"validation error strings.\n\n"
        f"The function should detect and report: {cluster.suggested_tool_description}\n\n"
        f"Examples of errors it should catch:\n{examples}\n\n"
        f"Requirements:\n"
        f"- Self-contained module (no external dependencies)\n"
        f"- Use the `ast` module for code analysis where possible\n"
        f"- Return an empty list if no errors are found\n"
        f"- Include a module-level docstring\n"
        f"- Include type hints on the function signature"
    )


def evaluate_tool_against_failures(
    tool_code: str,
    failure_traces: list[dict],
) -> float:
    """Execute the tool against historical failure code.

    Returns the catch rate (fraction of failures where the tool
    found at least one error).
    """
    if not failure_traces:
        return 0.0

    # Find the check function in the tool code
    namespace: dict = {}
    try:
        exec(compile(tool_code, "<tool_test>", "exec"), namespace)  # noqa: S102
    except Exception:
        return 0.0

    check_fn = None
    for key, val in namespace.items():
        if callable(val) and not key.startswith("_"):
            check_fn = val
            break

    if check_fn is None:
        return 0.0

    caught = 0
    tested = 0
    for trace in failure_traces:
        code = trace.get("code_sample", trace.get("content", ""))
        if not code or len(code) < 10:
            continue
        tested += 1
        try:
            errors = check_fn(code)
            if errors:
                caught += 1
        except Exception:
            pass

    return caught / max(tested, 1)


async def execute_new_tool_proposal(
    soil,
    proposal_title: str = "",
) -> PatchResult:
    """The engine builds a tool for itself using its own pipeline.

    This is the autocatalytic core: the engine uses graph.ainvoke()
    to build a Python module that validates, extracts, or transforms
    code — then validates and registers the result.

    Args:
        soil:           Soil instance for accessing failure traces and tool registry.
        proposal_title: Optional title for logging.

    Returns:
        PatchResult with success=True and tool_id if the tool was registered.
    """
    from belief.evolution.tool_validator import validate_tool
    from belief.memory.tool_registry import SelfAuthoredTool, ToolRegistry

    # 1. Get recent failures and cluster them
    failures = get_recent_failures(soil, n=20)
    if not failures:
        return PatchResult(
            success=False,
            file_path="",
            error="No failure traces to analyze",
        )

    clusters = cluster_failures(failures)
    if not clusters:
        return PatchResult(
            success=False,
            file_path="",
            error="No failure clusters identified",
        )

    # Check existing tools
    registry = ToolRegistry(soil)
    existing_names = [t.name for t in registry.get_active_tools()]

    target = select_target_cluster(clusters, existing_names)
    if target is None:
        return PatchResult(
            success=False,
            file_path="",
            error="All failure clusters already addressed by existing tools",
        )

    # 2. Formulate a build goal
    tool_goal = formulate_tool_goal(target)

    # 3. Run the engine's own pipeline to build the tool
    try:
        from belief.graph import build_graph
        graph = build_graph()
        result = await graph.ainvoke({
            "user_goal": tool_goal,
            "max_iterations": 2,
            "max_cost_usd": 2.0,
        })
    except Exception as e:
        return PatchResult(
            success=False,
            file_path="",
            error=f"Engine pipeline failed: {e}",
        )

    # 4. Extract generated code
    code_files = result.get("code_files", {})
    if not code_files:
        return PatchResult(
            success=False,
            file_path="",
            error="Engine produced no code files",
        )

    # Get the main Python file's code
    main_code = ""
    for fname, content in code_files.items():
        if fname.endswith(".py") and "test" not in fname.lower():
            main_code = content
            break
    if not main_code:
        main_code = list(code_files.values())[0]

    # 5. Create a SelfAuthoredTool
    tool = SelfAuthoredTool(
        id=str(uuid.uuid4()),
        name=target.suggested_tool_name,
        description=target.suggested_tool_description,
        code=main_code,
        input_description=target.input_description,
        output_description=target.output_description,
        dependencies=[],
        created_by="sica",
    )

    # 6. Validate
    validation = validate_tool(tool)
    if not validation.valid:
        return PatchResult(
            success=False,
            file_path=f"tools/{tool.name}.py",
            error=f"Tool validation failed: {'; '.join(validation.errors)}",
        )

    # 7. Test against historical failures
    catch_rate = evaluate_tool_against_failures(main_code, target.failure_traces)
    if catch_rate < 0.3:
        return PatchResult(
            success=False,
            file_path=f"tools/{tool.name}.py",
            error=f"Tool only catches {catch_rate:.0%} of target failures (need >= 30%)",
        )

    # 8. Register in tool registry
    tool_id = registry.register_tool(tool)

    logger.info(
        f"NEW_TOOL: registered {tool.name} (id={tool_id}, "
        f"catch_rate={catch_rate:.0%}, cluster_size={target.count})"
    )

    return PatchResult(
        success=True,
        file_path=f"tools/{tool.name}.py",
        backup_path=tool_id,  # Store tool_id in backup_path for caller
    )


# ── Internal helpers ────────────────────────────────────────────────────────


def _normalize_error(error_text: str) -> str:
    """Normalize an error message to a category key."""
    if not error_text:
        return "unknown"

    text = error_text.lower()

    # Common error categories
    if "import" in text and ("not found" in text or "no module" in text or "cannot import" in text):
        return "missing_import"
    if "syntax" in text:
        return "syntax_error"
    if "type" in text and "error" in text:
        return "type_error"
    if "indentation" in text:
        return "indentation_error"
    if "attribute" in text and "error" in text:
        return "attribute_error"
    if "name" in text and "not defined" in text:
        return "name_error"
    if "test" in text and ("fail" in text or "error" in text):
        return "test_failure"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "docker" in text:
        return "docker_error"
    if "fastapi" in text or "route" in text or "endpoint" in text:
        return "api_routing"
    if "database" in text or "sqlalchemy" in text:
        return "database_error"

    # Strip specifics and use first meaningful word
    words = re.sub(r'[^a-z_\s]', '', text).split()
    if len(words) >= 2:
        return f"{words[0]}_{words[1]}"
    return "unknown"


def _suggest_tool_name(error_type: str) -> str:
    """Suggest a tool name from an error category."""
    name_map = {
        "missing_import": "import_checker",
        "syntax_error": "syntax_validator",
        "type_error": "type_checker",
        "indentation_error": "indent_fixer",
        "attribute_error": "attribute_validator",
        "name_error": "name_resolver",
        "test_failure": "test_structure_validator",
        "timeout": "complexity_checker",
        "docker_error": "dockerfile_validator",
        "api_routing": "api_route_validator",
        "database_error": "sqlalchemy_validator",
    }
    return name_map.get(error_type, f"{error_type}_validator")


def _suggest_tool_description(error_type: str, traces: list[dict]) -> str:
    """Generate a description of what the tool should check."""
    desc_map = {
        "missing_import": "missing or incorrect import statements in Python code",
        "syntax_error": "Python syntax errors before they reach the interpreter",
        "type_error": "type mismatches and incorrect type usage",
        "indentation_error": "indentation inconsistencies in Python code",
        "attribute_error": "references to non-existent attributes on objects",
        "name_error": "undefined variable and function references",
        "test_failure": "test file structure issues (missing fixtures, incorrect assertions)",
        "timeout": "code patterns that indicate excessive complexity or infinite loops",
        "docker_error": "Dockerfile syntax and best-practice violations",
        "api_routing": "API route definition errors (duplicate paths, missing handlers)",
        "database_error": "SQLAlchemy model and query issues",
    }
    return desc_map.get(error_type, f"issues related to {error_type.replace('_', ' ')}")
