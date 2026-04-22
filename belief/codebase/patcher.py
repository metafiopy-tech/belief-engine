"""Patcher — generates search/replace edits for existing codebases.

Tier 7's equivalent of the builder. Instead of generating files from scratch,
the patcher takes:
  1. A codebase (indexed via Codebase.from_directory)
  2. An issue/feature description
  3. Localized files and functions (from Agentless localization)

And produces search/replace edits that implement the change.

This is the same search/replace pattern used by the debugger and
the water cycle fixer, extended to handle feature additions and
multi-file changes.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger("belief.codebase.patcher")


@dataclass
class PatchEdit:
    """A single search/replace edit."""
    file_path: str
    old_str: str
    new_str: str
    explanation: str
    validated: bool = False


@dataclass
class PatchPlan:
    """A plan for modifying an existing codebase."""
    description: str
    edits: list[PatchEdit] = field(default_factory=list)
    new_files: dict[str, str] = field(default_factory=dict)  # path → content
    affected_tests: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.edits) + len(self.new_files)


class _PatchResponse(BaseModel):
    """LLM response for a single file patch."""
    edits: list[dict] = Field(default_factory=list)
    new_files: list[dict] = Field(default_factory=list)
    explanation: str = ""


PATCHER_SYSTEM = """You are a code patcher for an existing codebase. You make MINIMAL,
SURGICAL edits to implement a feature or fix a bug.

RULES:
1. Use search/replace blocks — never regenerate entire files
2. The old_str must match the existing code EXACTLY (including whitespace and indentation)
3. Make the SMALLEST change that correctly implements the requirement
4. Preserve existing code style, naming conventions, and patterns
5. Add imports at the top of the file if needed (as a separate edit)
6. If a completely new file is needed, provide it in new_files

Respond ONLY with valid JSON:
{
    "edits": [
        {"file": "path/to/file.py", "old_str": "exact existing code", "new_str": "replacement code", "explanation": "why"}
    ],
    "new_files": [
        {"file": "path/to/new_file.py", "content": "full file content", "explanation": "why"}
    ],
    "explanation": "overall change summary"
}"""


PATCHER_PROMPT = """Implement this change in the existing codebase:

## Change Request
{issue}

## Relevant Files
{relevant_files}

## Repository Map (other files in the project)
{repo_map}

## Affected Tests (run these after patching)
{affected_tests}

Generate search/replace edits that implement the change.
Each edit must have old_str that EXACTLY matches existing code."""


async def generate_patch(
    issue: str,
    codebase,
    relevant_files: list[str],
    llm=None,
) -> PatchPlan:
    """Generate a patch plan for an issue against an existing codebase.

    Args:
        issue: Description of the feature/bug to implement/fix
        codebase: Codebase object with indexed files
        relevant_files: File paths identified by localization
        llm: LLMClient instance (created if None)
    """
    from belief.config import ModelRouter
    from belief.llm import LLMClient

    if llm is None:
        router = ModelRouter()
        llm = LLMClient(router)

    # Build context: full content of relevant files
    file_context_parts = []
    for fpath in relevant_files:
        content = codebase.get_file_content(fpath)
        if content:
            file_context_parts.append(f"### {fpath}\n```\n{content}\n```")

    # Get repo map for broader context
    repo_map = codebase.generate_repo_map(max_tokens=1500)

    # Find affected tests
    affected = set()
    for fpath in relevant_files:
        affected.update(codebase.get_affected_tests(fpath))
    affected_tests = "\n".join(f"  - {t}" for t in sorted(affected)) or "  (none found)"

    prompt = PATCHER_PROMPT.format(
        issue=issue,
        relevant_files="\n\n".join(file_context_parts),
        repo_map=repo_map,
        affected_tests=affected_tests,
    )

    try:
        result = await llm.generate_structured(
            role="builder",
            system=PATCHER_SYSTEM,
            prompt=prompt,
            response_schema=_PatchResponse,
            temperature=0.2,
            max_tokens=4000,
        )

        plan = PatchPlan(description=result.explanation)

        # Process edits
        for edit_dict in result.edits:
            fpath = edit_dict.get("file", "")
            old_str = edit_dict.get("old_str", "")
            new_str = edit_dict.get("new_str", "")
            explanation = edit_dict.get("explanation", "")

            if not fpath or not old_str:
                continue

            # Validate old_str exists in the file
            content = codebase.get_file_content(fpath)
            if old_str not in content:
                # Try stripped match
                old_stripped = old_str.strip()
                if old_stripped in content:
                    old_str = old_stripped
                else:
                    logger.warning(f"Patcher: old_str not found in {fpath}, skipping edit")
                    continue

            # Validate the edit produces valid code
            new_content = content.replace(old_str, new_str, 1)
            validated = True
            if fpath.endswith(".py"):
                try:
                    ast.parse(new_content)
                except SyntaxError:
                    logger.warning(f"Patcher: edit produces invalid Python in {fpath}")
                    validated = False

            plan.edits.append(PatchEdit(
                file_path=fpath,
                old_str=old_str,
                new_str=new_str,
                explanation=explanation,
                validated=validated,
            ))

        # Process new files
        for new_dict in result.new_files:
            fpath = new_dict.get("file", "")
            content = new_dict.get("content", "")
            if fpath and content:
                plan.new_files[fpath] = content

        # Track affected tests
        plan.affected_tests = sorted(affected)

        logger.info(
            f"Patcher: {len(plan.edits)} edits, {len(plan.new_files)} new files, "
            f"{len(plan.affected_tests)} affected tests"
        )

        return plan

    except Exception as e:
        logger.warning(f"Patcher failed: {e}")
        return PatchPlan(description=f"Patch generation failed: {e}")

    finally:
        await llm.close()


def apply_patch(codebase, plan: PatchPlan) -> dict[str, str]:
    """Apply a patch plan to a codebase, returning modified files.

    Returns dict of filepath → new content for all modified and new files.
    Only applies validated edits.
    """
    modified = {}

    for edit in plan.edits:
        if not edit.validated:
            logger.warning(f"Skipping unvalidated edit on {edit.file_path}")
            continue

        # Get current content (may have been modified by a previous edit)
        if edit.file_path in modified:
            content = modified[edit.file_path]
        else:
            content = codebase.get_file_content(edit.file_path)

        if edit.old_str in content:
            modified[edit.file_path] = content.replace(edit.old_str, edit.new_str, 1)
            logger.info(f"Applied edit to {edit.file_path}: {edit.explanation[:60]}")
        else:
            logger.warning(f"Could not apply edit to {edit.file_path}: old_str not found")

    # Add new files
    for fpath, content in plan.new_files.items():
        modified[fpath] = content
        logger.info(f"Created new file: {fpath}")

    return modified
