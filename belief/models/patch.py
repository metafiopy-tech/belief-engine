"""Patch Models — Unified Diff, Candidates, and Ranking.

Data models for Tier 7 brownfield patching:
- UnifiedDiff: represents a patch in unified diff format
- PatchCandidate: a candidate patch with validation results
- PatchRanking: CodeT-style ranking of multiple candidates
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PatchStatus(str, Enum):
    """Status of a patch candidate through the validation pipeline."""
    GENERATED = "generated"
    SYNTAX_VALID = "syntax_valid"
    SYNTAX_INVALID = "syntax_invalid"
    TESTS_PASSED = "tests_passed"
    TESTS_FAILED = "tests_failed"
    REGRESSION = "regression"
    SELECTED = "selected"
    REJECTED = "rejected"


class DiffHunk(BaseModel):
    """A single hunk in a unified diff."""
    old_start: int = Field(description="Starting line in the original file")
    old_count: int = Field(description="Number of lines in the original")
    new_start: int = Field(description="Starting line in the new file")
    new_count: int = Field(description="Number of lines in the new file")
    content: str = Field(default="", description="The hunk content with +/- prefixes")


class UnifiedDiff(BaseModel):
    """A patch in unified diff format."""
    file_path: str = Field(description="Path to the file being patched")
    hunks: list[DiffHunk] = Field(default_factory=list)
    old_content: str = Field(default="", description="Original file content")
    new_content: str = Field(default="", description="Patched file content")

    @property
    def is_valid(self) -> bool:
        """Check if the diff produces valid Python."""
        if not self.file_path.endswith(".py"):
            return True
        try:
            import ast
            ast.parse(self.new_content)
            return True
        except SyntaxError:
            return False

    def to_unified_format(self) -> str:
        """Render as a standard unified diff string."""
        lines = [
            f"--- a/{self.file_path}",
            f"+++ b/{self.file_path}",
        ]
        for hunk in self.hunks:
            lines.append(f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@")
            lines.append(hunk.content)
        return "\n".join(lines)

    @classmethod
    def from_search_replace(
        cls, file_path: str, old_content: str, old_str: str, new_str: str
    ) -> UnifiedDiff:
        """Create a UnifiedDiff from a search/replace operation."""
        new_content = old_content.replace(old_str, new_str, 1)

        # Compute hunk
        old_lines = old_content.split("\n")
        new_lines = new_content.split("\n")

        # Find the first and last differing lines
        start = 0
        while start < min(len(old_lines), len(new_lines)) and old_lines[start] == new_lines[start]:
            start += 1

        old_end = len(old_lines) - 1
        new_end = len(new_lines) - 1
        while old_end > start and new_end > start and old_lines[old_end] == new_lines[new_end]:
            old_end -= 1
            new_end -= 1

        # Build hunk content
        context_before = max(0, start - 3)
        context_after_old = min(len(old_lines), old_end + 4)
        context_after_new = min(len(new_lines), new_end + 4)

        hunk_lines = []
        for i in range(context_before, start):
            hunk_lines.append(f" {old_lines[i]}")
        for i in range(start, old_end + 1):
            hunk_lines.append(f"-{old_lines[i]}")
        for i in range(start, new_end + 1):
            hunk_lines.append(f"+{new_lines[i]}")
        for i in range(old_end + 1, context_after_old):
            hunk_lines.append(f" {old_lines[i]}")

        hunk = DiffHunk(
            old_start=context_before + 1,
            old_count=context_after_old - context_before,
            new_start=context_before + 1,
            new_count=context_after_new - context_before,
            content="\n".join(hunk_lines),
        )

        return cls(
            file_path=file_path,
            hunks=[hunk],
            old_content=old_content,
            new_content=new_content,
        )


class PatchCandidateModel(BaseModel):
    """A candidate patch with its validation state."""
    id: int = 0
    diff: UnifiedDiff = Field(default_factory=lambda: UnifiedDiff(file_path=""))
    explanation: str = ""
    status: PatchStatus = PatchStatus.GENERATED
    temperature: float = 0.0

    # Validation results
    syntax_valid: bool = False
    tests_passed: list[str] = Field(default_factory=list)
    tests_failed: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)

    # CodeT ranking
    agreement_signature: frozenset[str] = Field(default_factory=frozenset)
    codet_score: float = 0.0


class PatchRanking(BaseModel):
    """Result of CodeT dual execution ranking across candidates."""
    candidates: list[PatchCandidateModel] = Field(default_factory=list)
    selected_id: int = -1
    ranking_method: str = "codet"
    total_tests: int = 0

    @property
    def selected(self) -> Optional[PatchCandidateModel]:
        for c in self.candidates:
            if c.id == self.selected_id:
                return c
        return None

    @property
    def success_rate(self) -> float:
        sel = self.selected
        if not sel or not self.total_tests:
            return 0.0
        return len(sel.tests_passed) / self.total_tests
