"""Water Cycle Refinement Loop — targeted polishing of working code.

After the executor passes but the validator finds quality issues (fail_fixable),
the refinement loop runs up to 3 cycles of:
  1. analyze_failures — verbal self-reflection on test output (Reflexion pattern)
  2. generate_fix — search/replace edit on ONE file
  3. revalidate — re-run tests, check for progress or regression

The water never leaves the system. Code is polished, not rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CycleRecord:
    """Record of one refinement cycle."""

    cycle: int
    passed_count: int
    total_count: int
    failed_test_ids: list[str]
    file_modified: str
    diagnosis: str
    fix_summary: str
    regression: bool = False
    reflection: str = ""  # Reflexion: why did this fix work/fail?


@dataclass
class RefinementState:
    """State for the refinement subgraph."""

    code_files: dict[str, str]
    test_files: dict[str, str]
    test_output: str = ""
    cycle: int = 0
    max_cycles: int = 3
    test_history: list[CycleRecord] = field(default_factory=list)
    previous_fixes: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)  # Reflexion episodic memory
    best_snapshot: dict[str, str] = field(default_factory=dict)
    best_pass_count: int = 0
    initial_pass_count: int = 0
    exit_reason: str = ""  # resolved / regression / plateau / max_cycles
    verdict: str = "fail_fixable"
