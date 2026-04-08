"""Water Cycle Runner — subgraph builder, revalidation, routing, lesson storage.

Assembles the refinement loop as a self-contained async pipeline:
  analyze_failures → generate_fix → revalidate → [should_continue?]

Wired into the main graph between validator and END when:
  - verdict == fail_fixable
  - executor passed (code runs, just has quality issues)
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from belief.refinement import RefinementState, CycleRecord
from belief.refinement.analyzer import analyze_failures, parse_test_results
from belief.refinement.fixer import generate_fix

logger = logging.getLogger("belief.refinement.runner")


# ── Revalidate ───────────────────────────────────────────────────────────────

def _run_tests(code_files: dict[str, str], test_files: dict[str, str]) -> tuple[str, int, int, list[str]]:
    """Run pytest in a temp directory and return (output, passed, total, failed_ids).
    
    Uses --tb=short for concise tracebacks and --no-header for cleaner parsing.
    """
    with tempfile.TemporaryDirectory(prefix="belief_refine_") as tmp:
        tmp_path = Path(tmp)
        
        # Write all files
        for files_dict in [code_files, test_files]:
            for fname, content in files_dict.items():
                fpath = tmp_path / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)
        
        # Ensure __init__.py exists in all package dirs
        for dirpath, dirnames, filenames in os.walk(tmp_path):
            py_files = [f for f in filenames if f.endswith(".py")]
            if py_files:
                init = Path(dirpath) / "__init__.py"
                if not init.exists():
                    init.write_text("")
        
        # Run pytest
        python = sys.executable
        try:
            proc = subprocess.run(
                [python, "-m", "pytest", "--tb=short", "-q", "--no-header"],
                capture_output=True, text=True,
                timeout=60, cwd=str(tmp_path),
                env={**os.environ, "PYTHONPATH": str(tmp_path)},
            )
            output = proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired:
            output = "TIMEOUT: pytest timed out after 60s"
        except Exception as e:
            output = f"ERROR: {e}"
    
    passed, total, failed_ids, _ = parse_test_results(output)
    return output, passed, total, failed_ids


# ── The refinement loop ─────────────────────────────────────────────────────

async def run_refinement_loop(
    code_files: dict[str, str],
    test_files: dict[str, str],
    initial_test_output: str,
    max_cycles: int = 3,
) -> dict[str, Any]:
    """Run the water cycle refinement loop.
    
    Returns dict with:
        code_files: refined code
        verdict: "pass" or "fail_fixable"
        exit_reason: "resolved" / "regression" / "plateau" / "max_cycles"
        cycles_used: number of cycles executed
        test_history: list of CycleRecord
        lessons: list of refinement lessons for ChromaDB
    """
    from belief.config import ModelRouter
    from belief.llm import LLMClient
    
    router = ModelRouter()
    llm = LLMClient(router)
    
    state = RefinementState(
        code_files=dict(code_files),  # Working copy
        test_files=dict(test_files),
        test_output=initial_test_output,
        max_cycles=max_cycles,
    )
    
    # Parse initial test results
    initial_passed, initial_total, initial_failed, _ = parse_test_results(initial_test_output)
    state.initial_pass_count = initial_passed
    state.best_pass_count = initial_passed
    state.best_snapshot = dict(code_files)
    
    logger.info(
        f"Refinement: starting water cycle — {initial_passed}/{initial_total} tests passing, "
        f"max {max_cycles} cycles"
    )
    
    if initial_total == 0 or initial_passed == initial_total:
        # Nothing to refine
        return {
            "code_files": code_files,
            "verdict": "pass" if initial_passed == initial_total else "fail_fixable",
            "exit_reason": "resolved" if initial_passed == initial_total else "no_tests",
            "cycles_used": 0,
            "test_history": [],
            "lessons": [],
        }
    
    try:
        for cycle in range(max_cycles):
            state.cycle = cycle
            
            # ── Step 1: Analyze failures ──
            analysis = await analyze_failures(state, llm)
            diagnosis = analysis["diagnosis"]
            target_file = analysis["target_file"]
            
            # ── Step 2: Generate fix ──
            fix = await generate_fix(state, diagnosis, target_file, llm)
            
            if not fix.get("success"):
                logger.warning(f"Refinement cycle {cycle + 1}: fix generation failed — {fix.get('explanation')}")
                state.previous_fixes.append(f"Cycle {cycle + 1}: FAILED — {fix.get('explanation', 'unknown')}")
                
                record = CycleRecord(
                    cycle=cycle + 1,
                    passed_count=state.best_pass_count,
                    total_count=initial_total,
                    failed_test_ids=initial_failed,
                    file_modified="",
                    diagnosis=diagnosis,
                    fix_summary=f"FAILED: {fix.get('explanation', 'unknown')}",
                )
                state.test_history.append(record)
                continue
            
            # ── Step 3: Apply fix ──
            pre_fix_content = state.code_files[target_file]
            state.code_files[target_file] = fix["new_content"]
            
            # ── Step 4: Revalidate ──
            output, passed, total, failed_ids = _run_tests(state.code_files, state.test_files)
            state.test_output = output
            
            fix_summary = fix["explanation"]
            state.previous_fixes.append(f"Cycle {cycle + 1}: {target_file} — {fix_summary}")
            
            # ── Step 5: Check for regression ──
            if state.test_history:
                prev_failed = set(state.test_history[-1].failed_test_ids)
                curr_failed = set(failed_ids)
                newly_broken = curr_failed - prev_failed
                if newly_broken and passed < state.best_pass_count:
                    # REGRESSION — rollback
                    logger.warning(
                        f"Refinement cycle {cycle + 1}: REGRESSION — "
                        f"{len(newly_broken)} new failures, rolling back"
                    )
                    state.code_files[target_file] = pre_fix_content
                    
                    record = CycleRecord(
                        cycle=cycle + 1, passed_count=passed, total_count=total,
                        failed_test_ids=failed_ids, file_modified=target_file,
                        diagnosis=diagnosis, fix_summary=fix_summary, regression=True,
                    )
                    state.test_history.append(record)
                    state.exit_reason = "regression"
                    break
            
            record = CycleRecord(
                cycle=cycle + 1, passed_count=passed, total_count=total,
                failed_test_ids=failed_ids, file_modified=target_file,
                diagnosis=diagnosis, fix_summary=fix_summary,
            )
            state.test_history.append(record)
            
            logger.info(
                f"Refinement cycle {cycle + 1}: {passed}/{total} tests "
                f"(was {state.best_pass_count}/{total}) — {target_file}"
            )
            
            # Update best snapshot
            if passed > state.best_pass_count:
                state.best_pass_count = passed
                state.best_snapshot = dict(state.code_files)
            
            # ── Step 6: Check stop conditions ──
            if passed == total:
                state.exit_reason = "resolved"
                state.verdict = "pass"
                logger.info(f"Refinement: ALL TESTS PASSING after {cycle + 1} cycles")
                break
            
            # Plateau: no improvement for 2 consecutive cycles
            if len(state.test_history) >= 2:
                recent = [r.passed_count for r in state.test_history[-2:]]
                if recent[-1] <= recent[-2] and recent[-1] <= state.best_pass_count:
                    state.exit_reason = "plateau"
                    logger.info(f"Refinement: plateau detected after {cycle + 1} cycles")
                    break
            
            # Oscillation: current failures overlap with 2 cycles ago
            if len(state.test_history) >= 3:
                two_ago = set(state.test_history[-3].failed_test_ids)
                current = set(failed_ids)
                if len(two_ago & current) > len(current) * 0.5:
                    state.exit_reason = "oscillation"
                    logger.info(f"Refinement: oscillation detected after {cycle + 1} cycles")
                    break
        else:
            state.exit_reason = "max_cycles"
        
        # Use best snapshot
        final_files = state.best_snapshot if state.best_snapshot else state.code_files
        
        # Determine final verdict
        if state.best_pass_count == initial_total:
            state.verdict = "pass"
        elif state.best_pass_count > state.initial_pass_count:
            state.verdict = "fail_fixable"  # Improved but not perfect
        else:
            state.verdict = "fail_fixable"
        
        # Build lessons for ChromaDB
        lessons = _build_lessons(state)
        
        improvement = state.best_pass_count - state.initial_pass_count
        logger.info(
            f"Refinement complete: {state.exit_reason} — "
            f"{state.initial_pass_count} → {state.best_pass_count}/{initial_total} tests "
            f"(+{improvement}) in {len(state.test_history)} cycles"
        )
        
        return {
            "code_files": final_files,
            "verdict": state.verdict,
            "exit_reason": state.exit_reason,
            "cycles_used": len(state.test_history),
            "test_history": state.test_history,
            "lessons": lessons,
            "improvement": improvement,
            "best_pass_count": state.best_pass_count,
            "total_tests": initial_total,
        }
    
    finally:
        await llm.close()


# ── Lesson extraction ────────────────────────────────────────────────────────

def _build_lessons(state: RefinementState) -> list[dict]:
    """Extract refinement lessons for ChromaDB soil.
    
    Each successful fix becomes a pattern nutrient.
    Each failed fix or regression becomes an antipattern.
    """
    lessons = []
    
    for record in state.test_history:
        if record.regression:
            lessons.append({
                "nutrient_type": "antipattern",
                "content": (
                    f"Refinement regression: fixing {record.file_modified} with "
                    f"'{record.fix_summary}' caused {len(record.failed_test_ids)} new failures. "
                    f"Diagnosis was: {record.diagnosis}"
                ),
                "tags": ["refinement", "regression"],
            })
        elif record.passed_count > state.initial_pass_count:
            lessons.append({
                "nutrient_type": "pattern",
                "content": (
                    f"Refinement success: {record.file_modified} — {record.fix_summary}. "
                    f"Tests improved from {state.initial_pass_count} to {record.passed_count}. "
                    f"Diagnosis: {record.diagnosis}"
                ),
                "tags": ["refinement", "fix"],
            })
    
    return lessons


# ── Soil integration ─────────────────────────────────────────────────────────

async def store_refinement_lessons(lessons: list[dict]) -> int:
    """Deposit refinement lessons into ChromaDB soil."""
    if not lessons:
        return 0
    
    deposited = 0
    try:
        from belief.memory.soil import Soil
        from belief.memory.nutrients import Nutrient, NutrientType
        
        soil = Soil(Path("~/.belief-engine/soil").expanduser())
        
        for lesson in lessons:
            ntype = NutrientType(lesson["nutrient_type"])
            nutrient = Nutrient(
                nutrient_type=ntype,
                tier=1,  # Refinement lessons start at tier 1
                content=lesson["content"],
                embedding_text=lesson["content"][:200],
                tags=lesson.get("tags", []),
            )
            soil.deposit(nutrient)
            deposited += 1
            logger.info(f"Refinement lesson deposited: {ntype.value}")
    
    except Exception as e:
        logger.warning(f"Failed to store refinement lessons: {e}")
    
    return deposited
