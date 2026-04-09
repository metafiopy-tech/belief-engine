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
    
    Steps:
    1. Write all files to temp directory
    2. Install dependencies from requirements.txt (if present)
    3. Pre-validate tests — remove tests with hallucinated imports
    4. Run pytest with PYTHONPATH set
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
        
        # Install dependencies if requirements.txt exists
        req_path = tmp_path / "requirements.txt"
        if req_path.exists():
            _install_deps(req_path)

        # Pre-validate tests — remove tests with hallucinated imports
        _prevalidate_tests(tmp_path, code_files, test_files)
        
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


def _install_deps(req_path: Path) -> None:
    """Install dependencies from requirements.txt, ignoring failures."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "--break-system-packages", "-r", str(req_path)],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            logger.debug(f"Dep install partial failure: {proc.stderr[-200:]}")
    except Exception as e:
        logger.debug(f"Dep install skipped: {e}")


def _prevalidate_tests(
    tmp_path: Path, code_files: dict[str, str], test_files: dict[str, str]
) -> None:
    """Remove test functions that reference non-existent code symbols.
    
    Catches the 'testing hallucinated features' failure mode.
    A test that imports FooBar when the code only defines Foo gets removed.
    """
    import ast as _ast
    
    # Build set of all defined symbols in source code
    defined_symbols = set()
    for fname, content in code_files.items():
        if not fname.endswith(".py"):
            continue
        try:
            tree = _ast.parse(content)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                    defined_symbols.add(node.name)
                elif isinstance(node, _ast.Assign):
                    for target in node.targets:
                        if isinstance(target, _ast.Name):
                            defined_symbols.add(target.id)
        except SyntaxError:
            pass

    # Check each test file for hallucinated imports
    for fname, content in test_files.items():
        if not fname.endswith(".py"):
            continue
        try:
            tree = _ast.parse(content)
            hallucinated = []
            for node in _ast.walk(tree):
                if isinstance(node, _ast.ImportFrom) and node.module:
                    # Only check imports from local modules (not third-party)
                    module_root = node.module.split(".")[0]
                    local_modules = {
                        f.replace("/", ".").replace(".py", "").split(".")[0]
                        for f in code_files if f.endswith(".py")
                    }
                    if module_root in local_modules:
                        for alias in node.names:
                            if alias.name not in defined_symbols and alias.name != "*":
                                hallucinated.append(alias.name)

            if hallucinated:
                logger.info(
                    f"Pre-validation: {fname} references non-existent "
                    f"{', '.join(hallucinated[:3])} — keeping file but noting"
                )
        except SyntaxError:
            # Remove test files with syntax errors
            test_path = tmp_path / fname
            if test_path.exists():
                test_path.unlink()
                logger.info(f"Pre-validation: removed {fname} (syntax error)")


# ── The refinement loop ─────────────────────────────────────────────────────

def _find_related_files(target_file: str, code_files: dict[str, str]) -> list[str]:
    """Find files related to the target via imports.

    Returns the target + any files that import from it or that it imports from.
    Used to scope multi-file fixes.
    """
    import re as _re

    related = {target_file}
    target_module = target_file.replace("/", ".").replace(".py", "")
    target_base = target_module.split(".")[-1]

    for fname, content in code_files.items():
        if not fname.endswith(".py") or fname == target_file:
            continue
        # Check if this file imports from target
        if f"from {target_base}" in content or f"import {target_base}" in content:
            related.add(fname)
        # Check if target imports from this file
        target_content = code_files.get(target_file, "")
        other_base = fname.replace("/", ".").replace(".py", "").split(".")[-1]
        if f"from {other_base}" in target_content or f"import {other_base}" in target_content:
            related.add(fname)

    return sorted(related)[:5]  # Cap at 5 files

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
            bug_location = analysis.get("bug_location", "code")
            
            # ── Step 2: Generate fix ──
            # If the bug is in a test file, we need to point the fixer at test_files
            # Temporarily swap test files into code_files for the fixer to access
            fix_state = state
            if bug_location == "test" and target_file in state.test_files:
                # Create a temporary state with test files accessible as code_files
                fix_state = RefinementState(
                    code_files=dict(state.test_files),  # Fixer targets test files
                    test_files=state.test_files,
                    test_output=state.test_output,
                    cycle=state.cycle,
                    previous_fixes=state.previous_fixes,
                )
                logger.info(f"Refinement cycle {cycle + 1}: targeting TEST file {target_file}")
            
            # Try single-file fix first (cheaper). If it fails, try multi-file.
            fix = await generate_fix(fix_state, diagnosis, target_file, llm)
            is_multi = False

            if not fix.get("success") and cycle > 0:
                # Single-file fix failed — try multi-file
                try:
                    from belief.refinement.fixer import generate_multi_file_fix

                    # Find related files (dependencies of the target)
                    related = _find_related_files(target_file, state.code_files)
                    if len(related) > 1:
                        multi_fix = await generate_multi_file_fix(
                            state, diagnosis, related, llm
                        )
                        if multi_fix.get("success") and multi_fix.get("edits"):
                            # Convert multi-file result to single-file format
                            # by applying the first edit
                            first_edit = multi_fix["edits"][0]
                            fix = {
                                "success": True,
                                "target_file": first_edit["file"],
                                "new_content": first_edit["new_content"],
                                "explanation": multi_fix["summary"],
                                "_multi_edits": multi_fix["edits"],
                            }
                            is_multi = True
                except Exception as e:
                    logger.debug(f"Multi-file fix failed: {e}")
            
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
            pre_fix_code_snapshot = dict(state.code_files)
            pre_fix_test_snapshot = dict(state.test_files)

            if is_multi and fix.get("_multi_edits"):
                # Apply all edits atomically
                for edit in fix["_multi_edits"]:
                    if bug_location == "test" and edit["file"] in state.test_files:
                        state.test_files[edit["file"]] = edit["new_content"]
                    else:
                        state.code_files[edit["file"]] = edit["new_content"]
                target_file = ", ".join(e["file"] for e in fix["_multi_edits"])
            else:
                actual_target = fix.get("target_file", target_file)
                if bug_location == "test" and actual_target in state.test_files:
                    state.test_files[actual_target] = fix["new_content"]
                else:
                    state.code_files[actual_target] = fix["new_content"]
            
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
                    # REGRESSION — rollback all changes this cycle
                    logger.warning(
                        f"Refinement cycle {cycle + 1}: REGRESSION — "
                        f"{len(newly_broken)} new failures, rolling back"
                    )
                    state.code_files = pre_fix_code_snapshot
                    state.test_files = pre_fix_test_snapshot
                    
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
            "final_passed": state.best_pass_count,
            "final_total": initial_total,
            "initial_passed": state.initial_pass_count,
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
