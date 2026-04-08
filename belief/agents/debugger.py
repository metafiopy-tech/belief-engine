"""Debugger Agent — surgical error fixing from execution failures.

Source: forge/agents/debugger.py
IRON LAW: NO FIXES WITHOUT ROOT CAUSE FIRST.

Deterministic fixes (no LLM needed):
- SyntaxError → ast.parse() to find exact line, attempt auto-fix
- ModuleNotFoundError → check missing __init__.py or wrong import path
- ImportError → check if symbol exists in target module via ast

If deterministic fix fails, falls back to LLM debugger.
"""

from __future__ import annotations

import ast
import logging
import re

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.state import Phase, UnifiedState
from belief.prompts import DEBUGGER_SYSTEM, DEBUGGER_PROMPT

logger = logging.getLogger("belief.agents.debugger")

DEBUGGER_SEARCH_REPLACE_SYSTEM = """You are the Debugger Agent. Fix execution errors using MINIMAL search/replace edits.

IRON LAW: NO FIXES WITHOUT ROOT CAUSE FIRST.
1. Read the error traceback carefully
2. Identify the EXACT root cause (wrong import, missing function, type error, etc.)
3. Generate a search/replace edit that fixes ONLY the root cause
4. The old_str must match the file content EXACTLY (including whitespace)
5. The new_str should be as small as possible — fix the bug, don't refactor

Respond ONLY with valid JSON containing old_str, new_str, and explanation."""


class DebuggerAgent(BaseAgent):
    role = ModelRole.DEBUGGER
    name = "Debugger"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.BUILDING

        exec_result = state.execution_result
        if not exec_result or exec_result.success:
            logger.info("Debugger: no errors to fix")
            state.phase = Phase.EXECUTING
            return state

        error = exec_result.stderr or exec_result.stdout or exec_result.error_summary
        if not error:
            logger.info("Debugger: no error output to analyze")
            state.phase = Phase.EXECUTING
            return state

        # ── ITEM 2: Try deterministic fixes first (no LLM cost) ──
        fixed = _try_deterministic_fix(error, state.code_files)
        if fixed:
            fname, new_code, fix_type = fixed
            state.code_files[fname] = new_code
            logger.info(f"Debugger: deterministic fix ({fix_type}) on {fname}")
            state.phase = Phase.EXECUTING
            return state

        # ── Fallback: LLM-assisted debugging via search/replace ──
        target_file = _find_error_file(error, state.code_files)
        if not target_file:
            state.warnings.append("Debugger: could not identify error file")
            state.phase = Phase.EXECUTING
            return state

        code = state.code_files.get(target_file, "")
        if not code:
            state.phase = Phase.EXECUTING
            return state

        llm = LLMClient(self.router)
        try:
            # Use search/replace for files over 2000 chars, full replacement for small files
            if len(code) > 2000:
                fixed_code = await self._fix_via_search_replace(
                    llm, state.user_goal, error, target_file, code, state.complexity_score
                )
            else:
                fixed_code = await self._fix_via_full_replacement(
                    llm, state.user_goal, error, target_file, code, state.complexity_score
                )

            if fixed_code and fixed_code != code:
                state.code_files[target_file] = fixed_code
                logger.info(f"Debugger: LLM fix on {target_file} ({len(fixed_code)} chars)")
            else:
                logger.warning(f"Debugger: could not fix {target_file}")

        except Exception as e:
            logger.warning(f"Debugger failed: {e}")
            state.warnings.append(f"Debugger failed: {e}")
        finally:
            await llm.close()

        state.phase = Phase.EXECUTING
        return state

    async def _fix_via_search_replace(
        self, llm, goal: str, error: str, filename: str, code: str, complexity: int
    ) -> str | None:
        """Fix a file using search/replace blocks — handles large files without truncation.

        This is the same approach used by the water cycle fixer, Aider, and Claude Code.
        The LLM only generates the changed portion, not the entire file.
        """
        prompt = f"""Fix this execution error using a search/replace edit:

ERROR:
{error[-1500:]}

FILE: {filename}
```
{code}
```

Instructions:
1. Identify the ROOT CAUSE of the error
2. Generate a MINIMAL search/replace edit
3. The old_str must match the file content EXACTLY
4. The new_str should fix ONLY the error

Respond ONLY with valid JSON:
{{"old_str": "exact text to find", "new_str": "replacement text", "explanation": "what this fixes"}}"""

        try:
            from pydantic import BaseModel
            class _FixResult(BaseModel):
                old_str: str = ""
                new_str: str = ""
                explanation: str = ""

            result = await llm.generate_structured(
                role=self.role,
                system=DEBUGGER_SEARCH_REPLACE_SYSTEM,
                prompt=prompt,
                response_schema=_FixResult,
                temperature=0.1,
                max_tokens=2000,
                complexity=complexity,
            )

            if not result.old_str or not result.new_str:
                return None

            # Check that old_str exists in the file
            if result.old_str not in code:
                # Try stripping whitespace
                old_stripped = result.old_str.strip()
                if old_stripped in code:
                    result.old_str = old_stripped
                else:
                    logger.warning(f"Debugger: search string not found in {filename}")
                    return None

            # Apply the edit
            new_code = code.replace(result.old_str, result.new_str, 1)

            # Validate
            if filename.endswith(".py"):
                try:
                    ast.parse(new_code)
                except SyntaxError:
                    logger.warning(f"Debugger: search/replace produced invalid Python for {filename}")
                    return None

            logger.info(f"Debugger: search/replace fix — {result.explanation[:60]}")
            return new_code

        except Exception as e:
            logger.debug(f"Debugger: search/replace failed ({e}), trying full replacement")
            # Fall back to full replacement for small enough files
            if len(code) <= 4000:
                return await self._fix_via_full_replacement(
                    llm, goal, error, filename, code, complexity
                )
            return None

    async def _fix_via_full_replacement(
        self, llm, goal: str, error: str, filename: str, code: str, complexity: int
    ) -> str | None:
        """Fix a small file by regenerating it entirely (legacy behavior)."""
        prompt = DEBUGGER_PROMPT.format(
            goal=goal,
            error=error[-1500:],
            filename=filename,
            code=code[:4000],
        )
        fixed_code = await llm.generate_text(
            role=self.role, system=DEBUGGER_SYSTEM, prompt=prompt,
            temperature=0.1, complexity=complexity,
        )
        fixed_code = fixed_code.strip()
        fixed_code = re.sub(r"^```(?:python)?\s*\n?", "", fixed_code)
        fixed_code = re.sub(r"\n?```\s*$", "", fixed_code)

        if filename.endswith(".py"):
            try:
                ast.parse(fixed_code)
            except SyntaxError:
                logger.warning(f"Debugger: LLM returned invalid Python for {filename}, keeping original")
                return None

        return fixed_code


# ── Deterministic fixes (no LLM needed) ──────────────────────────────────


def _try_deterministic_fix(
    error: str, code_files: dict[str, str]
) -> tuple[str, str, str] | None:
    """Try to fix the error without an LLM call.

    Returns (filename, fixed_code, fix_type) or None.
    """
    error_lower = error.lower()

    # Fix 1: SyntaxError — find and attempt to fix
    if "syntaxerror" in error_lower:
        return _fix_syntax_error(error, code_files)

    # Fix 2: ModuleNotFoundError — add missing __init__.py or fix import
    if "modulenotfounderror" in error_lower:
        return _fix_module_not_found(error, code_files)

    # Fix 3: ImportError (cannot import name) — fix wrong symbol name
    if "importerror" in error_lower and "cannot import name" in error_lower:
        return _fix_import_error(error, code_files)

    return None


def _fix_syntax_error(
    error: str, code_files: dict[str, str]
) -> tuple[str, str, str] | None:
    """Fix common syntax errors deterministically."""
    # Find which file has the syntax error
    for fname, code in code_files.items():
        if not fname.endswith(".py"):
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            # Common fix: truncated file — remove the last incomplete line
            lines = code.splitlines()
            if e.lineno and e.lineno >= len(lines):
                # File is truncated — remove the last line
                fixed = "\n".join(lines[:e.lineno - 1]) + "\n"
                try:
                    ast.parse(fixed)
                    return (fname, fixed, "truncated_file")
                except SyntaxError:
                    pass

            # Common fix: missing colon after except/else/elif/finally
            if e.lineno and e.lineno <= len(lines):
                line = lines[e.lineno - 1]
                for kw in ("except", "else", "elif", "finally"):
                    if line.strip().startswith(kw) and not line.rstrip().endswith(":"):
                        lines[e.lineno - 1] = line.rstrip() + ":"
                        fixed = "\n".join(lines)
                        try:
                            ast.parse(fixed)
                            return (fname, fixed, f"missing_colon_{kw}")
                        except SyntaxError:
                            lines[e.lineno - 1] = line  # revert

    return None


def _fix_module_not_found(
    error: str, code_files: dict[str, str]
) -> tuple[str, str, str] | None:
    """Fix ModuleNotFoundError by adding missing __init__.py."""
    # Extract the module name: "No module named 'pipeline'"
    match = re.search(r"No module named '([^']+)'", error)
    if not match:
        return None

    missing_module = match.group(1).split(".")[0]

    # Check if we have files in that package but no __init__.py
    package_files = [f for f in code_files if f.startswith(missing_module + "/")]
    init_path = f"{missing_module}/__init__.py"

    if package_files and init_path not in code_files:
        # Add empty __init__.py
        return (init_path, "", "missing_init_py")

    return None


def _fix_import_error(
    error: str, code_files: dict[str, str]
) -> tuple[str, str, str] | None:
    """Fix 'cannot import name X from Y' by checking available symbols."""
    # Extract: cannot import name 'Foo' from 'module'
    match = re.search(r"cannot import name '(\w+)' from '([^']+)'", error)
    if not match:
        return None

    symbol_name = match.group(1)
    module_path = match.group(2).replace(".", "/") + ".py"

    # Check if the module file exists and what it exports
    if module_path not in code_files:
        return None

    module_code = code_files[module_path]
    try:
        tree = ast.parse(module_code)
    except SyntaxError:
        return None

    # Find all defined names in the module
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)

    if symbol_name in defined:
        return None  # Symbol exists, error is something else

    # Find closest match (simple case-insensitive check)
    for name in defined:
        if name.lower() == symbol_name.lower():
            # Fix the importing file — find and replace the import
            for fname, code in code_files.items():
                if f"import {symbol_name}" in code:
                    fixed = code.replace(symbol_name, name)
                    return (fname, fixed, f"wrong_case_{symbol_name}_to_{name}")

    return None


# ── File identification ──────────────────────────────────────────────────


def _find_error_file(error: str, code_files: dict[str, str]) -> str | None:
    """Identify which source file caused the error from traceback."""
    # Look for explicit file references in the traceback
    for fname in code_files:
        if fname in error:
            return fname

    # Look for common patterns
    match = re.search(r'File ".*?/([^/]+\.py)"', error)
    if match:
        candidate = match.group(1)
        if candidate in code_files:
            return candidate
        # Try with package prefix
        for fname in code_files:
            if fname.endswith("/" + candidate):
                return fname

    # Fall back to entry point or first .py file
    for fname in code_files:
        if fname in ("main.py", "app.py", "server.py"):
            return fname

    py_files = [f for f in code_files if f.endswith(".py") and "/test" not in f]
    return py_files[0] if py_files else None
