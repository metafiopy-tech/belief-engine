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

        # ── Move 6: Architect/Editor split for multi-file debugging ──
        # Phase 1 (Architect — Sonnet): Analyze root cause across ALL files
        # Phase 2 (Editor — Haiku): Apply targeted search/replace edits

        llm = LLMClient(self.router)
        try:
            # Build repo map for context
            repo_context = ""
            try:
                from belief.agents.repo_map import RepoMap
                repo_map = RepoMap.from_code_files(state.code_files)
                repo_context = repo_map.format_overview(max_tokens=1500)
            except Exception:
                pass

            # ── Phase 1: Architect diagnoses across all files (Sonnet) ──
            diagnosis = await self._architect_diagnose(
                llm, error, state.code_files, repo_context, state.complexity_score
            )

            # Skeleton files are ADDITIVE-ONLY for the debugger: we allow
            # edits, but reject any edit that removes a top-level export.
            # This lets the debugger add a missing symbol to `database.py`
            # (e.g. `get_db`) while still blocking edits that would
            # clobber the generator's engine/Base/SessionLocal scaffolding.
            skeleton_files = set()
            if hasattr(state, 'skeleton_files') and state.skeleton_files:
                skeleton_files = set(state.skeleton_files.keys())

            if not diagnosis or not diagnosis.get("files_to_fix"):
                # Fallback: single-file fix on the error file
                target_file = _find_error_file(error, state.code_files)
                if target_file:
                    code = state.code_files.get(target_file, "")
                    if code:
                        fixed_code = await self._fix_via_search_replace(
                            llm, state.user_goal, error, target_file, code,
                            state.complexity_score, code_files=state.code_files,
                        )
                        fixed_code = _accept_if_additive(
                            target_file, code, fixed_code,
                            is_skeleton=target_file in skeleton_files,
                        )
                        if fixed_code and fixed_code != code:
                            state.code_files[target_file] = fixed_code
                            logger.info(f"Debugger: single-file fix on {target_file}")
            else:
                # ── Phase 2: Editor applies fixes (uses debugger role = may be Haiku) ──
                fixes_applied = 0
                for fix_spec in diagnosis["files_to_fix"][:3]:  # Max 3 files per cycle
                    fname = fix_spec.get("file", "")
                    instruction = fix_spec.get("instruction", "")
                    code = state.code_files.get(fname, "")

                    if not code or not instruction:
                        continue

                    fixed_code = await self._editor_apply(
                        llm, fname, code, instruction, state.complexity_score
                    )
                    fixed_code = _accept_if_additive(
                        fname, code, fixed_code,
                        is_skeleton=fname in skeleton_files,
                    )

                    if fixed_code and fixed_code != code:
                        state.code_files[fname] = fixed_code
                        fixes_applied += 1
                        logger.info(f"Debugger: editor fix on {fname}")

                if fixes_applied > 0:
                    logger.info(
                        f"Debugger: architect/editor — {fixes_applied} files fixed, "
                        f"root cause: {diagnosis.get('root_cause', 'unknown')[:80]}"
                    )
                else:
                    logger.warning("Debugger: architect diagnosed but editor couldn't apply fixes")

        except Exception as e:
            logger.warning(f"Debugger failed: {e}")
            state.warnings.append(f"Debugger failed: {e}")
        finally:
            await llm.close()

        state.phase = Phase.EXECUTING
        return state

    async def _architect_diagnose(
        self, llm, error: str, code_files: dict[str, str],
        repo_context: str, complexity: int,
    ) -> dict | None:
        """Phase 1: Architect analyzes root cause across all files (Sonnet).

        Returns a diagnosis dict:
        {
            "root_cause": "description of the root cause",
            "files_to_fix": [
                {"file": "models.py", "instruction": "Add missing Mapped import"},
                {"file": "main.py", "instruction": "Fix import path from models to database"}
            ]
        }
        """
        # Show the first 200 lines of each file involved
        file_context = []
        for fname, content in sorted(code_files.items()):
            if not fname.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
                continue
            lines = content.split("\n")[:150]
            truncated = "\n".join(lines)
            lang = "typescript" if fname.endswith((".ts", ".tsx")) else "javascript" if fname.endswith((".js", ".jsx")) else "python"
            file_context.append(f"### {fname}\n```{lang}\n{truncated}\n```")

        # Cap total context
        context_str = "\n\n".join(file_context)
        if len(context_str) > 12000:
            context_str = context_str[:12000] + "\n... (truncated)"

        from pydantic import BaseModel, Field

        class _FileFix(BaseModel):
            file: str = ""
            instruction: str = ""

        class _Diagnosis(BaseModel):
            root_cause: str = ""
            files_to_fix: list[_FileFix] = Field(default_factory=list)

        try:
            result = await llm.generate_structured(
                role=self.role,  # Uses debugger role (Sonnet)
                system=(
                    "You are the Architect Debugger. Analyze this error and identify "
                    "the ROOT CAUSE across ALL project files. Determine which files "
                    "need changes and give a specific instruction for each file.\n\n"
                    "RULES:\n"
                    "1. Trace the error to its TRUE origin — often the error is in file A "
                    "   but the fix is in file B (e.g., missing export, wrong import path)\n"
                    "2. List ALL files that need changes, not just the one that threw\n"
                    "3. Each instruction must be specific: 'Add import X from Y', "
                    "   'Change class Base to DeclarativeBase', etc.\n"
                    "4. Max 3 files per fix cycle"
                ),
                prompt=(
                    f"## Error\n{error[-2000:]}\n\n"
                    f"## Available Modules\n{repo_context}\n\n"
                    f"## Project Files\n{context_str}\n\n"
                    f"Diagnose the root cause and list files to fix."
                ),
                response_schema=_Diagnosis,
                temperature=0.1,
                max_tokens=1000,
                complexity=complexity,
            )
            return {
                "root_cause": result.root_cause,
                "files_to_fix": [{"file": f.file, "instruction": f.instruction} for f in result.files_to_fix],
            }
        except Exception as e:
            logger.debug(f"Architect diagnosis failed: {e}")
            return None

    async def _editor_apply(
        self, llm, filename: str, code: str, instruction: str, complexity: int,
    ) -> str | None:
        """Phase 2: Editor applies a specific fix to one file (Haiku-eligible).

        Takes a specific instruction from the architect and applies it via
        search/replace edit.
        """
        from pydantic import BaseModel

        class _EditResult(BaseModel):
            old_str: str = ""
            new_str: str = ""
            explanation: str = ""

        try:
            result = await llm.generate_structured(
                role="gap_analyst",  # Uses Haiku (cheaper for mechanical edits)
                system=(
                    "You are the Editor. Apply the given instruction as a MINIMAL "
                    "search/replace edit. Match the old_str EXACTLY. Change only what "
                    "the instruction says."
                ),
                prompt=(
                    f"INSTRUCTION: {instruction}\n\n"
                    f"FILE: {filename}\n```python\n{code}\n```\n\n"
                    f"Respond with JSON: {{\"old_str\": \"...\", \"new_str\": \"...\", \"explanation\": \"...\"}}"
                ),
                response_schema=_EditResult,
                temperature=0.1,
                max_tokens=1500,
                complexity=1,  # Always low complexity for editor
            )

            if not result.old_str or not result.new_str:
                return None

            if result.old_str not in code:
                stripped = result.old_str.strip()
                if stripped in code:
                    result.old_str = stripped
                else:
                    return None

            new_code = code.replace(result.old_str, result.new_str, 1)

            # Validate
            if filename.endswith(".py"):
                try:
                    import ast
                    ast.parse(new_code)
                except SyntaxError:
                    return None
            elif filename.endswith((".ts", ".tsx", ".js", ".jsx")):
                # Basic brace/bracket validation for TypeScript
                if abs(new_code.count("{") - new_code.count("}")) > 1:
                    return None
                if abs(new_code.count("(") - new_code.count(")")) > 1:
                    return None

            logger.info(f"Debugger: editor — {result.explanation[:60]}")
            return new_code

        except Exception as e:
            logger.debug(f"Editor apply failed on {filename}: {e}")
            return None

    async def _fix_via_search_replace(
        self, llm, goal: str, error: str, filename: str, code: str, complexity: int,
        code_files: dict[str, str] | None = None,
    ) -> str | None:
        """Fix a file using search/replace blocks — handles large files without truncation."""
        # Build repo map context so debugger knows what's importable
        repo_context = ""
        if code_files:
            try:
                from belief.agents.repo_map import RepoMap
                repo_map = RepoMap.from_code_files(code_files)
                repo_context = repo_map.format_overview(max_tokens=1000)
                if repo_context:
                    repo_context = f"\nAVAILABLE MODULES AND SYMBOLS:\n{repo_context}\n"
            except Exception:
                pass

        prompt = f"""Fix this execution error using a search/replace edit:

ERROR:
{error[-1500:]}

FILE: {filename}
```
{code}
```
{repo_context}
Instructions:
1. Identify the ROOT CAUSE of the error
2. Generate a MINIMAL search/replace edit
3. The old_str must match the file content EXACTLY
4. The new_str should fix ONLY the error
5. Only import from modules listed in AVAILABLE MODULES above

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
            elif filename.endswith((".ts", ".tsx", ".js", ".jsx")):
                if abs(new_code.count("{") - new_code.count("}")) > 1:
                    logger.warning(f"Debugger: search/replace produced unmatched braces in {filename}")
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


# ── Skeleton-file safety: additive-only edits ────────────────────────────


def _top_level_exports(source: str) -> set[str]:
    """Return the set of top-level names defined in a Python source file.

    Includes classes, functions, async functions, and module-level
    assignments. Used to verify that a debugger edit doesn't remove any
    symbols a skeleton file was generated to export.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _accept_if_additive(
    filename: str,
    original: str,
    fixed: str | None,
    *,
    is_skeleton: bool,
) -> str | None:
    """Gate an edit against the additive-only rule for skeleton files.

    For non-skeleton files the edit passes through unchanged. For
    skeleton files we require that every top-level export present in
    the original survives in the fixed version — edits may only ADD
    symbols, never remove them. This replaces the old blanket
    "skeleton files are immutable" block, which made the debugger
    unable to add a missing `get_db` to a generator-emitted `database.py`.
    """
    if fixed is None or fixed == original:
        return fixed
    if not is_skeleton:
        return fixed

    original_exports = _top_level_exports(original)
    new_exports = _top_level_exports(fixed)
    missing = original_exports - new_exports
    if missing:
        logger.info(
            f"Debugger: rejecting edit to skeleton file {filename} — "
            f"would remove exports {sorted(missing)}"
        )
        return original
    logger.info(
        f"Debugger: additive edit to skeleton file {filename} "
        f"(+{sorted(new_exports - original_exports)})"
    )
    return fixed


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
