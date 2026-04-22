"""Generate Fix — search/replace edit on ONE file guided by verbal diagnosis.

Uses search/replace blocks (not full file regeneration) because:
- 86% reduction in output tokens vs full regeneration
- Fewer regressions — only the targeted section changes
- Claude's str_replace format is built into its model weights

Each fix is validated via ast.parse before being accepted.
"""

from __future__ import annotations

import ast
import logging

from pydantic import BaseModel, Field

from belief.refinement import RefinementState

logger = logging.getLogger("belief.refinement.fixer")


FIXER_SYSTEM = """You are a surgical code fixer. You receive a diagnosis of a test failure
and the FULL content of the file that needs fixing.

Your job is to produce a MINIMAL search/replace edit that fixes the issue.
Do NOT rewrite the entire file. Change only what is necessary.

RULES:
1. Output EXACTLY one search/replace block
2. The old_str must match the file content EXACTLY (including whitespace)
3. The new_str should fix the diagnosed issue and nothing else
4. Do NOT add unrelated features, refactor other code, or change formatting
5. The result must be valid Python (will be checked via ast.parse)
6. Do NOT repeat a fix that was already tried and failed

Respond ONLY with valid JSON:
{
    "old_str": "the exact text to find in the file",
    "new_str": "the replacement text",
    "explanation": "what this fix does and why"
}"""


FIXER_PROMPT = """## Diagnosis
{diagnosis}

## Target File: {target_file}
```python
{file_content}
```

## Previous Fixes Already Tried (DO NOT repeat)
{previous_fixes}

Generate a search/replace edit to fix the diagnosed issue."""


class _FixResult(BaseModel):
    old_str: str = ""
    new_str: str = ""
    explanation: str = ""


async def generate_fix(
    state: RefinementState,
    diagnosis: str,
    target_file: str,
    llm=None,
) -> dict:
    """Generate a search/replace fix for the target file.
    
    Returns dict with 'old_str', 'new_str', 'explanation', 'success'.
    """
    file_content = state.code_files.get(target_file, "")
    if not file_content:
        return {"success": False, "explanation": f"File not found: {target_file}"}
    
    previous = "\n".join(f"  - {p}" for p in state.previous_fixes) if state.previous_fixes else "  (none — first cycle)"
    
    prompt = FIXER_PROMPT.format(
        diagnosis=diagnosis,
        target_file=target_file,
        file_content=file_content,
        previous_fixes=previous,
    )
    
    try:
        from belief.config import ModelRouter
        from belief.llm import LLMClient
        
        if llm is None:
            router = ModelRouter()
            llm = LLMClient(router)
        
        result = await llm.generate_structured(
            role="debugger",
            system=FIXER_SYSTEM,
            prompt=prompt,
            response_schema=_FixResult,
            temperature=0.2,
            max_tokens=2000,
        )
        
        old_str = result.old_str
        new_str = result.new_str
        explanation = result.explanation
        
        if not old_str or not new_str:
            return {"success": False, "explanation": "Empty search/replace block"}
        
        # Check that old_str exists in the file
        if old_str not in file_content:
            # Try fuzzy match — strip whitespace differences
            old_stripped = old_str.strip()
            if old_stripped in file_content:
                old_str = old_stripped
            else:
                logger.warning(f"Fixer: old_str not found in {target_file}")
                return {"success": False, "explanation": f"Search string not found in {target_file}"}
        
        # Apply the edit
        new_content = file_content.replace(old_str, new_str, 1)
        
        # Validate the result is valid Python
        if target_file.endswith(".py"):
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                logger.warning(f"Fixer: edit produces invalid Python: {e}")
                return {"success": False, "explanation": f"Edit produces syntax error: {e}"}
        
        logger.info(
            f"Fixer: {target_file} — {explanation[:60]}... "
            f"({len(old_str)} chars → {len(new_str)} chars)"
        )
        
        return {
            "success": True,
            "target_file": target_file,
            "old_str": old_str,
            "new_str": new_str,
            "new_content": new_content,
            "explanation": explanation,
        }
        
    except Exception as e:
        logger.warning(f"Fixer failed: {e}")
        return {"success": False, "explanation": str(e)}


# ── Multi-file fix (architect/editor pattern) ────────────────────────────────

MULTI_FIXER_SYSTEM = """You are a surgical code fixer that can edit MULTIPLE files at once.
You receive a diagnosis of a test failure and the content of ALL relevant files.

Your job is to produce MINIMAL search/replace edits across one or more files
that fix the issue ATOMICALLY — all edits succeed together or none apply.

RULES:
1. Output one or more search/replace blocks, each targeting a specific file
2. Each old_str must match its file content EXACTLY
3. Fix ONLY what the diagnosis identifies — no refactoring
4. If fixing file A requires a corresponding change in file B, include BOTH edits
5. All results must be valid Python

Respond ONLY with valid JSON:
{
    "edits": [
        {"file": "models.py", "old_str": "...", "new_str": "...", "explanation": "..."},
        {"file": "crud.py", "old_str": "...", "new_str": "...", "explanation": "..."}
    ],
    "summary": "overall explanation"
}"""


class _MultiFixResult(BaseModel):
    edits: list[dict] = Field(default_factory=list)
    summary: str = ""


async def generate_multi_file_fix(
    state: RefinementState,
    diagnosis: str,
    target_files: list[str],
    llm=None,
) -> dict:
    """Generate coordinated search/replace fixes across multiple files.

    Returns dict with 'edits' (list of per-file fixes), 'success', 'summary'.
    """
    file_context = []
    for f in target_files:
        content = state.code_files.get(f, "")
        if content:
            file_context.append(f"### {f}\n```python\n{content}\n```")

    if not file_context:
        return {"success": False, "edits": [], "summary": "No files found"}

    previous = "\n".join(f"  - {p}" for p in state.previous_fixes) if state.previous_fixes else "  (none)"

    prompt = f"""## Diagnosis
{diagnosis}

## Files to fix
{chr(10).join(file_context)}

## Previous Fixes Already Tried (DO NOT repeat)
{previous}

Generate search/replace edits across one or more files to fix this atomically."""

    try:
        from belief.config import ModelRouter
        from belief.llm import LLMClient

        if llm is None:
            router = ModelRouter()
            llm = LLMClient(router)

        result = await llm.generate_structured(
            role="debugger",
            system=MULTI_FIXER_SYSTEM,
            prompt=prompt,
            response_schema=_MultiFixResult,
            temperature=0.2,
            max_tokens=3000,
        )

        applied_edits = []

        for edit in result.edits:
            fpath = edit.get("file", "")
            old_str = edit.get("old_str", "")
            new_str = edit.get("new_str", "")
            explanation = edit.get("explanation", "")

            if not fpath or not old_str:
                continue

            content = state.code_files.get(fpath, "")
            if old_str not in content:
                old_stripped = old_str.strip()
                if old_stripped in content:
                    old_str = old_stripped
                else:
                    logger.warning(f"Multi-fixer: old_str not found in {fpath}")
                    continue

            new_content = content.replace(old_str, new_str, 1)

            if fpath.endswith(".py"):
                try:
                    ast.parse(new_content)
                except SyntaxError as e:
                    logger.warning(f"Multi-fixer: edit invalid in {fpath}: {e}")
                    continue

            applied_edits.append({
                "file": fpath,
                "old_str": old_str,
                "new_str": new_str,
                "new_content": new_content,
                "explanation": explanation,
            })

        if applied_edits:
            logger.info(f"Multi-fixer: {len(applied_edits)} edits across {len(set(e['file'] for e in applied_edits))} files — {result.summary[:60]}")

        return {
            "success": len(applied_edits) > 0,
            "edits": applied_edits,
            "summary": result.summary,
        }

    except Exception as e:
        logger.warning(f"Multi-fixer failed: {e}")
        return {"success": False, "edits": [], "summary": str(e)}
