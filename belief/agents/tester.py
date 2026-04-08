"""Tester Agent — generate pytest tests from code and acceptance criteria."""

from __future__ import annotations
import json, logging, os, re
from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.llm import LLMClient
from belief.models.state import Phase, UnifiedState
from belief.prompts import TESTER_SYSTEM, TESTER_PROMPT

logger = logging.getLogger("belief.agents.tester")

class TesterAgent(BaseAgent):
    role = ModelRole.TESTER
    name = "Tester"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.TESTING
        if not state.code_files:
            state.phase = Phase.EXECUTING
            return state

        spec = state.requirement_spec
        if not spec or not spec.acceptance_criteria:
            state.phase = Phase.EXECUTING
            return state

        llm = LLMClient(self.router)
        try:
            # Build code context — prioritize actual imports and signatures
            # over raw code to help the tester generate accurate imports
            code_context = self._build_code_context(state)

            prompt = TESTER_PROMPT.format(
                goal=spec.goal,
                acceptance_criteria="\n".join(f"  {i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)),
                code_files=code_context,
            )
            raw = await llm.generate_text(
                role=self.role, system=TESTER_SYSTEM, prompt=prompt,
                temperature=0.2, complexity=state.complexity_score,
            )
            # Parse ###FILE: format
            test_files = _parse_test_files(raw)
            state.test_files = test_files
            logger.info(f"Tester: {len(test_files)} test file(s)")

        except Exception as e:
            logger.warning(f"Tester skipped: {e}")
            state.warnings.append(f"Tester skipped: {e}")
        finally:
            await llm.close()

        state.phase = Phase.EXECUTING
        return state

    def _build_code_context(self, state: UnifiedState) -> str:
        """Build code context that shows the tester exactly what to import.

        Priority order:
        1. Skeleton registry context (typed interfaces — most accurate)
        2. AST-extracted signatures from actual code files
        3. Raw code truncated to 2000 chars (fallback)

        The key insight: tests fail because they import wrong class names.
        Giving the tester the exact import paths and class names prevents this.
        """
        parts = []

        # 1. Skeleton registry — the typed interface contracts
        if state.skeleton_registry_context:
            parts.append(
                "## IMPORTABLE INTERFACES (use these exact import paths and class names)\n"
                f"```python\n{state.skeleton_registry_context}\n```"
            )

        # 2. Extract signatures from all built code files via AST
        import_map = self._extract_imports_and_exports(state.code_files)
        if import_map:
            parts.append("## ACTUAL CODE EXPORTS (verified via AST)")
            for fname, exports in sorted(import_map.items()):
                if exports:
                    module = fname.replace("/", ".").replace(".py", "")
                    parts.append(f"  {fname} → from {module} import {', '.join(exports)}")

        # 3. Full code for key files (entry points, routes, services — under 3000 chars)
        parts.append("\n## CODE FILES")
        for f, c in sorted(state.code_files.items()):
            if f == "requirements.txt":
                continue
            # Show full content for small files, truncated for large ones
            if len(c) <= 3000:
                parts.append(f"--- {f} ---\n{c}")
            else:
                parts.append(f"--- {f} ({len(c)} chars, truncated) ---\n{c[:2000]}\n... (truncated)")

        return "\n\n".join(parts)

    def _extract_imports_and_exports(self, code_files: dict[str, str]) -> dict[str, list[str]]:
        """Extract public class/function names from each file via language adapter.

        Uses the LanguageAdapter pattern for multi-language support.
        Each adapter knows how to parse exports for its language.
        """
        from belief.languages import get_adapter, detect_language

        result = {}
        for fname, code in code_files.items():
            lang = detect_language(fname)
            adapter = get_adapter(lang)

            if not adapter.is_source_file(fname):
                continue
            if adapter.is_test_file(fname) or "/test" in fname:
                continue

            try:
                symbols = adapter.parse_exports(code, fname)
                if symbols:
                    result[fname] = [s.name for s in symbols]
            except Exception:
                pass

        return result

def _parse_test_files(raw: str) -> dict[str, str]:
    if "###FILE:" not in raw:
        return {}
    files = {}
    parts = re.split(r"###FILE:\s*", raw)
    for part in parts[1:]:
        nl = part.find("\n")
        if nl == -1:
            continue
        fname = part[:nl].strip()
        if not fname:
            continue
        content = re.sub(r"###END\s*$", "", part[nl + 1:])
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        if os.path.basename(fname).startswith("test_"):
            files[fname] = content
    return files
