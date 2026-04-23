"""Hierarchical Localizer — Agentless 3-Phase Fault Localization.

Implements the Agentless approach (arXiv 2407.01489):
  Phase 1: File-level — rank files by BM25+PPR, LLM selects top 5
  Phase 2: Class/function-level — show code skeletons, LLM narrows to symbols
  Phase 3: Line-level — show full function bodies, LLM identifies edit locations

Each phase uses cheaper models for broad ranking, more expensive for precise narrowing.

Research basis:
- Agentless: $0.70/issue, 50.8% on SWE-bench Verified
- Kimi-Dev: 3 patches × 3 tests > 40 patches majority voting
- RGFL: 69% file-level exact match with reasoning-guided localization

Usage:
    from belief.codebase.localizer import HierarchicalLocalizer
    localizer = HierarchicalLocalizer()
    locations = await localizer.localize(codebase, "fix the login endpoint validation")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("belief.codebase.localizer")


@dataclass
class EditLocation:
    """A specific location in the codebase to edit."""

    file_path: str
    function_name: str = ""
    class_name: str = ""
    start_line: int = 0
    end_line: int = 0
    context: str = ""  # Surrounding code for LLM context
    confidence: float = 0.0


class HierarchicalLocalizer:
    """3-phase Agentless localization: file → symbol → line."""

    async def localize(
        self,
        codebase,
        issue: str,
        max_locations: int = 3,
        llm=None,
    ) -> list[EditLocation]:
        """Run the full 3-phase localization pipeline.

        Args:
            codebase: Codebase object (from codebase/__init__.py)
            issue: Natural language issue/feature description
            max_locations: Maximum edit locations to return
            llm: LLMClient instance (optional — uses Haiku for phase 1-2, Sonnet for 3)

        Returns:
            List of EditLocations with file paths, function names, and line ranges
        """
        # Phase 1: File-level ranking (BM25 + PPR)
        ranked_files = await self._phase1_files(codebase, issue, llm)
        if not ranked_files:
            logger.warning("Localizer: Phase 1 found no relevant files")
            return []

        logger.info(f"Localizer Phase 1: {len(ranked_files)} candidate files")

        # Phase 2: Function/class-level narrowing
        candidates = await self._phase2_symbols(codebase, ranked_files[:5], issue, llm)
        if not candidates:
            # Fallback: return file-level locations
            return [EditLocation(file_path=f, confidence=0.5) for f in ranked_files[:max_locations]]

        logger.info(f"Localizer Phase 2: {len(candidates)} candidate symbols")

        # Phase 3: Line-level identification
        locations = await self._phase3_lines(codebase, candidates[:5], issue, llm)

        logger.info(f"Localizer Phase 3: {len(locations)} edit locations identified")
        return locations[:max_locations]

    async def _phase1_files(self, codebase, issue: str, llm=None) -> list[str]:
        """Phase 1: Rank files by relevance using BM25 + PageRank hybrid."""
        try:
            from belief.codebase.repo_graph import RepoGraph

            graph = RepoGraph.from_codebase(codebase)
            ranked = graph.localize(issue, max_files=20)
            file_paths = [r.path for r in ranked]
        except Exception as e:
            logger.debug(f"RepoGraph ranking failed, falling back: {e}")
            file_paths = codebase.localize_files(issue, max_files=20)

        if not file_paths or not llm:
            return file_paths[:5]

        # LLM re-ranking: show file list + repo map, ask which files to examine
        repo_map = codebase.generate_repo_map(max_tokens=1500)
        file_list = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(file_paths[:20]))

        prompt = f"""Given this issue:
{issue}

And this repository structure:
{repo_map}

These files were identified as potentially relevant:
{file_list}

Which 5 files are MOST LIKELY to contain the code that needs to be modified?
Return ONLY the file paths, one per line, most relevant first."""

        try:
            response = await llm.generate_text(
                role="latios",  # Haiku — cheap for ranking
                system="You are a fault localization expert. Return only file paths.",
                prompt=prompt,
                temperature=0.1,
                max_tokens=500,
            )
            # Parse file paths from response
            selected = []
            for line in response.strip().split("\n"):
                line = line.strip().lstrip("0123456789.-) ")
                if line in {f for f in file_paths}:
                    selected.append(line)
                elif any(line.endswith(f.split("/")[-1]) for f in file_paths):
                    # Partial match
                    for f in file_paths:
                        if line.endswith(f.split("/")[-1]):
                            selected.append(f)
                            break

            return selected[:5] if selected else file_paths[:5]

        except Exception:
            return file_paths[:5]

    async def _phase2_symbols(
        self, codebase, files: list[str], issue: str, llm=None
    ) -> list[EditLocation]:
        """Phase 2: Narrow to specific classes/functions within the top files."""
        candidates = []

        for fpath in files:
            # Get exports for this file
            symbols = codebase.localize_functions(fpath, issue, max_functions=5)

            if symbols:
                for sym in symbols:
                    candidates.append(
                        EditLocation(
                            file_path=fpath,
                            function_name=sym.name if sym.kind == "function" else "",
                            class_name=sym.name if sym.kind == "class" else "",
                            confidence=0.6,
                        )
                    )
            else:
                # No symbol match — include file-level
                candidates.append(
                    EditLocation(
                        file_path=fpath,
                        confidence=0.4,
                    )
                )

        if not llm or not candidates:
            return candidates

        # LLM narrowing: show code skeletons, ask which symbols to edit
        skeleton_context = []
        for fpath in files[:5]:
            content = codebase.get_file_content(fpath)
            if content:
                # Show first 100 lines as skeleton
                lines = content.split("\n")[:100]
                skeleton_context.append(f"### {fpath}\n```python\n{chr(10).join(lines)}\n```")

        if not skeleton_context:
            return candidates

        prompt = f"""Given this issue:
{issue}

Here are the code skeletons of the most relevant files:
{chr(10).join(skeleton_context[:3])}

Which specific functions or classes need to be modified to fix this issue?
For each, specify the file path and function/class name.

Return as a list:
file_path::function_or_class_name"""

        try:
            response = await llm.generate_text(
                role="latios",
                system="You are a code analysis expert. Identify exact functions/classes to modify.",
                prompt=prompt,
                temperature=0.1,
                max_tokens=500,
            )

            refined = []
            for line in response.strip().split("\n"):
                line = line.strip().lstrip("- ")
                if "::" in line:
                    parts = line.split("::", 1)
                    fpath = parts[0].strip()
                    symbol = parts[1].strip()
                    # Validate file exists
                    if fpath in {c.file_path for c in candidates}:
                        refined.append(
                            EditLocation(
                                file_path=fpath,
                                function_name=symbol,
                                confidence=0.7,
                            )
                        )

            return refined if refined else candidates

        except Exception:
            return candidates

    async def _phase3_lines(
        self, codebase, candidates: list[EditLocation], issue: str, llm=None
    ) -> list[EditLocation]:
        """Phase 3: Identify exact line ranges to modify."""
        if not llm:
            # Without LLM, add full function context and return
            for loc in candidates:
                content = codebase.get_file_content(loc.file_path)
                loc.context = content[:3000] if content else ""
            return candidates

        refined = []
        for loc in candidates[:3]:  # Max 3 to keep cost down
            content = codebase.get_file_content(loc.file_path)
            if not content:
                continue

            # Find the specific function/class body
            target_code = content
            if loc.function_name:
                target_code = _extract_function_body(content, loc.function_name) or content[:2000]
            elif loc.class_name:
                target_code = _extract_class_body(content, loc.class_name) or content[:2000]

            prompt = f"""Given this issue:
{issue}

Here is the code from {loc.file_path}:
```python
{target_code[:3000]}
```

Identify the EXACT lines that need to be modified. Return:
START_LINE: <number>
END_LINE: <number>
WHAT_TO_CHANGE: <brief description>"""

            try:
                response = await llm.generate_text(
                    role="debugger",  # Sonnet for precise line identification
                    system="You are a surgical code editor. Identify exact edit locations.",
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=300,
                )

                import re

                start_match = re.search(r"START_LINE:\s*(\d+)", response)
                end_match = re.search(r"END_LINE:\s*(\d+)", response)

                if start_match and end_match:
                    loc.start_line = int(start_match.group(1))
                    loc.end_line = int(end_match.group(1))
                    loc.confidence = 0.8

                loc.context = target_code[:3000]
                refined.append(loc)

            except Exception:
                loc.context = target_code[:3000]
                refined.append(loc)

        return refined


def _extract_function_body(code: str, func_name: str) -> str | None:
    """Extract a function's full source code by name."""
    import ast

    try:
        tree = ast.parse(code)
        lines = code.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    start = node.lineno - 1
                    end = node.end_lineno or (start + 20)
                    return "\n".join(lines[start:end])
    except SyntaxError:
        pass
    return None


def _extract_class_body(code: str, class_name: str) -> str | None:
    """Extract a class's full source code by name."""
    import ast

    try:
        tree = ast.parse(code)
        lines = code.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == class_name:
                    start = node.lineno - 1
                    end = node.end_lineno or (start + 50)
                    return "\n".join(lines[start:end])
    except SyntaxError:
        pass
    return None
