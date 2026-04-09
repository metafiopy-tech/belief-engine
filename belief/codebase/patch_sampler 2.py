"""Patch Sampler — Multi-Patch Generation with Self-Play Ranking.

Implements two key research findings:
1. Kimi-Dev self-play: 3 patches × 3 tests beats 40 patches with majority voting
2. CodeT dual execution: score = |solutions_in_group| × |tests_passed_by_group|

Pipeline:
  1. BugFixer generates N candidate patches (diff format)
  2. TestWriter generates M reproduction tests per patch
  3. Execute all patches against all tests → agreement matrix
  4. CodeT ranking selects the best patch

Research basis:
- Agentless: generates 40 patches (1 greedy + 39 sampled at temp=0.8)
- Kimi-Dev: 3×3 self-play achieves 60.4% on SWE-bench Verified
- CodeT: +18.8% on HumanEval via dual execution agreement

Usage:
    from belief.codebase.patch_sampler import PatchSampler
    sampler = PatchSampler()
    best_patch = await sampler.sample_and_rank(
        codebase, location, issue, llm, n_patches=3, n_tests=3
    )
"""

from __future__ import annotations

import ast
import asyncio
import logging
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.codebase.patch_sampler")


@dataclass
class PatchCandidate:
    """A single candidate patch."""
    id: int
    file_path: str
    old_code: str
    new_code: str
    explanation: str
    syntax_valid: bool = False
    tests_passed: frozenset = field(default_factory=frozenset)
    codet_score: float = 0.0


@dataclass
class ReproductionTest:
    """A test designed to fail on unfixed code, pass on fixed code."""
    id: int
    code: str
    name: str


@dataclass
class SamplerResult:
    """Result of the self-play sampling process."""
    best_patch: PatchCandidate | None = None
    candidates: list[PatchCandidate] = field(default_factory=list)
    tests: list[ReproductionTest] = field(default_factory=list)
    agreement_matrix: dict[int, frozenset] = field(default_factory=dict)
    ranking_method: str = "codet"


class PatchSampler:
    """Multi-patch generation with Kimi-Dev self-play and CodeT ranking."""

    async def sample_and_rank(
        self,
        codebase,
        location,  # EditLocation
        issue: str,
        llm=None,
        n_patches: int = 3,
        n_tests: int = 3,
    ) -> SamplerResult:
        """Generate patches and tests, cross-validate, rank by CodeT agreement.

        The Kimi-Dev insight: generating diverse tests alongside patches
        and cross-validating produces better selection than majority voting
        on patches alone.
        """
        result = SamplerResult()

        if not llm:
            logger.warning("PatchSampler: no LLM provided")
            return result

        # Get the current code
        current_code = codebase.get_file_content(location.file_path)
        if not current_code:
            return result

        # Step 1: Generate N candidate patches (BugFixer role)
        patches = await self._generate_patches(
            location, issue, current_code, llm, n_patches
        )
        result.candidates = patches
        valid_patches = [p for p in patches if p.syntax_valid]

        if not valid_patches:
            logger.warning("PatchSampler: no valid patches generated")
            return result

        logger.info(f"PatchSampler: {len(valid_patches)}/{len(patches)} valid patches")

        # Step 2: Generate M reproduction tests (TestWriter role)
        tests = await self._generate_reproduction_tests(
            location, issue, current_code, llm, n_tests
        )
        result.tests = tests

        if not tests:
            # No tests — fall back to first valid patch
            result.best_patch = valid_patches[0]
            result.ranking_method = "first_valid"
            return result

        logger.info(f"PatchSampler: {len(tests)} reproduction tests generated")

        # Step 3: Execute all patches against all tests (agreement matrix)
        for patch in valid_patches:
            passed = await self._execute_patch_tests(
                codebase, location, patch, tests
            )
            patch.tests_passed = frozenset(passed)

        # Step 4: CodeT ranking
        best = self._codet_rank(valid_patches)
        result.best_patch = best
        result.agreement_matrix = {p.id: p.tests_passed for p in valid_patches}

        if best:
            logger.info(
                f"PatchSampler: selected patch {best.id} "
                f"(passes {len(best.tests_passed)}/{len(tests)} tests, "
                f"CodeT score={best.codet_score:.1f})"
            )

        return result

    async def _generate_patches(
        self, location, issue: str, current_code: str, llm, n: int
    ) -> list[PatchCandidate]:
        """Generate N candidate patches at varying temperatures."""
        patches = []

        # Temperature schedule: 1 greedy + (n-1) sampled
        temps = [0.0] + [0.2 + i * 0.2 for i in range(n - 1)]

        context = location.context or current_code[:3000]

        tasks = [
            self._generate_one_patch(location, issue, context, llm, i, temps[i])
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, PatchCandidate):
                patches.append(r)

        return patches

    async def _generate_one_patch(
        self, location, issue: str, context: str, llm, patch_id: int, temperature: float
    ) -> PatchCandidate:
        """Generate a single patch candidate."""
        prompt = f"""Fix this issue in {location.file_path}:

ISSUE: {issue}

CURRENT CODE:
```python
{context}
```

{f"Focus on function: {location.function_name}" if location.function_name else ""}

Provide your fix as a search/replace edit:
OLD_CODE:
<the exact code to replace>
NEW_CODE:
<the replacement code>
EXPLANATION:
<what this fix does>"""

        try:
            response = await llm.generate_text(
                role="debugger",
                system="You are a surgical code fixer. Produce minimal, correct edits.",
                prompt=prompt,
                temperature=temperature,
                max_tokens=2000,
            )

            old_code, new_code, explanation = _parse_patch_response(response)

            # Validate syntax of the patched file
            full_content = context
            if old_code and old_code in full_content:
                patched = full_content.replace(old_code, new_code, 1)
                try:
                    ast.parse(patched)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
            else:
                syntax_valid = False

            return PatchCandidate(
                id=patch_id,
                file_path=location.file_path,
                old_code=old_code,
                new_code=new_code,
                explanation=explanation,
                syntax_valid=syntax_valid,
            )

        except Exception as e:
            return PatchCandidate(
                id=patch_id,
                file_path=location.file_path,
                old_code="",
                new_code="",
                explanation=f"Generation failed: {e}",
                syntax_valid=False,
            )

    async def _generate_reproduction_tests(
        self, location, issue: str, current_code: str, llm, n: int
    ) -> list[ReproductionTest]:
        """Generate tests that should FAIL on current code, PASS on fixed code."""
        prompt = f"""Write {n} pytest test functions that REPRODUCE this bug:

ISSUE: {issue}
FILE: {location.file_path}

The tests should:
1. FAIL on the current buggy code
2. PASS once the bug is fixed
3. Be minimal and focused on the specific bug

Return each test as a separate function starting with test_.
Import from the module being tested."""

        try:
            response = await llm.generate_text(
                role="tester",
                system="You are a test writer for bug reproduction. Write minimal, focused tests.",
                prompt=prompt,
                temperature=0.2,
                max_tokens=1500,
            )

            tests = []
            import re
            # Extract test functions
            func_pattern = re.compile(r'((?:async\s+)?def\s+test_\w+\s*\([^)]*\).*?)(?=(?:async\s+)?def\s+test_|\Z)', re.DOTALL)
            matches = func_pattern.findall(response)

            for i, match in enumerate(matches[:n]):
                test_code = match.strip()
                name_match = re.search(r'def\s+(test_\w+)', test_code)
                name = name_match.group(1) if name_match else f"test_repro_{i}"
                tests.append(ReproductionTest(id=i, code=test_code, name=name))

            return tests

        except Exception as e:
            logger.debug(f"Reproduction test generation failed: {e}")
            return []

    async def _execute_patch_tests(
        self, codebase, location, patch: PatchCandidate, tests: list[ReproductionTest]
    ) -> set[int]:
        """Apply patch, run tests, return set of passing test IDs."""
        if not patch.syntax_valid or not patch.old_code:
            return set()

        current_code = codebase.get_file_content(location.file_path)
        if not current_code or patch.old_code not in current_code:
            return set()

        patched_code = current_code.replace(patch.old_code, patch.new_code, 1)

        # Build test file
        test_imports = f"import sys\nsys.path.insert(0, '.')\n"
        test_content = test_imports + "\n\n".join(t.code for t in tests)

        passed = set()

        with tempfile.TemporaryDirectory(prefix="belief_patch_") as tmp:
            tmp_path = Path(tmp)

            # Write all codebase files
            for fpath in codebase.files:
                content = codebase.get_file_content(fpath)
                if content:
                    out = tmp_path / fpath
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(content)

            # Overwrite with patched file
            patched_path = tmp_path / location.file_path
            patched_path.write_text(patched_code)

            # Write test file
            test_path = tmp_path / "test_repro.py"
            test_path.write_text(test_content)

            # Run each test individually
            for test in tests:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "pytest", "-xvs",
                         f"test_repro.py::{test.name}"],
                        capture_output=True, text=True,
                        timeout=15, cwd=str(tmp_path),
                        env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)},
                    )
                    if proc.returncode == 0:
                        passed.add(test.id)
                except Exception:
                    continue

        return passed

    def _codet_rank(self, patches: list[PatchCandidate]) -> PatchCandidate | None:
        """CodeT dual execution agreement ranking.

        Score = |solutions_in_group| × |tests_passed_by_group|
        Group patches by identical test-pass signatures.
        Select from the highest-scoring group.
        """
        if not patches:
            return None

        # Group by identical test-pass signature
        groups: dict[frozenset, list[PatchCandidate]] = defaultdict(list)
        for patch in patches:
            groups[patch.tests_passed].append(patch)

        # Score each group
        best_score = -1
        best_group = None

        for sig, group_patches in groups.items():
            score = len(group_patches) * len(sig)
            for p in group_patches:
                p.codet_score = score
            if score > best_score:
                best_score = score
                best_group = group_patches

        if best_group:
            # Return the first patch in the best group (greedy preferred)
            best_group.sort(key=lambda p: p.id)
            return best_group[0]

        # Fallback: return patch that passes the most tests
        return max(patches, key=lambda p: len(p.tests_passed))


def _parse_patch_response(response: str) -> tuple[str, str, str]:
    """Parse OLD_CODE/NEW_CODE/EXPLANATION from LLM response."""
    import re

    old_code = ""
    new_code = ""
    explanation = ""

    # Try structured markers
    old_match = re.search(r'OLD_CODE:\s*\n(.*?)(?=NEW_CODE:)', response, re.DOTALL)
    new_match = re.search(r'NEW_CODE:\s*\n(.*?)(?=EXPLANATION:|$)', response, re.DOTALL)
    exp_match = re.search(r'EXPLANATION:\s*\n(.*?)$', response, re.DOTALL)

    if old_match:
        old_code = old_match.group(1).strip().strip("`").strip()
    if new_match:
        new_code = new_match.group(1).strip().strip("`").strip()
    if exp_match:
        explanation = exp_match.group(1).strip()

    # Fallback: try search/replace JSON format
    if not old_code:
        try:
            import json
            match = re.search(r'\{[^{}]*"old_str"[^{}]*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                old_code = data.get("old_str", "")
                new_code = data.get("new_str", "")
                explanation = data.get("explanation", "")
        except Exception:
            pass

    return old_code, new_code, explanation
