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
            # Build code context with repo map
            code_context = self._build_code_context(state)

            # Inject API contract if available (Move 5)
            contract_context = ""
            skeleton = state.skeleton_artifact
            if skeleton:
                from belief.models.skeleton import SkeletonArtifact
                if isinstance(skeleton, dict):
                    try:
                        skeleton = SkeletonArtifact.model_validate(skeleton)
                    except Exception:
                        skeleton = None
                if skeleton and hasattr(skeleton, 'format_contract'):
                    contract = skeleton.format_contract()
                    if contract:
                        contract_context = f"\n{contract}\n"

            prompt = TESTER_PROMPT.format(
                goal=spec.goal,
                acceptance_criteria="\n".join(f"  {i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)),
                code_files=code_context + contract_context,
            )
            raw = await llm.generate_text(
                role=self.role, system=TESTER_SYSTEM, prompt=prompt,
                temperature=0.2, complexity=state.complexity_score,
            )
            # Parse ###FILE: format
            test_files = _parse_test_files(raw)

            # Post-process: validate test imports and generate conftest if needed
            test_files = self._postprocess_tests(test_files, state.code_files)

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

        Uses the repo map to provide a definitive list of importable symbols.
        This prevents the tester from importing non-existent modules or classes.
        """
        parts = []

        # 1. Repo map — definitive list of what's importable (Move 4)
        try:
            from belief.agents.repo_map import RepoMap
            repo_map = RepoMap.from_code_files(state.code_files)
            overview = repo_map.format_overview(max_tokens=2000)
            if overview:
                parts.append(
                    "## PROJECT API MAP (these are the ONLY importable symbols)\n"
                    f"{overview}"
                )
        except Exception:
            pass

        # 2. Extract import paths from all built code files via AST
        import_map = self._extract_imports_and_exports(state.code_files)
        if import_map:
            parts.append("## IMPORTABLE SYMBOLS (use these exact paths)")
            for fname, exports in sorted(import_map.items()):
                if exports:
                    module = fname.replace("/", ".").replace(".py", "")
                    parts.append(f"  from {module} import {', '.join(exports)}")

        # 3. Full code for key files (entry points, routes — under 3000 chars)
        parts.append("\n## CODE FILES")
        for f, c in sorted(state.code_files.items()):
            if f == "requirements.txt":
                continue
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

    def _postprocess_tests(
        self, test_files: dict[str, str], code_files: dict[str, str]
    ) -> dict[str, str]:
        """Validate and fix generated test files.

        1. Remove test files with syntax errors
        2. Check for conftest.py imports — generate conftest if needed
        3. Remove tests that import non-existent modules
        4. Ensure tests/__init__.py exists
        """
        import ast

        processed = {}
        needs_conftest = False
        conftest_imports = set()

        for fname, content in test_files.items():
            # Skip non-Python files
            if not fname.endswith(".py"):
                processed[fname] = content
                continue

            # Remove files with syntax errors
            try:
                tree = ast.parse(content)
            except SyntaxError:
                logger.warning(f"Tester: removed {fname} (syntax error)")
                continue

            # Check for conftest imports (explicit)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "conftest":
                    needs_conftest = True
                    for alias in node.names:
                        conftest_imports.add(alias.name)

            # Check for pytest fixture usage (implicit via function parameters)
            # Tests like `def test_health(client):` need a conftest with a `client` fixture
            _KNOWN_FIXTURES = {"client", "runner", "run_cli", "db", "session", "app", "tmp_path", "tmpdir"}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        for arg in node.args.args:
                            if arg.arg in _KNOWN_FIXTURES:
                                # tmp_path and tmpdir are built-in pytest fixtures
                                if arg.arg not in ("tmp_path", "tmpdir"):
                                    needs_conftest = True
                                    conftest_imports.add(arg.arg)

            # Check for imports from non-existent local modules
            valid = True
            source_modules = {
                f.replace("/", ".").replace(".py", "").split(".")[-1]
                for f in code_files if f.endswith(".py")
            }
            source_modules.add("conftest")  # Will be generated if needed

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                    # Check if it's a local module that doesn't exist
                    if root not in source_modules and root not in _KNOWN_PACKAGES:
                        logger.warning(f"Tester: {fname} imports non-existent '{node.module}'")

            processed[fname] = content

        # Generate conftest.py if tests need fixtures
        if needs_conftest and "conftest.py" not in processed and "tests/conftest.py" not in processed:
            conftest = self._generate_conftest(conftest_imports, code_files)

            # Always generate at root level — pytest discovers fixtures from conftest.py
            # in the rootdir regardless of where test files live
            processed["conftest.py"] = conftest
            logger.info(f"Tester: generated conftest.py with fixtures: {', '.join(sorted(conftest_imports))}")

            # Also generate in tests/ subdir if tests live there
            test_dirs = {fname.split("/")[0] for fname in processed if "/" in fname and fname.endswith(".py")}
            for d in test_dirs:
                if d.startswith("test"):
                    subdir_path = f"{d}/conftest.py"
                    if subdir_path not in processed:
                        processed[subdir_path] = conftest

        # Ensure __init__.py
        test_dirs = {fname.rsplit("/", 1)[0] for fname in processed if "/" in fname}
        for d in test_dirs:
            init_path = f"{d}/__init__.py"
            if init_path not in processed:
                processed[init_path] = ""

        return processed

    def _generate_conftest(
        self, needed_fixtures: set[str], code_files: dict[str, str]
    ) -> str:
        """Generate a conftest.py with commonly needed fixtures."""
        lines = ['"""Auto-generated conftest.py with shared test fixtures."""', "",
                 "import pytest", ""]

        # Detect if it's a FastAPI project
        is_fastapi = any("FastAPI" in c or "fastapi" in c for c in code_files.values())
        is_click = any("click" in c or "Click" in c for c in code_files.values())

        if is_fastapi:
            lines.extend([
                "from fastapi.testclient import TestClient",
                "",
                "# Import the app — adjust path if needed",
            ])
            # Find the app
            for fname, content in code_files.items():
                if "app = FastAPI" in content or "app=FastAPI" in content:
                    module = fname.replace("/", ".").replace(".py", "")
                    lines.append(f"from {module} import app")
                    break
            else:
                lines.append("from main import app")

            lines.extend([
                "",
                "",
                "@pytest.fixture",
                "def client():",
                '    """Test client for FastAPI app."""',
                "    with TestClient(app) as c:",
                "        yield c",
                "",
            ])

        if is_click:
            lines.extend([
                "from click.testing import CliRunner",
                "",
                "",
                "@pytest.fixture",
                "def runner():",
                '    """Click CLI test runner."""',
                "    return CliRunner()",
                "",
            ])
            # Find the CLI entry point
            for fname, content in code_files.items():
                if "@click.group" in content or "@click.command" in content:
                    module = fname.replace("/", ".").replace(".py", "")
                    # Find the group/command name
                    import re
                    match = re.search(r"def (\w+)\(", content)
                    if match:
                        func_name = match.group(1)
                        lines.extend([
                            f"from {module} import {func_name}",
                            "",
                            "",
                            "@pytest.fixture",
                            "def run_cli(runner):",
                            '    """Helper to invoke CLI commands."""',
                            f"    def _run(*args):",
                            f"        return runner.invoke({func_name}, args)",
                            "    return _run",
                            "",
                        ])
                    break

        # Add any other needed fixtures as stubs
        existing = {"client", "runner", "run_cli"}
        for fixture in needed_fixtures:
            if fixture not in existing:
                lines.extend([
                    f"@pytest.fixture",
                    f"def {fixture}():",
                    f'    """Auto-generated fixture stub for {fixture}."""',
                    f"    pass  # TODO: implement",
                    "",
                ])

        return "\n".join(lines)


_KNOWN_PACKAGES = {
    "pytest", "click", "fastapi", "pydantic", "sqlalchemy", "uvicorn",
    "httpx", "starlette", "requests", "rich", "typer", "os", "sys",
    "json", "pathlib", "datetime", "typing", "re", "tempfile", "unittest",
    "collections", "functools", "dataclasses", "enum", "abc", "io",
    "uuid", "hashlib", "time", "logging", "math", "random", "shutil",
    "subprocess", "asyncio", "contextlib", "copy", "textwrap",
}

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
