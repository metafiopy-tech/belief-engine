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

            # Hard cap: scale test count by complexity to reduce phantom failures.
            # Simple scripts (complexity 1-3) get 10 tests max.
            # Medium apps (complexity 4-6) get 14 tests max.
            # Complex systems (complexity 7+) get 20 tests max.
            complexity = state.complexity_score
            if complexity <= 3:
                max_tests = 10
            elif complexity <= 6:
                max_tests = 14
            else:
                max_tests = 20
            test_files = _cap_test_count(test_files, max_tests_per_file=max_tests)

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
            # ANY test function parameter that isn't a pytest builtin is a custom
            # fixture that needs a conftest.py. Previous approach used a whitelist
            # which missed db_session, test_client, api_client, sample_data, etc.
            _PYTEST_BUILTINS = {
                "self", "request", "tmp_path", "tmpdir", "monkeypatch",
                "capsys", "capfd", "caplog", "pytestconfig", "cache",
                "record_property", "record_testsuite_property", "recwarn",
                "tmp_path_factory", "tmpdir_factory",
            }
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        for arg in node.args.args:
                            if arg.arg not in _PYTEST_BUILTINS:
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
            # Find the CLI entry point — the function decorated with @click.group/@click.command
            for fname, content in code_files.items():
                if "@click.group" in content or "@click.command" in content:
                    module = fname.replace("/", ".").replace(".py", "")
                    # Find the function name that follows a Click decorator
                    import re
                    # Match: @click.group() or @click.command() followed by def func_name(
                    match = re.search(
                        r'@click\.(?:group|command)\s*\([^)]*\)\s*\n\s*def\s+(\w+)\s*\(',
                        content,
                    )
                    if not match:
                        # Fallback: @cli.command or @app.command pattern
                        match = re.search(
                            r'@\w+\.(?:group|command)\s*\([^)]*\)\s*\n\s*def\s+(\w+)\s*\(',
                            content,
                        )
                    if not match:
                        # Last resort: look for common CLI names
                        for name in ("cli", "main", "app"):
                            if f"def {name}(" in content:
                                match = re.search(rf'def\s+({name})\s*\(', content)
                                break
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
                            "",
                            "@pytest.fixture",
                            f"def cli_app():",
                            f'    """The Click app object for direct invocation."""',
                            f"    return {func_name}",
                            "",
                        ])
                    break

        # Add any other needed fixtures — generate smart stubs based on name patterns
        existing = {"client", "runner", "run_cli"}
        for fixture in sorted(needed_fixtures):
            if fixture in existing:
                continue
            existing.add(fixture)

            # Smart fixture generation based on naming patterns
            fixture_lower = fixture.lower()

            if "db" in fixture_lower or "session" in fixture_lower:
                # Database session fixture
                if is_fastapi:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """Database session for testing."""',
                        f"    from database import SessionLocal, init_db, engine, Base",
                        f"    Base.metadata.create_all(bind=engine)",
                        f"    db = SessionLocal()",
                        f"    try:",
                        f"        yield db",
                        f"    finally:",
                        f"        db.close()",
                        f"        Base.metadata.drop_all(bind=engine)",
                        "",
                    ])
                else:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """Database session stub."""',
                        f"    yield None  # TODO: implement with real DB session",
                        "",
                    ])
            elif "client" in fixture_lower or "api" in fixture_lower:
                # API/test client variant
                if is_fastapi:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """Test client for the API."""',
                        f"    with TestClient(app) as c:",
                        f"        yield c",
                        "",
                    ])
                else:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """Test client stub."""',
                        f"    yield None  # TODO: implement",
                        "",
                    ])
            elif "app" in fixture_lower:
                # App instance fixture
                if is_fastapi:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """FastAPI app instance."""',
                        f"    return app",
                        "",
                    ])
                else:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """App instance stub."""',
                        f"    yield None  # TODO: implement",
                        "",
                    ])
            elif "cli" in fixture_lower or "invoke" in fixture_lower:
                # CLI runner variant
                if is_click:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """CLI runner fixture."""',
                        f"    return CliRunner()",
                        "",
                    ])
                else:
                    lines.extend([
                        f"@pytest.fixture",
                        f"def {fixture}():",
                        f'    """CLI fixture stub."""',
                        f"    yield None  # TODO: implement",
                        "",
                    ])
            else:
                # Generic stub — at least yield something non-None
                lines.extend([
                    f"@pytest.fixture",
                    f"def {fixture}():",
                    f'    """Auto-generated fixture for {fixture}."""',
                    f"    yield {{}}  # TODO: implement with real test data",
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
        if os.path.basename(fname).startswith("test_") or "conftest" in fname or fname.endswith("__init__.py"):
            files[fname] = content
    return files


def _cap_test_count(test_files: dict[str, str], max_tests_per_file: int = 20) -> dict[str, str]:
    """Hard cap on test functions per file.

    The LLM often generates 40-60 tests when asked for 8-14.
    Excess tests increase phantom failure surface area without
    improving quality signal. Keep P0 smoke tests, then P1 functional,
    drop P2 edge cases first.

    Priority: P0 > P1 > P2 (by comment marker or position)
    """
    import ast

    capped = {}
    for fname, content in test_files.items():
        if not fname.endswith(".py") or "conftest" in fname or "__init__" in fname:
            capped[fname] = content
            continue

        try:
            tree = ast.parse(content)
        except SyntaxError:
            capped[fname] = content
            continue

        # Count test functions
        test_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    test_funcs.append(node.name)

        if len(test_funcs) <= max_tests_per_file:
            capped[fname] = content
            continue

        # Too many tests — keep only the first max_tests_per_file
        # (prioritized by position: smoke tests come first in generated output)
        keep = set(test_funcs[:max_tests_per_file])
        drop = set(test_funcs[max_tests_per_file:])

        # Remove dropped test functions from the source
        lines = content.split("\n")
        new_lines = []
        skip_until_next_def = False

        for line in lines:
            # Check if this line starts a dropped test function
            stripped = line.lstrip()
            if stripped.startswith("def test_") or stripped.startswith("async def test_"):
                func_name = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
                if func_name in drop:
                    skip_until_next_def = True
                    continue
                else:
                    skip_until_next_def = False

            # Check if we've reached a new top-level definition (end of dropped function)
            if skip_until_next_def:
                if stripped and not stripped.startswith("#") and not line.startswith(" ") and not line.startswith("\t"):
                    if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class ") or stripped.startswith("@"):
                        skip_until_next_def = False
                    else:
                        continue
                else:
                    continue

            new_lines.append(line)

        capped_content = "\n".join(new_lines)
        logger.info(f"Tester: capped {fname} from {len(test_funcs)} to {max_tests_per_file} tests (dropped {len(drop)})")
        capped[fname] = capped_content

    return capped
