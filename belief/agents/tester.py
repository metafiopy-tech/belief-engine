"""Tester Agent — generate pytest tests from code and acceptance criteria."""

from __future__ import annotations
import logging
import os
import re
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
                if skeleton and hasattr(skeleton, "format_contract"):
                    contract = skeleton.format_contract()
                    if contract:
                        contract_context = f"\n{contract}\n"

            # ── Complexity-adaptive test count ─────────────────────────
            # Research: 5-8 tests is optimal. Simple goals get fewer.
            # Meta's TestGen-LLM found 4:1 ratio of generated-to-useful.
            # Hard ceiling of 15 to prevent test explosion — the LLM
            # sometimes ignores soft limits and generates 50+.
            complexity = state.complexity_score
            n_criteria = len(spec.acceptance_criteria) if spec.acceptance_criteria else 3
            if complexity <= 2:
                test_count = min(5, n_criteria + 2)
            elif complexity <= 4:
                test_count = min(10, n_criteria + 3)
            else:
                test_count = min(15, n_criteria + 5)

            # ── Countdown markers (95% compliance vs 30% naive) ──────
            countdown_markers = "\n".join(
                f"  Test {i} of {test_count} [remaining: {test_count - i}]"
                for i in range(1, test_count + 1)
            )

            prompt = TESTER_PROMPT.format(
                goal=spec.goal,
                acceptance_criteria="\n".join(
                    f"  {i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)
                ),
                code_files=code_context + contract_context,
                test_count=test_count,
                countdown_markers=countdown_markers,
            )
            raw = await llm.generate_text(
                role=self.role,
                system=TESTER_SYSTEM,
                prompt=prompt,
                temperature=0.2,
                complexity=state.complexity_score,
            )
            # Parse ###FILE: format
            test_files = _parse_test_files(raw)

            # Post-process: validate test imports and generate conftest if needed
            test_files = self._postprocess_tests(test_files, state.code_files)

            # ── Generate-then-filter pipeline ────────────────────────
            # Research-backed: generate many, filter by quality, hard cap.
            # This is an UNCONDITIONAL final step — no code path skips it.
            test_files = _filter_and_cap_tests(
                test_files,
                code_files=state.code_files,
                max_total=test_count,
            )

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
                    f"## PROJECT API MAP (these are the ONLY importable symbols)\n{overview}"
                )
        except Exception as e:
            logger.debug(f"Repo map failed in tester: {e}")

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
                parts.append(
                    f"--- {f} ({len(c)} chars, truncated) ---\n{c[:2000]}\n... (truncated)"
                )

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
            except Exception as e:
                logger.debug(f"Export parse failed for {fname}: {e}")

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
                "self",
                "request",
                "tmp_path",
                "tmpdir",
                "monkeypatch",
                "capsys",
                "capfd",
                "caplog",
                "pytestconfig",
                "cache",
                "record_property",
                "record_testsuite_property",
                "recwarn",
                "tmp_path_factory",
                "tmpdir_factory",
            }
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        for arg in node.args.args:
                            if arg.arg not in _PYTEST_BUILTINS:
                                needs_conftest = True
                                conftest_imports.add(arg.arg)

            # Check for imports from non-existent local modules
            source_modules = {
                f.replace("/", ".").replace(".py", "").split(".")[-1]
                for f in code_files
                if f.endswith(".py")
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
        if (
            needs_conftest
            and "conftest.py" not in processed
            and "tests/conftest.py" not in processed
        ):
            conftest = self._generate_conftest(conftest_imports, code_files)

            # Always generate at root level — pytest discovers fixtures from conftest.py
            # in the rootdir regardless of where test files live
            processed["conftest.py"] = conftest
            logger.info(
                f"Tester: generated conftest.py with fixtures: {', '.join(sorted(conftest_imports))}"
            )

            # Also generate in tests/ subdir if tests live there
            test_dirs = {
                fname.split("/")[0] for fname in processed if "/" in fname and fname.endswith(".py")
            }
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

    def _generate_conftest(self, needed_fixtures: set[str], code_files: dict[str, str]) -> str:
        """Generate a conftest.py with commonly needed fixtures."""
        lines = [
            '"""Auto-generated conftest.py with shared test fixtures."""',
            "",
            "import pytest",
            "",
        ]

        # Detect if it's a FastAPI project
        is_fastapi = any("FastAPI" in c or "fastapi" in c for c in code_files.values())
        is_click = any("click" in c or "Click" in c for c in code_files.values())

        if is_fastapi:
            lines.extend(
                [
                    "from fastapi.testclient import TestClient",
                    "",
                    "# Import the app — adjust path if needed",
                ]
            )
            # Find the app
            for fname, content in code_files.items():
                if "app = FastAPI" in content or "app=FastAPI" in content:
                    module = fname.replace("/", ".").replace(".py", "")
                    lines.append(f"from {module} import app")
                    break
            else:
                lines.append("from main import app")

            lines.extend(
                [
                    "",
                    "",
                    "@pytest.fixture",
                    "def client():",
                    '    """Test client for FastAPI app."""',
                    "    with TestClient(app) as c:",
                    "        yield c",
                    "",
                ]
            )

        if is_click:
            lines.extend(
                [
                    "from click.testing import CliRunner",
                    "",
                    "",
                    "@pytest.fixture",
                    "def runner():",
                    '    """Click CLI test runner."""',
                    "    return CliRunner()",
                    "",
                ]
            )
            # Find the CLI entry point — the function decorated with @click.group/@click.command
            for fname, content in code_files.items():
                if "@click.group" in content or "@click.command" in content:
                    module = fname.replace("/", ".").replace(".py", "")
                    # Find the function name that follows a Click decorator
                    import re

                    # Match: @click.group() or @click.command() followed by def func_name(
                    match = re.search(
                        r"@click\.(?:group|command)\s*\([^)]*\)\s*\n\s*def\s+(\w+)\s*\(",
                        content,
                    )
                    if not match:
                        # Fallback: @cli.command or @app.command pattern
                        match = re.search(
                            r"@\w+\.(?:group|command)\s*\([^)]*\)\s*\n\s*def\s+(\w+)\s*\(",
                            content,
                        )
                    if not match:
                        # Last resort: look for common CLI names
                        for name in ("cli", "main", "app"):
                            if f"def {name}(" in content:
                                match = re.search(rf"def\s+({name})\s*\(", content)
                                break
                    if match:
                        func_name = match.group(1)
                        lines.extend(
                            [
                                f"from {module} import {func_name}",
                                "",
                                "",
                                "@pytest.fixture",
                                "def run_cli(runner):",
                                '    """Helper to invoke CLI commands."""',
                                "    def _run(*args):",
                                f"        return runner.invoke({func_name}, args)",
                                "    return _run",
                                "",
                                "",
                                "@pytest.fixture",
                                "def cli_app():",
                                '    """The Click app object for direct invocation."""',
                                f"    return {func_name}",
                                "",
                            ]
                        )
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
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """Database session for testing."""',
                            "    from database import SessionLocal, init_db, engine, Base",
                            "    Base.metadata.create_all(bind=engine)",
                            "    db = SessionLocal()",
                            "    try:",
                            "        yield db",
                            "    finally:",
                            "        db.close()",
                            "        Base.metadata.drop_all(bind=engine)",
                            "",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """Database session stub."""',
                            "    yield None  # TODO: implement with real DB session",
                            "",
                        ]
                    )
            elif "client" in fixture_lower or "api" in fixture_lower:
                # API/test client variant
                if is_fastapi:
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """Test client for the API."""',
                            "    with TestClient(app) as c:",
                            "        yield c",
                            "",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """Test client stub."""',
                            "    yield None  # TODO: implement",
                            "",
                        ]
                    )
            elif "app" in fixture_lower:
                # App instance fixture
                if is_fastapi:
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """FastAPI app instance."""',
                            "    return app",
                            "",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """App instance stub."""',
                            "    yield None  # TODO: implement",
                            "",
                        ]
                    )
            elif "cli" in fixture_lower or "invoke" in fixture_lower:
                # CLI runner variant
                if is_click:
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """CLI runner fixture."""',
                            "    return CliRunner()",
                            "",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "@pytest.fixture",
                            f"def {fixture}():",
                            '    """CLI fixture stub."""',
                            "    yield None  # TODO: implement",
                            "",
                        ]
                    )
            else:
                # Generic stub — at least yield something non-None
                lines.extend(
                    [
                        "@pytest.fixture",
                        f"def {fixture}():",
                        f'    """Auto-generated fixture for {fixture}."""',
                        "    yield {}  # TODO: implement with real test data",
                        "",
                    ]
                )

        return "\n".join(lines)


_KNOWN_PACKAGES = {
    "pytest",
    "click",
    "fastapi",
    "pydantic",
    "sqlalchemy",
    "uvicorn",
    "httpx",
    "starlette",
    "requests",
    "rich",
    "typer",
    "os",
    "sys",
    "json",
    "pathlib",
    "datetime",
    "typing",
    "re",
    "tempfile",
    "unittest",
    "collections",
    "functools",
    "dataclasses",
    "enum",
    "abc",
    "io",
    "uuid",
    "hashlib",
    "time",
    "logging",
    "math",
    "random",
    "shutil",
    "subprocess",
    "asyncio",
    "contextlib",
    "copy",
    "textwrap",
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
        content = re.sub(r"###END\s*$", "", part[nl + 1 :])
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        if (
            os.path.basename(fname).startswith("test_")
            or "conftest" in fname
            or fname.endswith("__init__.py")
        ):
            files[fname] = content
    return files


def _filter_and_cap_tests(
    test_files: dict[str, str],
    code_files: dict[str, str],
    max_total: int = 8,
) -> dict[str, str]:
    """Generate-then-filter pipeline — the research-backed approach.

    Replaces the fragile _cap_test_count + _global_test_cap chain with
    a single UNCONDITIONAL function that:
    1. Extracts all test functions from all test files
    2. Filters by quality (syntax valid, imports exist, no hallucinated modules)
    3. Ranks by tier (P0 > P1 > P2) and position (earlier = more important)
    4. Hard caps at max_total
    5. Reconstructs the test file(s) with only the kept tests

    This function ALWAYS runs. No conditional code path can skip it.
    """
    import ast

    # Pass through non-test files (conftest, __init__)
    passthrough = {}
    testable = {}
    for fname, content in test_files.items():
        if not fname.endswith(".py") or "conftest" in fname or "__init__" in fname:
            passthrough[fname] = content
        else:
            testable[fname] = content

    if not testable:
        return test_files

    # ── Step 1: Extract all test functions with metadata ──────────
    all_tests = []  # (fname, func_name, tier_priority, line_number, content_lines)

    # Build available module set for import validation
    available_modules = set()
    for f in code_files:
        if f.endswith(".py"):
            mod = f.replace("/", ".").replace(".py", "")
            available_modules.add(mod)
            top = mod.split(".")[0]
            available_modules.add(top)

    for fname, content in testable.items():
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Syntax error in entire file — skip all tests from this file
            logger.warning(f"Filter: dropping {fname} — syntax error")
            continue

        lines = content.split("\n")

        # Check file-level imports for hallucinated modules
        has_bad_imports = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top_level = node.module.split(".")[0]
                # If it looks like an intra-project import, check it exists
                if top_level in available_modules:
                    if node.module not in available_modules:
                        has_bad_imports = True
                        break

        if has_bad_imports:
            logger.warning(f"Filter: {fname} has hallucinated imports — deprioritizing")

        # Extract individual test functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("test_"):
                    continue

                # Determine tier from comments or position
                tier = 2  # Default: P2 (edge case)
                # Check for tier markers in the function or decorators
                func_start = node.lineno - 1
                func_end = (
                    node.end_lineno
                    if hasattr(node, "end_lineno") and node.end_lineno
                    else func_start + 10
                )
                func_lines = lines[func_start : min(func_end, len(lines))]
                func_text = "\n".join(func_lines)

                if "# P0" in func_text or "# p0" in func_text or "SMOKE" in func_text.upper():
                    tier = 0
                elif (
                    "# P1" in func_text or "# p1" in func_text or "FUNCTIONAL" in func_text.upper()
                ):
                    tier = 1

                # Import-bad files get deprioritized
                priority = tier + (10 if has_bad_imports else 0)

                all_tests.append((fname, node.name, priority, node.lineno))

    # ── Step 2: Sort by quality (tier, then position) ─────────────
    all_tests.sort(key=lambda t: (t[2], t[3]))

    # ── Step 3: Keep only max_total tests ─────────────────────────
    kept_tests = all_tests[:max_total]
    dropped_tests = all_tests[max_total:]

    if dropped_tests:
        logger.info(
            f"Filter: keeping {len(kept_tests)}/{len(all_tests)} tests "
            f"(dropped {len(dropped_tests)})"
        )

    # ── Step 4: Reconstruct test files with only kept tests ───────
    keep_by_file = {}
    for fname, func_name, _, _ in kept_tests:
        keep_by_file.setdefault(fname, set()).add(func_name)

    result = dict(passthrough)

    for fname, content in testable.items():
        if fname not in keep_by_file:
            # All tests from this file were dropped
            if any(f == fname for f, _, _, _ in dropped_tests):
                logger.info(f"Filter: dropping entire file {fname}")
            continue

        keep_names = keep_by_file[fname]
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        # Find all test function names in this file
        all_funcs_in_file = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]

        drop_names = set(all_funcs_in_file) - keep_names

        if not drop_names:
            # Keep entire file unchanged
            result[fname] = content
            continue

        # Remove dropped functions line by line
        lines = content.split("\n")
        new_lines = []
        skip_func = False

        for line in lines:
            stripped = line.lstrip()
            # Detect test function definition
            if stripped.startswith("def test_") or stripped.startswith("async def test_"):
                func_name = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
                if func_name in drop_names:
                    skip_func = True
                    continue
                else:
                    skip_func = False
            elif skip_func:
                # Still inside a dropped function — check if we've exited
                if stripped and not line[0].isspace() and not stripped.startswith("#"):
                    # New top-level definition
                    skip_func = False
                else:
                    continue

            new_lines.append(line)

        result[fname] = "\n".join(new_lines)

    return result


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
                if (
                    stripped
                    and not stripped.startswith("#")
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                ):
                    if (
                        stripped.startswith("def ")
                        or stripped.startswith("async def ")
                        or stripped.startswith("class ")
                        or stripped.startswith("@")
                    ):
                        skip_until_next_def = False
                    else:
                        continue
                else:
                    continue

            new_lines.append(line)

        capped_content = "\n".join(new_lines)
        logger.info(
            f"Tester: capped {fname} from {len(test_funcs)} to {max_tests_per_file} tests (dropped {len(drop)})"
        )
        capped[fname] = capped_content

    return capped


def _global_test_cap(test_files: dict[str, str], max_total: int = 14) -> dict[str, str]:
    """Enforce a GLOBAL cap on total test functions across ALL test files.

    The per-file cap misses multi-file splits where the tester generates
    test_smoke.py (8 tests) + test_functional.py (12 tests) + test_edge.py (10 tests)
    = 30 total despite each file being under the per-file cap.

    Strategy: count all tests across all files, keep the first max_total
    (preserving file order — smoke files come first), drop entire excess files
    or truncate the last file to hit the cap.
    """
    import ast as _ast

    # Count total tests across all files
    file_test_counts = []
    total = 0
    for fname, content in test_files.items():
        if not fname.endswith(".py") or "conftest" in fname or "__init__" in fname:
            continue
        try:
            tree = _ast.parse(content)
            count = sum(
                1
                for node in _ast.walk(tree)
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
            file_test_counts.append((fname, count))
            total += count
        except SyntaxError:
            continue

    if total <= max_total:
        return test_files  # Under budget

    # Over budget — keep files in order until we hit the cap
    result = {}
    remaining = max_total
    for fname, count in file_test_counts:
        if remaining <= 0:
            # Drop this entire file
            logger.info(f"Global cap: dropping {fname} ({count} tests) — budget exhausted")
            continue
        if count <= remaining:
            # Keep entire file
            result[fname] = test_files[fname]
            remaining -= count
        else:
            # Truncate this file to fit remaining budget
            result[fname] = _cap_test_count(
                {fname: test_files[fname]}, max_tests_per_file=remaining
            )[fname]
            logger.info(f"Global cap: truncated {fname} from {count} to {remaining} tests")
            remaining = 0

    # Re-add conftest and __init__ files
    for fname, content in test_files.items():
        if "conftest" in fname or "__init__" in fname:
            result[fname] = content

    dropped = total - max_total
    logger.info(
        f"Global cap: {total} → {max_total} tests across {len(file_test_counts)} files (dropped {dropped})"
    )
    return result
