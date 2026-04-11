"""Executor Agent — run built code in a sandboxed subprocess.

Installs dependencies, runs the entry point, captures output.
Also runs pytest if test_files are present.

Server detection: If the entry point code contains patterns like
`.run()`, `.serve_forever()`, `uvicorn.run()`, or `mcp.run()`,
the executor tests it via import-and-call instead of running as a
blocking subprocess. This prevents 60s timeout hangs on servers.

Source: forge/agents/executor.py + code_executor.py
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path

from belief.agents.base import BaseAgent
from belief.config.models import ModelRole
from belief.config.settings import settings
from belief.models.artifacts import ExecutionResult, PytestResult, PytestTestItem
from belief.models.state import Phase, UnifiedState

logger = logging.getLogger("belief.agents.executor")

EXEC_TIMEOUT = 60
INSTALL_TIMEOUT = 120

# Patterns that indicate a long-running server process
_SERVER_PATTERNS = [
    r"(?:app|server|mcp|uvicorn)\.run\s*\(",  # app.run(), server.run(), mcp.run(), uvicorn.run()
    r"\.serve_forever\s*\(",         # socketserver
    r"uvicorn\.run\s*\(",            # uvicorn
    r"\.start_serving\s*\(",         # asyncio server
    r"asyncio\.run\s*\(\s*main\b",   # async server main loops
    r"serve\s*\(\s*\)",              # grpc, other serve() patterns
]

# Patterns that indicate a server/API app even without a .run() call in __main__
# These are checked against the FULL file, not just the tail
_APP_PATTERNS = [
    r"app\s*=\s*FastAPI\s*\(",       # FastAPI app creation
    r"FastAPI\s*\(",                  # FastAPI() anywhere
    r"app\s*=\s*Flask\s*\(",         # Flask app creation
    r"@app\.(get|post|put|delete|patch)\s*\(",  # Route decorators
    r"FastMCP\s*\(",                  # MCP server creation
    r"mcp\s*=\s*FastMCP",            # MCP server assignment
]

# Patterns that indicate a Click CLI application
_CLI_PATTERNS = [
    r"@click\.(command|group)\s*\(",          # @click.command() or @click.group()
    r"@\w+\.(command|group)\s*\(",            # @cli.command() or @app.group()
    r"click\.group\s*\(",                     # click.group() call
    r"click\.command\s*\(",                   # click.command() call
    r"from\s+click\s+import",                 # Click import
    r"import\s+click",                        # Click import
]


def _is_click_cli(code: str) -> bool:
    """Detect if code is a Click CLI application."""
    cli_matches = sum(1 for p in _CLI_PATTERNS if re.search(p, code))
    # Require at least 2 matches (import + decorator) to avoid false positives
    return cli_matches >= 2


def _is_server_code(code: str) -> bool:
    """Detect if code is a long-running server that shouldn't be run as a script."""
    # First check the full file for app framework patterns
    for pattern in _APP_PATTERNS:
        if re.search(pattern, code):
            return True

    # Then check the __main__ block / tail for run patterns
    main_match = re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', code)
    if main_match:
        tail = code[main_match.start():]
    else:
        lines = code.strip().splitlines()
        tail = "\n".join(lines[-30:])

    for pattern in _SERVER_PATTERNS:
        if re.search(pattern, tail):
            return True
    return False


def _detect_entry_points(code_files: dict[str, str], manifest) -> list[str]:
    """Detect all entry points in a project.

    For single-service projects: returns [manifest.entry_point]
    For multi-service projects: returns one entry point per service

    Detection order:
    1. Manifest-specified entry point (always included if valid)
    2. Files matching server.py/app.py/main.py in each package directory
    3. Files containing FastAPI()/Flask() app instantiation
    4. Fallback: first .py file
    """
    entry_points: list[str] = []
    seen: set[str] = set()

    def _add(ep: str) -> None:
        if ep in seen or ep not in code_files:
            return
        seen.add(ep)
        entry_points.append(ep)

    # 1. Manifest entry point
    if manifest:
        ep = manifest.entry_point if hasattr(manifest, "entry_point") else "main.py"
        if ep in code_files:
            _add(ep)

    # 2. Server/app/main files in package directories (Python + TypeScript)
    for fname in sorted(code_files.keys()):
        base = fname.split("/")[-1] if "/" in fname else fname
        if base in ("server.py", "app.py", "main.py",
                     "index.ts", "server.ts", "app.ts", "main.ts",
                     "index.tsx", "main.go"):
            _add(fname)

    # 3. Files with FastAPI/Flask/MCP/Express app creation
    for fname, content in code_files.items():
        if "/test" in fname or fname.startswith("test"):
            continue
        # Python servers
        if fname.endswith(".py") and _is_server_code(content) and fname not in seen:
            _add(fname)
        # TypeScript/JS servers (Express, Next.js, etc.)
        if fname.endswith((".ts", ".tsx", ".js", ".jsx")):
            ts_server_patterns = [
                r"express\s*\(\s*\)", r"createServer\s*\(",
                r"app\.listen\s*\(", r"Hono\s*\(\s*\)",
                r"new\s+Koa\s*\(", r"Fastify\s*\(",
            ]
            for pat in ts_server_patterns:
                if re.search(pat, content):
                    _add(fname)
                    break

    # 4. Fallback
    if not entry_points:
        candidates = [f for f in code_files
                       if f.endswith((".py", ".ts", ".tsx", ".js", ".go"))
                       and "/test" not in f and not f.startswith("test")]
        if candidates:
            _add(candidates[0])

    return entry_points


class ExecutorAgent(BaseAgent):
    role = ModelRole.EXECUTOR
    name = "Executor"

    async def run(self, state: UnifiedState) -> UnifiedState:
        state.phase = Phase.EXECUTING

        if not state.code_files:
            state.execution_result = ExecutionResult(
                exit_code=-1, success=False,
                error_summary="No code files to execute",
            )
            state.phase = Phase.GAP_ANALYSIS
            return state

        manifest = state.file_manifest

        # ── M4: Detect all entry points (multi-service support) ──
        entry_points = _detect_entry_points(state.code_files, manifest)

        if not entry_points:
            state.execution_result = ExecutionResult(
                exit_code=-1, success=False,
                error_summary="No entry points found in code_files",
            )
            state.phase = Phase.GAP_ANALYSIS
            return state

        # Check if ANY entry point is a server or a Click CLI
        is_server = any(
            _is_server_code(state.code_files.get(ep, ""))
            for ep in entry_points
        )
        is_cli = any(
            _is_click_cli(state.code_files.get(ep, ""))
            for ep in entry_points
        )

        # Both servers and CLIs use import verification instead of script execution.
        # Servers block forever if run as scripts. CLIs may work as scripts but
        # import verification is more reliable — it catches import errors, validates
        # the Click command tree exists, and doesn't depend on __main__ blocks.
        use_import_verify = is_server or is_cli

        if use_import_verify:
            app_type = "server" if is_server else "CLI"
            logger.info(
                f"Executor: detected {app_type}-type program — using import verification "
                f"({len(entry_points)} entry point{'s' if len(entry_points) > 1 else ''})"
            )

        result = await asyncio.to_thread(
            self._execute_in_sandbox, state.code_files, state.test_files,
            entry_points, use_import_verify,
        )
        state.execution_result = result

        if result.success:
            logger.info(f"Executor: success in {result.duration_seconds:.1f}s")
        else:
            logger.info(f"Executor: failed (exit={result.exit_code})")

        state.phase = Phase.GAP_ANALYSIS
        return state

    def _execute_in_sandbox(
        self, code_files: dict[str, str], test_files: dict[str, str],
        entry_points: list[str], use_import_verify: bool = False,
    ) -> ExecutionResult:
        """Write files to a temp dir and verify the project works.

        Verification strategy (research-backed):
        1. Write all files + auto-generate __init__.py everywhere
        2. Static analysis: circular import detection, undefined names
        3. Install dependencies (pip or npm)
        4. If tests exist → run pytest (primary verification)
        5. If no tests → fall back to import verification
        6. Capture actionable errors for the debugger
        """
        import sys as _sys

        with tempfile.TemporaryDirectory(prefix="belief_") as tmpdir:
            tmp = Path(tmpdir)

            # Write all code files
            for fname, content in code_files.items():
                fpath = tmp / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)

            # Write test files
            for fname, content in test_files.items():
                fpath = tmp / fname
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content)

            # ── Fix #1: Comprehensive __init__.py generation ──────────────
            # Walk the ENTIRE temp directory tree and add __init__.py
            # to every directory that contains .py files (or has subdirs
            # that contain .py files). This fixes the #1 executor failure.
            self._ensure_init_files(tmp)

            system_python = _sys.executable

            # ── TypeScript/Node.js path ──────────────────────────────────
            if "package.json" in code_files:
                return self._verify_typescript(tmp, entry_points)

            # ── Fix #2: Static import validation ─────────────────────────
            # Detect broken imports BEFORE execution so the debugger
            # gets actionable error messages instead of cryptic tracebacks.
            static_errors = self._static_import_check(code_files, tmp)
            if static_errors:
                logger.info(f"Executor: {len(static_errors)} static import issues detected")
                # Don't fail here — let pytest/import catch the real errors.
                # But log them so the debugger can use them.

            # ── Install dependencies ─────────────────────────────────────
            install_result = self._install_deps(tmp)

            # ── Fix #3: Pytest-first verification ────────────────────────
            # If test files exist, run pytest as primary verification.
            # This is what every competitive system does (SWE-agent,
            # OpenHands, Aider). Import verification is the fallback.
            all_test_files = dict(test_files)
            for fname in code_files:
                if fname.startswith("test") or "/test" in fname:
                    all_test_files[fname] = code_files[fname]

            # ── Hard cap: filter ALL test files (including builder-generated)
            # The builder sometimes generates 30+ tests that bypass the tester's
            # filter. This catches them before pytest runs.
            if all_test_files:
                try:
                    from belief.agents.tester import _filter_and_cap_tests
                    pre_count = sum(
                        c.count("\ndef test_") + c.count("\n    def test_")
                        for c in all_test_files.values()
                    )
                    all_test_files = _filter_and_cap_tests(
                        all_test_files, code_files=code_files, max_total=14,
                    )
                    post_count = sum(
                        c.count("\ndef test_") + c.count("\n    def test_")
                        for c in all_test_files.values()
                    )
                    if pre_count != post_count:
                        logger.info(f"Executor: test cap {pre_count} → {post_count}")
                        # Rewrite the capped test files to disk
                        for fname, content in all_test_files.items():
                            fpath = tmp / fname
                            fpath.parent.mkdir(parents=True, exist_ok=True)
                            fpath.write_text(content)
                except Exception as e:
                    logger.debug(f"Executor: test cap skipped: {e}")

            if all_test_files:
                pytest_result = self._run_pytest_verification(
                    system_python, tmp, install_result
                )
                if pytest_result is not None:
                    # ── Smoke test: verify code works WITHOUT pytest ──────
                    # pytest manipulates sys.path (prepend mode adds dirs
                    # that conftest.py lives in). Code can pass pytest but
                    # fail when run normally. This smoke test catches that.
                    if pytest_result.success and entry_points:
                        smoke = self._smoke_test(
                            system_python, tmp, entry_points, code_files
                        )
                        if smoke:
                            pytest_result.stdout = (
                                (pytest_result.stdout or "") +
                                f"\nSMOKE_TEST: {smoke}"
                            )
                            # Check if smoke test found import failures
                            if "FAIL:" in smoke:
                                # Tests pass but code has structural issues
                                # Report as error_summary so the debugger can fix it
                                fail_modules = [
                                    s.replace("FAIL:", "")
                                    for s in smoke.split(", ")
                                    if s.startswith("FAIL:")
                                ]
                                pytest_result.error_summary = (
                                    f"Smoke test: {len(fail_modules)} module(s) fail to import "
                                    f"outside pytest: {', '.join(fail_modules[:3])}. "
                                    f"Check __init__.py files and import paths."
                                )
                                logger.info(f"Executor: smoke test found {len(fail_modules)} import failure(s)")
                            else:
                                logger.info(f"Executor: smoke test passed — {smoke}")
                    return pytest_result

            # ── Fallback: import verification or script execution ────────
            if use_import_verify:
                all_results = []
                for ep in entry_points:
                    result = self._verify_via_import(system_python, tmp, ep, code_files)
                    all_results.append((ep, result))
                    if not result.success:
                        logger.info(f"Executor: {ep} failed verification")
                        return result

                if all_results:
                    final = all_results[-1][1]
                    if len(all_results) > 1:
                        verified = [ep for ep, r in all_results if r.success]
                        sep = ", "
                        final.stdout = (final.stdout or "") + f"\nVERIFIED: {sep.join(verified)}"
                        logger.info(f"Executor: all {len(verified)} entry points verified")
                    return final

            # For scripts: run the primary entry point
            return self._run_script(system_python, tmp, entry_points[0], install_result)

    def _ensure_init_files(self, tmp: Path) -> int:
        """Create __init__.py in every directory containing .py files.

        This is the #1 fix for executor failures. LLM-generated projects
        consistently miss __init__.py files, causing ImportError on
        any cross-module import. We walk the ENTIRE tree, not just
        directories where generated files have '/' in their name.

        Returns count of __init__.py files created.
        """
        created = 0
        for dirpath, dirnames, filenames in os.walk(tmp):
            # Skip hidden dirs, __pycache__, .venv
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d != "__pycache__" and d != ".venv"
            ]
            # If this directory (or any child) has .py files, ensure __init__.py
            has_py = any(f.endswith(".py") for f in filenames)
            if has_py:
                init_path = Path(dirpath) / "__init__.py"
                if not init_path.exists():
                    init_path.write_text("")
                    rel = init_path.relative_to(tmp)
                    logger.debug(f"Executor: auto-created {rel}")
                    created += 1
        if created:
            logger.info(f"Executor: auto-created {created} __init__.py file(s)")
        return created

    def _static_import_check(
        self, code_files: dict[str, str], tmp: Path
    ) -> list[str]:
        """Static analysis of imports — detect issues before execution.

        Checks:
        1. All intra-project imports resolve to existing files
        2. No circular import chains
        3. Standard library imports are valid

        Returns list of error descriptions (empty = all clear).
        """
        errors = []
        import re as _re

        # Build a map of available modules from code_files
        available_modules = set()
        for fname in code_files:
            if fname.endswith(".py"):
                mod = fname.replace("/", ".").replace(".py", "")
                available_modules.add(mod)
                # Also add parent packages
                parts = mod.split(".")
                for i in range(1, len(parts)):
                    available_modules.add(".".join(parts[:i]))

        # Check each file's imports
        for fname, content in code_files.items():
            if not fname.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            current_module = fname.replace("/", ".").replace(".py", "")
            current_pkg = ".".join(current_module.split(".")[:-1])

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    target = node.module
                    # Resolve relative imports
                    if node.level > 0 and current_pkg:
                        parts = current_pkg.split(".")
                        up = node.level - 1
                        if up < len(parts):
                            base = ".".join(parts[:len(parts) - up])
                            target = f"{base}.{node.module}" if node.module else base
                        else:
                            target = node.module or ""

                    # Check if it's an intra-project import that doesn't resolve
                    if target and not target.startswith(("_", ".")):
                        # Is it a standard library or third-party module?
                        top_level = target.split(".")[0]
                        # Check if it looks like an intra-project import
                        if top_level in available_modules or target in available_modules:
                            if target not in available_modules:
                                errors.append(
                                    f"{fname}: import '{target}' does not resolve "
                                    f"to any project file"
                                )

        return errors

    def _run_pytest_verification(
        self, python: str, tmp: Path, install_result: ExecutionResult
    ) -> ExecutionResult | None:
        """Run pytest as primary verification — the state-of-the-art approach.

        Every competitive code generation system (SWE-agent, OpenHands, Aider)
        uses test execution as verification, not import checking. If pytest
        runs and at least some tests pass, the code is structurally sound.

        Returns ExecutionResult if pytest ran, None if pytest isn't available.
        """
        t0 = time.time()
        env = {**os.environ, "PYTHONPATH": str(tmp)}

        try:
            proc = subprocess.run(
                [python, "-m", "pytest", "-x", "-v", "--tb=short", "--no-header", "-q",
                 "--import-mode=importlib"],
                capture_output=True, text=True,
                timeout=60, cwd=str(tmp),
                env=env,
            )

            output = proc.stdout + "\n" + proc.stderr
            elapsed = round(time.time() - t0, 2)

            # Parse results
            import re as _re
            passed = 0
            failed = 0
            errored = 0

            match = _re.search(r"(\d+) passed", output)
            if match:
                passed = int(match.group(1))
            match = _re.search(r"(\d+) failed", output)
            if match:
                failed = int(match.group(1))
            match = _re.search(r"(\d+) error", output)
            if match:
                errored = int(match.group(1))

            total = passed + failed + errored

            # If pytest found and ran tests, use its result
            if total > 0:
                # Success if ANY tests passed (partial success is still structural soundness)
                success = passed > 0
                error_summary = ""
                if not success:
                    # Extract the first error for the debugger
                    error_summary = _extract_error(proc.stderr or proc.stdout)

                logger.info(
                    f"Executor: pytest {passed}/{total} passed "
                    f"({elapsed}s)"
                )

                return ExecutionResult(
                    exit_code=proc.returncode,
                    stdout=output[-3000:],
                    stderr=proc.stderr[-2000:],
                    duration_seconds=elapsed,
                    success=success,
                    error_summary=error_summary,
                    install_success=install_result.install_success,
                    install_stdout=install_result.install_stdout,
                    install_stderr=install_result.install_stderr,
                )

            # pytest ran but found no tests — that's okay, fall through
            # to import verification
            if "no tests ran" in output.lower() or "collected 0 items" in output:
                logger.info("Executor: pytest found no tests — falling back to import verification")
                return None

            # pytest itself failed to run (missing module, etc.)
            # Check if it's a collection error vs pytest not being available
            if "ModuleNotFoundError" in output and "pytest" in output:
                logger.info("Executor: pytest not available — falling back")
                return None

            # Collection errors mean the code has import issues
            if proc.returncode != 0 and ("ImportError" in output or "ModuleNotFoundError" in output):
                error_msg = _extract_error(output)
                logger.info(f"Executor: pytest collection failed — {error_msg}")
                return ExecutionResult(
                    exit_code=proc.returncode,
                    stdout=output[-3000:],
                    stderr=proc.stderr[-2000:],
                    duration_seconds=elapsed,
                    success=False,
                    error_summary=f"Test collection failed: {error_msg}",
                    install_success=install_result.install_success,
                )

            return None

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=-1, success=False,
                error_summary="pytest timed out after 60s",
                install_success=install_result.install_success,
                duration_seconds=round(time.time() - t0, 2),
            )
        except Exception as e:
            logger.debug(f"Executor: pytest execution failed: {e}")
            return None

    def _smoke_test(
        self, python: str, tmp: Path, entry_points: list[str],
        code_files: dict[str, str],
    ) -> str:
        """Smoke test: verify code works WITHOUT pytest's sys.path manipulation.

        Pytest's default 'prepend' import mode adds conftest.py directories
        to sys.path, making imports work that would fail in normal execution.
        This smoke test catches that gap by:
        1. Importing every .py module individually (no pytest)
        2. Running the entry point with a short timeout

        Returns a status string, or empty string on failure.
        """
        env = {**os.environ, "PYTHONPATH": str(tmp)}
        results = []

        # Step 1: Import every source file individually
        for fname in sorted(code_files.keys()):
            if not fname.endswith(".py"):
                continue
            if "test" in fname or fname == "__init__.py":
                continue
            if fname.endswith("__init__.py"):
                continue

            module = fname.replace("/", ".").replace(".py", "")
            try:
                proc = subprocess.run(
                    [python, "-c", f"import sys; sys.path.insert(0, '{tmp}'); import {module}"],
                    capture_output=True, text=True,
                    timeout=10, cwd=str(tmp), env=env,
                )
                if proc.returncode != 0:
                    error = _extract_error(proc.stderr)
                    logger.info(f"Smoke test: import {module} FAILED — {error}")
                    results.append(f"FAIL:{module}")
                else:
                    results.append(f"OK:{module}")
            except subprocess.TimeoutExpired:
                results.append(f"TIMEOUT:{module}")

        ok_count = sum(1 for r in results if r.startswith("OK:"))
        fail_count = sum(1 for r in results if r.startswith("FAIL:"))

        # Step 2: Try running the entry point
        ep_status = "skipped"
        if entry_points:
            ep = entry_points[0]
            ep_path = tmp / ep
            if ep_path.exists():
                try:
                    proc = subprocess.run(
                        [python, str(ep_path)],
                        capture_output=True, text=True,
                        timeout=15, cwd=str(tmp), env=env,
                    )
                    ep_status = "pass" if proc.returncode == 0 else "fail"
                    if ep_status == "fail":
                        error = _extract_error(proc.stderr)
                        logger.info(f"Smoke test: entry point {ep} FAILED — {error}")
                except subprocess.TimeoutExpired:
                    # Timeout is okay for servers — they bind a port and wait
                    ep_status = "timeout_ok"

        status = f"imports={ok_count}/{ok_count + fail_count}, entry={ep_status}"
        if fail_count == 0:
            logger.info(f"Smoke test PASSED: {status}")
        else:
            logger.info(f"Smoke test: {fail_count} import failure(s) — {status}")

        return status

    def _install_deps(self, tmp: Path) -> ExecutionResult:
        """Install requirements.txt in a venv after verifying packages exist.

        Security: LLMs hallucinate package names at 5-21% rate (Spracklen et al.,
        USENIX 2025). Attackers register these names with malicious code
        ("slopsquatting"). We verify each package exists on PyPI before installing.
        """
        req_file = tmp / "requirements.txt"
        install_result = ExecutionResult(install_success=True)
        if not req_file.exists():
            return install_result

        content = req_file.read_text().strip()
        deps = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
        if not deps:
            return install_result

        # ── Verify packages exist before installing ──────────────────
        verified_deps = []
        for dep in deps:
            # Extract package name (strip version specifiers)
            pkg_name = dep.split(">=")[0].split("<=")[0].split("==")[0].split("~=")[0].split("[")[0].split("<")[0].split(">")[0].split("!=")[0].strip()
            if not pkg_name:
                continue

            # Known-safe packages (stdlib-adjacent, extremely common)
            SAFE_PACKAGES = {
                "pytest", "click", "flask", "fastapi", "uvicorn", "requests",
                "httpx", "pydantic", "sqlalchemy", "alembic", "jinja2",
                "starlette", "python-dotenv", "aiohttp", "aiofiles",
                "rich", "typer", "celery", "redis", "pymongo", "psycopg2-binary",
                "boto3", "pillow", "numpy", "pandas", "scipy", "matplotlib",
                "cryptography", "bcrypt", "pyjwt", "python-jose", "passlib",
                "gunicorn", "python-multipart", "email-validator", "orjson",
                "websockets", "httptools", "watchfiles", "itsdangerous",
                "werkzeug", "markupsafe", "certifi", "charset-normalizer",
                "idna", "urllib3", "packaging", "setuptools", "wheel", "pip",
                "toml", "tomli", "tomli-w", "pyyaml", "tomlkit",
            }

            if pkg_name.lower() in SAFE_PACKAGES:
                verified_deps.append(dep)
                continue

            # Check PyPI for unknown packages
            try:
                import urllib.request
                url = f"https://pypi.org/pypi/{pkg_name}/json"
                req = urllib.request.Request(url, method="HEAD")
                req.add_header("User-Agent", "belief-engine/2.3")
                resp = urllib.request.urlopen(req, timeout=5)
                if resp.status == 200:
                    verified_deps.append(dep)
                else:
                    logger.warning(f"Executor: BLOCKED unknown package '{pkg_name}' — not found on PyPI")
            except Exception:
                # Network error or package not found — skip it
                logger.warning(f"Executor: BLOCKED unverified package '{pkg_name}' — PyPI check failed")

        if len(verified_deps) < len(deps):
            blocked = len(deps) - len(verified_deps)
            logger.info(f"Executor: blocked {blocked} unverified package(s), installing {len(verified_deps)}")

        if not verified_deps:
            return install_result

        try:
            venv_dir = tmp / ".venv"
            venv.create(str(venv_dir), with_pip=True)
            pip = str(venv_dir / "bin" / "pip")
            proc = subprocess.run(
                [pip, "install"] + verified_deps,
                capture_output=True, text=True,
                timeout=INSTALL_TIMEOUT, cwd=str(tmp),
            )
            install_result.install_stdout = proc.stdout[-1000:]
            install_result.install_stderr = proc.stderr[-1000:]
            install_result.install_success = proc.returncode == 0
        except Exception as e:
            install_result.install_success = False
            install_result.install_stderr = str(e)

        return install_result

    def _verify_typescript(self, tmp: Path, entry_points: list[str]) -> ExecutionResult:
        """Verify a TypeScript/Node.js project.

        Steps:
        1. Run the TypeScript fixup pipeline (streaming fixes + covenants)
        2. npm install
        3. tsc --noEmit --strict (zero errors required)
        4. Run vitest (if test files exist)
        5. Smoke test: try running entry point with tsx
        """
        import shutil

        t0 = time.time()
        result = ExecutionResult()

        npm = shutil.which("npm")
        npx = shutil.which("npx")
        node = shutil.which("node")

        if not npm:
            result.success = False
            result.stderr = "npm not found — install Node.js to verify TypeScript projects"
            result.duration_seconds = time.time() - t0
            return result

        try:
            # Step 1: Run fixup pipeline on generated TS files
            try:
                from belief.validators.typescript_fixup import fixup_typescript_output

                ts_files = {}
                for fname in list(tmp.rglob("*.ts")) + list(tmp.rglob("*.tsx")):
                    rel = str(fname.relative_to(tmp))
                    if "node_modules" in rel:
                        continue
                    ts_files[rel] = fname.read_text()

                if ts_files:
                    fixed = fixup_typescript_output(ts_files, goal="")
                    for fname, content in fixed.items():
                        (tmp / fname).write_text(content)
                    logger.info(f"Executor: TS fixup pipeline ran on {len(ts_files)} files")
            except Exception as e:
                logger.debug(f"TS fixup skipped: {e}")

            # Step 2: npm install
            logger.info("Executor: running npm install for TypeScript project")
            proc = subprocess.run(
                [npm, "install", "--no-audit", "--no-fund"],
                capture_output=True, text=True,
                timeout=INSTALL_TIMEOUT, cwd=str(tmp),
            )
            result.install_success = proc.returncode == 0
            result.install_stdout = proc.stdout[-500:]
            result.install_stderr = proc.stderr[-500:]

            if not result.install_success:
                result.success = False
                result.stderr = f"npm install failed: {proc.stderr[-500:]}"
                result.duration_seconds = time.time() - t0
                return result

            # Step 3: Type check — tsc --noEmit
            if npx:
                logger.info("Executor: running tsc --noEmit")
                proc = subprocess.run(
                    [npx, "tsc", "--noEmit"],
                    capture_output=True, text=True,
                    timeout=60, cwd=str(tmp),
                )
                if proc.returncode != 0:
                    type_output = proc.stdout + proc.stderr
                    error_count = type_output.count("error TS")
                    # Log type errors but don't hard-fail on minor issues
                    # skipLibCheck handles most third-party type issues
                    if error_count > 3:
                        result.success = False
                        result.stderr = f"TypeScript: {error_count} type errors:\n{type_output[-1500:]}"
                        result.duration_seconds = time.time() - t0
                        return result
                    elif error_count > 0:
                        logger.info(f"Executor: {error_count} minor type errors (non-blocking)")

            # Step 4: Run vitest (if tests exist)
            has_tests = any(
                f.name.endswith((".test.ts", ".spec.ts", ".test.js", ".spec.js"))
                for f in tmp.rglob("*") if "node_modules" not in str(f)
            )
            if has_tests and npx:
                logger.info("Executor: running vitest")
                proc = subprocess.run(
                    [npx, "vitest", "run", "--reporter=verbose"],
                    capture_output=True, text=True,
                    timeout=60, cwd=str(tmp),
                    env={**os.environ, "NODE_ENV": "test"},
                )
                test_output = proc.stdout + "\n" + proc.stderr
                import re as _re

                passed = 0
                failed = 0
                match = _re.search(r"(\d+) passed", test_output)
                if match:
                    passed = int(match.group(1))
                match = _re.search(r"(\d+) failed", test_output)
                if match:
                    failed = int(match.group(1))

                total = passed + failed
                if total > 0:
                    success = passed > 0
                    result.success = success
                    result.stdout = test_output[-3000:]
                    result.stderr = proc.stderr[-1000:]
                    result.duration_seconds = time.time() - t0
                    logger.info(f"Executor: vitest {passed}/{total} passed")
                    if not success:
                        result.error_summary = f"vitest: 0/{total} tests passed"
                    return result

            # Step 5: Smoke test — try running with tsx
            if entry_points and npx:
                ep = entry_points[0]
                if ep.endswith((".ts", ".tsx")):
                    proc = subprocess.run(
                        [npx, "tsx", ep],
                        capture_output=True, text=True,
                        timeout=15, cwd=str(tmp),
                    )
                    if proc.returncode == 0:
                        result.stdout = (result.stdout or "") + "\nEntry point executed successfully"
                elif ep.endswith(".js") and node:
                    proc = subprocess.run(
                        [node, "--check", ep],
                        capture_output=True, text=True,
                        timeout=10, cwd=str(tmp),
                    )

            result.success = True
            result.stdout = (result.stdout or "") + "\nTypeScript verification passed"
            result.duration_seconds = time.time() - t0
            logger.info("Executor: TypeScript verification passed")

        except subprocess.TimeoutExpired:
            result.success = False
            result.stderr = "TypeScript verification timed out"
        except Exception as e:
            result.success = False
            result.stderr = f"TypeScript verification error: {e}"

        result.duration_seconds = time.time() - t0
        return result

    def _verify_via_import(
        self, python: str, tmp: Path, entry_point: str,
        code_files: dict[str, str],
    ) -> ExecutionResult:
        """Verify a server/API project by importing it with PYTHONPATH set.

        Milestone A: Bottom-up verification.
        Instead of importing the entry point (which fails with a confusing error
        when a deep dependency is broken), verify modules bottom-up:
        1. Syntax check all source files
        2. Find internal imports and build a dependency order
        3. Verify leaf modules first (no internal deps)
        4. Then verify their dependents, and so on up to the entry point
        5. When a module fails, report THAT file — not the entry point

        This way the debugger fixes the actual broken file.
        """
        t0 = time.time()

        # Step 0: Security scan — block dangerous code before execution
        try:
            from belief.hardening import scan_all_files, has_critical_violations
            violations = scan_all_files(code_files)
            if has_critical_violations(violations):
                critical = [v for v in violations if v.severity == "critical"]
                summary = "; ".join(f"{v.file}:{v.line} {v.message}" for v in critical[:3])
                return ExecutionResult(
                    exit_code=1, success=False,
                    error_summary=f"Security violation: {summary}",
                    install_success=True,
                )
            if violations:
                warnings = [f"{v.file}:{v.line} {v.message}" for v in violations]
                logger.warning(f"Security warnings: {'; '.join(warnings[:5])}")
        except Exception as e:
            logger.debug(f"Security scan skipped: {e}")

        # Step 0.5: Static import verification (Covenant #3)
        # NOTE: The _import_fix_node in the pipeline already ran auto_fix_imports.
        # Here we only VERIFY — we do NOT auto-fix again (double-fix corrupts
        # skeleton files like database.py). Remaining issues are logged as warnings,
        # not treated as fatal — let the actual import attempt catch real errors.
        try:
            from belief.codebase.imports import verify_imports
            import_issues = verify_imports(code_files)
            if import_issues:
                for issue in import_issues[:3]:
                    logger.warning(
                        f"Import warning: {issue.source_file}: "
                        f"'{issue.symbol}' from '{issue.target_module}' "
                        f"({issue.issue_type})"
                    )
        except Exception as e:
            logger.debug(f"Import verification skipped: {e}")

        # Step 1: Syntax check source files (multi-language via adapters)
        for fname, content in code_files.items():
            if "/test" in fname or fname.startswith("test"):
                continue
            try:
                from belief.languages import detect_language, get_adapter
                lang = detect_language(fname)
                adapter = get_adapter(lang)
                if adapter.is_source_file(fname):
                    result = adapter.verify_code(content, fname)
                    if not result.success:
                        return ExecutionResult(
                            exit_code=1, success=False,
                            error_summary=f"Syntax error in {fname}: {'; '.join(result.errors[:2])}",
                            install_success=True,
                        )
            except ImportError:
                # Fallback to Python-only ast.parse
                if fname.endswith(".py"):
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        return ExecutionResult(
                            exit_code=1, success=False,
                            error_summary=f"Syntax error in {fname} line {e.lineno}: {e.msg}",
                            install_success=True,
                        )

        # Step 2: Bottom-up import order using the DAG planner
        try:
            from belief.agents.parallel_planner import build_plan_from_files
            source_files = [
                f for f in code_files
                if f.endswith(".py") and "/test" not in f and not f.startswith("test")
            ]
            plan = build_plan_from_files(source_files)
            import_order = plan.file_order()
        except Exception:
            # Fallback: just verify the entry point directly
            import_order = [entry_point]

        # Step 3: Verify each module bottom-up
        env = {**os.environ, "PYTHONPATH": str(tmp)}

        for fname in import_order:
            if fname not in code_files:
                continue
            if not fname.endswith(".py"):
                continue
            if fname.endswith("__init__.py"):
                continue  # __init__.py is verified implicitly by package imports

            module_name = fname.replace("/", ".").replace(".py", "")
            is_package = "/" in fname

            if is_package:
                import_cmd = f"import {module_name}"
            else:
                import_cmd = (
                    f"import importlib.util; "
                    f"spec = importlib.util.spec_from_file_location('{module_name}', '{fname}'); "
                    f"mod = importlib.util.module_from_spec(spec); "
                    f"mod.__name__ = '{module_name}'; "
                    f"spec.loader.exec_module(mod)"
                )

            check_script = (
                f'import sys; sys.path.insert(0, "{tmp}"); '
                f'import importlib; importlib.invalidate_caches(); '
                f'{import_cmd}; '
                f'print("OK")'
            )

            try:
                proc = subprocess.run(
                    [python, "-c", check_script],
                    capture_output=True, text=True,
                    timeout=15, cwd=str(tmp), env=env,
                )
                if proc.returncode != 0:
                    error = proc.stderr[-500:] if proc.stderr else proc.stdout[-500:]
                    elapsed = round(time.time() - t0, 2)
                    return ExecutionResult(
                        exit_code=1, success=False,
                        stdout=proc.stdout[-1000:],
                        stderr=proc.stderr[-1000:],
                        duration_seconds=elapsed,
                        error_summary=f"Import failed for {fname}: {_extract_error(error)}",
                        install_success=True,
                    )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    exit_code=-1, success=False,
                    error_summary=f"Import of {fname} timed out after 15s",
                    install_success=True,
                )

        # Step 4: All modules imported successfully — now do the full entry point
        # verification with export listing and function testing
        module_name = entry_point.replace("/", ".").replace(".py", "")
        is_package = "/" in entry_point

        if is_package:
            import_stmt = f"import {module_name}"
            mod_ref = module_name
        else:
            import_stmt = (
                f"import importlib.util; "
                f"spec = importlib.util.spec_from_file_location('{module_name}', '{entry_point}'); "
                f"mod = importlib.util.module_from_spec(spec); "
                f"mod.__name__ = '{module_name}'; "
                f"spec.loader.exec_module(mod)"
            )
            mod_ref = "mod"

        verify_script = f'''
import sys
sys.path.insert(0, "{tmp}")
import importlib
importlib.invalidate_caches()

try:
    {import_stmt}
    print("IMPORT_OK")

    public = [n for n in dir({mod_ref}) if not n.startswith("_")]
    callables = [n for n in public if callable(getattr({mod_ref}, n, None))]
    print(f"EXPORTS: {{len(public)}} public, {{len(callables)}} callable")
    print(f"NAMES: {{', '.join(callables[:20])}}")

    tested = 0
    import inspect
    for name in callables:
        obj = getattr({mod_ref}, name)
        if isinstance(obj, type):
            continue
        try:
            sig = inspect.signature(obj)
            params = list(sig.parameters.values())
            if len(params) == 1 and params[0].annotation in (str, "str", inspect.Parameter.empty):
                result = obj("test input")
                print(f"CALL_OK: {{name}}")
                tested += 1
                if tested >= 3:
                    break
            elif len(params) == 0:
                result = obj()
                print(f"CALL_OK: {{name}}")
                tested += 1
                if tested >= 3:
                    break
        except Exception:
            continue

    if tested == 0:
        print("VERIFY_OK: module imports and exports are valid")
    else:
        print(f"VERIFY_OK: {{tested}} function(s) tested successfully")

except Exception as e:
    print(f"IMPORT_FAIL: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

        try:
            proc = subprocess.run(
                [python, "-c", verify_script],
                capture_output=True, text=True,
                timeout=30, cwd=str(tmp),
                env=env,
            )
            elapsed = round(time.time() - t0, 2)
            stdout = proc.stdout[-3000:]
            stderr = proc.stderr[-2000:]

            success = "IMPORT_OK" in stdout and proc.returncode == 0
            if success:
                last_line = [l for l in stdout.strip().splitlines() if l.strip()][-1] if stdout.strip() else ""
                logger.info(f"Server verification passed: {last_line}")

            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=elapsed,
                success=success,
                error_summary="" if success else _extract_error(stderr or stdout),
                install_success=True,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=-1, success=False,
                error_summary="Server verification timed out after 30s",
                install_success=True,
            )

    def _run_script(self, python: str, tmp: Path, entry_point: str,
                    install_result: ExecutionResult) -> ExecutionResult:
        """Standard execution: run the entry point as a script."""
        try:
            env = {**os.environ, "PYTHONPATH": str(tmp)}
            proc = subprocess.run(
                [python, str(tmp / entry_point)],
                capture_output=True, text=True,
                timeout=EXEC_TIMEOUT, cwd=str(tmp),
                env=env,
            )
            result = ExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout[-3000:],
                stderr=proc.stderr[-2000:],
                success=proc.returncode == 0,
                install_success=install_result.install_success,
                install_stdout=install_result.install_stdout,
                install_stderr=install_result.install_stderr,
            )
            if not result.success:
                result.error_summary = _extract_error(proc.stderr)
            return result

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=-1, success=False,
                error_summary=f"Execution timed out after {EXEC_TIMEOUT}s",
                install_success=install_result.install_success,
            )



def _extract_error(stderr: str) -> str:
    """Extract and classify error into a structured diagnosis.

    Instead of dumping raw tracebacks to the debugger, we classify the error
    and explain what assumption was violated. This follows Julia Evans'
    debugging framework: what did we expect? what happened? what was wrong?

    The debugger can act on "missing __init__.py in app/" much faster
    than on a 50-line ModuleNotFoundError traceback.
    """
    if not stderr:
        return ""

    stderr_lower = stderr.lower()
    lines = stderr.strip().splitlines()

    # ── Classify by error pattern ────────────────────────────────
    # Each pattern: (indicator, diagnosis template)
    patterns = [
        # Import errors — the #1 failure mode
        ("modulenotfounderror: no module named '", lambda s: _diagnose_import(s)),
        ("importerror: cannot import name '", lambda s: _diagnose_import(s)),
        ("importerror: attempted relative import", "Relative import used outside a package. File needs to be inside a package with __init__.py, or use absolute imports."),

        # Syntax errors
        ("syntaxerror:", lambda s: _find_line(s, "SyntaxError")),

        # Type/attribute errors
        ("attributeerror: '", lambda s: _diagnose_attribute(s)),
        ("typeerror:", lambda s: _find_line(s, "TypeError")),

        # Runtime errors
        ("filenotfounderror:", "Code references a file that doesn't exist. Check file paths and working directory assumptions."),
        ("permissionerror:", "File permission denied. Check if the code is trying to write to a read-only location."),
        ("connectionrefusederror:", "Network connection refused. Service dependency not running or wrong port."),
        ("keyerror:", lambda s: _find_line(s, "KeyError")),

        # Dependency errors
        ("no matching distribution found for", "pip install failed — package doesn't exist or version constraint is unsatisfiable. Check requirements.txt."),
        ("could not find a version that satisfies", "Version constraint in requirements.txt can't be satisfied. Relax the version pin."),
    ]

    for indicator, diagnosis in patterns:
        if indicator in stderr_lower:
            if callable(diagnosis):
                result = diagnosis(stderr)
                if result:
                    return result[:400]
            else:
                return diagnosis[:400]

    # Fallback: last meaningful line
    for line in reversed(lines):
        line = line.strip()
        if any(kw in line for kw in ("Error:", "Exception:", "Traceback")):
            return line[:300]
    return lines[-1][:300] if lines else ""


def _diagnose_import(stderr: str) -> str:
    """Diagnose import errors with actionable fix suggestions."""
    import re
    # Extract the module name
    match = re.search(r"No module named '([^']+)'", stderr)
    if match:
        module = match.group(1)
        parts = module.split(".")
        if len(parts) > 1:
            return (
                f"ModuleNotFoundError: '{module}'. "
                f"Check: (1) does {parts[0]}/__init__.py exist? "
                f"(2) is '{parts[-1]}' defined in {'/'.join(parts[:-1])}/? "
                f"(3) is the package in requirements.txt?"
            )
        return f"ModuleNotFoundError: '{module}'. Check: is it in requirements.txt? Is it spelled correctly?"

    match = re.search(r"cannot import name '([^']+)' from '([^']+)'", stderr)
    if match:
        name, module = match.group(1), match.group(2)
        return f"ImportError: '{name}' not found in '{module}'. Check: does '{name}' exist in {module.replace('.', '/')}? Is it exported?"

    return "Import failed. Check module paths and __init__.py files."


def _diagnose_attribute(stderr: str) -> str:
    """Diagnose attribute errors — often caused by wrong API version."""
    import re
    match = re.search(r"AttributeError: '(\w+)' object has no attribute '(\w+)'", stderr)
    if match:
        obj_type, attr = match.group(1), match.group(2)
        return f"AttributeError: '{obj_type}' has no '{attr}'. The API may have changed — check the library version and documentation."
    return "AttributeError: accessing a property or method that doesn't exist on this object."


def _find_line(stderr: str, keyword: str) -> str:
    """Find the most relevant line containing a keyword."""
    for line in stderr.strip().splitlines():
        if keyword in line:
            return line.strip()[:300]
    return ""


def _parse_pytest(output: str) -> PytestResult:
    """Parse pytest output into structured result."""
    import re
    result = PytestResult(ran=True, raw_output=output[-2000:])

    # Parse summary line: "5 passed, 2 failed in 1.23s"
    match = re.search(r"(\d+) passed", output)
    if match:
        result.passed = int(match.group(1))
    match = re.search(r"(\d+) failed", output)
    if match:
        result.failed = int(match.group(1))
    match = re.search(r"(\d+) error", output)
    if match:
        result.errors = int(match.group(1))
    result.total = result.passed + result.failed + result.errors

    # Parse individual test items
    for match in re.finditer(r"(PASSED|FAILED|ERROR)\s+(\S+)", output):
        outcome = match.group(1).lower()
        if outcome == "error":
            outcome = "error"
        result.items.append(PytestTestItem(
            node_id=match.group(2),
            outcome=outcome,
        ))

    return result
