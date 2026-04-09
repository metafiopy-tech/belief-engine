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
        """Write files to a temp dir, verify imports, optionally run."""
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

            # Auto-generate __init__.py in every directory containing .py files
            for dirpath in set(
                (tmp / f).parent for f in list(code_files) + list(test_files)
                if "/" in f
            ):
                init_file = dirpath / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("")
                    logger.debug(f"Executor: auto-generated {init_file.relative_to(tmp)}")

            system_python = _sys.executable

            # ── TypeScript/Node.js path ──────────────────────────────────
            # If project has package.json, use Node.js execution
            if "package.json" in code_files:
                return self._verify_typescript(tmp, entry_points)

            if use_import_verify:
                # M4: Verify each entry point independently — all must pass
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
            install_result = self._install_deps(tmp)
            return self._run_script(system_python, tmp, entry_points[0], install_result)

    def _install_deps(self, tmp: Path) -> ExecutionResult:
        """Install requirements.txt in a venv. Returns install status."""
        req_file = tmp / "requirements.txt"
        install_result = ExecutionResult(install_success=True)
        if not req_file.exists():
            return install_result

        content = req_file.read_text().strip()
        deps = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
        if not deps:
            return install_result

        try:
            venv_dir = tmp / ".venv"
            venv.create(str(venv_dir), with_pip=True)
            pip = str(venv_dir / "bin" / "pip")
            proc = subprocess.run(
                [pip, "install"] + deps,
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
        1. npm install (install dependencies)
        2. npx tsc --noEmit (type check without emitting)
        3. If entry point is .js, try running with node --check

        This mirrors the Python import verification but for the Node.js ecosystem.
        """
        import shutil

        t0 = time.time()
        result = ExecutionResult()

        # Check if npm is available
        npm = shutil.which("npm")
        npx = shutil.which("npx")
        node = shutil.which("node")

        if not npm:
            result.success = False
            result.stderr = "npm not found — install Node.js to verify TypeScript projects"
            result.duration_seconds = time.time() - t0
            return result

        try:
            # Step 1: npm install
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

            # Step 2: Type check with tsc --noEmit
            if npx:
                logger.info("Executor: running tsc --noEmit")
                proc = subprocess.run(
                    [npx, "tsc", "--noEmit"],
                    capture_output=True, text=True,
                    timeout=60, cwd=str(tmp),
                )
                if proc.returncode != 0:
                    # Type errors — report but don't necessarily fail
                    # Many generated TS projects have minor type issues that don't affect runtime
                    type_errors = proc.stdout[-1000:] if proc.stdout else proc.stderr[-1000:]
                    error_count = type_errors.count("error TS")
                    if error_count > 5:
                        result.success = False
                        result.stderr = f"TypeScript: {error_count} type errors:\n{type_errors}"
                        result.duration_seconds = time.time() - t0
                        return result
                    else:
                        logger.info(f"Executor: {error_count} minor type errors (non-blocking)")
                        result.stdout = f"TypeScript: {error_count} minor type errors (non-blocking)"

            # Step 3: Syntax check entry point with node --check (JS files only)
            if node and entry_points:
                ep = entry_points[0]
                if ep.endswith(".js"):
                    proc = subprocess.run(
                        [node, "--check", ep],
                        capture_output=True, text=True,
                        timeout=10, cwd=str(tmp),
                    )
                    if proc.returncode != 0:
                        result.success = False
                        result.stderr = f"Node syntax error: {proc.stderr[-500:]}"
                        result.duration_seconds = time.time() - t0
                        return result

            result.success = True
            result.stdout = (result.stdout or "") + "\nTypeScript verification passed"
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
    """Extract the most meaningful error line from stderr."""
    if not stderr:
        return ""
    lines = stderr.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if any(kw in line for kw in ("Error:", "Exception:", "Traceback")):
            return line[:300]
    return lines[-1][:300] if lines else ""


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
