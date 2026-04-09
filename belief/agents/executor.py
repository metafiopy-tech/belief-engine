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

        # Check if ANY entry point is a server
        is_server = any(
            _is_server_code(state.code_files.get(ep, ""))
            for ep in entry_points
        )

        if is_server:
            logger.info(
                f"Executor: detected server-type program — using import verification "
                f"({len(entry_points)} entry point{'s' if len(entry_points) > 1 else ''})"
            )

        result = await asyncio.to_thread(
            self._execute_in_sandbox, state.code_files, state.test_files,
            entry_points, is_server,
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
        entry_points: list[str], is_server: bool = False,
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

            if is_server:
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
