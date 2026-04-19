"""Docker Compose Orchestrator — Multi-Service Lifecycle Management.

Handles the full lifecycle for integration testing:
  1. Write docker-compose.yml + Dockerfiles to temp dir
  2. docker compose build
  3. docker compose up -d
  4. Wait for health checks (all services healthy)
  5. Run integration tests
  6. Capture logs on failure
  7. docker compose down (always, even on error)

Uses subprocess for docker compose commands. python-on-whales is preferred
if available (Docker Inc endorsed), falls back to raw subprocess.

Usage:
    from belief.tools.compose_orchestrator import ComposeStack

    async with ComposeStack(compose_yaml, service_files) as stack:
        if stack.healthy:
            results = await stack.run_tests(test_fn)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("belief.tools.compose")

HEALTH_TIMEOUT = 60  # seconds to wait for all services to be healthy
BUILD_TIMEOUT = 120  # seconds for docker compose build
TEARDOWN_TIMEOUT = 30


class ComposeStack:
    """Context manager for docker-compose lifecycle.

    Guarantees teardown even on errors via __aexit__.
    """

    def __init__(
        self,
        compose_yaml: str,
        service_files: dict[str, dict[str, str]],
        project_name: str = "belief_test",
    ):
        """
        Args:
            compose_yaml: docker-compose.yml content
            service_files: {package_name: {filename: content}} per service
            project_name: docker compose project name (for isolation)
        """
        self.compose_yaml = compose_yaml
        self.service_files = service_files
        self.project_name = project_name
        self.tmp_dir: Path | None = None
        self.healthy = False
        self._up = False

    async def __aenter__(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="belief_compose_"))

        try:
            # Write docker-compose.yml
            (self.tmp_dir / "docker-compose.yml").write_text(self.compose_yaml)

            # Write service files
            for package, files in self.service_files.items():
                pkg_dir = self.tmp_dir / package
                pkg_dir.mkdir(parents=True, exist_ok=True)

                for fname, content in files.items():
                    fpath = pkg_dir / fname
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(content)

                # Generate Dockerfile if not provided
                if "Dockerfile" not in files:
                    dockerfile = _generate_dockerfile(package, files)
                    (pkg_dir / "Dockerfile").write_text(dockerfile)

            # Build
            logger.info(f"Compose: building in {self.tmp_dir}")
            build_ok = await self._run_compose("build", timeout=BUILD_TIMEOUT)
            if not build_ok:
                logger.warning("Compose: build failed")
                return self

            # Up
            up_ok = await self._run_compose("up", "-d", timeout=30)
            if not up_ok:
                logger.warning("Compose: up failed")
                return self
            self._up = True

            # Wait for health
            self.healthy = await self._wait_for_health()
            if self.healthy:
                logger.info("Compose: all services healthy")
            else:
                logger.warning("Compose: health check timeout")

        except Exception as e:
            logger.warning(f"Compose setup failed: {e}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._up:
            try:
                # Capture logs before teardown
                logs = await self._get_logs()
                if logs:
                    logger.debug(f"Compose logs:\n{logs[:3000]}")

                await self._run_compose("down", "-v", "--remove-orphans", timeout=TEARDOWN_TIMEOUT)
            except Exception as e:
                logger.warning(f"Compose teardown error: {e}")
                # Force kill
                try:
                    await self._run_compose("kill", timeout=10)
                    await self._run_compose("down", "-v", timeout=10)
                except Exception as e:
                    logger.debug(f"Force kill failed during teardown: {e}")

        # Clean up temp dir
        if self.tmp_dir and self.tmp_dir.exists():
            try:
                shutil.rmtree(self.tmp_dir)
            except Exception as e:
                logger.debug(f"Temp dir cleanup failed: {e}")

    async def run_tests(self, test_fn: Callable) -> dict[str, Any]:
        """Run integration tests against the running services.

        test_fn receives a dict of {service_name: base_url} and returns test results.
        """
        if not self.healthy:
            return {"success": False, "error": "Services not healthy"}

        # Build base URLs from compose config
        base_urls = self._get_base_urls()

        try:
            if asyncio.iscoroutinefunction(test_fn):
                results = await test_fn(base_urls)
            else:
                results = await asyncio.to_thread(test_fn, base_urls)
            return results
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_base_urls(self) -> dict[str, str]:
        """Extract base URLs for each service from compose config."""
        import yaml
        try:
            compose = yaml.safe_load(self.compose_yaml)
            urls = {}
            for name, svc in compose.get("services", {}).items():
                ports = svc.get("ports", [])
                if ports:
                    host_port = str(ports[0]).split(":")[0]
                    urls[name] = f"http://localhost:{host_port}"
            return urls
        except Exception:
            return {}

    async def _run_compose(self, *args: str, timeout: int = 60) -> bool:
        """Run a docker compose command."""
        cmd = [
            "docker", "compose",
            "-p", self.project_name,
            "-f", str(self.tmp_dir / "docker-compose.yml"),
            *args,
        ]
        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True,
                timeout=timeout, cwd=str(self.tmp_dir),
            )
            if proc.returncode != 0:
                logger.debug(f"Compose {args[0]} stderr: {proc.stderr[-500:]}")
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning(f"Compose {args[0]} timed out after {timeout}s")
            return False
        except FileNotFoundError:
            logger.warning("Docker compose not found — is Docker installed?")
            return False

    async def _wait_for_health(self) -> bool:
        """Poll docker compose ps until all services are healthy or timeout."""
        deadline = time.time() + HEALTH_TIMEOUT

        while time.time() < deadline:
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "compose", "-p", self.project_name,
                     "-f", str(self.tmp_dir / "docker-compose.yml"),
                     "ps", "--format", "json"],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(self.tmp_dir),
                )

                if proc.returncode == 0 and proc.stdout.strip():
                    import json
                    lines = proc.stdout.strip().split("\n")
                    all_healthy = True
                    for line in lines:
                        try:
                            container = json.loads(line)
                            health = container.get("Health", "")
                            state = container.get("State", "")
                            if state != "running":
                                all_healthy = False
                            elif health and health != "healthy":
                                all_healthy = False
                        except json.JSONDecodeError:
                            continue

                    if all_healthy and lines:
                        return True

            except Exception:
                pass

            await asyncio.sleep(2)

        return False

    async def _get_logs(self) -> str:
        """Capture logs from all containers."""
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["docker", "compose", "-p", self.project_name,
                 "-f", str(self.tmp_dir / "docker-compose.yml"),
                 "logs", "--tail=50"],
                capture_output=True, text=True, timeout=10,
                cwd=str(self.tmp_dir),
            )
            return proc.stdout
        except Exception:
            return ""


def _generate_dockerfile(package: str, files: dict[str, str]) -> str:
    """Generate a basic Dockerfile for a Python FastAPI service."""
    has_requirements = "requirements.txt" in files

    return f"""FROM python:3.12-slim

WORKDIR /app

{"COPY requirements.txt ." if has_requirements else ""}
{"RUN pip install --no-cache-dir -r requirements.txt" if has_requirements else "RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy"}

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
