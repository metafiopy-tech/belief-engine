"""Deployment Pipeline — Tier 8 Autonomous Operation.

Takes generated code and deploys it to a live URL:
  1. Write files to a temp directory
  2. Generate Dockerfile if missing
  3. Deploy via Railway CLI (`railway up`) or Docker push
  4. Return the live URL

The engine already generates Dockerfiles, docker-compose.yml, and
railway.toml via the synthesizer. This module connects those artifacts
to actual deployment infrastructure.

Supports:
  - Railway (primary) — `railway up` from project directory
  - Docker local — `docker compose up -d` for local testing
  - Fly.io (future) — `fly deploy`
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger("belief.deploy")


class DeployTarget(str, Enum):
    RAILWAY = "railway"
    DOCKER_LOCAL = "docker_local"
    FLY = "fly"


class DeployStatus(str, Enum):
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    LIVE = "live"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeployResult:
    """Result of a deployment attempt."""

    status: DeployStatus
    url: str = ""
    deploy_id: str = ""
    target: DeployTarget = DeployTarget.RAILWAY
    duration_seconds: float = 0.0
    error: str = ""
    logs: str = ""


@dataclass
class DeployConfig:
    """Configuration for deployment."""

    target: DeployTarget = DeployTarget.RAILWAY
    project_name: str = ""
    railway_token: str = ""
    auto_domain: bool = True
    health_check_path: str = "/health"
    health_check_timeout: int = 120  # seconds to wait for healthy


# ── Railway deployment ───────────────────────────────────────────────────────


async def deploy_to_railway(
    code_files: dict[str, str],
    config: DeployConfig,
) -> DeployResult:
    """Deploy generated code to Railway.

    Requires: `railway` CLI installed and authenticated.
    Uses `railway up` which builds and deploys from source.
    """
    t0 = time.time()
    result = DeployResult(status=DeployStatus.PENDING, target=DeployTarget.RAILWAY)

    # Check Railway CLI
    if not shutil.which("railway"):
        result.status = DeployStatus.FAILED
        result.error = "Railway CLI not installed. Run: npm install -g @railway/cli"
        return result

    # Use persistent directory
    deploy_dir = Path.home() / ".belief-engine" / "deploy" / config.project_name
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous deploy
    for f in deploy_dir.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    # Write all code files
    for fname, content in code_files.items():
        fpath = deploy_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)

    # Ensure .env exists
    if ".env" not in code_files:
        env_content = code_files.get(".env.example", "PORT=8000\n")
        (deploy_dir / ".env").write_text(env_content)

    # Ensure Dockerfile exists
    if "Dockerfile" not in code_files:
        dockerfile = _generate_dockerfile(code_files)
        (deploy_dir / "Dockerfile").write_text(dockerfile)

    # Ensure railway.toml exists
    if "railway.toml" not in code_files:
        toml = _generate_railway_toml(config)
        (deploy_dir / "railway.toml").write_text(toml)

    # Deploy
    result.status = DeployStatus.DEPLOYING
    env = {**os.environ}
    if config.railway_token:
        env["RAILWAY_TOKEN"] = config.railway_token

    try:
        proc = subprocess.run(
            ["railway", "up", "--detach"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(deploy_dir),
            env=env,
        )
        result.logs = proc.stdout + "\n" + proc.stderr

        if proc.returncode == 0:
            result.status = DeployStatus.LIVE
            # Try to extract URL from output
            url = _extract_railway_url(proc.stdout + proc.stderr)
            result.url = url or f"https://{config.project_name}.up.railway.app"
            logger.info(f"Deploy: Railway deployment successful → {result.url}")
        else:
            result.status = DeployStatus.FAILED
            result.error = proc.stderr[-500:] if proc.stderr else "Unknown error"
            logger.warning(f"Deploy: Railway failed — {result.error[:100]}")

    except subprocess.TimeoutExpired:
        result.status = DeployStatus.FAILED
        result.error = "Deployment timed out after 300s"
    except Exception as e:
        result.status = DeployStatus.FAILED
        result.error = str(e)

    result.duration_seconds = time.time() - t0
    return result


# ── Docker local deployment ──────────────────────────────────────────────────


async def deploy_local_docker(
    code_files: dict[str, str],
    config: DeployConfig,
) -> DeployResult:
    """Deploy generated code locally via Docker Compose.

    Good for testing before pushing to Railway.
    Writes to a persistent .deploy/ directory inside the build output
    so Docker can reference the files after the function returns.
    """
    t0 = time.time()
    result = DeployResult(status=DeployStatus.PENDING, target=DeployTarget.DOCKER_LOCAL)

    if not shutil.which("docker"):
        result.status = DeployStatus.FAILED
        result.error = "Docker not installed"
        return result

    # Use a persistent directory instead of tempdir
    deploy_dir = Path.home() / ".belief-engine" / "deploy" / config.project_name
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous deploy
    for f in deploy_dir.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    for fname, content in code_files.items():
        fpath = deploy_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content)

    # Ensure .env exists (docker-compose may reference it)
    if ".env" not in code_files:
        env_content = code_files.get(".env.example", "PORT=8000\n")
        (deploy_dir / ".env").write_text(env_content)

    if "Dockerfile" not in code_files:
        (deploy_dir / "Dockerfile").write_text(_generate_dockerfile(code_files))

    if "docker-compose.yml" not in code_files:
        compose = _generate_compose(config)
        (deploy_dir / "docker-compose.yml").write_text(compose)

    result.status = DeployStatus.BUILDING
    try:
        # Stop any existing container with same project name
        subprocess.run(
            ["docker", "compose", "-p", config.project_name, "down", "--remove-orphans"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(deploy_dir),
        )

        # Build and start
        proc = subprocess.run(
            ["docker", "compose", "-p", config.project_name, "up", "-d", "--build"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(deploy_dir),
        )
        result.logs = proc.stdout + "\n" + proc.stderr

        if proc.returncode == 0:
            result.status = DeployStatus.LIVE
            result.url = "http://localhost:8000"
            logger.info("Deploy: local Docker deployment successful")
        else:
            result.status = DeployStatus.FAILED
            result.error = proc.stderr[-500:]

    except Exception as e:
        result.status = DeployStatus.FAILED
        result.error = str(e)

    result.duration_seconds = time.time() - t0
    return result


# ── Unified deploy interface ─────────────────────────────────────────────────


async def deploy(
    code_files: dict[str, str],
    config: DeployConfig | None = None,
) -> DeployResult:
    """Deploy generated code to the configured target.

    Tries Railway first, falls back to local Docker.
    """
    if config is None:
        config = DeployConfig()
        # Auto-detect project name from code files
        for fname in code_files:
            if fname == "pyproject.toml":
                import re

                match = re.search(r'name\s*=\s*"([^"]+)"', code_files[fname])
                if match:
                    config.project_name = match.group(1)
                    break
        if not config.project_name:
            config.project_name = "belief-build"

    if config.target == DeployTarget.RAILWAY:
        return await deploy_to_railway(code_files, config)
    elif config.target == DeployTarget.DOCKER_LOCAL:
        return await deploy_local_docker(code_files, config)
    else:
        return DeployResult(
            status=DeployStatus.FAILED,
            error=f"Unsupported target: {config.target}",
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _generate_dockerfile(code_files: dict[str, str]) -> str:
    """Generate a Dockerfile for the project."""
    has_requirements = "requirements.txt" in code_files

    # Detect entry point
    entry = "main.py"
    for f in code_files:
        if f.endswith("main.py"):
            entry = f
            break
        if f.endswith("server.py"):
            entry = f

    return f"""FROM python:3.12-slim

WORKDIR /app

{"COPY requirements.txt ." if has_requirements else ""}
{"RUN pip install --no-cache-dir -r requirements.txt" if has_requirements else ""}

COPY . .

EXPOSE 8000

CMD ["python", "{entry}"]
"""


def _generate_railway_toml(config: DeployConfig) -> str:
    return f"""[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "{config.health_check_path}"
healthcheckTimeout = {config.health_check_timeout}
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
"""


def _generate_compose(config: DeployConfig) -> str:
    return """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    restart: unless-stopped
"""


def _extract_railway_url(output: str) -> str:
    """Extract the deployment URL from Railway CLI output."""
    import re

    # Railway outputs URLs like: https://xxx.up.railway.app
    match = re.search(r"https://[\w.-]+\.railway\.app", output)
    if match:
        return match.group(0)
    # Also try: https://xxx.railway.app
    match = re.search(r"https://[\w.-]+\.railway\.\w+", output)
    return match.group(0) if match else ""
