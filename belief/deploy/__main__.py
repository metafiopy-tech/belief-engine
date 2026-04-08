"""Deploy CLI — deploy built projects to Railway or Docker.

Usage:
    # Deploy the most recent build
    python3 -m belief.deploy

    # Deploy a specific build
    python3 -m belief.deploy --build belief-7ec9b0cb

    # Deploy to local Docker instead of Railway
    python3 -m belief.deploy --target docker_local

    # Deploy with a custom project name
    python3 -m belief.deploy --name my-url-shortener

    # Check health of a deployed service
    python3 -m belief.deploy --health https://myapp.up.railway.app

    # List recent builds available for deployment
    python3 -m belief.deploy --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from belief.config.settings import settings


def _configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "\033[90m%(asctime)s\033[0m \033[36m%(levelname)-8s\033[0m \033[90m%(name)-28s\033[0m %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("belief")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def _find_builds(output_dir: Path) -> list[dict]:
    """Find all completed builds, sorted by modification time (newest first)."""
    builds = []
    if not output_dir.exists():
        return builds

    for d in output_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("belief-"):
            continue

        # Count files and total size
        files = list(d.rglob("*"))
        source_files = [f for f in files if f.is_file() and f.suffix in (".py", ".ts", ".go", ".html", ".css", ".js")]
        all_files = [f for f in files if f.is_file()]

        # Get modification time
        mtime = max((f.stat().st_mtime for f in all_files), default=0)

        builds.append({
            "id": d.name,
            "path": str(d),
            "files": len(all_files),
            "source_files": len(source_files),
            "mtime": mtime,
        })

    builds.sort(key=lambda b: b["mtime"], reverse=True)
    return builds


def _load_build_files(build_path: Path) -> dict[str, str]:
    """Load all files from a build directory."""
    files = {}
    for f in build_path.rglob("*"):
        if not f.is_file():
            continue
        rel = str(f.relative_to(build_path))
        try:
            files[rel] = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    return files


async def _deploy(args):
    """Main deploy logic."""
    output_dir = settings.output_path

    # ── List builds ──
    if args.list:
        builds = _find_builds(output_dir)
        if not builds:
            print("No builds found in ./output/")
            return

        print(f"\n  {'Build ID':<25} {'Files':>5}  {'Source':>6}  Path")
        print(f"  {'-'*25} {'-'*5}  {'-'*6}  {'-'*30}")
        for b in builds[:15]:
            marker = " ←" if b == builds[0] else ""
            print(f"  {b['id']:<25} {b['files']:>5}  {b['source_files']:>6}  {b['path']}{marker}")
        print(f"\n  {len(builds)} builds total. Latest: {builds[0]['id']}\n")
        return

    # ── Health check ──
    if args.health:
        from belief.deploy.monitor import HealthMonitor
        monitor = HealthMonitor(url=args.health, check_interval=5)
        print(f"\n  Checking health of {args.health}...\n")

        for i in range(args.checks or 3):
            check = await monitor.check_once()
            status_icon = "✓" if check.status.value == "healthy" else "✗"
            color = "\033[32m" if check.status.value == "healthy" else "\033[31m"
            print(f"  {color}{status_icon}\033[0m {check.status.value:<12} {check.response_time_ms:>6.0f}ms  {check.diagnosis or 'OK'}")

        print(f"\n{monitor.summary()}\n")
        return

    # ── Find the build to deploy ──
    if args.build:
        build_path = output_dir / args.build
        if not build_path.exists():
            print(f"Build not found: {args.build}")
            print(f"Run: python3 -m belief.deploy --list")
            sys.exit(1)
        build_id = args.build
    else:
        # Find the latest build
        builds = _find_builds(output_dir)
        if not builds:
            print("No builds found. Run a build first:")
            print("  python3 -m belief.cli --goal \"your goal\"")
            sys.exit(1)
        build_id = builds[0]["id"]
        build_path = Path(builds[0]["path"])

    # Load files
    code_files = _load_build_files(build_path)
    if not code_files:
        print(f"No files found in {build_path}")
        sys.exit(1)

    # ── Deploy ──
    from belief.deploy import deploy, DeployConfig, DeployTarget, DeployStatus

    project_name = args.name or build_id
    target = DeployTarget(args.target)

    print(f"\n{'═' * 60}")
    print(f"  DEPLOYING: {build_id}")
    print(f"{'═' * 60}")
    print(f"  Target:  {target.value}")
    print(f"  Name:    {project_name}")
    print(f"  Files:   {len(code_files)}")
    print(f"  Path:    {build_path}")
    print()

    config = DeployConfig(
        target=target,
        project_name=project_name,
        railway_token=os.environ.get("RAILWAY_TOKEN", ""),
    )

    result = await deploy(code_files, config)

    if result.status == DeployStatus.LIVE:
        print(f"\n  \033[32m✓ DEPLOYED SUCCESSFULLY\033[0m")
        print(f"  URL: {result.url}")
        print(f"  Time: {result.duration_seconds:.1f}s")

        if args.monitor:
            print(f"\n  Starting health monitoring ({args.monitor_duration}s)...")
            from belief.deploy.monitor import HealthMonitor
            monitor = HealthMonitor(url=result.url, check_interval=10)

            async def _on_check(check):
                icon = "✓" if check.status.value == "healthy" else "✗"
                print(f"  {icon} {check.status.value} ({check.response_time_ms:.0f}ms)")

            await monitor.run_continuous(max_checks=args.monitor_duration // 10)
            print(f"\n{monitor.summary()}")

    else:
        print(f"\n  \033[31m✗ DEPLOY FAILED\033[0m")
        print(f"  Error: {result.error}")
        if result.logs:
            print(f"\n  Logs:\n{result.logs[-500:]}")
        sys.exit(1)

    print()


def main():
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Deploy Belief Engine builds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m belief.deploy              # Deploy latest build to Railway
  python3 -m belief.deploy --list       # List available builds
  python3 -m belief.deploy --target docker_local  # Deploy locally
  python3 -m belief.deploy --health https://myapp.up.railway.app
        """,
    )
    parser.add_argument("--build", help="Build ID to deploy (default: latest)")
    parser.add_argument("--name", help="Project name for deployment")
    parser.add_argument("--target", default="railway", choices=["railway", "docker_local"],
                        help="Deploy target (default: railway)")
    parser.add_argument("--list", action="store_true", help="List available builds")
    parser.add_argument("--health", metavar="URL", help="Check health of a deployed service")
    parser.add_argument("--checks", type=int, default=3, help="Number of health checks (default: 3)")
    parser.add_argument("--monitor", action="store_true", help="Monitor after deploy")
    parser.add_argument("--monitor-duration", type=int, default=60, help="Monitor duration in seconds")

    args = parser.parse_args()
    asyncio.run(_deploy(args))


if __name__ == "__main__":
    main()
