"""Belief Engine Miner for Bittensor Subnet 62 (Ridges).

Ridges operates as a winner-take-all agent competition, not real-time
synapse responses. Miners submit agent codebases that are evaluated
against SWE-bench and Polyglot problem sets.

Architecture:
  1. Ridges validator sends a SWE-bench problem (issue + repo + tests)
  2. This agent clones the repo, runs the Belief Engine brownfield pipeline
  3. Outputs a unified diff patch
  4. Validator applies patch, runs tests, scores the agent

Ridges infrastructure:
  - Docker: sweagent/swe-agent:latest sandbox
  - Framework: ridgesai/abstract-agent-runner
  - Networking: Fiber (not bt.Axon/bt.Dendrite)
  - Timeout: 25 minutes per problem
  - Internet: limited during execution

Usage:
  # As standalone agent (for testing)
  python -m belief.bittensor.miner solve --instance /path/to/instance.json

  # As Ridges-compatible agent
  belief mine --netuid 62

  # Test on SWE-bench locally
  python -m belief.bittensor.miner test --dataset swebench_verified --limit 5
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("belief.bittensor.miner")


@dataclass
class SWEBenchInstance:
    """A single SWE-bench problem instance."""

    instance_id: str = ""
    repo: str = ""  # e.g., "django/django"
    base_commit: str = ""  # commit to check out
    problem_statement: str = ""  # issue description
    hints_text: str = ""  # optional hints
    test_patch: str = ""  # gold test patch (for evaluation)
    patch: str = ""  # gold fix patch (for reference, not given to agent)
    version: str = ""  # repo version tag

    @classmethod
    def from_dict(cls, d: dict) -> SWEBenchInstance:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str | Path) -> SWEBenchInstance:
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            data = data[0]
        return cls.from_dict(data)


@dataclass
class AgentResult:
    """Result of solving one SWE-bench instance."""

    instance_id: str
    model_patch: str = ""  # unified diff output
    success: bool = False
    error: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0


class BeliefMiner:
    """Ridges-compatible SWE-bench agent powered by the Belief Engine.

    Wraps the brownfield pipeline (Agentless localization + Kimi-Dev
    self-play patching) as a SWE-bench agent runner.
    """

    def __init__(
        self,
        netuid: int = 62,
        wallet_name: str = "miner",
        hotkey: str = "default",
        network: str = "finney",
        axon_port: int = 7999,
        max_cost_per_problem: float = 1.00,
        timeout_seconds: int = 1500,  # 25 minutes
    ):
        self.netuid = netuid
        self.wallet_name = wallet_name
        self.hotkey = hotkey
        self.network = network
        self.axon_port = axon_port
        self.max_cost = max_cost_per_problem
        self.timeout = timeout_seconds

        # Stats
        self.problems_received = 0
        self.problems_solved = 0
        self.total_cost = 0.0

    async def solve(self, instance: SWEBenchInstance) -> AgentResult:
        """Solve a SWE-bench instance using the Belief Engine brownfield pipeline.

        Steps:
        1. Clone the repo at the specified commit
        2. Run fix_issue with the problem statement
        3. Convert the patch to unified diff format
        4. Return the diff
        """
        t0 = time.time()
        self.problems_received += 1
        result = AgentResult(instance_id=instance.instance_id)

        with tempfile.TemporaryDirectory(prefix="belief_swe_") as tmpdir:
            repo_path = Path(tmpdir) / "repo"

            try:
                # Step 1: Clone the repo
                repo_url = f"https://github.com/{instance.repo}.git"
                logger.info(f"Miner: cloning {instance.repo} @ {instance.base_commit[:8]}")

                proc = subprocess.run(
                    ["git", "clone", repo_url, str(repo_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode != 0:
                    result.error = f"Clone failed: {proc.stderr[:200]}"
                    result.duration_seconds = time.time() - t0
                    return result

                # Checkout the base commit
                if instance.base_commit:
                    subprocess.run(
                        ["git", "checkout", instance.base_commit],
                        capture_output=True,
                        cwd=str(repo_path),
                        timeout=30,
                    )

                # Step 2: Run the brownfield pipeline
                logger.info(f"Miner: solving {instance.instance_id}")
                from belief.agents.brownfield_agent import fix_issue

                fix_result = await fix_issue(
                    repo_path=repo_path,
                    issue=instance.problem_statement,
                    n_patches=3,
                    n_tests=3,
                    max_agentless_iterations=3,
                    escalate_to_agentic=False,
                )

                # Step 3: Convert to unified diff
                if fix_result.success and fix_result.patch_file:
                    diff = self._make_unified_diff(
                        fix_result.patch_file,
                        fix_result.patch_old,
                        fix_result.patch_new,
                    )
                    result.model_patch = diff
                    result.success = True
                    self.problems_solved += 1
                    logger.info(f"Miner: SOLVED {instance.instance_id} ({time.time() - t0:.1f}s)")
                else:
                    result.error = fix_result.error or "No valid patch found"
                    logger.info(f"Miner: FAILED {instance.instance_id}: {result.error}")

                result.cost_usd = fix_result.cost_usd if hasattr(fix_result, "cost_usd") else 0.0
                self.total_cost += result.cost_usd

            except asyncio.TimeoutError:
                result.error = f"Timeout after {self.timeout}s"
            except Exception as e:
                result.error = str(e)
                logger.error(f"Miner: error on {instance.instance_id}: {e}")

        result.duration_seconds = time.time() - t0
        return result

    def _make_unified_diff(
        self,
        file_path: str,
        old_code: str,
        new_code: str,
    ) -> str:
        """Convert old/new code to unified diff format.

        SWE-bench expects unified diff patches that can be applied with `git apply`.
        """
        import difflib

        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        return "\n".join(diff)

    async def solve_batch(
        self,
        instances: list[SWEBenchInstance],
    ) -> list[AgentResult]:
        """Solve multiple SWE-bench instances sequentially."""
        results = []
        for i, instance in enumerate(instances):
            logger.info(f"Miner: [{i + 1}/{len(instances)}] {instance.instance_id}")
            result = await asyncio.wait_for(
                self.solve(instance),
                timeout=self.timeout,
            )
            results.append(result)
        return results

    def start(self):
        """Start as a Ridges-compatible miner.

        Ridges uses Fiber for networking, not traditional bt.Axon.
        The miner registers on the subnet and waits for evaluation.

        For now, this logs the configuration and waits — the actual
        Fiber integration requires studying ridgesai/ridges repo.
        """
        print(f"\n  Belief Engine Miner — Subnet {self.netuid}")
        print(f"  Wallet: {self.wallet_name} / {self.hotkey}")
        print(f"  Network: {self.network}")
        print(f"  Port: {self.axon_port}")
        print(f"  Max cost/problem: ${self.max_cost:.2f}")
        print(f"  Timeout: {self.timeout}s")
        print()
        print("  To register on Ridges:")
        print(f"    btcli subnet register --netuid {self.netuid} \\")
        print(f"      --wallet.name {self.wallet_name} --wallet.hotkey {self.hotkey}")
        print()
        print("  To run with Fiber:")
        print(f"    fiber-post-ip --netuid {self.netuid} \\")
        print(f"      --external_ip <YOUR_IP> --external_port {self.axon_port}")
        print()
        print("  Note: Ridges evaluates agents asynchronously.")
        print("  Submit your agent codebase via their platform.")
        print("  See: https://github.com/ridgesai/abstract-agent-runner")
        print()

        # Keep alive for monitoring
        try:
            while True:
                time.sleep(60)
                logger.info(
                    f"Miner: solved={self.problems_solved}/{self.problems_received} "
                    f"cost=${self.total_cost:.4f}"
                )
        except KeyboardInterrupt:
            logger.info("Miner shutting down")

    @property
    def stats(self) -> dict:
        return {
            "problems_received": self.problems_received,
            "problems_solved": self.problems_solved,
            "solve_rate": self.problems_solved / max(self.problems_received, 1),
            "total_cost_usd": self.total_cost,
        }


def main():
    """CLI entry point for the Belief Engine miner."""
    import argparse

    parser = argparse.ArgumentParser(description="Belief Engine — Ridges Miner")
    subparsers = parser.add_subparsers(dest="action")

    # Solve a single SWE-bench instance
    solve_parser = subparsers.add_parser("solve", help="Solve one SWE-bench instance")
    solve_parser.add_argument("--instance", required=True, help="Path to instance JSON")
    solve_parser.add_argument("--output", help="Output patch file path")

    # Test on SWE-bench dataset
    test_parser = subparsers.add_parser("test", help="Test on SWE-bench instances")
    test_parser.add_argument("--dataset", required=True, help="Path to dataset JSONL")
    test_parser.add_argument("--limit", type=int, default=5, help="Max instances to solve")

    # Start as Ridges miner
    start_parser = subparsers.add_parser("start", help="Start as Ridges miner")
    start_parser.add_argument("--netuid", type=int, default=62)
    start_parser.add_argument("--wallet-name", default="miner")
    start_parser.add_argument("--hotkey", default="default")
    start_parser.add_argument("--network", default="finney")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    miner = BeliefMiner()

    if args.action == "solve":
        instance = SWEBenchInstance.from_json(args.instance)
        result = asyncio.run(miner.solve(instance))
        if result.success:
            print(f"\n✓ Solved: {result.instance_id}")
            if args.output:
                Path(args.output).write_text(result.model_patch)
                print(f"  Patch saved to: {args.output}")
            else:
                print(result.model_patch)
        else:
            print(f"\n✗ Failed: {result.error}")

    elif args.action == "test":
        instances = []
        dataset_path = Path(args.dataset)
        if dataset_path.suffix == ".jsonl":
            for line in dataset_path.read_text().strip().split("\n")[: args.limit]:
                instances.append(SWEBenchInstance.from_dict(json.loads(line)))
        elif dataset_path.suffix == ".json":
            data = json.loads(dataset_path.read_text())
            if isinstance(data, list):
                instances = [SWEBenchInstance.from_dict(d) for d in data[: args.limit]]

        results = asyncio.run(miner.solve_batch(instances))
        solved = sum(1 for r in results if r.success)
        print(f"\nResults: {solved}/{len(results)} solved")
        for r in results:
            status = "✓" if r.success else "✗"
            print(f"  {status} {r.instance_id}: {r.duration_seconds:.1f}s, ${r.cost_usd:.4f}")

    elif args.action == "start":
        miner = BeliefMiner(
            netuid=args.netuid,
            wallet_name=args.wallet_name,
            hotkey=args.hotkey,
            network=args.network,
        )
        miner.start()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
