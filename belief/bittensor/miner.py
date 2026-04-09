"""Belief Engine Miner for Bittensor Subnet 62 (Ridges).

Wraps the Belief Engine's autonomous code generation pipeline as a
Bittensor miner neuron. When a validator sends a coding challenge
(issue description, test suite, repo context), the miner runs the
Belief Engine pipeline and returns generated code.

Architecture:
  Validator → sends challenge (synapse) → Miner
  Miner    → runs Belief Engine pipeline  → returns code
  Validator → runs tests on code          → assigns score
  Score    → feeds into Yuma Consensus   → TAO emissions

Requirements:
  pip install bittensor>=10.0.0
  ANTHROPIC_API_KEY set in environment

Usage:
  # Register on testnet first
  btcli subnet register --netuid 62 --wallet.name miner --wallet.hotkey default --subtensor.network test

  # Run the miner
  python -m belief.bittensor.miner --netuid 62 --wallet.name miner --wallet.hotkey default

  # Or use the CLI
  belief mine --netuid 62 --wallet.name miner --wallet.hotkey default
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("belief.bittensor.miner")


class BeliefMiner:
    """Bittensor miner that uses the Belief Engine to solve coding challenges.

    The miner:
    1. Registers on a subnet (default: 62 / Ridges)
    2. Listens for coding challenges from validators
    3. Runs each challenge through the Belief Engine pipeline
    4. Returns generated code files to the validator
    5. Validator scores the code by running tests
    """

    def __init__(
        self,
        netuid: int = 62,
        wallet_name: str = "miner",
        hotkey: str = "default",
        network: str = "finney",
        axon_port: int = 8091,
        max_cost_per_challenge: float = 0.50,
        max_iterations: int = 2,
    ):
        self.netuid = netuid
        self.wallet_name = wallet_name
        self.hotkey = hotkey
        self.network = network
        self.axon_port = axon_port
        self.max_cost = max_cost_per_challenge
        self.max_iterations = max_iterations

        # Stats
        self.challenges_received = 0
        self.challenges_solved = 0
        self.total_cost = 0.0
        self.total_time = 0.0

    async def solve_challenge(self, challenge: dict[str, Any]) -> dict[str, Any]:
        """Solve a coding challenge using the Belief Engine.

        Args:
            challenge: Dict containing the challenge specification.
                Expected keys (adapt to subnet protocol):
                - goal / description / issue: natural language description
                - test_code / tests: test suite to pass
                - repo_context / context: existing code context
                - language: target language (python/typescript)

        Returns:
            Dict with:
                - code_files: {filename: content} mapping
                - test_files: {filename: content} mapping
                - success: bool
                - time_seconds: float
                - cost_usd: float
        """
        t0 = time.time()
        self.challenges_received += 1

        # Extract goal from various possible fields
        goal = (
            challenge.get("goal")
            or challenge.get("description")
            or challenge.get("issue")
            or challenge.get("prompt")
            or ""
        )

        if not goal:
            return {
                "code_files": {},
                "test_files": {},
                "success": False,
                "time_seconds": time.time() - t0,
                "cost_usd": 0.0,
                "error": "No goal/description in challenge",
            }

        # Append test context if provided
        test_code = challenge.get("test_code") or challenge.get("tests") or ""
        if test_code:
            goal += f"\n\nTESTS TO PASS:\n{test_code[:3000]}"

        # Append repo context if provided
        repo_context = challenge.get("repo_context") or challenge.get("context") or ""
        if repo_context:
            goal += f"\n\nEXISTING CODE CONTEXT:\n{repo_context[:5000]}"

        try:
            from belief.cli import run as run_pipeline

            result = await run_pipeline(
                goal=goal,
                max_cost=self.max_cost,
                max_iterations=self.max_iterations,
            )

            code_files = result.get("code_files", {})
            test_files = result.get("test_files", {})

            # Extract cost from token usage
            token_usage = result.get("token_usage")
            cost = 0.0
            if token_usage:
                cost = getattr(token_usage, "total_cost_usd", 0.0)
                if isinstance(token_usage, dict):
                    cost = token_usage.get("total_cost_usd", 0.0)

            elapsed = time.time() - t0
            self.total_cost += cost
            self.total_time += elapsed

            success = bool(code_files)
            if success:
                self.challenges_solved += 1

            logger.info(
                f"Miner: challenge {'solved' if success else 'failed'} "
                f"({len(code_files)} files, {elapsed:.1f}s, ${cost:.4f})"
            )

            return {
                "code_files": code_files,
                "test_files": test_files,
                "success": success,
                "time_seconds": elapsed,
                "cost_usd": cost,
            }

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"Miner: pipeline failed: {e}")
            return {
                "code_files": {},
                "test_files": {},
                "success": False,
                "time_seconds": elapsed,
                "cost_usd": 0.0,
                "error": str(e),
            }

    def start(self):
        """Start the miner as a Bittensor neuron.

        This registers on the subnet, creates an Axon server,
        and begins listening for validator requests.
        """
        try:
            import bittensor as bt
        except ImportError:
            logger.error(
                "bittensor not installed. Run: pip install bittensor>=10.0.0"
            )
            sys.exit(1)

        # Configure logging
        bt.logging.set_trace(True)

        # Load wallet
        wallet = bt.wallet(name=self.wallet_name, hotkey=self.hotkey)
        logger.info(f"Miner wallet: {wallet}")

        # Connect to subtensor
        subtensor = bt.subtensor(network=self.network)
        logger.info(f"Connected to {self.network}")

        # Check registration
        metagraph = subtensor.metagraph(self.netuid)
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            logger.error(
                f"Hotkey {wallet.hotkey.ss58_address} not registered on subnet {self.netuid}. "
                f"Run: btcli subnet register --netuid {self.netuid} "
                f"--wallet.name {self.wallet_name} --wallet.hotkey {self.hotkey}"
            )
            sys.exit(1)

        uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        logger.info(f"Registered as UID {uid} on subnet {self.netuid}")

        # Create Axon (server)
        axon = bt.axon(wallet=wallet, port=self.axon_port)

        # Register the forward function
        # Note: The exact synapse type depends on the subnet's protocol.
        # For Ridges (SN62), this would be their specific challenge synapse.
        # This is a generic handler that works with any dict-based protocol.
        async def forward_fn(synapse):
            """Handle incoming challenge from validator."""
            try:
                # Extract challenge data from synapse
                challenge_data = {}
                if hasattr(synapse, "challenge"):
                    challenge_data = synapse.challenge
                elif hasattr(synapse, "prompt"):
                    challenge_data = {"goal": synapse.prompt}
                elif hasattr(synapse, "issue"):
                    challenge_data = {"goal": synapse.issue}
                else:
                    # Try to extract from any available field
                    for field in ["goal", "description", "task", "problem"]:
                        if hasattr(synapse, field):
                            challenge_data = {"goal": getattr(synapse, field)}
                            break

                # Solve the challenge
                result = await self.solve_challenge(challenge_data)

                # Pack result back into synapse
                if hasattr(synapse, "response"):
                    synapse.response = result
                if hasattr(synapse, "code"):
                    # Flatten code files into a single string if needed
                    all_code = "\n\n".join(
                        f"# === {fname} ===\n{content}"
                        for fname, content in result.get("code_files", {}).items()
                    )
                    synapse.code = all_code
                if hasattr(synapse, "code_files"):
                    synapse.code_files = json.dumps(result.get("code_files", {}))

            except Exception as e:
                logger.error(f"Forward failed: {e}\n{traceback.format_exc()}")

            return synapse

        # Register forward function
        # The actual registration depends on the subnet's protocol definition
        axon.attach(forward_fn=forward_fn)

        # Serve axon
        axon.serve(netuid=self.netuid, subtensor=subtensor)
        axon.start()

        logger.info(
            f"Miner running on port {self.axon_port} | "
            f"Subnet {self.netuid} | UID {uid}"
        )

        # Main loop — keep running and log stats
        try:
            step = 0
            while True:
                time.sleep(60)
                step += 1

                # Refresh metagraph periodically
                if step % 5 == 0:
                    metagraph.sync()
                    incentive = metagraph.I[uid].item() if uid < len(metagraph.I) else 0.0
                    logger.info(
                        f"Miner step {step} | "
                        f"challenges={self.challenges_received} "
                        f"solved={self.challenges_solved} "
                        f"cost=${self.total_cost:.4f} "
                        f"incentive={incentive:.6f}"
                    )

        except KeyboardInterrupt:
            logger.info("Miner shutting down")
            axon.stop()

    @property
    def stats(self) -> dict:
        return {
            "challenges_received": self.challenges_received,
            "challenges_solved": self.challenges_solved,
            "solve_rate": self.challenges_solved / max(self.challenges_received, 1),
            "total_cost_usd": self.total_cost,
            "total_time_seconds": self.total_time,
            "avg_time_per_challenge": self.total_time / max(self.challenges_received, 1),
        }


def main():
    """CLI entry point for the Belief Engine miner."""
    parser = argparse.ArgumentParser(description="Belief Engine Bittensor Miner")
    parser.add_argument("--netuid", type=int, default=62, help="Subnet ID (default: 62 / Ridges)")
    parser.add_argument("--wallet.name", dest="wallet_name", default="miner", help="Wallet name")
    parser.add_argument("--wallet.hotkey", dest="hotkey", default="default", help="Hotkey name")
    parser.add_argument("--subtensor.network", dest="network", default="finney",
                        help="Network (finney/test/local)")
    parser.add_argument("--axon.port", dest="axon_port", type=int, default=8091, help="Axon port")
    parser.add_argument("--max-cost", type=float, default=0.50,
                        help="Max USD per challenge (default: 0.50)")
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="Max build iterations per challenge (default: 2)")
    parser.add_argument("--logging.debug", dest="debug", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    miner = BeliefMiner(
        netuid=args.netuid,
        wallet_name=args.wallet_name,
        hotkey=args.hotkey,
        network=args.network,
        axon_port=args.axon_port,
        max_cost_per_challenge=args.max_cost,
        max_iterations=args.max_iterations,
    )
    miner.start()


if __name__ == "__main__":
    main()
