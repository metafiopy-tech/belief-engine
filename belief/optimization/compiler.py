"""
GEPA compilation wrapper for Belief Engine agents.

Optimizes agent prompts using DSPy's GEPA (reflective prompt evolution)
or MIPROv2 as fallback.  GEPA beats MIPROv2 by ~13% with 35x fewer
rollouts (ICLR 2026 oral).

DSPy is an optional dependency — this module raises ImportError if not
installed.

Usage:
    from belief.optimization.compiler import BeliefOptimizer
    optimizer = BeliefOptimizer(challenges)
    optimized, metrics = optimizer.compile_agent(module, "planner", trainset, valset)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("belief.optimization.compiler")


class BeliefOptimizer:
    """Optimizes agent prompts using DSPy/GEPA.

    Inner-loop optimizer within SICA's outer loop.  DSPy optimizes a
    FIXED computation graph (prompt text + few-shot demos) while SICA
    changes graph topology and builds tools.
    """

    def __init__(
        self,
        benchmark_challenges: Optional[list[dict]] = None,
        teacher_model: str = "claude-sonnet-4-6",
        student_model: str = "claude-haiku-4-5-20251001",
    ):
        self.challenges = benchmark_challenges or []
        self.teacher = teacher_model
        self.student = student_model

    def compile_agent(
        self,
        module: Any,
        agent_name: str,
        trainset: list[dict],
        valset: list[dict],
        num_candidates: int = 10,
        num_iterations: int = 5,
    ) -> tuple[Any, dict]:
        """Compile a single agent's prompts.

        Tries GEPA first, falls back to MIPROv2, then to no-op.

        Returns (optimized_module, metrics_dict).
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "dspy is required for prompt optimization. "
                "Install with: pip install 'belief-engine[optimize]'"
            )

        metric_fn = self._make_metric(agent_name)

        # Try GEPA first (best performance, fewest rollouts)
        optimizer = None
        optimizer_name = "none"

        try:
            from dspy.teleprompt import GEPA
            optimizer = GEPA(
                metric=metric_fn,
                num_candidates=num_candidates,
                num_iterations=num_iterations,
                max_bootstrapped_demos=3,
            )
            optimizer_name = "GEPA"
        except (ImportError, AttributeError):
            pass

        if optimizer is None:
            try:
                from dspy.teleprompt import MIPROv2
                optimizer = MIPROv2(
                    metric=metric_fn,
                    num_candidates=num_candidates,
                )
                optimizer_name = "MIPROv2"
            except (ImportError, AttributeError):
                pass

        if optimizer is None:
            try:
                from dspy.teleprompt import BootstrapFewShot
                optimizer = BootstrapFewShot(
                    metric=metric_fn,
                    max_bootstrapped_demos=3,
                )
                optimizer_name = "BootstrapFewShot"
            except (ImportError, AttributeError):
                pass

        if optimizer is None:
            logger.warning("No DSPy optimizer available, returning module unchanged")
            return module, {"agent": agent_name, "avg_score": 0.0, "optimizer": "none"}

        logger.info(f"Compiling {agent_name} with {optimizer_name}")

        try:
            optimized = optimizer.compile(module, trainset=trainset, valset=valset)
        except Exception as e:
            logger.warning(f"Compilation failed for {agent_name}: {e}")
            optimized = module

        metrics = self._evaluate(optimized, valset, agent_name)
        metrics["optimizer"] = optimizer_name
        return optimized, metrics

    def compile_all(
        self,
        modules: dict[str, Any],
        trainset: list[dict],
        valset: list[dict],
    ) -> dict[str, tuple[Any, dict]]:
        """Compile all agents.

        Returns {agent_name: (optimized_module, metrics)}.
        """
        results = {}
        for name, module in modules.items():
            logger.info(f"Optimizing {name}...")
            results[name] = self.compile_agent(module, name, trainset, valset)
        return results

    def extract_optimized_prompts(
        self,
        optimized_modules: dict[str, Any],
    ) -> dict[str, str]:
        """Extract optimized instruction strings from compiled modules.

        DSPy stores optimized instructions in the predictors' signatures.
        These can be saved and loaded into the non-DSPy agents.
        """
        prompts: dict[str, str] = {}

        for name, module in optimized_modules.items():
            if hasattr(module, "named_predictors"):
                for pred_name, predictor in module.named_predictors():
                    instructions = ""
                    if hasattr(predictor, "extended_signature"):
                        instructions = getattr(
                            predictor.extended_signature, "instructions", ""
                        )
                    elif hasattr(predictor, "signature"):
                        instructions = getattr(
                            predictor.signature, "instructions", ""
                        )
                    if instructions:
                        prompts[f"{name}.{pred_name}"] = instructions

        return prompts

    def save_optimized_prompts(self, prompts: dict[str, str], path: str) -> None:
        """Save optimized prompts to JSON for future use."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(prompts, indent=2))
        logger.info(f"Saved {len(prompts)} optimized prompts to {path}")

    def load_optimized_prompts(self, path: str) -> dict[str, str]:
        """Load previously optimized prompts."""
        return json.loads(Path(path).read_text())

    def _make_metric(self, agent_name: str):
        """Create a DSPy metric function for a specific agent."""

        def metric(example, prediction, trace=None):
            try:
                output = str(prediction)

                if agent_name == "planner":
                    return 1.0 if "step" in output.lower() else 0.0
                elif agent_name == "architect":
                    return 1.0 if "file" in output.lower() else 0.0
                elif agent_name == "builder":
                    return 1.0 if ("def " in output or "class " in output) else 0.0
                elif agent_name == "tester":
                    return 1.0 if "def test_" in output else 0.0
                elif agent_name == "debugger":
                    return 1.0 if "fix" in output.lower() else 0.0
                return 0.5
            except Exception:
                return 0.0

        return metric

    def _evaluate(self, module: Any, valset: list[dict], agent_name: str) -> dict:
        """Evaluate optimized module on validation set."""
        metric_fn = self._make_metric(agent_name)
        scores: list[float] = []

        for example in valset:
            try:
                pred = module.forward(**example)
                score = metric_fn(example, pred)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return {
            "agent": agent_name,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "n_examples": len(scores),
        }
