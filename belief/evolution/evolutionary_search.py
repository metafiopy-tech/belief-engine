"""Evolutionary Search — AlphaEvolve-Inspired Code Evolution.

Maintains a population of candidate solutions using MAP-Elites with an
island model. Uses an LLM ensemble (Haiku for breadth, Sonnet for depth)
as genetic operators to propose mutations.

Key research basis:
- AlphaEvolve (DeepMind, arXiv 2506.13131): MAP-Elites + island model + LLM ensemble
- FunSearch (Nature 2023): fixed scaffold + evolvable priority function
- CodeT (arXiv 2207.10397): dual execution for candidate ranking

The population stores programs binned by behavioral dimensions (complexity,
test pass rate, code size). The best individual per cell is retained.
Multiple islands evolve independently with periodic migration.

Usage:
    from belief.evolution.evolutionary_search import EvolutionarySearch
    search = EvolutionarySearch(evaluator=my_evaluator)
    best = await search.evolve(spec="Build a URL shortener", max_generations=5)
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("belief.evolution.search")


@dataclass
class Individual:
    """A single candidate solution in the population."""
    id: int
    code_files: dict[str, str]
    score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    complexity: int = 0  # Behavioral dimension: code complexity
    code_size: int = 0   # Behavioral dimension: total lines
    generation: int = 0
    parent_id: int = -1
    mutation_type: str = ""  # "breadth" (Haiku) or "depth" (Sonnet)


@dataclass
class Island:
    """An independently evolving sub-population."""
    id: int
    population: dict[tuple[int, int], Individual] = field(default_factory=dict)
    # Key is (complexity_bin, size_bin) — MAP-Elites behavioral grid

    def insert(self, individual: Individual) -> bool:
        """Insert into MAP-Elites grid. Returns True if it's a new best for its cell."""
        cell = self._get_cell(individual)
        existing = self.population.get(cell)
        if existing is None or individual.score > existing.score:
            self.population[cell] = individual
            return True
        return False

    def select_parent(self) -> Individual | None:
        """Fitness-proportional selection."""
        if not self.population:
            return None
        individuals = list(self.population.values())
        scores = [max(i.score, 0.01) for i in individuals]
        total = sum(scores)
        if total <= 0:
            return random.choice(individuals)
        weights = [s / total for s in scores]
        return random.choices(individuals, weights=weights, k=1)[0]

    def select_inspiration(self) -> Individual | None:
        """Diversity-biased selection — prefer individuals from less populated cells."""
        if not self.population:
            return None
        return random.choice(list(self.population.values()))

    def best(self) -> Individual | None:
        if not self.population:
            return None
        return max(self.population.values(), key=lambda i: i.score)

    def _get_cell(self, individual: Individual) -> tuple[int, int]:
        """Map an individual to its MAP-Elites grid cell."""
        complexity_bin = min(individual.complexity // 3, 4)  # 0-4
        size_bin = min(individual.code_size // 100, 4)       # 0-4
        return (complexity_bin, size_bin)


class EvolutionarySearch:
    """Population-based evolutionary search for code generation.

    Uses MAP-Elites with island model:
    - Multiple islands evolve independently
    - Periodic migration shares best individuals across islands
    - LLM ensemble: Haiku for breadth (many cheap mutations),
      Sonnet for depth (fewer quality mutations)
    - Cascade evaluation: cheap filter → expensive scoring
    """

    def __init__(
        self,
        evaluator: Callable | None = None,
        n_islands: int = 2,
        population_per_island: int = 10,
        migration_interval: int = 2,  # generations between migrations
    ):
        self.evaluator = evaluator
        self.n_islands = n_islands
        self.max_pop = population_per_island
        self.migration_interval = migration_interval
        self.islands = [Island(id=i) for i in range(n_islands)]
        self._next_id = 0
        self.generation = 0
        self.total_evaluations = 0

    async def evolve(
        self,
        spec: str,
        initial_candidates: list[dict[str, str]] | None = None,
        max_generations: int = 5,
        llm=None,
    ) -> Individual | None:
        """Run the evolutionary loop.

        Args:
            spec: Natural language specification
            initial_candidates: Seed population [{filename: code}, ...]
            max_generations: Max evolution generations
            llm: LLMClient for mutations

        Returns:
            Best Individual found, or None
        """
        if not self.evaluator:
            logger.warning("EvolutionarySearch: no evaluator provided")
            return None

        # Seed population
        if initial_candidates:
            for code_files in initial_candidates:
                ind = Individual(
                    id=self._next_id,
                    code_files=code_files,
                    code_size=sum(len(c) for c in code_files.values()),
                )
                self._next_id += 1

                # Evaluate
                score_result = await self._evaluate(ind)
                ind.score = score_result.get("score", 0.0)
                ind.tests_passed = score_result.get("tests_passed", 0)
                ind.tests_total = score_result.get("tests_total", 0)

                # Insert into random island
                island = random.choice(self.islands)
                island.insert(ind)

        logger.info(f"Evolution: seeded {sum(len(i.population) for i in self.islands)} individuals across {self.n_islands} islands")

        # Evolution loop
        best_score = self._global_best_score()
        plateau_count = 0

        for gen in range(max_generations):
            self.generation = gen
            gen_improvements = 0

            for island in self.islands:
                # Select parent + inspiration
                parent = island.select_parent()
                inspiration = island.select_inspiration()

                if not parent:
                    continue

                # Generate mutations — Haiku for breadth, Sonnet for depth
                mutations = []
                if llm:
                    # Breadth: 2 cheap Haiku mutations
                    for _ in range(2):
                        mutant = await self._mutate(
                            parent, spec, llm,
                            role="latios", temperature=0.7 + random.random() * 0.3,
                            inspiration=inspiration,
                        )
                        if mutant:
                            mutant.mutation_type = "breadth"
                            mutations.append(mutant)

                    # Depth: 1 quality Sonnet mutation
                    mutant = await self._mutate(
                        parent, spec, llm,
                        role="architect", temperature=0.3,
                        inspiration=inspiration,
                    )
                    if mutant:
                        mutant.mutation_type = "depth"
                        mutations.append(mutant)

                # Cascade evaluation + insertion
                for mutant in mutations:
                    # Cheap filter: syntax check
                    if not self._syntax_check(mutant):
                        continue

                    # Expensive evaluation
                    score_result = await self._evaluate(mutant)
                    mutant.score = score_result.get("score", 0.0)
                    mutant.tests_passed = score_result.get("tests_passed", 0)
                    mutant.tests_total = score_result.get("tests_total", 0)

                    if island.insert(mutant):
                        gen_improvements += 1

            # Migration
            if gen > 0 and gen % self.migration_interval == 0:
                self._migrate()

            # Check plateau
            new_best = self._global_best_score()
            if new_best <= best_score:
                plateau_count += 1
            else:
                plateau_count = 0
                best_score = new_best

            logger.info(
                f"Evolution gen {gen + 1}: best={best_score:.3f}, "
                f"improvements={gen_improvements}, plateau={plateau_count}"
            )

            if plateau_count >= 3:
                logger.info("Evolution: plateau detected, stopping")
                break

        return self._global_best()

    async def _mutate(
        self,
        parent: Individual,
        spec: str,
        llm,
        role: str = "latios",
        temperature: float = 0.5,
        inspiration: Individual | None = None,
    ) -> Individual | None:
        """Use LLM to generate a mutation of the parent."""
        # Build context: show parent code + spec + optional inspiration
        parent_summary = "\n".join(
            f"### {fname}\n```python\n{code[:500]}\n```"
            for fname, code in list(parent.code_files.items())[:3]
        )

        inspiration_text = ""
        if inspiration and inspiration.id != parent.id:
            inspiration_summary = "\n".join(
                f"### {fname}\n```python\n{code[:300]}\n```"
                for fname, code in list(inspiration.code_files.items())[:2]
            )
            inspiration_text = f"\n\nAnother approach (score={inspiration.score:.2f}):\n{inspiration_summary}"

        prompt = f"""Improve this code to better match the specification.

SPECIFICATION: {spec}

CURRENT CODE (score={parent.score:.2f}, {parent.tests_passed}/{parent.tests_total} tests):
{parent_summary}
{inspiration_text}

Generate an IMPROVED version. Output each file as:
###FILE: filename
<code>
###END"""

        try:
            response = await llm.generate_text(
                role=role,
                system="You are a code improver. Make targeted improvements to increase test pass rate.",
                prompt=prompt,
                temperature=temperature,
                max_tokens=3000,
            )

            # Parse files from response
            import re
            code_files = dict(parent.code_files)  # Start with parent's files
            parts = re.split(r"###FILE:\s*", response)
            for part in parts[1:]:
                nl = part.find("\n")
                if nl == -1:
                    continue
                fname = part[:nl].strip()
                content = re.sub(r"###END\s*$", "", part[nl + 1:]).strip()
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)
                if fname and content:
                    code_files[fname] = content

            mutant = Individual(
                id=self._next_id,
                code_files=code_files,
                code_size=sum(len(c) for c in code_files.values()),
                generation=self.generation,
                parent_id=parent.id,
            )
            self._next_id += 1
            return mutant

        except Exception as e:
            logger.debug(f"Mutation failed: {e}")
            return None

    async def _evaluate(self, individual: Individual) -> dict[str, Any]:
        """Evaluate an individual using the provided evaluator."""
        self.total_evaluations += 1
        try:
            if asyncio.iscoroutinefunction(self.evaluator):
                return await self.evaluator(individual.code_files)
            return self.evaluator(individual.code_files)
        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            return {"score": 0.0, "tests_passed": 0, "tests_total": 0}

    def _syntax_check(self, individual: Individual) -> bool:
        """Cheap filter: check all Python files parse."""
        import ast
        for fname, code in individual.code_files.items():
            if fname.endswith(".py"):
                try:
                    ast.parse(code)
                except SyntaxError:
                    return False
        return True

    def _migrate(self):
        """Share best individuals between islands."""
        bests = [island.best() for island in self.islands]
        bests = [b for b in bests if b is not None]

        for island in self.islands:
            for best in bests:
                if best not in island.population.values():
                    island.insert(best)

        logger.debug(f"Migration: shared {len(bests)} individuals across {self.n_islands} islands")

    def _global_best(self) -> Individual | None:
        bests = [island.best() for island in self.islands]
        bests = [b for b in bests if b is not None]
        return max(bests, key=lambda i: i.score) if bests else None

    def _global_best_score(self) -> float:
        best = self._global_best()
        return best.score if best else 0.0
