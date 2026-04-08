# CLAUDE.md — Belief Engine v2.0

## What This Is

An autonomous multi-agent build system (68 Python files, ~15,500 lines) that takes a natural language goal and builds working software from research through deployment. It uses an incompleteness-driven convergence loop — Latios finds what's missing, Latias protects what matters, and the tension between them drives the build forward.

## How to Run

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add ANTHROPIC_API_KEY
python3 -m belief.cli --goal "Build a task management API with FastAPI"
```

## Project Structure

```
belief/
  agents/                  — 11 LangGraph agents
    intake.py              — Goal → RequirementSpec (criteria, complexity)
    research.py            — Web search + GitHub scout + package search
    planner.py             — RequirementSpec → ImplementationPlan
    architect.py           — Plan → SkeletonArtifact (typed interface layer)
    skeleton_builder.py    — Deterministic skeleton generation (zero LLM)
    skeleton_pass1.py      — Orchestrates skeleton Pass 1
    builder.py             — Manifest → code_files (parallel, DAG-ordered)
    tester.py              — Code → test files (AST-informed imports)
    executor.py            — Import verification + security scan + static import check
    debugger.py            — Search/replace fixes for large files, full replacement for small
    gap_analyst.py         — Deterministic + LLM gap analysis
    synthesizer.py         — Polish (truncation guard) + deployment artifacts
    validator.py           — Adversarial review + scoring
    parallel_planner.py    — DAG ordering for parallel file generation
    repo_map.py            — AST-based context compression
    base.py                — BaseAgent ABC

  memory/                  — Metabolization (food chain memory)
    nutrients.py           — Nutrient model with FSRS decay
    soil.py                — ChromaDB store (deposit, retrieve, decay, archive)
    decomposer.py          — Post-build nutrient extraction
    recomposer.py          — Pre-build nutrient injection
    lineage.py             — Cross-build correlation, covenant promotion
    store.py               — Build history (SQLite)

  refinement/              — Water Cycle (test-driven polish)
    __init__.py            — RefinementState, CycleRecord
    analyzer.py            — Verbal self-reflection on test failures
    fixer.py               — Search/replace edit generation
    runner.py              — 3-cycle loop with regression detection

  codebase/                — Tier 7 brownfield support
    __init__.py            — Codebase ingestion, Agentless localization, repo map
    patcher.py             — Search/replace patch generation for existing repos
    imports.py             — Static import verifier + auto-fixer (Covenant #3)

  languages/               — Tier 6 multi-language support
    __init__.py            — LanguageAdapter ABC, registry, detection
    python_adapter.py      — PythonAdapter (AST parsing, verification)
    typescript_adapter.py  — TypeScriptAdapter (package.json, tsc, regex parsing)
    types.py               — Cross-language type pipeline (Pydantic → TS/Go)

  models/                  — Pydantic models
    state.py               — UnifiedState (the single state object)
    artifacts.py           — All artifact models (proven, from Forge)
    skeleton.py            — SkeletonArtifact, FileRole, ModelChain
    service_architecture.py — Multi-service descriptor (Tier 5)
    symbol_registry.py     — Symbol tracking for skeleton context
    dependency_dag.py      — DAG utilities

  polarity/                — Incompleteness engine
    incompleteness.py      — Latios — extract_remainder()
    belief.py              — Latias — extract_covenant()
    convergence.py         — Oscillation detection, circuit breakers
    crosstalk.py           — Gap ↔ world feedback
    frequency.py           — Moment-to-moment coherence

  evolution/               — SEED self-improvement
    __init__.py            — SEED reads covenants/antipatterns from soil

  config/                  — Settings, model routing
  tools/                   — Composition planner, deployment generator
  prompts/                 — System + user prompt templates
  daemons/                 — Health monitoring
  hardening.py             — BuildBudget, AsyncTokenBucket, SecurityScanner
  graph.py                 — LangGraph pipeline (all routing + nodes)
  llm.py                   — Anthropic API client with JSON repair
  cli.py                   — CLI entry point
```

## Pipeline Flow

```
recomposer → intake → research → planner → architect → skeleton_pass1
→ builder → import_fix → tester → executor → gap_analyst
→ [debugger loop if needed]
→ synthesizer → validator
→ [refinement loop if fail_fixable + exec passed]
→ decomposer → END
```

## Key Architecture Decisions

- **Skeleton-first**: typed interfaces before implementations. Skeletons generated deterministically (zero LLM). Builder generates against contracts.
- **DAG ordering**: models → ABCs → implementations → servers. Parallel within each level.
- **Covenants**: immutable rules self-learned from repeated failures. Currently 3 active. Injected into architect AND builder prompts.
- **Water cycle**: when code runs but tests fail, 3 cycles of analyze→fix→revalidate. Uses search/replace (not full regen). Rollback on regression.
- **Search/replace debugger**: files >2000 chars get surgical edits. Files <2000 chars get full replacement. Prevents truncation of large files.
- **Static import verifier**: traces every `from X import Y` and auto-fixes name mismatches before execution. Implements Covenant #3.
- **Food chain**: every build deposits nutrients. Patterns from successes, antipatterns from failures. 3+ antipatterns with same root → covenant.
- **No rebuild on working code**: if executor passes, Latios can't force a rebuild. Refinement polishes instead.

## Commands

```bash
# Standard build
python3 -m belief.cli --goal "your goal here"

# With budget cap
python3 -m belief.cli --goal "your goal" --max-cost 5.0

# Check soil
python3 -c "
from belief.memory.soil import Soil
from pathlib import Path
soil = Soil(Path('~/.belief-engine/soil'))
print(f'Nutrients: {soil.count()}')
print(f'By type: {soil.count_by_type()}')
"
```

## Environment

- Python 3.14 on Mac (python3/pip3 for system, python/pip inside venv)
- pyproject.toml build backend: setuptools.build_meta
- All Belief components in ~/Desktop/belief-engine/
