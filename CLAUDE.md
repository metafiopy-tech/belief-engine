# CLAUDE.md — Belief Engine v2.0

## What This Is

An autonomous multi-agent build system (73 Python files, ~17,300 lines) that takes a natural language goal and produces working, deployed software. It uses an incompleteness-driven convergence loop — Latios finds what's missing, Latias protects what matters, and the tension between them drives the build forward. The system learns from every build through a metabolization architecture: patterns, antipatterns, and covenants accumulate in ChromaDB soil, making each subsequent build smarter.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add ANTHROPIC_API_KEY

# Build
python3 -m belief.cli --goal "Build a bookmark API with FastAPI and SQLite"

# Build + deploy in one command
python3 -m belief.cli --goal "Build a URL shortener" --deploy docker_local --deploy-name shortener

# Deploy a previous build
python3 -m belief.deploy --list
python3 -m belief.deploy --target docker_local --name myapp

# Health check a deployed service
python3 -m belief.deploy --health http://localhost:8000

# Check soil (learning state)
python3 -c "
from belief.memory.soil import Soil
from pathlib import Path
soil = Soil(Path('~/.belief-engine/soil'))
print(f'Nutrients: {soil.count()}')
print(f'By type: {soil.count_by_type()}')
"
```

## Project Structure

```
belief/
  agents/                  — 11+ LangGraph agents
    intake.py              — Goal → RequirementSpec (criteria, complexity)
    research.py            — Web search + GitHub scout + package search
    planner.py             — RequirementSpec → ImplementationPlan
    architect.py           — Plan → SkeletonArtifact (typed interface layer)
    skeleton_builder.py    — Deterministic skeleton generation (zero LLM)
    skeleton_pass1.py      — Orchestrates skeleton Pass 1
    builder.py             — Manifest → code_files (parallel, DAG-ordered)
    tester.py              — Spec-first tiered tests (P0 smoke, P1 functional, P2 edge)
    executor.py            — Import verification + security scan + static import check
    debugger.py            — Search/replace for large files, full replacement for small
    gap_analyst.py         — Deterministic + LLM gap analysis
    synthesizer.py         — Polish + deployment artifacts (Dockerfile, CI/CD, railway.toml)
    validator.py           — Weighted scoring (smoke=3x, functional=2x, edge=1x, env=0x)
    parallel_planner.py    — DAG ordering for parallel file generation
    repo_map.py            — AST-based context compression

  memory/                  — Metabolization (food chain memory)
    nutrients.py           — Nutrient model with FSRS decay
    soil.py                — ChromaDB store (deposit, retrieve, reinforce, archive)
    decomposer.py          — Post-build nutrient extraction
    recomposer.py          — Pre-build nutrient injection
    lineage.py             — Cross-build correlation, covenant promotion
    store.py               — Build history (SQLite)

  refinement/              — Water Cycle (test-driven polish)
    analyzer.py            — Verbal self-reflection on test failures
    fixer.py               — Single-file + multi-file search/replace edits
    runner.py              — 3-cycle loop with regression detection + dep install

  deploy/                  — Tier 8 autonomous deployment
    __init__.py            — Deploy pipeline (Railway, Docker local)
    __main__.py            — Deploy CLI (python3 -m belief.deploy)
    monitor.py             — Health monitoring (HTTP checks, degradation detection)
    remediation.py         — Auto-remediation (diagnose → fix → test → redeploy)
    autonomous.py          — Full loop: build → deploy → monitor → heal

  codebase/                — Tier 7 brownfield support
    __init__.py            — Codebase ingestion, Agentless localization, repo map
    patcher.py             — Search/replace patch generation for existing repos
    imports.py             — Static import verifier + auto-fixer (Covenant #3)

  languages/               — Tier 6 multi-language support
    __init__.py            — LanguageAdapter ABC, registry, detection
    python_adapter.py      — PythonAdapter (AST parsing, verification, signatures)
    typescript_adapter.py  — TypeScriptAdapter (package.json, tsc, regex parsing)
    types.py               — Cross-language type pipeline (Pydantic → TS/Go)

  models/                  — Pydantic models
    state.py               — UnifiedState (the single state object)
    artifacts.py           — Artifacts + TestCase with tiers + weighted ValidationResult
    skeleton.py            — SkeletonArtifact, FileRole, ModelChain
    service_architecture.py — Multi-service descriptor (Tier 5, multi-language)

  polarity/                — Incompleteness engine (Latios/Latias)
  evolution/               — SEED self-improvement (triggers every 5 builds)
  config/                  — Settings, model routing
  tools/                   — Composition planner, deployment generator
  prompts/                 — System + user prompt templates
  daemons/                 — Health monitoring
  hardening.py             — BuildBudget, rate limiter, SecurityScanner
  graph.py                 — LangGraph pipeline (all routing + nodes)
  llm.py                   — Anthropic API client with JSON repair
  cli.py                   — CLI entry point (--goal, --deploy, --max-cost)
```

## Pipeline Flow

```
recomposer → intake → research → planner → architect → skeleton_pass1
→ builder → import_fix → tester → executor → gap_analyst
→ [debugger loop: search/replace for large files]
→ synthesizer → validator (weighted scoring)
→ [refinement loop if fail_fixable + exec passed: multi-file capable]
→ decomposer → END
```

## Key Architecture Decisions

- **Skeleton-first**: typed interfaces before implementations. Skeletons generated deterministically (zero LLM).
- **DAG ordering**: models → ABCs → implementations → servers. Parallel within each level.
- **5 covenants self-learned**: explicit imports, 200-line limit, static import verification, SQLAlchemy type annotations, Mapped/mapped_column imports.
- **Water cycle**: analyze → fix → revalidate. Search/replace edits. Multi-file when single-file plateaus. Rollback on regression.
- **Weighted scoring**: smoke=3x, functional=2x, edge=1x, environment=0x. Score ≥0.85 + all smoke passing → PASS verdict.
- **Food chain**: builds deposit nutrients → patterns from successes, antipatterns from failures → 3+ antipatterns cluster → covenant.
- **SEED**: triggers every 5 builds, reads soil antipatterns, proposes improvements. Propose-only mode (human approval required).
- **SQLAlchemy-aware**: database.py skeleton, no `__future__` with Mapped types, builder warns against the conflict.
- **Import fix node**: statically traces `from X import Y`, auto-fixes Pipeline→DataPipeline mismatches before tester runs.

## Build Tiers

| Tier | Description | Status |
|------|-------------|--------|
| 1 | Single file scripts | ✅ |
| 2 | MCP servers, simple APIs | ✅ |
| 3 | Package-structured apps | ✅ ($0.25-0.55) |
| 4 | Multi-component systems | ✅ |
| 5 | Distributed microservices | ✅ |
| 6 | Multi-language (Python + TypeScript) | ✅ Adapters wired |
| 7 | Extend existing codebases | ✅ Codebase ingestion + patcher |
| 8 | Autonomous deploy + monitor + heal | ✅ Docker + Railway |

## Environment

- Python 3.14 on Mac (python3/pip3 for system)
- pyproject.toml build backend: setuptools.build_meta
- All Belief components in ~/Desktop/belief-engine/
