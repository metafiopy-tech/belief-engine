# Belief Engine v2.0

An autonomous multi-agent code generation system that takes a natural language goal and builds working software — from research through deployment.

Describe what you want. Belief builds it.

## What It Does

```bash
python3 -m belief.cli --goal "Build a task management API with FastAPI, SQLAlchemy models, service layer, and tests"
```

The engine runs 11 LangGraph agents in a convergence loop:

**recomposer** → **intake** → **research** → **planner** → **architect** → **skeleton** → **builder** → **tester** → **executor** → **gap analyst** → **synthesizer** → **validator** → **refinement** → **decomposer**

Each agent has a specific role. The architect designs the file structure. The skeleton generator creates typed interfaces deterministically (zero LLM calls). The builder implements each file with compressed cross-file context. The executor verifies the code actually runs. If it doesn't, the debugger fixes it and loops back. If it runs but has quality issues, the **water cycle** refines it through up to 3 cycles of targeted fixes.

## Key Features

- **Skeleton-first generation** — typed interfaces before implementations, based on the CodeS paper
- **Dependency DAG ordering** — models before base classes before implementations before servers
- **Parallel file generation** — independent files at each dependency level build simultaneously
- **AST context compression** — function signatures extracted via ast.parse(), not full file contents
- **Multi-entry-point verification** — each service in a multi-service project verified independently
- **Multi-service architecture** — Tier 5 projects with ServiceArchitecture descriptors, Docker Compose
- **Metabolization architecture** — a ChromaDB-backed memory system where every build deposits nutrients (patterns, antipatterns, skeletons, covenants) that feed future builds
- **FSRS confidence decay** — nutrients that aren't reinforced fade over time; the system forgets what it doesn't use
- **Self-learned covenants** — immutable rules extracted from repeated failures (e.g., "no file over 200 lines")
- **Bottom-up import verification** — leaf modules verified first, so the debugger fixes the actual broken file
- **Deterministic error classifier** — ~40% of import/syntax errors fixed without LLM calls
- **Water cycle refinement** — test-driven polish loop using verbal self-reflection (Reflexion pattern)
- **Security scanner** — AST-based scan blocking eval/exec/os.system before execution
- **Rate limiting** — token bucket + exponential backoff for parallel generation

## The Water Cycle

When code runs but tests don't all pass, the refinement loop activates:

```
validator says fail_fixable + executor passed
    → analyze failure (verbal self-reflection — WHY did the test fail?)
    → generate fix (search/replace edit on ONE file)
    → revalidate (run tests, check for progress)
    → repeat up to 3 cycles
    → deposit refinement lessons into soil
```

The water never leaves the system. Code is polished, not rebuilt. If a fix causes regression (breaks a previously passing test), it's immediately rolled back. Each successful fix becomes a pattern nutrient; each regression becomes an antipattern.

## The Food Chain

Every build decomposes its results into atomic nutrients:
- **Patterns** — what worked (verified by passing builds)
- **Antipatterns** — what failed and why (verified by concrete errors)
- **Skeletons** — file structures that produced clean code
- **Covenants** — immutable rules from repeated failures

These nutrients are stored in ChromaDB with FSRS-based confidence decay. Before each new build, the recomposer retrieves relevant nutrients and injects them into the architect's context. The system gets smarter with every build.

```
Soil → Plant → Caterpillar → Bird → Soil
Build 1 → Nutrients → Build 2 → More Nutrients → Build 3 → ...
Nothing is lost. Everything is transformed.
```

## Quick Start

```bash
# Clone
git clone https://github.com/metafiopy-tech/belief-engine.git
cd belief-engine

# Install
pip install -e ".[dev]"
pip install chromadb

# Configure
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run
python3 -m belief.cli --goal "Build a hello world FastAPI server with a health check endpoint"
```

## Project Structure

```
belief/
  agents/          — 11 LangGraph agents (intake through validator)
  memory/          — Metabolization (nutrients, soil, decomposer, recomposer, lineage)
  models/          — Pydantic models (state, artifacts, skeleton, service architecture)
  refinement/      — Water cycle (analyzer, fixer, runner)
  codebase/        — Tier 7: ingestion, Agentless localization, patcher, import verifier
  languages/       — Tier 6: LanguageAdapter (Python, TypeScript), type pipeline
  config/          — Settings, model routing
  polarity/        — Incompleteness engine (Latios/Latias convergence)
  evolution/       — SEED self-improvement (reads covenants from soil)
  tools/           — Composition planner, deployment generator
  hardening.py     — Budget, rate limiter, security scanner
  graph.py         — LangGraph pipeline wiring
  cli.py           — Command-line interface
  llm.py           — Anthropic API client with JSON repair
```

## Build Tiers

| Tier | Description | Status |
|------|-------------|--------|
| 1 | Single file scripts | ✅ |
| 2 | MCP servers, simple APIs | ✅ |
| 3 | Package-structured apps | ✅ ($0.40-0.60) |
| 4 | Multi-component systems | ✅ |
| 5 | Distributed microservices | ✅ |
| 6 | Multi-language (Python + TypeScript) | ✅ |
| 7 | Extend existing codebases | ✅ |
| 8 | Autonomous deploy + monitor + heal | ✅ |

## Soil Status

124+ nutrients, 5 covenants self-learned from ~44 builds:

```bash
python3 -c "
from belief.memory.soil import Soil
from pathlib import Path
soil = Soil(Path('~/.belief-engine/soil'))
print(f'Nutrients: {soil.count()}')
print(f'By type: {soil.count_by_type()}')
"
```

## Deploy

```bash
# Build + deploy in one command
python3 -m belief.cli --goal "your goal" --deploy docker_local --deploy-name myapp

# Or deploy a previous build
python3 -m belief.deploy --list
python3 -m belief.deploy --target docker_local --name myapp

# Health check
python3 -m belief.deploy --health http://localhost:8000
```

## License

MIT

## Author

Built by [metafiopy-tech](https://github.com/metafiopy-tech)
