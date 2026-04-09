# 🧠 Belief Engine v2.0

**An autonomous AI system that turns a sentence into deployed software.**

Describe what you want. Belief Engine builds it, tests it, deploys it, and learns from every build.

```bash
python3 -m belief.cli \
  --goal "Build a bookmark manager API with FastAPI — CRUD with tags, GET /random. SQLite." \
  --deploy docker_local \
  --deploy-name bookmarks

# 240 seconds later → http://localhost:8000
```

---

## How It Works

```
You: "Build a todo app with Click"
  ↓
11 AI agents collaborate in a convergence loop:
  intake → research → planner → architect → skeleton → builder
  → covenant enforce → import fix → tester → executor → debugger
  → synthesizer → validator (real pytest) → water cycle → deploy
  ↓
Working software, tested, Dockerized, deployed.
```

The engine doesn't just generate code — it **builds, tests, debugs, deploys, and learns**. Every build deposits knowledge into ChromaDB soil. Patterns from successes, antipatterns from failures, and covenants from repeated mistakes feed future builds. Build 51 is smarter than build 1.

## Key Numbers

| Metric | Value |
|--------|-------|
| Files | 74 Python files |
| Lines | ~18,650 |
| Builds completed | 51+ |
| Nutrients learned | 135+ |
| Covenants self-learned | 7 |
| Cost per build | **$0.18** (was $0.87) |
| Build time | ~240s |
| LLM calls in validator | **0** (fully deterministic) |

## What Makes This Different

### It Learns From Every Build
ChromaDB-backed metabolization. Patterns, antipatterns, skeletons, and covenants accumulate in "soil" with FSRS confidence decay. Nutrients that aren't reinforced fade. The system forgets what it doesn't use.

### Incompleteness Drives Convergence
Latios finds what's missing. Latias protects what matters. The tension between them drives builds forward — the "remainder" after each operation seeds the next.

### Covenants Are Structural, Not Suggestions
Self-learned rules enforced via AST validators — not prompt injection. When the engine learned that `from __future__ import annotations` breaks SQLAlchemy's `Mapped` types, it didn't just add a note. It added a deterministic AST check that removes the offending line automatically. Zero LLM tokens. Permanent fix.

### Real Tests, Not Imagination
The validator runs actual `pytest` in a sandbox. Real pass/fail. Real error messages. Weighted scoring: smoke tests = 3x weight, functional = 2x, edge cases = 1x, environment errors (missing deps) = 0x.

### Skeleton-First Architecture
Typed interfaces generated deterministically before any implementation. Models → ABCs → implementations → servers. Parallel generation within each dependency level.

### SEED Self-Improvement
Every 5 builds, the engine analyzes its own failure patterns and proposes improvements to its own agents. Propose-only mode — human approval required.

## Architecture

```
belief/
  agents/          — 11+ LangGraph agents (intake → validator)
  validators/      — AST covenant enforcers (deterministic, zero LLM)
  memory/          — ChromaDB metabolization (nutrients, soil, decay)
  refinement/      — Water cycle (analyze → fix → revalidate)
  deploy/          — Docker + Railway deployment + health monitoring
  codebase/        — Brownfield support (Agentless localization, patcher)
  languages/       — Multi-language (Python, TypeScript adapters)
  evolution/       — SEED self-improvement engine
  polarity/        — Latios/Latias incompleteness engine
  models/          — Pydantic models (state, artifacts, skeleton, contracts)
  hardening.py     — Budget limits, rate limiter, security scanner, audit log
  graph.py         — LangGraph pipeline wiring
  llm.py           — Anthropic API client with prompt caching + JSON repair
```

## Quick Start

```bash
git clone https://github.com/metafiopy-tech/belief-engine.git
cd belief-engine
pip install -e ".[dev]"
cp .env.example .env   # Add your ANTHROPIC_API_KEY

# Build something
python3 -m belief.cli --goal "Build a URL shortener with FastAPI and SQLite"

# Build + deploy
python3 -m belief.cli --goal "Build a REST API" --deploy docker_local --deploy-name myapi

# Check what the engine has learned
python3 -c "
from belief.memory.soil import Soil
from pathlib import Path
soil = Soil(Path('~/.belief-engine/soil'))
print(f'Nutrients: {soil.count()}')
print(f'By type: {soil.count_by_type()}')
"
```

## The 7 Optimization Moves

The engine went through a research-driven optimization cycle that cut costs 55% and improved quality:

| Move | What | Impact |
|------|------|--------|
| 1 | Real pytest validator | Accurate verdicts from execution, not LLM guessing |
| 2 | AST covenant enforcers | SQLAlchemy bugs die permanently — deterministic fixes |
| 3 | Prompt caching + Haiku routing | $0.42 → $0.18 per build (55% reduction) |
| 4 | Repo map in tester/debugger | Phantom imports eliminated — tests import real modules |
| 5 | Contract-first generation | API contracts are source of truth for code AND tests |
| 6 | Architect/editor debugger | Sonnet diagnoses across all files, Haiku applies fixes |
| 7 | Safety infrastructure | Resource limits, audit logging, SEED approval gates |

## Build Tiers

| Tier | Description | Status |
|------|-------------|--------|
| 1 | Single file scripts | ✅ |
| 2 | MCP servers, simple APIs | ✅ |
| 3 | Package-structured apps (6-15 files) | ✅ |
| 4 | Multi-component systems | ✅ |
| 5 | Distributed microservices | ✅ |
| 6 | Multi-language (Python + TypeScript) | ✅ |
| 7 | Extend existing codebases | ✅ |
| 8 | Autonomous deploy + monitor + heal | ✅ |

## The Food Chain

Every build decomposes its results into atomic nutrients:

```
Soil → Plant → Caterpillar → Bird → Soil
Build 1 → Nutrients → Build 2 → More Nutrients → Build 3 → ...
Nothing is lost. Everything is transformed.
```

- **Patterns** — what worked (verified by passing builds)
- **Antipatterns** — what failed and why (linked to concrete errors)
- **Skeletons** — file structures that produced clean code
- **Covenants** — immutable rules from repeated failures (currently 7)

## Deploy CLI

```bash
# List previous builds
python3 -m belief.deploy --list

# Deploy a specific build
python3 -m belief.deploy --target docker_local --name myapp

# Health check
python3 -m belief.deploy --health http://localhost:8000
```

## Tech Stack

- **Python 3.14** on macOS Apple Silicon
- **LangGraph** for agent orchestration
- **Anthropic Claude** (Sonnet 4.6 for reasoning, Haiku 4.5 for mechanical tasks)
- **ChromaDB** for metabolization memory
- **Docker** for deployment
- **Railway** for cloud deployment (optional)

## Model Routing

| Agent | Model | Role |
|-------|-------|------|
| Research, Planner, Architect, Builder, Debugger | Sonnet 4.6 | Deep reasoning |
| Intake, Tester, Gap Analyst, Synthesizer, Validator, Latios, Executor | Haiku 4.5 | Mechanical tasks |
| Skeleton, Covenant Enforcer, Import Fix, Validator (core) | None | Deterministic (zero tokens) |

Prompt caching provides 90% savings on repeated system prompts. Combined with Haiku routing, builds cost **$0.15-0.25** compared to $0.87 before optimization.

## License

MIT

## Author

Built by [Fio](https://github.com/metafiopy-tech)

*"90% of anything buildable already exists as composable pieces. The intelligence is in finding, routing, and stitching — not generating."*
