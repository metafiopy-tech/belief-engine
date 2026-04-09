# 🧠 Belief Engine

**An autonomous AI system that turns a sentence into working, tested software.**

Describe what you want. Belief Engine builds it, tests it, deploys it, and learns from every build.

```bash
pip install belief-engine
```

```bash
belief --goal "Build a bookmark manager API with FastAPI — CRUD with tags, GET /random. SQLite." \
  --deploy docker_local
```

---

## Benchmark: 85% Pass Rate

Tested on 20 challenges spanning single-file scripts to workflow DAG engines.

```
Pass rate:     17/20 (85%)
Avg weighted:  0.86
Cost per build: $0.18
Build time:    ~5 minutes

Tier 1 (scripts):        2/3
Tier 2 (CLIs + APIs):    4/4  ← perfect
Tier 3 (CRUD apps):      4/5
Tier 4 (multi-component): 3/4
Tier 5 (complex systems): 4/4  ← perfect
```

The engine builds complex systems (workflow engines, inventory managers, quiz platforms) more reliably than simple scripts. Tier 5 has been at 100% for three consecutive benchmark runs.

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

The engine doesn't just generate code — it **builds, tests, debugs, deploys, and learns**. Every build deposits knowledge into ChromaDB soil. Patterns, antipatterns, and covenants feed future builds. Build 50 is smarter than build 1.

## Key Numbers

| Metric | Value |
|--------|-------|
| Codebase | 76 Python files, ~19,800 lines |
| Benchmark | **17/20 (85%)** on 20-challenge suite |
| Builds completed | 53+ |
| Nutrients learned | 140+ |
| Self-learned covenants | 7 |
| Cost per build | **$0.18** (was $0.87 — 80% reduction) |
| Build time | ~5 minutes |
| LLM calls in validator | **0** (fully deterministic) |

## What Makes This Different

### It Learns From Every Build
ChromaDB-backed metabolization. Patterns, antipatterns, skeletons, and covenants accumulate in "soil" with FSRS confidence decay. After failed builds, the engine generates verbal self-critiques (Reflexion) that inform future similar builds.

### Incompleteness Drives Convergence
Latios finds what's missing. Latias protects what matters. The tension between them drives builds forward — the "remainder" after each operation seeds the next.

### Covenants Are Structural, Not Suggestions
Self-learned rules enforced via AST validators — not prompt injection. When the engine learned that `from __future__ import annotations` breaks SQLAlchemy's `Mapped` types, it added a deterministic AST check that removes the offending line automatically. Zero LLM tokens. Permanent fix.

### Real Tests, Not Imagination
The validator runs actual `pytest` in a sandbox. Real pass/fail. Weighted scoring: smoke tests = 3x weight, functional = 2x, edge cases = 1x, environment errors = 0x.

### SEED Self-Improvement
Every 5 builds, the engine analyzes its own failure patterns and proposes improvements. HIGH confidence proposals targeting prompt and configuration files are auto-applied with rollback on failure.

## Quick Start

```bash
pip install belief-engine

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Build something
belief --goal "Build a URL shortener with FastAPI and SQLite"

# Build + deploy
belief --goal "Build a REST API" --deploy docker_local --deploy-name myapi

# Run the benchmark
python3 -m belief.benchmark --tier=1,2,3,4,5
```

### From Source

```bash
git clone https://github.com/metafiopy-tech/belief-engine.git
cd belief-engine
pip install -e ".[dev]"
```

## Architecture

```
belief/
  agents/          — 11+ LangGraph agents (intake → validator)
  validators/      — AST covenant enforcers (deterministic, zero LLM)
  memory/          — ChromaDB metabolization (nutrients, soil, reflexion)
  refinement/      — Water cycle (analyze → fix → revalidate)
  deploy/          — Docker + Railway deployment
  codebase/        — Brownfield support (localization, patcher)
  languages/       — Multi-language adapters (Python, TypeScript)
  evolution/       — SEED self-improvement engine
  polarity/        — Latios/Latias incompleteness engine
  models/          — Pydantic models (state, artifacts, skeleton, contracts)
  hardening.py     — Budget limits, rate limiter, security scanner, audit log
  graph.py         — LangGraph pipeline wiring
  llm.py           — Anthropic API client with prompt caching + JSON repair
```

## Model Routing

| Agent | Model | Role |
|-------|-------|------|
| Research, Planner, Architect, Builder, Debugger | Sonnet 4.6 | Deep reasoning |
| Intake, Tester, Gap Analyst, Synthesizer, Validator, Latios | Haiku 4.5 | Mechanical tasks |
| Skeleton, Covenant Enforcer, Import Fix, Validator core | None | Deterministic (zero tokens) |

Prompt caching provides 90% savings on repeated system prompts. Combined with Haiku routing, builds cost **$0.15-0.25**.

## The Benchmark Suite

20 challenges across 5 tiers of complexity:

| Tier | Challenges | Examples |
|------|-----------|----------|
| 1 | 3 | FizzBuzz, Fibonacci, word count |
| 2 | 4 | Todo CLI, health API, calculator CLI, CSV stats |
| 3 | 5 | URL shortener, bookmark API, notes API, expense tracker, contacts |
| 4 | 4 | Blog engine, Kanban board, file vault, poll system |
| 5 | 4 | Event system, inventory manager, quiz engine, workflow DAG |

## Tech Stack

- **Python 3.11+** (tested on 3.14)
- **LangGraph** for agent orchestration
- **Anthropic Claude** (Sonnet 4.6 + Haiku 4.5)
- **ChromaDB** for learning memory
- **Docker** for deployment

## License

MIT

## Author

Built by [Fio](https://github.com/metafiopy-tech) — solo, from scratch, while making pizzas.

*"The remainder after every operation drives the next cycle."*
