# Belief Engine

**An autonomous AI system that turns a sentence into working, tested software — and improves itself after every build.**

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
Tier 2 (CLIs + APIs):    4/4
Tier 3 (CRUD apps):      4/5
Tier 4 (multi-component): 3/4
Tier 5 (complex systems): 4/4
```

The engine builds complex systems (workflow engines, inventory managers, quiz platforms) more reliably than simple scripts. Tier 5 has been at 100% for three consecutive benchmark runs.

## How It Works

```
You: "Build a todo app with Click"
  |
11 AI agents collaborate in a convergence loop:
  intake -> research -> planner -> architect -> skeleton -> builder
  -> covenant enforce -> import fix -> tester -> executor -> debugger
  -> synthesizer -> validator (real pytest) -> water cycle -> deploy
  |
Working software, tested, Dockerized, deployed.
```

The engine doesn't just generate code — it **builds, tests, debugs, deploys, and learns**. Every build deposits knowledge into ChromaDB soil. Patterns, antipatterns, and covenants feed future builds. Build 50 is smarter than build 1.

## v3.0: Autocatalytic Self-Improvement

v3.0 adds a full self-improvement loop. The engine builds tools for itself, discovers its own rules, and measures its own progress.

```
           Jitterbug Cycle
          /               \
    Expansion          Integration
   (diverse builds)    (accept/prune)
        |                   |
    Compression        Validation
   (cluster failures)  (regression check)
        |
   Reconstruction
   (build tools, crystallize covenants)
```

**5 new subsystems:**

| Subsystem | What it does |
|-----------|-------------|
| **FSRS Memory** | Spaced-repetition decay on all knowledge. Stale patterns fade; reinforced ones strengthen. |
| **Evolutionary Archive** | SQLite DAG of every agent version. DGM-style parent selection preserves stepping stones. |
| **Crystallizer** | Discovers covenants from build traces. Template sweep (Daikon) + Houdini filter + promotion. |
| **Autocatalytic NEW_TOOL** | The engine uses its own pipeline to build tools for itself. Failure clusters drive tool goals. |
| **Safety Guardrails** | Async overseer, evaluator integrity hashes, Goodhart canary (held-out benchmark), cost monitors. |

## Key Numbers

| Metric | Value |
|--------|-------|
| Codebase | 131 Python files, ~37,800 lines |
| Benchmark | **17/20 (85%)** on 20-challenge suite |
| Builds completed | 53+ |
| Nutrients learned | 900+ |
| Self-learned covenants | 7 static + dynamic discovery |
| Cost per build | **$0.18** (was $0.87 -- 80% reduction) |
| Build time | ~5 minutes |
| ChromaDB collections | 5 (tools, episodes, principles, failures, covenants) |

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
belief benchmark --tiers 1 2 3 4 5
```

### From Source

```bash
git clone https://github.com/metafiopy-tech/belief-engine.git
cd belief-engine
pip install -e ".[dev]"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `belief --goal "..."` | Build software from a goal |
| `belief benchmark` | Run benchmark challenges |
| `belief sica --iterations N` | Run SICA self-improvement |
| `belief jitterbug` | Run compression-reconstruction cycle |
| `belief jitterbug --dry-run` | Expansion + compression only |
| `belief progression` | Display generative chain stage |
| `belief optimize [agent]` | DSPy/GEPA prompt optimization |
| `belief dashboard` | Metrics dashboard |
| `belief dashboard --json` | Metrics as JSON |
| `belief fix --repo PATH --issue "..."` | Fix an issue in existing code |

## Architecture

```
belief/
  agents/          -- 11+ LangGraph agents (intake -> validator)
  validators/      -- AST covenant enforcers + dynamic covenant registry
  memory/          -- ChromaDB metabolization (5 collections, FSRS decay)
  refinement/      -- Water cycle (analyze -> fix -> revalidate)
  evolution/       -- SICA, archive, crystallizer, jitterbug, progression
  optimization/    -- DSPy/GEPA prompt optimization (optional)
  safety/          -- Overseer, probes, Goodhart canary
  metrics/         -- Dashboard, growth analysis
  deploy/          -- Docker + Railway deployment
  codebase/        -- Brownfield support (localization, patcher)
  languages/       -- Multi-language adapters (Python, TypeScript)
  polarity/        -- Latios/Latias incompleteness engine
  models/          -- Pydantic models (state, artifacts, skeleton, contracts)
  hardening.py     -- Budget limits, rate limiter, security scanner, audit log
  graph.py         -- LangGraph pipeline wiring
  llm.py           -- Anthropic API client with prompt caching + JSON repair
```

## Model Routing

| Agent | Model | Role |
|-------|-------|------|
| Research, Planner, Architect, Builder, Debugger | Sonnet 4.6 | Deep reasoning |
| Intake, Tester, Gap Analyst, Synthesizer, Validator, Latios | Haiku 4.5 | Mechanical tasks |
| Skeleton, Covenant Enforcer, Import Fix, Validator core | None | Deterministic (zero tokens) |

Prompt caching provides 90% savings on repeated system prompts. Combined with Haiku routing, builds cost **$0.15-0.25**.

## Tech Stack

- **Python 3.11+** (tested on 3.14)
- **LangGraph** for agent orchestration
- **Anthropic Claude** (Sonnet 4.6 + Haiku 4.5)
- **ChromaDB** for learning memory (5 collections with FSRS)
- **SQLite** for evolutionary archive
- **Docker** for deployment
- **DSPy** (optional) for prompt optimization

## License

MIT

## Author

Built by [Fio](https://github.com/metafiopy-tech) -- solo, from scratch, while making pizzas.

*"The remainder after every operation drives the next cycle."*
