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

## Validation: Does accumulated knowledge help a local model?

**Research question.** The engine stores patterns, antipatterns, covenants, and skeletons in ChromaDB soil after every build. Does that accumulated knowledge cause a measurable quality lift when the engine is paired with a local model — or is the lift just noise from running more computation against the same weights?

**Protocol.** Four paired A/B runs over 2026-04-22. Same model (qwen2.5-coder:14b, Q4_K_M), same hardware (MacBook Air M2 16GB), same challenge set (five tier-1/tier-2 problems rotating between runs). The only variable between the two arms: whether the engine's ChromaDB soil, covenants, and debug memory are connected to the model on inference.

**Results.**

| Run (timestamp)     | Engine + local | Raw local | Δ   |
|---------------------|----------------|-----------|-----|
| 02:46               | 5 / 5          | 2 / 5     | +60% |
| 07:03               | 5 / 5          | 2 / 5     | +60% |
| 08:03               | 5 / 5          | 3 / 5     | +40% |
| 08:52               | 5 / 5          | 4 / 5     | +20% |
| **Cumulative n=20** | **20 / 20**    | **11 / 20** | **+45%** |

Fisher's exact test on the paired n=20 gives **p < 0.001**.

A fifth run the next morning on a fresh three-challenge sample reproduced the pattern: engine 3/3 vs raw 1/3, +66.7% lift. By the end of the experiment window the archive held 424 builds, 37 covenants, and had extracted ~100 new nutrients in the previous 24 hours.

**What this means.** For *this* local model, on *this* paired benchmark, a ChromaDB-backed context layer with FSRS-decayed nutrients and AST-enforced covenants produces a statistically significant quality lift. The local-14B pipeline solved problems it could not solve without the engine's accumulated knowledge.

**Honest limitations.**
- n=20 is below publication-grade for a strong claim across all 20 benchmark challenges; the next milestone is n=50 paired with per-domain analysis.
- Challenges rotate, so the raw-local scores drift between runs (easier challenges rotate in as the engine's coverage grows).
- Engine wall clock is 10-15× slower per build (~255-900s vs ~30-70s raw). Quality/time tradeoff, not a free lunch.
- Factorial ablation (soil × covenants × debug × skeleton) is needed to attribute the lift — which subsystem is load-bearing is still an open question.

**Reproducibility.** Raw data: `~/.belief-engine/experiments.db`. Methodology and statistical protocol: `docs/validation/v3.1.0-consistency-results.md`.

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

### Local-only quick start (v3.1)

No API key, no cloud calls, no per-build cost. Everything runs on
your laptop against [Ollama](https://ollama.com). Requires ~16 GB
of RAM for the default model.

```bash
# One-command setup (installs Ollama, pulls qwen2.5-coder, runs a smoke build):
curl -fsSL https://raw.githubusercontent.com/metafiopy-tech/belief-engine/main/scripts/belief-setup.sh | bash

# Or, step by step:
curl -fsSL https://ollama.ai/install.sh | sh     # one-off
ollama pull qwen2.5-coder:14b                    # ~8 GB download
pip install "belief-engine[full]"

# Point every agent at the local model:
export BELIEF_MODEL_MODE=local
belief --goal "Build a Python script that prints hello world"
```

Hybrid mode (mix local + Claude) is one env var away — see
[Adding Claude for hard tasks](#adding-claude-for-hard-tasks-hybrid-mode)
below.

### From Source

```bash
git clone https://github.com/metafiopy-tech/belief-engine.git
cd belief-engine
pip install -e ".[dev]"
```

### How the soil compounds over time

Every build deposits knowledge — patterns, antipatterns, skeletons,
covenants — into the ChromaDB soil at `~/.belief-engine/soil`. The
soil is the engine's working memory. Build N is smarter than
build N-1 because build N-1 left behind what worked, what didn't,
and why.

Decay is FSRS-4.5 spaced repetition with **clade-productivity
weighting** (v3.1): a nutrient's retention is proportional to how
often its descendants succeed in later builds. Nutrients whose
downstream uses keep working stay sharp; orphans fade. Contradicted
nutrients are soft-deleted with a `valid_until` timestamp, never
purged — `belief manifold` can show the soil as it was on any
historical date.

You can watch this happen:

```bash
belief dashboard        # metrics: pass rate, cost, nutrients, covenants
belief manifold         # clusters by domain + coverage gaps (v3.1)
```

### Checking progression per vertical

The generative-chain progression tracker (Session 7) scores each of
eight verticals independently — `fastapi`, `cli`, `mcp`, `data`,
`async`, `library`, `script`, `general` — so you can see which
domains the engine has matured in and which it hasn't touched yet.

```bash
belief progression
```

Output lists every domain and its current stage (Seed → Cluster →
Tessellation → Basis → Connectivity → Archetypes). Domains stuck at
Seed are the ones to target with the next round of builds.

### Adding Photosynthesis for autonomous goal generation

The Grinder daemon (Session 8) picks goals out of a queue and
builds them continuously. The Photosynthesis daemon (Sessions 3–5)
populates that queue by harvesting candidate build goals from
GitHub, PyPI, HN, Stack Overflow, RSS feeds, and ArXiv, then
filtering them through a four-stage cascade (novelty band → ACCEL
heap → LLM judge). Together they turn the engine into a
self-running research workshop:

```bash
# Background the grinder (drains the goal queue):
belief grinder start --max-builds 100

# Photosynthesis lives in its own package extras:
pip install "belief-engine[photosynthesis]"
```

### Adding Claude for hard tasks (hybrid mode)

Hybrid mode routes mechanical agents (intake, tester, synthesizer,
validator) to the local model and keeps reasoning agents (research,
planner, architect, builder, debugger) on Claude — the same
quality ceiling as cloud mode at roughly 1/4 the cost.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BELIEF_MODEL_MODE=hybrid
belief --goal "Build a distributed task queue with priority lanes"
```

v3.1 additionally introduces a **confidence-probe-gated
escalation** path: when the Session-10 probe judges the local model
unlikely to succeed on a given call (confidence < 0.4), that single
call escalates to Claude automatically. Local-first; Claude is only
paid for when needed.

## CLI Commands

| Command | Description |
|---------|-------------|
| `belief --goal "..."` | Build software from a goal |
| `belief benchmark` | Run benchmark challenges |
| `belief sica --iterations N` | Run SICA self-improvement |
| `belief jitterbug` | Run compression-reconstruction cycle |
| `belief jitterbug --dry-run` | Expansion + compression only |
| `belief progression` | Per-domain generative-chain stage |
| `belief manifold` | Knowledge topology: clusters, cross-links, gaps (v3.1) |
| `belief manifold --json` | Manifold as machine-readable JSON |
| `belief optimize [agent]` | DSPy/GEPA prompt optimization |
| `belief dashboard` | Metrics dashboard |
| `belief dashboard --json` | Metrics as JSON |
| `belief library` | Named library of promoted tools (v3.0) |
| `belief grinder start` | Autonomous build loop |
| `belief models` | Show active model routing table |
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
