# 🧠 Belief Engine

**An autonomous AI system that turns a sentence into working, tested software.**

Describe what you want. Belief Engine builds it, tests it, debugs it, deploys it, and learns from every build.

```bash
pip install belief-engine
```

```bash
belief build --goal "Build a bookmark manager API with FastAPI — CRUD with tags, GET /random. SQLite."
```

---

## Benchmark: 90% Pass Rate

Tested on 35 challenges across 8 tiers — from FizzBuzz to multi-service architectures and TypeScript.

```
Pass rate:     27/30 (90%) on Tiers 1-7
Avg weighted:  0.93
Cost per build: $0.50
Build time:    ~5 minutes

Tier 1 (scripts):            5/5
Tier 2 (CLIs + APIs):        4/4  ← perfect
Tier 3 (CRUD apps):          3/5
Tier 4 (multi-component):    4/5
Tier 5 (complex systems):    4/4  ← perfect
Tier 6 (multi-service):      5/5  ← perfect (was 0%)
Tier 7 (brownfield):         5/5  ← perfect
Tier 8 (TypeScript):         TBD
```

## How It Works

```
You: "Build a todo app with Click"
  ↓
11+ AI agents collaborate in a convergence loop:
  intake → research → planner → architect → skeleton → builder
  → covenant enforce → import fix → tester → executor → debugger
  → synthesizer → validator (real pytest) → water cycle → deploy
  ↓
Working software, tested, Dockerized, deployed.
```

The engine doesn't just generate code — it **builds, tests, debugs, deploys, and learns**. Every build deposits knowledge into ChromaDB soil. Patterns, antipatterns, and covenants feed future builds.

## Key Numbers

| Metric | Value |
|--------|-------|
| Codebase | 118 Python files, ~28,000 lines |
| Benchmark | **27/30 (90%)** on 35-challenge suite |
| Tiers | 8 (scripts → multi-service → TypeScript) |
| Cost per build | **$0.50** |
| Build time | ~5 minutes |
| Self-learned covenants | 8+ |
| LLM calls in validator | **0** (fully deterministic) |

## What Makes This Different

### It Learns From Every Build
ChromaDB-backed metabolization. Patterns, antipatterns, skeletons, and covenants accumulate in "soil" with FSRS confidence decay. After failed builds, the engine generates verbal self-critiques (Reflexion) that inform future similar builds.

### Reflexion-Based Refinement
Each refinement cycle generates a structured reflection: what was tried, what assumption was wrong, what to try differently. These reflections accumulate in episodic memory and inform subsequent cycles. Without reflection, retry shows zero improvement (Shinn et al., NeurIPS 2023).

### OTP-Style Error Classification
Errors are classified into four categories with appropriate recovery: TRANSIENT (retry with backoff), REPAIRABLE (feed to debugger), TERMINAL (fail fast), DEGRADED (circuit break). Inspired by Erlang/OTP supervision trees.

### Anti-Slopsquatting Package Verification
Before `pip install`, every dependency is verified against a safe-list of ~50 known packages or checked against PyPI. LLMs hallucinate package names at 5-21% (Spracklen et al., USENIX 2025). Attackers register those names with malicious code.

### Covenants Are Structural, Not Suggestions
Self-learned rules enforced via AST validators — not prompt injection. Bare `except:` → `except Exception:`. Missing `__init__.py` → auto-created. SQLAlchemy + `__future__` annotations → removed. Zero LLM tokens.

### Real Tests, Not Imagination
The validator runs actual `pytest` (with `--import-mode=importlib` to match real execution) in a sandbox. Real pass/fail. Weighted scoring. Post-pytest smoke test imports every module and runs the entry point.

### SICA Self-Improvement
Automated self-improvement loop: benchmark → propose → apply → re-benchmark → accept/reject with regression gating.

```bash
belief sica --iterations 10 --tiers 1 2 3 --dry-run           # Preview proposals
belief sica --iterations 10 --tiers 1 2 3                      # Run for real
belief sica --iterations 10 --tiers 1 2 3 --validate-tiers 4 5 # With held-out validation
```

## Quick Start

```bash
pip install belief-engine

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Build something
belief build --goal "Build a URL shortener with FastAPI and SQLite"

# Run the benchmark
belief benchmark --tiers 1 2 3 4 5 6 7

# Fix an existing repo
belief fix --repo ./my-project --issue "Login endpoint returns 500"
belief fix --repo https://github.com/user/repo --issue "Fix the bug"
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
    error_classifier.py — OTP-style error routing
  validators/      — AST covenant enforcers (deterministic, zero LLM)
    typescript_fixup.py — v0-style streaming fixes for TS builds
  memory/          — ChromaDB metabolization (nutrients, soil, reflexion)
  refinement/      — Water cycle with Reflexion episodic memory
  deploy/          — Docker + Railway deployment
  codebase/        — Brownfield support (localization, patcher)
  languages/       — Multi-language adapters (Python, TypeScript)
  evolution/       — SEED + SICA self-improvement engine
  polarity/        — Latios/Latias incompleteness engine
  models/          — Pydantic models (state, artifacts, skeleton, contracts)
  hardening.py     — Budget limits, rate limiter, security scanner
  graph.py         — LangGraph pipeline wiring
  llm.py           — Anthropic API client with prompt caching
```

## CLI Commands

```bash
belief build --goal "..."                              # Build from description
belief benchmark --tiers 1 2 3 4 5 6 7 8              # Run benchmark suite
belief sica --iterations 10 --tiers 1 2 3              # Self-improvement cycle
belief fix --repo ./path --issue "..."                 # Fix an existing project
belief fix --repo https://github.com/... --issue "..." # Fix a GitHub repo
```

## Model Routing

| Agent | Model | Role |
|-------|-------|------|
| Research, Planner, Architect, Builder, Debugger | Sonnet | Deep reasoning |
| Intake, Tester, Gap Analyst, Synthesizer, Latios | Haiku | Mechanical tasks |
| Skeleton, Covenant Enforcer, Import Fix, Validator | None | Deterministic |

## The Benchmark Suite

35 challenges across 8 tiers:

| Tier | Challenges | Examples |
|------|-----------|----------|
| 1 | 5 | FizzBuzz, Fibonacci, word count, calculator, converter |
| 2 | 4 | Todo CLI, health API, calculator CLI, CSV stats |
| 3 | 5 | URL shortener, bookmark API, notes API, expense tracker, contacts |
| 4 | 5 | Blog engine, Kanban board, file vault, poll system, task board |
| 5 | 4 | Event system, inventory manager, quiz engine, workflow DAG |
| 6 | 5 | API gateway, event-driven, shared auth, data pipeline, microservice CRUD |
| 7 | 5 | Brownfield fixes on existing repos |
| 8 | 5 | TypeScript: Express API, CLI tool, ethers v6, MCP server, fullstack |

## Tech Stack

- **Python 3.11+** (tested on 3.14)
- **LangGraph** for agent orchestration
- **Anthropic Claude** (Sonnet + Haiku)
- **ChromaDB** for learning memory
- **Docker** for deployment

## License

MIT

## Author

Built by [Fio](https://github.com/metafiopy-tech) — solo, from scratch, while making pizzas.

*"The remainder after every operation drives the next cycle."*
