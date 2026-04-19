# CLAUDE.md -- Belief Engine v3.0 Technical Reference

## What This Is

Autocatalytic multi-agent build system (131 Python files, ~37,800 lines). Takes a natural language goal, produces working deployed software, and improves itself after every build. Uses LangGraph for agent orchestration, Anthropic Claude for LLM calls, ChromaDB for persistent learning memory with FSRS decay.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add ANTHROPIC_API_KEY

# Build
belief --goal "Build a bookmark API with FastAPI and SQLite"

# Build + deploy
belief --goal "Build a URL shortener" --deploy docker_local --deploy-name shortener

# Self-improvement
belief sica --iterations 5
belief jitterbug
belief dashboard
belief progression
```

## Pipeline Flow

```
recomposer -> intake -> research -> planner -> architect -> skeleton_pass1
-> builder -> covenant_enforce -> import_fix -> tester -> executor -> gap_analyst
-> [debugger loop: architect diagnoses, editor applies fixes]
-> synthesizer -> validator (real pytest + lint + security)
-> [refinement loop: water cycle with multi-file fixes]
-> decomposer (episode recording) -> END
```

## v3.0 Architecture (Sessions 1-7)

### Session 1: ChromaDB 5-Collection Architecture
- `belief/memory/fsrs.py` -- FSRS-4.5 spaced repetition (retrievability, stability, difficulty)
- `belief/memory/collections.py` -- 5 collections: belief_tools, belief_episodes, belief_principles, belief_failures, belief_covenants
- `belief/memory/soil.py` -- Updated for multi-collection routing with FSRS metadata

### Session 2: Evolutionary Archive
- `belief/evolution/archive.py` -- SQLite DAG of AgentVersion + BenchmarkResult, DGM parent selection, MAP-Elites niches
- `belief/evolution/cascade.py` -- 4-gate evaluation: canary, smoke, full benchmark, regression check

### Session 3: Crystallization Pipeline
- `belief/evolution/crystallizer.py` -- 4-stage covenant discovery: template sweep (15 invariants), Claude proposer, Houdini filter, promotion
- `belief/validators/covenant_registry.py` -- Static (6 hand-written) + dynamic (ChromaDB) covenant management
- `belief/memory/episode_recorder.py` -- Build trace recording with 15+ structural features

### Session 4: Autocatalytic NEW_TOOL
- `belief/memory/tool_registry.py` -- SelfAuthoredTool lifecycle (register, retrieve, usage tracking, FSRS)
- `belief/evolution/tool_validator.py` -- 7-point validation (syntax, no belief imports, no dangerous calls, docstring, length, importability)
- `belief/evolution/self_improvement.py` -- FailureCluster, goal formulation, execute_new_tool_proposal (uses own pipeline)

### Session 5: Jitterbug Cycle
- `belief/evolution/jitterbug.py` -- LangGraph StateGraph: expansion -> compression -> reconstruction -> validation -> integration
- `belief/evolution/progression.py` -- Generative chain stage tracking (0-5: Seed -> Cluster -> Tessellation -> Basis -> Connectivity -> Archetypes)

### Session 6: DSPy Prompt Optimization
- `belief/optimization/dspy_modules.py` -- 5 DSPy ChainOfThought wrappers (planner, architect, builder, tester, debugger)
- `belief/optimization/compiler.py` -- GEPA/MIPROv2/BootstrapFewShot cascade, metric functions, prompt extraction
- `belief/optimization/prompt_store.py` -- Version-tagged prompt persistence (~/.belief-engine/prompts/)

### Session 7: Safety Guardrails
- `belief/safety/overseer.py` -- AsyncOverseer on Haiku (different model than agent)
- `belief/safety/probes.py` -- Evaluator integrity (SHA-256), test harness edit detection, env tampering, cost monitors
- `belief/safety/goodhart_canary.py` -- 3 held-out challenges, divergence detection
- `belief/metrics/dashboard.py` -- JSONL metrics, linear/exponential growth analysis, formatted dashboard

## Project Structure

```
belief/
  agents/          -- 17 agents (intake through brownfield + reproduction_tester)
  validators/      -- AST covenant enforcers + dynamic covenant registry
  memory/          -- ChromaDB soil (5 collections), FSRS, tool registry, episode recorder
  refinement/      -- Water cycle (analyzer, fixer, runner)
  evolution/       -- SICA, archive, crystallizer, jitterbug, progression, tool_validator
  optimization/    -- DSPy/GEPA prompt optimization (optional dependency)
  safety/          -- Overseer, probes, Goodhart canary
  metrics/         -- Dashboard, growth analysis
  deploy/          -- Docker + Railway deployment + monitoring
  codebase/        -- Brownfield (localizer, repo_graph, patch_sampler, change_impact)
  languages/       -- Python + TypeScript adapters
  polarity/        -- Latios/Latias incompleteness engine
  models/          -- State, artifacts, skeleton, contracts, service_architecture
  config/          -- Settings, model routing
  tools/           -- Composition planner, deployment generator
  prompts/         -- All agent prompts + protocol skeletons
  hardening.py     -- Budget, rate limiter, security scanner, audit log
  graph.py         -- LangGraph pipeline (all nodes + edges)
  llm.py           -- Anthropic API client (caching, JSON repair)
  cli.py           -- CLI entry point
```

## CLI Commands

```bash
belief --goal "..."                    # Build from goal
belief benchmark --tiers 1 2 3        # Run benchmark
belief sica --iterations 5            # Self-improvement
belief jitterbug                      # Compression-reconstruction cycle
belief jitterbug --dry-run            # Expansion + compression only
belief progression                    # Generative chain stage
belief optimize builder               # DSPy optimization (requires dspy)
belief optimize --all                 # Optimize all agents
belief dashboard                      # Metrics dashboard
belief dashboard --json               # Metrics as JSON
belief fix --repo PATH --issue "..."  # Brownfield fix
```

## Model Routing

| Role | Model | Why |
|------|-------|-----|
| research, planner, architect, builder, debugger | Sonnet 4.6 | Deep reasoning |
| intake, tester, gap_analyst, synthesizer, validator, latios, executor | Haiku 4.5 | Mechanical tasks |
| skeleton, covenant_enforce, import_fix, validator core | None | Deterministic |
| safety overseer | Haiku 4.5 | Different model than agent (prevents self-deception) |

## Self-Learned Covenants (7 static + dynamic)

Static (hand-written):
1. Explicit stdlib imports
2. No file over 200 lines
3. No `__future__` with SQLAlchemy ORM
4. SQLAlchemy Mapped/mapped_column imports
5. No stdlib in requirements.txt
6. No bare except clauses

Dynamic (crystallized from build traces):
- Discovered automatically by the crystallization pipeline
- Stored in belief_covenants ChromaDB collection
- Loaded and fired by CovenantRegistry alongside static covenants

## Environment

- Python 3.14 on Mac (python3/pip3 for system)
- pyproject.toml build backend: setuptools.build_meta
- All components in ~/Desktop/belief-engine/
- Soil stored in ~/.belief-engine/soil (ChromaDB)
- Archive stored in ~/.belief-engine/archive.db (SQLite)
- Metrics stored in ~/.belief-engine/metrics.jsonl
- Optimized prompts in ~/.belief-engine/prompts/
- Audit logs in ~/.belief-engine/audit/ (JSONL)

## Running Tests

```bash
# All unit tests (no API key needed)
python -m pytest tests/ -v --timeout=120

# Individual test suites
python -m pytest tests/test_fsrs.py tests/test_collections.py -v     # Session 1
python -m pytest tests/test_archive.py -v                             # Session 2
python -m pytest tests/test_crystallizer.py -v                        # Session 3
python -m pytest tests/test_new_tool.py -v                            # Session 4
python -m pytest tests/test_jitterbug.py -v                           # Session 5
python -m pytest tests/test_dspy.py -v                                # Session 6
python -m pytest tests/test_safety.py -v                              # Session 7
python -m pytest tests/test_e2e_autocatalytic.py -v                   # Integration
```
