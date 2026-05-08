# CLAUDE.md -- Belief Engine v3.3 Technical Reference

## What This Is

Autocatalytic multi-agent build system (~220 Python files in `belief/`, ~64,600 lines as of v3.3 Session 5). Takes a natural language goal, produces working deployed software, and improves itself after every build. Uses LangGraph for agent orchestration, Anthropic Claude for LLM calls, ChromaDB for persistent learning memory with FSRS decay.

**Current ring:** v3.3.0 in flight. Tier 1 of the ecology layer (Economist, Predator, Sleep, GC) is live; first Tier 2 organ (Curiosity, suggest-only) is live. Speciator / Storyteller / Red-team / Body remain unscheduled.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add ANTHROPIC_API_KEY

# Build
belief build --goal "Build a bookmark API with FastAPI and SQLite"

# Build + deploy
belief build --goal "Build a URL shortener" --deploy docker_local --deploy-name shortener

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

## v3.2 Architecture (Sessions 1-8)

Hardening pass on top of v3.0. Key landings:

- `belief/llm.py::AsyncOllamaClient` -- streaming + per-chunk inactivity watchdog + per-role wall-clock budgets + pybreaker circuit breaker per model. Replaces the single 300s cap that crashed the architect overnight.
- `belief/covenants/pydantic_v2.py` -- LibCST-based covenant proposer for Pydantic v1 -> v2 migration patterns.
- `belief/covenants/precision_gate.py` -- gate that blocks auto-merge of proposed covenants below precision threshold; human approval required.
- `belief/synthesizer_router.py` -- router for the polish-pass synthesizer (1.5B local fallback path).
- `belief/archive/` -- DGM-style ChromaDB-backed retrieval layer. Planner injects top-3 priors per build.
- `belief/repomap/` -- tree-sitter + PageRank symbol map for the engine's own codebase. CLI: `belief repomap`.

## v3.3 Architecture (Sessions 1-5, ecology layer)

Adds the "ecology organs" -- background processes that maintain soil quality, manage spend, and propose what to build next. All organs consult the Economist contract before spending.

### Session 1: Economist (Tier 0 contract)
- `belief/ecology/economist.py` -- daily USD budget contract every other organ consults. Single audit trail for spend across organs. Quote / register / commit primitives.

### Session 2: Predator (Tier 1)
- `belief/ecology/predator.py` -- utility-driven culling of low-value soil. Soft-tombstones nutrients via existing `Soil.invalidate_nutrient` (no new tombstone path invented). Min-age + first-run cap to prevent overcorrection.
- `belief/memory/soil.py::revalidate` -- companion path to bring borderline tombstones back if utility recovers.

### Session 3: Sleep (Tier 1, first LLM-spending organ)
- `belief/ecology/sleep.py` -- Phase A (replay) + Phase B (crystallize). Consults Economist before each phase. Uses inlined FSRS formula to avoid pydantic import friction. Crystallizes covenants via the existing `belief.evolution.crystallizer.propose_invariants` Claude proposer.

### Session 4: Garbage Collector (Tier 1)
- `belief/ecology/garbage_collector.py` -- removes broken tools, invalid covenants, duplicate tool sources. Different decision rule from Predator (binary correctness, not utility). Caught a real upstream bug on first live run: NEW_TOOL pipeline was re-depositing the same tool 13x without dedup.

### Session 5: Curiosity Gate (Tier 2, suggest-only)
- `belief/ecology/curiosity.py` -- proposes build goals that fill gaps in the soil (file extensions, frameworks, covenant-sparse niches). Currently suggest-only; auto_build deferred to Session 5b. Info-gain estimator is token-match against historical builds.
- `belief/ecology/_information_gain.py` -- v1 info-gain heuristic.
- `belief/ecology/_utility.py` -- shared utility-scoring used by Predator + Curiosity.

## Project Structure

```
belief/
  agents/          -- agents (intake through brownfield + reproduction_tester)
  validators/      -- AST covenant enforcers + dynamic covenant registry
  covenants/       -- LibCST proposers + precision gate (v3.2)
  memory/          -- ChromaDB soil (5 collections), FSRS, tool registry, episode recorder, library inductor, trophic levels
  ecology/         -- v3.3 organs: economist, predator, sleep, garbage_collector, curiosity (+ shared _utility / _information_gain helpers)
  refinement/      -- Water cycle (analyzer, fixer, runner)
  evolution/       -- SICA, archive, crystallizer, jitterbug, progression, tool_validator
  archive/         -- v3.2 DGM-style retrieval layer (planner priors)
  repomap/         -- v3.2 tree-sitter + PageRank repo map
  optimization/    -- DSPy/GEPA prompt optimization (optional dependency)
  safety/          -- Overseer, probes, Goodhart canary, confidence probes
  metrics/         -- Dashboard, growth analysis
  deploy/          -- Docker + Railway deployment + monitoring
  codebase/        -- Brownfield (localizer, repo_graph, patch_sampler, change_impact)
  languages/       -- Python + TypeScript adapters
  polarity/        -- Latios/Latias incompleteness engine
  models/          -- State, artifacts, skeleton, contracts, service_architecture
  config/          -- Settings, model routing, local model table
  tools/           -- Composition planner, deployment generator
  prompts/         -- All agent prompts + protocol skeletons
  hardening.py     -- Budget, rate limiter, security scanner, audit log  [IMMUTABLE]
  benchmark.py     -- Scoring logic                                       [IMMUTABLE]
  graph.py         -- LangGraph pipeline (all nodes + edges)
  llm.py           -- Anthropic + Ollama clients (streaming, retry, breaker, JSON repair)
  cli.py           -- CLI entry point (subcommands listed below)
```

## CLI Commands

```bash
# Build / fix
belief build --goal "..."             # Build from goal
belief fix --repo PATH --issue "..."  # Brownfield fix
belief benchmark --tiers 1 2 3        # Run benchmark

# Self-improvement
belief sica --iterations 5            # Self-improvement
belief jitterbug                      # Compression-reconstruction cycle
belief jitterbug --dry-run            # Expansion + compression only
belief progression                    # Generative chain stage
belief optimize builder               # DSPy optimization (requires dspy)
belief optimize --all                 # Optimize all agents

# v3.2 inspection
belief archive ...                    # Inspect DGM-style build archive
belief repomap                        # PageRank-ranked symbol map
belief covenants                      # Review / approve auto-proposed covenants
belief models                         # Show active model routing table

# v3.3 ecology organs
belief economy                        # Economist daily-budget tracker
belief predator                       # Soft-tombstone low-utility nutrients
belief sleep                          # Offline soil consolidation cycles
belief gc                             # Tombstone broken tools / invalid covenants / duplicate sources
belief curiosity                      # Suggest build goals filling soil gaps

# Observability
belief dashboard                      # Metrics dashboard
belief dashboard --json               # Metrics as JSON
belief probe ...                      # Confidence probe commands
belief grinder ...                    # Grinder daemon commands
```

## Model Routing

| Role | Model | Why |
|------|-------|-----|
| research, planner, architect, builder, debugger | Sonnet 4.6 | Deep reasoning |
| intake, tester, gap_analyst, synthesizer, validator, latios, executor | Haiku 4.5 | Mechanical tasks |
| skeleton, covenant_enforce, import_fix, validator core | None | Deterministic |
| safety overseer | Haiku 4.5 | Different model than agent (prevents self-deception) |

## Self-Learned Covenants

Static (hand-written, in `belief/validators/`):
1. Explicit stdlib imports
2. No file over 200 lines
3. No `__future__` with SQLAlchemy ORM
4. SQLAlchemy Mapped/mapped_column imports
5. No stdlib in requirements.txt
6. No bare except clauses
7. Pydantic v1 -> v2 migration covenant (v3.2 Session 2, LibCST-based, in `belief/covenants/pydantic_v2.py`)

Dynamic (crystallized from build traces):
- Proposed by `belief/evolution/crystallizer.py` (4-stage pipeline: template sweep, Claude proposer on Haiku via `gap_analyst` role, Houdini filter, promotion)
- Sleep (v3.3) drives the crystallizer offline against accumulated traces
- Auto-proposed covenants pass through `belief/covenants/precision_gate.py` and require human approval via `belief covenants` -- no auto-merge
- Stored in `belief_covenants` ChromaDB collection; loaded and fired by `CovenantRegistry` alongside static covenants

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
# All unit tests (no API key needed). Hard gate: must stay green.
python3 -m pytest tests/ -q --timeout=60

# v3.0 individual suites
python3 -m pytest tests/test_fsrs.py tests/test_collections.py -v     # Session 1
python3 -m pytest tests/test_archive.py -v                             # Session 2
python3 -m pytest tests/test_crystallizer.py -v                        # Session 3
python3 -m pytest tests/test_new_tool.py -v                            # Session 4
python3 -m pytest tests/test_jitterbug.py -v                           # Session 5
python3 -m pytest tests/test_dspy.py -v                                # Session 6
python3 -m pytest tests/test_safety.py -v                              # Session 7
python3 -m pytest tests/test_e2e_autocatalytic.py -v                   # Integration

# v3.2 / v3.3 additions
python3 -m pytest tests/test_pydantic_covenant.py -v                   # v3.2 LibCST covenant
python3 -m pytest tests/test_covenant_proposer.py -v                   # v3.2 precision gate
python3 -m pytest tests/test_repomap.py -v                             # v3.2 repomap
python3 -m pytest tests/test_archive_*.py -v                           # v3.2 DGM archive
python3 -m pytest tests/test_economist.py -v                           # v3.3 Session 1
python3 -m pytest tests/test_predator.py -v                            # v3.3 Session 2
python3 -m pytest tests/test_sleep.py -v                               # v3.3 Session 3
python3 -m pytest tests/test_garbage_collector.py -v                   # v3.3 Session 4
python3 -m pytest tests/test_curiosity.py -v                           # v3.3 Session 5
```

## Hard constraints

- `belief/benchmark.py` scoring and `belief/hardening.py` are immutable. ruff `extend-exclude = ["belief/hardening.py"]`.
- Tests are a hard gate. On red, revert the session rather than patching forward.
- Sessions ship one at a time. Even when asked to do multiple, pause for commit + approval between each.
- All shell snippets on Joe's Mac use `python3` / `pip3`, never bare `python`.
- `belief` CLI is the `belief` console script from `pip install -e .`, not `python3 -m belief` (no `__main__.py`).
