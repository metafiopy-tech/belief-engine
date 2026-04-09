# BELIEF ENGINE v2.3.0 — SESSION HANDOFF DOCUMENT

## What This Is

The Belief Engine is an autonomous multi-agent code generation system (96 Python files, ~25,500 lines) that takes a natural language goal and produces working, tested, deployed software. Uses LangGraph for agent orchestration, Anthropic Claude for LLM calls, ChromaDB for persistent learning memory.

**GitHub:** https://github.com/metafiopy-tech/belief-engine
**PyPI:** https://pypi.org/project/belief-engine/ (PUBLISHED v2.2.2, v2.3.0 pending)

---

## CURRENT STATE (as of April 9, 2026)

### Key Metrics
- **96 files, ~25,500 lines** of Python
- **Benchmark: 17/20 (85%)** on Tiers 1-5 (v2.2.1)
- **30 benchmark challenges** across Tiers 1-7
- **Cost per build: $0.18** (was $0.87)
- **Build time: ~5 minutes** average
- **Published on PyPI:** `pip install belief-engine`
- **TypeScript generation:** full pipeline built (not yet benchmarked)

### Benchmark Progression (This Session)
| Version | Pass Rate | Avg Score | Key Changes |
|---------|-----------|-----------|-------------|
| v2.1.0 | 4/20 (20%) | 0.68 | Baseline |
| v2.1.1 | 8/20 (40%) | 0.74 | Database skeleton fix + test cap |
| v2.2.0 | 14/20 (70%) | 0.87 | 5 research fixes + threshold tuning |
| v2.2.1 | 17/20 (85%) | 0.86 | Global test cap + SEED autonomous + Reflexion |
| v2.2.2 | — | — | README update, re-published to PyPI |
| v2.3.0 | (running) | — | Tiers 6-8 infrastructure, TypeScript, 30 challenges |

---

## WHAT WAS BUILT THIS SESSION

### Phase 1: Benchmark Optimization (v2.1.0 → v2.2.1)

**v2.1.1 — Structural Fixes (7 fixes across 5 files)**
- Import fixer: `_NEVER_RENAME_SYMBOLS` blacklist, `_SKELETON_FILE_NAMES` guard
- Debugger: skeleton files fully immutable
- Executor: removed duplicate `auto_fix_imports` call
- Tester: fixture detection uses blacklist instead of whitelist
- Skeleton builder: sanitize field defaults

**v2.2.0 — Research-Driven Fixes (5 problems)**
- P1: Click CLI detection + import verification routing
- P2: Tester prompt tightened (8-10 hard limit, 14 negative rules)
- P3: Click conftest with decorator-aware entry point detection
- P4: Refinement analyzer classifies bugs as code vs test
- P5: Test classifier expanded with P0/P1/P2 markers

**v2.2.1 — Phase 1 Complete**
- Global test cap: `_global_test_cap()` enforces across ALL test files
- High-score override lowered to 0.80
- SEED autonomous mode: FunSearch pattern — HIGH confidence proposals auto-apply
- Reflexion: verbal self-critique after failed builds, stored in ChromaDB

### Phase 2: Tier 6 Multi-Service (5 new files)
- `models/openapi.py` — OpenAPI 3.1 spec from ServiceArchitecture
- `agents/contract_agent.py` — Deterministic contract generation
- `tools/compose_orchestrator.py` — Docker Compose lifecycle (python-on-whales pattern)
- `agents/integration_tester.py` — Cross-service contract test generation
- `graph_multi.py` — LangGraph Send() fan-out/fan-in with custom `merge_dicts` reducer

### Phase 3: Tier 7 Brownfield (1 new + 6 verified existing)
- `agents/reproduction_tester.py` — NEW, Kimi-Dev TestWriter (3×3 self-play)
- Verified existing: `repo_graph.py`, `localizer.py`, `patch_sampler.py`, `change_impact.py`, `brownfield_agent.py`, `models/patch.py`

### Phase 4: Tier 8+ Advanced (4 new + 3 verified existing)
- `evolution/scaffold.py` — FunSearch fixed scaffold + evolvable priority function
- `evolution/sica.py` — Self-modification with benchmark gating, version archive
- `verification/__init__.py` + `verification/property_tester.py` — Schemathesis + Hypothesis
- Verified existing: `evolutionary_search.py`, `q_value_store.py`, `benchmark_generator.py`

### Phase 5: TypeScript Generation (4 files modified, 1 new)
- `languages/typescript_adapter.py` — Rewritten: ESM-first, NodeNext resolution, protocol-aware scaffolding, 8 ESM covenants
- `agents/executor.py` — Added `_verify_typescript()`: npm install + tsc --noEmit
- `agents/validator.py` — Added `_validate_typescript()`: brace matching + npm install + tsc + vitest run
- `agents/skeleton_pass1.py` — TypeScript bypass: skips Python skeleton for .ts projects
- `agents/debugger.py` — Includes .ts files in context, brace-matching validation
- `prompts/__init__.py` — All prompts language-aware (research→npm, architect→src/, builder→ESM, tester→vitest)
- `prompts/protocol_skeletons.py` — NEW: minimum viable code for x402, MCP, A2A, ERC-8004

### Audit Fixes
- Architect `__call__` override injects `service_architecture` into graph state dict
- Benchmark cost tracking reads `token_usage.total_cost_usd` (was reading wrong field)
- Fixed duplicate brownfield deps in pyproject.toml

---

## CODEBASE STRUCTURE (96 files)

```
belief/
  agents/          — 17 agents (intake through brownfield + reproduction_tester)
  validators/      — AST covenant enforcers
  memory/          — ChromaDB soil, reflexion, q_value_store, store
  refinement/      — Water cycle (analyzer, fixer, runner)
  deploy/          — Docker + Railway deployment
  codebase/        — Brownfield (localizer, repo_graph, patch_sampler, change_impact)
  languages/       — Python + TypeScript adapters (TypeScript rewritten for ESM)
  evolution/       — SEED, evolutionary_search, scaffold, sica
  polarity/        — Latios/Latias incompleteness engine
  models/          — state, artifacts, skeleton, service_architecture, openapi, patch
  verification/    — property_tester (Schemathesis + Hypothesis)
  tools/           — composition_planner, deployment_generator, compose_orchestrator
  prompts/         — All agent prompts + protocol_skeletons (x402, MCP, A2A, ERC-8004)
  config/          — settings, model routing
  benchmark.py     — 30 challenges (Tiers 1-7)
  benchmark_generator.py — BeTaL procedural generation
  graph.py         — Single-service LangGraph pipeline
  graph_multi.py   — Multi-service Send() fan-out/fan-in pipeline
  hardening.py     — Budget, rate limiter, security scanner, SEED gates
  llm.py           — Anthropic API client
  cli.py           — CLI entry point
```

---

## WHAT'S NEXT

### Immediate
1. Benchmark v2.3.0 results (running now — 30 challenges, Tiers 1-7)
2. Publish v2.3.0 to PyPI
3. Test TypeScript generation with protocol goals (x402, MCP, A2A, ERC-8004)

### Short-term
- Wire `graph_multi.py` into CLI (currently only reachable via direct import)
- Test Tier 6 challenges (architect needs to learn ServiceArchitecture output)
- Test Tier 7 challenges against real codebases
- Run SICA self-improvement cycles

### Medium-term (from research)
- Build x402 payment-gated MCP server (first revenue-generating use of TS generation)
- Register Belief Engine on ERC-8004
- Bittensor subnet miner for code generation (Subnet 62 / RidgesAI pattern)

### Known Issues (lower priority)
- 51 silently swallowed exceptions (most intentionally non-fatal)
- 39 oversized functions (refactoring candidates)
- ModelRouter() instantiated 11 times (should be passed through)
- CLI doesn't import `graph_multi` yet
- Vitest test cap not enforced (TypeScript tests skip _cap_test_count)

---

## KEY RESEARCH CONDUCTED

### Research Report 1: Multi-Agent Code Generation (20 topics)
LangGraph Send(), OpenAPI generation, Docker orchestration, Agentless algorithm,
Kimi-Dev self-play (3×3 > 40), AlphaEvolve, FunSearch, SICA, MemRL Q-values,
CodeT ranking, RethinkMCTS, BeTaL benchmarks, Schemathesis, RepoGraph, pytest-testmon

### Research Report 2: Agentic Economy Protocols
x402 V2 SDK, MCP payment gating, ERC-8004 registration, A2A Agent Cards,
TypeScript ESM pitfalls, cross-protocol adapters, Bittensor subnets

### Research Report 3: Solo Builder Strategy
Protocol landscape convergence, revenue benchmarks, 3-horizon playbook,
MCP-to-x402 bridge as highest-leverage Year 1 play

---

## KEY FILE PATHS
- Tarball: `/mnt/user-data/outputs/belief-engine-v2.3.0-ts-validator.tar.gz`
- Architecture doc: `/home/claude/TIER_ARCHITECTURE.md`
- Session handoff: `/home/claude/SESSION_HANDOFF.md`
- All belief source: `/home/claude/belief/`
- pyproject.toml: `/home/claude/pyproject.toml`
