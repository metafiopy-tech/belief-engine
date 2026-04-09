# CLAUDE.md — Belief Engine v2.0 Technical Reference

## What This Is

Autonomous multi-agent build system (74 Python files, ~18,650 lines). Takes a natural language goal → produces working, deployed software. Uses an incompleteness-driven convergence loop with metabolization memory (ChromaDB soil). 51+ builds completed, 135+ nutrients, 7 self-learned covenants.

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add ANTHROPIC_API_KEY

# Build
python3 -m belief.cli --goal "Build a bookmark API with FastAPI and SQLite"

# Build + deploy
python3 -m belief.cli --goal "Build a URL shortener" --deploy docker_local --deploy-name shortener
```

## Pipeline Flow

```
recomposer → intake → research → planner → architect → skeleton_pass1
→ builder → covenant_enforce → import_fix → tester → executor → gap_analyst
→ [debugger loop: architect diagnoses, editor applies fixes]
→ synthesizer → validator (real pytest + lint + security)
→ [refinement loop: water cycle with multi-file fixes]
→ decomposer → END
```

## The 7 Optimization Moves (Research-Validated)

### Move 1: Real pytest validator
`belief/agents/validator.py` — Runs actual pytest, ruff-style lint, and security scan. Zero LLM tokens. Weighted scoring: smoke=3x, functional=2x, edge=1x, env=0x. Score ≥ 0.85 with all smoke passing → PASS verdict.

### Move 2: AST covenant enforcers  
`belief/validators/__init__.py` — Five deterministic AST validators:
- Remove `__future__` from SQLAlchemy files
- Add missing `Mapped`/`mapped_column` imports
- Remove stdlib from requirements.txt
- Add missing stdlib imports
- Warn on files over 200 lines

### Move 3: Prompt caching + model routing
`belief/config/models.py` + `belief/llm.py` — 7 of 12 agents on Haiku (3x cheaper). Prompt caching with `cache_control: ephemeral`. Cost: $0.42 → $0.18 per build.

### Move 4: Repo map in tester/debugger
`belief/agents/repo_map.py` — AST-based structural index injected into tester and debugger prompts. Tester instructed to ONLY import symbols from the repo map.

### Move 5: Contract-first generation
`belief/models/skeleton.py` — `APIContract` with `EndpointContract` and `CLIContract`. Both builder and tester reference the same contract as source of truth.

### Move 6: Architect/editor debugger
`belief/agents/debugger.py` — Sonnet architect diagnoses root cause across ALL files. Haiku editor applies targeted search/replace edits. Multi-file fixes in one cycle.

### Move 7: Safety infrastructure
`belief/hardening.py` — `AgentLimits` per role, `AuditLogger` (JSONL), `seed_requires_approval()` gate, `is_critical_file()` check. OWASP Agentic AI Top 10 aligned.

## Project Structure

```
belief/
  agents/          — 11+ agents (intake, research, planner, architect,
                     skeleton_builder, builder, tester, executor,
                     debugger, gap_analyst, synthesizer, validator)
  validators/      — AST covenant enforcers (Move 2)
  memory/          — ChromaDB metabolization (nutrients, soil, FSRS decay)
  refinement/      — Water cycle (analyzer, fixer, runner)
  deploy/          — Docker + Railway deployment + monitoring
  codebase/        — Brownfield (Agentless localization, patcher, imports)
  languages/       — Python + TypeScript adapters
  evolution/       — SEED self-improvement (propose-only)
  polarity/        — Latios/Latias incompleteness engine
  models/          — State, artifacts, skeleton, contracts
  config/          — Settings, model routing
  tools/           — Composition planner, deployment generator
  hardening.py     — Budget, rate limiter, security scanner, audit log
  graph.py         — LangGraph pipeline (all nodes + edges)
  llm.py           — Anthropic API client (caching, JSON repair)
  cli.py           — CLI entry point
```

## Model Routing

| Role | Model | Why |
|------|-------|-----|
| research, planner, architect, builder, debugger | Sonnet 4.6 | Deep reasoning |
| intake, tester, gap_analyst, synthesizer, validator, latios, executor | Haiku 4.5 | Mechanical tasks |
| skeleton, covenant_enforce, import_fix, validator core | None | Deterministic |

## Self-Learned Covenants (7 active)

1. Explicit stdlib imports
2. No file over 200 lines
3. Static import verification
4. SQLAlchemy type annotations at module level
5. SQLAlchemy Mapped/mapped_column imports
6. Entry point imports must resolve
7. Validate JSON completeness before downstream use

## Environment

- Python 3.14 on Mac (python3/pip3 for system)
- pyproject.toml build backend: setuptools.build_meta
- All components in ~/Desktop/belief-engine/
- Soil stored in ~/.belief-engine/soil (ChromaDB)
- Build history in ~/.belief-engine/builds.db (SQLite)
- Audit logs in ~/.belief-engine/audit/ (JSONL)
