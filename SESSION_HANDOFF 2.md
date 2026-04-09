# BELIEF ENGINE v2.1 — SESSION HANDOFF DOCUMENT

## What This Is

The Belief Engine is an autonomous multi-agent code generation system (75 Python files, ~19,200 lines) that takes a natural language goal and produces working, tested, deployed software. It uses LangGraph for agent orchestration, Anthropic Claude for LLM calls, and ChromaDB for persistent learning memory.

**GitHub:** https://github.com/metafiopy-tech/belief-engine

---

## CURRENT STATE (as of April 9, 2026)

### Key Metrics
- **75 files, ~19,200 lines** of Python
- **53+ builds completed**, 140+ nutrients in ChromaDB soil, 7 self-learned covenants
- **Cost per build: $0.18** (was $0.87 before optimization — 80% reduction)
- **Build time: ~250s** average
- **Benchmark: 4/20 pass (20%), 10/20 executor pass (50%), 0.68 avg weighted score**
- First benchmark running with v2.1.1 fixes (database skeleton + test cap) — results pending

### Benchmark Results (v2.1.0 — 20 challenges)
```
Pass rate:     4/20 (20%)
Executor rate: 10/20 (50%)
Avg weighted:  0.68
Total time:    104 min (~$5)

Tier 1: 1/3 pass    (fibonacci perfect, fizzbuzz/wordcount fail)
Tier 2: 1/4 pass    (health-api perfect 26/26, todo/calc/csv fail)
Tier 3: 0/5 pass    (url-shortener 0.77 near-miss, bookmark 0.33)
Tier 4: 1/4 pass    (task-board 0.975, file-vault 0.92 near-miss)
Tier 5: 1/4 pass    (workflow-engine PERFECT 18/18, inventory 0.95 near-miss)

Near-misses (0.85+ weighted but executor failed):
  t4-file-vault:      0.92 (11/12 tests)
  t5-event-system:    0.93 (13/14 tests)
  t5-inventory-system: 0.95 (20/21 tests)
```

### What Was Built This Session

#### The 7 Research Moves (all implemented)
1. **Move 1: Real pytest validator** — Runs actual pytest + lint + security scan. Zero LLM tokens. Weighted scoring: smoke=3x, functional=2x, edge=1x, env=0x.
2. **Move 2: AST covenant enforcers** — 5 deterministic validators: no __future__ with SQLAlchemy, add missing Mapped imports, remove stdlib from requirements, add missing stdlib imports, warn on 200+ line files.
3. **Move 3: Prompt caching + Haiku routing** — 7 of 12 agents on Haiku (3x cheaper). Cache-control headers on system prompts. Cost dropped from $0.42 to $0.18.
4. **Move 4: Repo map in tester/debugger** — AST-based structural index injected into prompts. Tester instructed to ONLY import symbols that exist in the repo map.
5. **Move 5: Contract-first generation** — APIContract model with EndpointContract and CLIContract. Architect generates contracts. Both builder and tester reference the same contract.
6. **Move 6: Architect/editor debugger** — Sonnet diagnoses root cause across ALL files. Haiku applies targeted search/replace edits. Multi-file fixes in one cycle.
7. **Move 7: Safety infrastructure** — AgentLimits per role, AuditLogger (JSONL), seed_requires_approval() gate, is_critical_file() check.

#### Other Features Built
- **SEED activated** — Triggers every 5 builds, reads soil antipatterns, propose-only mode. First proposal implemented (SEED-001: plan validation).
- **Database skeleton generator** — Generates correct SQLAlchemy 2.x setup with get_db, init_db, engine, Base, SessionLocal. Protected from debugger overwrites.
- **Auto-generated conftest.py** — Detects pytest fixture usage from function parameters (not just imports). Generates TestClient for FastAPI, CliRunner for Click.
- **Multi-file water cycle** — Fixer can emit atomic edits across multiple files.
- **Deploy CLI** — `--deploy docker_local` or `--deploy railway`. One-command build+deploy.
- **20-challenge benchmark suite** — Tiers 1-5, from FizzBuzz to workflow DAG engines.
- **Test cap** — Hard limit of 20 test functions per file to reduce phantom failures.
- **PyPI ready** — pyproject.toml with full metadata, LICENSE file, version 2.1.0.
- **GitHub docs** — README rewrite, CLAUDE.md, CONTRIBUTING.md.

---

## ARCHITECTURE

### Pipeline Flow
```
recomposer → intake → research → planner → architect → skeleton_pass1
→ builder → covenant_enforce → import_fix → tester → executor → gap_analyst
→ [debugger loop: architect diagnoses across all files, editor applies fixes]
→ synthesizer → validator (real pytest + lint + security)
→ [refinement loop: water cycle with multi-file fixes]
→ decomposer → END
```

### Model Routing
| Role | Model | Why |
|------|-------|-----|
| research, planner, architect, builder, debugger | Sonnet 4.6 | Deep reasoning |
| intake, tester, gap_analyst, synthesizer, validator, latios, executor | Haiku 4.5 | Mechanical tasks |
| skeleton, covenant_enforce, import_fix, validator core | None | Deterministic (zero tokens) |

### File Structure
```
belief/
  agents/          — 11+ agents (intake through validator + debugger + repo_map)
  validators/      — AST covenant enforcers (Move 2)
  memory/          — ChromaDB metabolization (nutrients, soil, FSRS decay)
  refinement/      — Water cycle (analyzer, fixer, runner)
  deploy/          — Docker + Railway deployment + monitoring
  codebase/        — Brownfield (Agentless localization, patcher, imports)
  languages/       — Python + TypeScript adapters
  evolution/       — SEED self-improvement (propose-only)
  polarity/        — Latios/Latias incompleteness engine
  models/          — State, artifacts, skeleton (with APIContract), service_architecture
  config/          — Settings, model routing
  tools/           — Composition planner, deployment generator
  hardening.py     — Budget, rate limiter, security scanner, audit log, agent limits
  graph.py         — LangGraph pipeline (all nodes + edges)
  llm.py           — Anthropic API client (caching, JSON repair)
  cli.py           — CLI entry point
  benchmark.py     — 20-challenge benchmark suite
```

### Self-Learned Covenants (7 active)
1. Explicit stdlib imports
2. No file over 200 lines
3. Static import verification
4. SQLAlchemy type annotations at module level
5. SQLAlchemy Mapped/mapped_column imports
6. Entry point imports must resolve
7. Validate JSON completeness before downstream use

---

## WHAT'S LEFT TO DO (prioritized)

### Immediate (next session)

1. **Review v2.1.1 benchmark results** — The benchmark with database skeleton fix + test cap is running. Compare to v2.1.0 baseline (4/20 → target 7-10/20).

2. **Fix the remaining executor failures** — The #1 issue: `main.py` imports symbols from `database.py` that the skeleton generates but the builder/debugger sometimes overwrites. If the v2.1.1 fixes don't solve it, the next step is making skeleton files truly immutable (never overwritten by any agent).

3. **Fix the conftest fixture issue in the water cycle** — The tester generates conftest.py correctly now, but the water cycle's test runner may not copy it to the temp directory. Verify conftest lands in the right place during refinement.

4. **Publish to PyPI** — pyproject.toml is ready. Just needs `python -m build && twine upload dist/*` with a PyPI account.

### Medium-term

5. **Implement SEED-001** — SEED proposed "validate JSON completeness in planner output." This was implemented but needs testing across builds. Monitor whether plan validation reduces silent degradation.

6. **Add more deterministic templates** — The config skeleton generator fails on requirements.txt (tagged as CONFIG but not a Python file). Add a requirements.txt generator that reads from `skeleton.external_dependencies`.

7. **Push for 50% pass rate on benchmark** — After the database/test fixes, the bottleneck shifts to:
   - CLI projects (Click) needing different test patterns than FastAPI
   - Complex projects (blog engine, quiz engine) needing better multi-step generation
   - The water cycle not finding the right files to fix

8. **Railway deployment testing** — Docker local works. Railway hasn't been tested with a real project. Need Railway CLI installed + token.

### Longer-term

9. **Tree-sitter integration** — Replace ast.parse with tree-sitter for more robust parsing. The repo map works with ast.parse but tree-sitter handles partial/broken files better.

10. **Property-based testing with Hypothesis** — The research identified this as a key quality lever. Generate property tests from API contracts instead of example-based tests.

11. **SWE-bench evaluation** — Not directly applicable (SWE-bench tests bug-fixing, engine does greenfield). But running the Belief Engine against SWE-bench Lite would test Tier 7 (extend existing codebases) capability.

12. **Full-stack generation** — The engine builds Python backends. Adding React/Next.js frontend generation from the same API contract would enable full-stack builds.

13. **Self-replication** — The research paper "Bootstrapping Coding Agents" validates this: maintain a formal specification, have the engine regenerate itself from spec. V1→V2→V3 bootstrap.

14. **Agent marketplace** — The Forge Network vision: package the engine as an MCP server, list on AWS/Google Cloud agent marketplaces, enable agent-to-agent invocation.

---

## KNOWN ISSUES

1. **Cost tracking shows $0.00 in benchmark** — The benchmark extracts cost from `build_budget` in final state, but the CLI doesn't store it there. Need to wire token usage into the returned state dict.

2. **Skeleton syntax errors on models.py** — When the architect defines model fields with complex types (e.g., `list[str]` with defaults), the skeleton generator sometimes produces invalid syntax. The f-string escaping breaks.

3. **Synthesizer rejects files** — The synthesizer (now on Haiku) sometimes produces syntax errors when "polishing" code. These are caught and the original is kept, but it wastes a Haiku call.

4. **Water cycle plateaus on conftest issues** — When tests fail because of missing fixtures, the water cycle tries to fix code files instead of generating conftest.py. It plateaus after 2 cycles.

5. **Import fix node too aggressive** — The import fixer changes `Base` to `BaseModel` in database.py, which breaks SQLAlchemy's DeclarativeBase inheritance. Need a blacklist for database-related symbols.

6. **Latios sometimes can't parse output** — `"Latios: could not parse output — treating as complete"` appears frequently. The JSON parsing for Latios's gap analysis is brittle.

---

## ENVIRONMENT

- **Python 3.14** on macOS Apple Silicon (M-series)
- `python3`/`pip3` for system commands, `python`/`pip` inside venvs
- `pyproject.toml` build backend: `setuptools.build_meta`
- All Belief components in `~/Desktop/belief-engine/`
- Soil stored in `~/.belief-engine/soil/` (ChromaDB)
- Build history in `~/.belief-engine/builds.db` (SQLite)
- Audit logs in `~/.belief-engine/audit/` (JSONL)
- Benchmark results in `~/.belief-engine/benchmark_results.json`
- SEED proposals in `~/.belief-engine/proposals.json`
- SEED counter in `~/.belief-engine/seed_counter.txt`

---

## RESEARCH REPORTS GENERATED

Three deep research reports were produced during this session:
1. "Scaling the Belief Engine to Autonomous Multi-Language Operation" — Tier 6/7/8 architecture
2. "Three Production-Validated Upgrades" — deployment CLI, SEED, test verdicts
3. "The Belief Engine: From Self-Improving Prototype to Production Software Factory" — comprehensive audit with 7-move roadmap (all implemented)

---

## SESSION HISTORY

The full conversation transcript is at:
`/mnt/transcripts/2026-04-08-23-47-44-belief-engine-v2-full-build-session.txt`

Previous session transcript:
`/mnt/transcripts/2026-04-08-04-09-21-belief-engine-v2-build-session.txt`

These contain every code change, every build result, every architectural decision.
