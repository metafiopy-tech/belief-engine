# Belief Engine: Tier 6-8+ Architecture Blueprint (Research-Validated)

*Updated with findings from deep research across 40+ papers and production systems.*
*See companion research reports for full citations and code examples.*

## How to Read This Document

Each tier section covers: what exists in the codebase, what's missing, the pipeline architecture, key research findings with specific algorithms, and implementation patterns.

---

## Current Codebase Inventory (75 files, ~19,200 lines)

```
Models:           service_architecture.py, skeleton.py, dependency_dag.py, symbol_registry.py
Brownfield:       codebase/__init__.py (Codebase class), imports.py, patcher.py
Self-Improvement: evolution/ (SEED, SelfPatch, Mentor), polarity/ (Latios/Latias), memory/soil.py
Multi-Language:   languages/ (Python adapter complete, TypeScript partial)
Deployment:       deploy/ (Docker, Railway), tools/ (compose, deployment gen)
```

---

## TIER 6: Multi-Service Systems

**Target**: Backend + worker + queue + shared DB from a single spec.

### What's Missing → What to Build

| New File | Purpose | Key Library |
|----------|---------|-------------|
| `models/openapi.py` | OpenAPI 3.1 spec via Pydantic | `openapi-pydantic` (v0.5.1) — `PydanticSchema` → `$ref` resolution |
| `agents/contract_agent.py` | Generate specs BEFORE code | Uses openapi-pydantic `construct_open_api_with_schema_class()` |
| `graph_multi.py` | Parallel service generation | LangGraph `Send()` with custom `merge_dicts` reducer |
| `tools/compose_orchestrator.py` | Docker lifecycle | `python-on-whales` — `compose.up(wait=True, wait_timeout=120)` |
| `agents/integration_tester.py` | Contract validation | `schemathesis` — `schema.as_state_machine()` for lifecycle tests |

### Research-Validated Architecture

```
Spec → Architect (ServiceArchitecture) → Contract Agent (OpenAPI per service)
  → Send() fan-out: [Service A pipeline, Service B pipeline, Service C pipeline]
  → Annotated[dict, merge_dicts] fan-in
  → Integration Test (python-on-whales compose.up → Schemathesis → compose.down)
  → Decomposer → END
```

**Critical implementation detail — custom dict reducer required:**
```python
def merge_dicts(left: dict | None, right: dict | None) -> dict:
    merged = (left or {}).copy()
    merged.update(right or {})
    return merged

class MultiServiceState(TypedDict):
    code_files: Annotated[dict, merge_dicts]  # Merges parallel branches
```

Without this, LangGraph raises `InvalidUpdateError`. Subgraphs do NOT inherit parent state — pass all data via `Send.arg`.

**Library choices (research-corrected):**
- docker-py does NOT support Compose. Use `python-on-whales` (Docker Inc endorsed).
- AsyncAPI Python codegen is immature (3-5 years behind OpenAPI). Use AsyncAPI as spec format, generate messaging code via LLM.
- Specmatic requires JRE 17+. For pure Python: Schemathesis with stateful testing chains POST→GET, DELETE→GET→404 automatically.

**Cost**: $0.35-0.50/build (3 services × $0.12 + contracts + integration)

---

## TIER 7: Brownfield Code Modification

**Target**: Given 10K+ line codebase + issue description, produce correct patch.

### What's Missing → What to Build

| New File | Purpose | Key Library |
|----------|---------|-------------|
| `codebase/repo_graph.py` | Tree-sitter parse + NetworkX + PPR | `tree-sitter`, `networkx` |
| `codebase/localizer.py` | 3-phase Agentless narrowing | LLM + BM25 + PPR hybrid |
| `codebase/patch_sampler.py` | Kimi-Dev self-play (3 patches × 3 tests) | CodeT ranking |
| `codebase/change_impact.py` | Test selection | `pytest-testmon` |
| `agents/brownfield_agent.py` | Agentless default → agentic escalation | LangGraph conditional |
| `agents/reproduction_tester.py` | Tests that prove the bug exists | Dual BugFixer/TestWriter |

### Research-Validated Architecture

```
Codebase + Issue
  → INGEST: tree-sitter parse, dependency graph, BM25+PPR hybrid ranking
  → LOCALIZE: File (top 20→5) → Class/Function → Lines (4 samples at temp=0.8, merged)
  → SELF-PLAY: 3 patches × 3 reproduction tests (beats 40-patch majority voting)
  → VALIDATE: CodeT ranking [f(S) = |solutions| × |tests_passed|], pytest-testmon
  → ESCALATE?: If Agentless fails 3x → agentic mode with warm start, $5 cap
  → OUTPUT: Unified diff
```

**Three breakthrough findings:**

1. **Kimi-Dev self-play: 3×3 > 40×1.** BugFixer generates 3 patches, TestWriter generates 3 reproduction tests. Cross-validate via agreement matrix. Scales from 48.0% (1×1) to 60.4% (40×40) on SWE-bench Verified. The key: invest in test generation alongside patches.

2. **Agentless generates 40 patches, not 5.** 1 greedy (temp=0) + 39 sampled (temp=0.8), unified diff format. Automatic selection: 32.0% vs oracle upper bound 42.0% — ranking is the bottleneck. Cost: $0.70/issue with Claude 3.5 Sonnet.

3. **Hybrid BM25+PPR outperforms either alone.** `0.6 × bm25 + 0.4 × ppr`. PageRank seeded on top BM25 results catches structurally connected files missing keyword matches. PRFL showed +39% Top-1 accuracy.

**Tree-sitter .scm queries for Python dependency extraction:**
```scheme
(import_from_statement module_name: (dotted_name) @import.source)
(class_definition name: (identifier) @class.name superclasses: (argument_list) @class.bases)
(function_definition name: (identifier) @func.name return_type: (type) @func.return)
(call function: (attribute object: (_) @call.obj attribute: (identifier) @call.method))
```

**CodeT ranking (30 lines of core logic):**
```python
def codet_rank(solutions, tests, execute_fn):
    groups = defaultdict(list)
    for i, sol in enumerate(solutions):
        sig = frozenset(j for j, t in enumerate(tests) if execute_fn(sol, t))
        groups[sig].append(i)
    best = max(groups, key=lambda s: len(groups[s]) * len(s))
    return solutions[groups[best][0]]
```

**Cost**: $0.40-0.70/issue (localization $0.08, self-play $0.30, validation $0.10)

---

## TIER 8+: Novel Algorithms, Full-Stack, Self-Improvement

**Target**: Genuinely novel solutions, full-stack apps, 50K+ line systems, autonomous self-improvement.

### What's Missing → What to Build

| New File | Purpose | Key Technique |
|----------|---------|---------------|
| `evolution/evolutionary_search.py` | AlphaEvolve-style population search | MAP-Elites + island model + LLM ensemble |
| `evolution/scaffold.py` | FunSearch decomposition | Fixed scaffold + evolvable priority function |
| `evolution/sica.py` | Self-modification | Docker sandbox + async overseer + version archive |
| `memory/q_value_store.py` | Value-aware retrieval | MemRL: `Q ← Q + α(r − Q)`, two-phase retrieval |
| `memory/reflexion.py` | Verbal self-critique | Store natural language reflections after failures |
| `verification/property_tester.py` | Property-based API tests | Schemathesis stateful + hypothesis-jsonschema |
| `benchmark_generator.py` | Procedural challenges | BeTaL calibration at ~50% target difficulty |

### Research-Validated Architectures

**Evolutionary Search (AlphaEvolve):**
```
Population (MAP-Elites, 10 per niche, island model)
  → Select parent (fitness-proportional) + inspiration (diversity-biased)
  → LLM ensemble mutations: Haiku for breadth (many cheap), Sonnet for depth (few quality)
  → Cascade evaluation: ast.parse (1s) → ruff+mypy (5s) → pytest (30s) → property tests (mins)
  → Archive update: store best per behavioral niche
  → Repeat until plateau (3 gens) or budget exceeded
```

**Self-Improvement (SICA + FunSearch hybrid):**
```
FIXED SCAFFOLD (never modified):
  graph.py, benchmark.py, hardening.py, models/

EVOLVABLE PRIORITY FUNCTIONS (modified by SEED):
  prompts/__init__.py, agents/validator.py, agents/tester.py,
  config/models.py, refinement/analyzer.py

CYCLE:
  1. Benchmark (20+ challenges) → record results
  2. SEED analyzes → proposes edits to EVOLVABLE files only
  3. Overseer reviews (async LLM, every 30s, Docker sandboxed)
  4. Apply → re-benchmark → accept if improved, rollback if not
  5. Q-value update on ChromaDB: Q ← Q + α(reward − Q)
```

**Key algorithms with exact formulas:**

*MemRL Q-value:* `Q_new ← Q_old + α · (r − Q_old)` — exponential moving average, no explicit decay. Two-phase retrieval: semantic filter (`cosine > δ`) then value re-rank (`score = 0.5·ẑ_sim + 0.5·Q̂`, z-score normalized). 56% relative improvement over RAG.

*CodeT consensus:* `score(group) = |solutions_in_group| × |tests_passed|`. Group by identical pass signatures. Robust to wrong tests. +18.8% absolute on HumanEval.

*RethinkMCTS:* MCTS over reasoning thoughts (not tokens). Value = `1.0 × pass_rate + 0.2 × llm_score`. P-UCB selection. "Rethink": trace execution failures to erroneous thought node, revise directly. 16 rollouts beats Reflexion/ToT. Expensive — best for offline/high-stakes.

*BeTaL benchmark calibration:* LLM "designer" iteratively adjusts parameterized specs. Target 50% difficulty = maximum information. 5.3-13.2% deviation from target. Dimensions: models (1-20), endpoints (2-50), business rules (0-20), auth level, data depth.

---

## Implementation Roadmap

| Phase | Timeline | Key Deliverables | Dependencies |
|-------|----------|-----------------|--------------|
| **1: Solidify Tier 5** | Now → 2 weeks | 50%+ benchmark, PyPI publish, SEED autonomous, Reflexion | None new |
| **2: Tier 6** | 2-4 weeks | Multi-service generation, contract testing, docker orchestration | openapi-pydantic, python-on-whales, schemathesis |
| **3: Tier 7** | 4-8 weeks | Brownfield modification, Agentless+agentic hybrid, SWE-bench Lite | tree-sitter, networkx, gitpython, pytest-testmon |
| **4: Tier 8+** | 8-16 weeks | Evolutionary search, self-improvement loop, procedural benchmarks | hypothesis, mypy, hypothesis-jsonschema |

---

## Cost Model

| Tier | Cost | Time | Strategy |
|------|------|------|----------|
| 5 | $0.18 | 5 min | 70% Haiku + 30% Sonnet |
| 6 | $0.35-0.50 | 10-15 min | Parallel service gen, Schemathesis validation |
| 7 | $0.40-0.70 | 5-10 min | Agentless default ($0.70), agentic escalation ($5 cap) |
| 8 | $0.55-0.90 | 15-25 min | Evolutionary: 5× base with cascade pruning |
| Self-improvement | $5-10 | 90-120 min | Full benchmark run |

**Levers**: Prompt caching (0.1× input), batch API (50% off), model routing (70/20/10 H/S/O), cascade eval.

---

## Key References

| System | Source | Finding |
|--------|--------|---------|
| Agentless | arXiv 2407.01489 | 40 patches at $0.70/issue, 50.8% SWE-bench Verified |
| Kimi-Dev | arXiv 2509.23045 | 3×3 self-play > 40 majority voting, 60.4% Verified |
| AlphaEvolve | arXiv 2506.13131 | MAP-Elites + LLM ensemble, improved Strassen after 56 years |
| FunSearch | Nature 2023 | Scaffold + priority function, novel math constructions |
| SICA | arXiv 2504.15228 | Self-modifying agent, 17%→53% in 15 iterations, $7k |
| MemRL | arXiv 2601.03192 | Q-value memory, 56% improvement over RAG |
| CodeT | arXiv 2207.10397 | |solutions|×|tests_passed|, +18.8% HumanEval |
| RethinkMCTS | arXiv 2409.09584 | Thought-level MCTS with rethink, beats Reflexion |
| RepoGraph | arXiv 2410.14684 | Line-level deps, 32.8% relative improvement |
| BeTaL | arXiv 2510.25039 | Procedural benchmark calibration, 5.3-13.2% deviation |
| Schemathesis | schemathesis.io | Property-based API testing, stateful operation chains |
| OpenEvolve | github.com/codelion/openevolve | Open-source AlphaEvolve replication |
