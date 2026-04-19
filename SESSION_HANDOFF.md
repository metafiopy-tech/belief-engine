# BELIEF ENGINE v3.0.0 -- SESSION HANDOFF DOCUMENT

## What This Is

The Belief Engine is an autocatalytic multi-agent code generation system (131 Python files, ~37,800 lines) that takes a natural language goal and produces working, tested, deployed software. It learns from every build and can build tools for itself.

**GitHub:** https://github.com/metafiopy-tech/belief-engine
**PyPI:** https://pypi.org/project/belief-engine/

---

## CURRENT STATE (v3.0.0)

### Key Metrics
- **131 files, ~37,800 lines** of Python
- **Benchmark: 17/20 (85%)** on Tiers 1-5 (measured on v2.2.1, v3.0 adds infrastructure only)
- **Cost per build: $0.18** (was $0.87)
- **Build time: ~5 minutes** average
- **ChromaDB: 5 collections** with FSRS decay (tools, episodes, principles, failures, covenants)
- **900+ nutrients** in soil (auto-migrated from legacy single collection)

### What Was Built (Sessions 1-8)

| Session | Subsystem | Files Created | Tests |
|---------|-----------|---------------|-------|
| 1 | FSRS + 5-collection ChromaDB | fsrs.py, collections.py, soil.py (modified) | 59 tests |
| 2 | Evolutionary archive | archive.py, cascade.py | 33 tests |
| 3 | Crystallization pipeline | crystallizer.py, covenant_registry.py, episode_recorder.py | 33 tests |
| 4 | Autocatalytic NEW_TOOL | tool_registry.py, tool_validator.py, self_improvement.py (modified) | 36 tests |
| 5 | Jitterbug cycle | jitterbug.py, progression.py | 32 tests |
| 6 | DSPy optimization | dspy_modules.py, compiler.py, prompt_store.py | 28 tests |
| 7 | Safety guardrails | overseer.py, probes.py, goodhart_canary.py, dashboard.py | 38 tests |
| 8 | Integration tests + docs | test_e2e_autocatalytic.py, README, CLAUDE.md | 30 tests |

### Test Suite Status
- **v3.0 tests (sessions 1-8):** All pass (289 tests across 9 test files)
- **Pre-existing tests:** 76/86 pass in test_stress.py (10 pre-existing failures unrelated to v3.0), 39/39 milestone456
- **Integration test:** test_e2e_autocatalytic.py covers all subsystems with synthetic data ($0 cost)

---

## ARCHITECTURE OVERVIEW

```
                    SICA Outer Loop
                   /              \
          propose improvement    evaluate + archive
               |                      |
    +----- NEW_TOOL ------+    Evolutionary Archive
    | (autocatalytic:      |   (SQLite DAG, DGM
    |  engine builds tools |    parent selection)
    |  for itself)         |
    +---------------------+
               |
         Jitterbug Cycle
        /        |        \
   Expand    Compress    Reconstruct
   (builds)  (cluster)   (tools + covenants)
        \        |        /
         Validate + Integrate
               |
         Safety Overseer
        (integrity, Goodhart, costs)
               |
         Metrics Dashboard
        (JSONL, growth analysis)
```

### ChromaDB Collections
| Collection | Purpose | Record Types |
|-----------|---------|-------------|
| belief_tools | Self-authored tools, validators | SelfAuthoredTool, skeletons |
| belief_episodes | Build traces | Episode records with 15+ features |
| belief_principles | Patterns, insights | Soft knowledge from successful builds |
| belief_failures | Failure traces | Antipatterns with root cause |
| belief_covenants | Hard rules | Static (6) + dynamically crystallized |

---

## KNOWN ISSUES

### Pre-existing (not from v3.0)
- 10 test_stress.py failures: SICA safety (3), codebase health (4), bittensor miner (3) -- all pre-existing
- 51 silently swallowed exceptions in the codebase (intentionally non-fatal)
- ModelRouter() instantiated multiple times (should be passed through)

### v3.0 Limitations
- DSPy is an optional dependency -- GEPA/MIPROv2 optimization requires `pip install dspy>=2.6.0`
- Jitterbug expansion builds make real API calls ($2/build cap, $10/cycle)
- Autocatalytic tool building catch rate threshold (30%) may be too aggressive for some failure types
- Progression tracker's sklearn dependency is optional -- falls back to threshold clustering
- Canary challenges are defined in code, not in a config file

---

## WHAT'S NEXT

### Immediate
- Run SICA with jitterbug cycles to accumulate tools and covenants
- Benchmark after 10+ jitterbug cycles to measure compounding
- Publish v3.0.0 to PyPI

### Short-term
- Wire optimized DSPy prompts back into agent system prompts
- Expand invariant templates beyond 15 (cover more failure patterns)
- Add more canary challenges for better Goodhart detection
- Dashboard web UI (currently CLI only)

### Medium-term
- Population-based self-improvement (evolutionary_search.py + archive)
- Cross-language tool generation (TypeScript tools)
- Bittensor subnet integration with tool library
