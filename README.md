# Belief Engine

**An autonomous AI system that turns a sentence into working, tested software — and improves itself after every build.**

```bash
pip install belief-engine
```

```bash
belief build --goal "Build a bookmark manager API with FastAPI — CRUD with tags, GET /random. SQLite." \
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
17 AI agents collaborate in a convergence loop:
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

## v3.2: Hardening, Defense, Archive, Repo-Map

Eight focused sessions on top of v3.1:

| Session | What shipped |
|--------|-------------|
| **Ollama hardening** | Retry + circuit breaker on every local LLM call. No more silent stalls when the model server hiccups. |
| **LibCST Pydantic v2 covenant** | AST-level rewrite covenant for v1→v2 migrations. 30 hermetic tests. |
| **Package validator** | 6-layer slopsquatting defense (canonicalize → stdlib → hallucination check → typo guard → guarddog → pip-audit). 23 tests. |
| **Synthesizer router** | Cyclomatic-complexity routing between full Haiku synthesis and a 1.5B local polish fallback. Ablation harness for KEEP/ROUTE/DELETE decisions. |
| **DGM-style agent archive** | ChromaDB-backed retrieval of prior successful builds. The planner injects top-3 priors into every plan. 16 tests. |
| **tree-sitter + PageRank repo-map** | Aider-style repo summary for brownfield work. New `belief repomap` command. |
| **Covenant auto-extraction** | Failure-cluster proposer + precision gate + human-review CLI. **No auto-merge** — `belief covenants approve` is mandatory before a discovered rule lands. |
| **Validation writeup** | The v3.1.0 paired-A/B results above, expanded into a 2-page technical note (`docs/validation/v3.1.0-consistency-results.md`). |

## v3.3: The Ecology Layer

v3.3 is the current release. Background organs that maintain soil quality, manage spend, and propose what to build next. Every organ consults the Economist contract before spending — single audit trail across the system.

| Organ | Tier | What it does |
|-------|------|-------------|
| **Economist** | Tier 0 | Daily USD budget contract. Quote / register / commit primitives every other organ goes through. |
| **Predator** | Tier 1 | Utility-driven culling of low-value soil. Soft-tombstones nutrients via the existing `Soil.invalidate_nutrient` path; companion `Soil.revalidate` brings borderline tombstones back if utility recovers. Min-age + first-run cap prevent overcorrection. |
| **Sleep** | Tier 1 | Phase A (replay) + Phase B (crystallize). First organ to spend LLM money — consults the Economist before each phase. Drives the covenant crystallizer against accumulated build traces. |
| **Garbage Collector** | Tier 1 | Removes broken tools, invalid covenants, and duplicate tool sources. Binary correctness rule, not utility. Caught a real upstream bug on its first live run: the NEW_TOOL pipeline was re-depositing the same tool 13× without dedup. |
| **Curiosity Gate** | Tier 2 (suggest-only) | Proposes build goals that fill gaps in the soil — file extensions, frameworks, covenant-sparse niches. Token-match info-gain estimator against historical builds. `auto_build` deferred. |

## Synthesis Engine: cross-domain word-set input

A sibling input pathway. Instead of a sentence, give the engine two or more concept words and it synthesizes a typed `StructuralMechanism` — predicate signature, Marr-level role arguments, named higher-order relations, open implementation probes — and feeds it into the build pipeline as constraints + acceptance criteria.

```bash
# 1. Synthesize a mechanism from a word set (writes a sidecar):
belief synth words "mantis_shrimp,camera" --no-novelty-gate

# 2. Build from the sidecar:
belief build --goal "..." --sidecar ~/.belief-engine/pending_sessions/<id>.json
```

The four-pass synthesizer runs on Sonnet (brainstorm → predicate-form-forcing → anti-rationalization → final structurer); the 8-check Chain-of-Verification critic runs on Haiku. Novelty is gated against a 6th ChromaDB collection (`belief_biological_primitives`) seeded from the AskNature taxonomy; 14-20 incompleteness probes per mechanism with 2× loopback through the research dispatcher fill in implementation detail before the structural mechanism reaches the build pipeline.

End-to-end demonstrated 2026-05-13: `belief synth words "mantis_shrimp,camera"` → sidecar → `belief build --sidecar ...` produced 13 files of working Python at 0.82 weighted validator score for $0.94 in 967s.

## Mycorrhizal: shared substrate across agents

Eight stages of soil-layer infrastructure shared across every agent in the system.

| Stage | What shipped |
|-------|-------------|
| **1 — Reciprocity ledger** | SQLite-backed per-agent give/take ledger; decomposer hooks; `belief reciprocity`. |
| **2 — Niche ledger** | Per-niche modification ledger at `~/.belief-engine/niches.db` with downstream-reference credit at 0.1/ref to the constructor; `belief niches`. |
| **3 — Snapshot + cold-start** | `SoilSnapshot` (tree-copy for ChromaDB + SQLite backup API for ledgers + atomic 3-rename restore); APScheduler hook in the photosynthesis daemon at 6h cadence with GFS 10/10/10 rotation; `belief snapshot {take,restore,list,verify}` and `belief cold-start --snapshot PATH`. |
| **4 — Signal alphabet + capacity** | Pydantic v2 `Signal` model (closed 5-token Literal), SQLite `SignalStore` with exponential-decay concentration + circular buffer, trigger registry, capacity-measurement plug-in MI estimator; `belief signal {capacity,emit,show}`. |
| **5 — Hub topology + routing + sanctions** | `belief/routing/` package — `HubRegistry`, `Router`, `SanctionsEngine`, `TopologyDiagnostics` on a shared `routing.db`; advisory recomposer hook; enforcement behind `BELIEF_ROUTING_ENFORCE` (default OFF); `belief topology`. |
| **6 — Defense priming + onboarding** | `belief/safety/priming.py` (`Warning` model + `WarningStore` with decay-on-read + `PrimingPropagator`) and `belief/routing/onboarding.py` (demo-task gate, graveyard re-entry → manual approval); `belief warnings`. |
| **7 — Decomposition + succession** | 3-tier decomposition (easy AST fragments / structural import+call edges / recalcitrant failure signatures), quarantine gate, succession modes (PIONEER / MID / MATURE) with policy + humus consolidation; `belief quarantine` and `belief succession`. |
| **8 — Protocol v1 + offline probe** | `docs/PROTOCOL_v1.md` canonical surface spec, `belief/protocol/` with `compatibility_check` (warns-never-refuses), `WeeklyOfflineProbe` with daemon hook; `belief probe offline`. |

## Post-stress hardening

Four sessions that landed after a paid stress-test surfaced specific failure modes — each commit is one knob:

- **Hard `--max-cost` ceiling.** The cost limit is now enforced at agent and tail-node level via a contextvar ceiling; a build whose projected cost exceeds the cap aborts instead of finishing over-budget.
- **No-tests score cap.** A testable-but-untested build can't score 1.00 anymore — the validator caps at 0.6 and tags `FAIL_FIXABLE`.
- **Coverage gate.** New gate after `compile_gate` enforces planned-vs-produced symbol coverage and detects hollow stubs; caps the score at the coverage fraction. The hollow-stub detector recognises trivial returns (`return None`, `return []`, placeholder strings like `'TODO'`) as filler while exempting subclass-only modules where the declaration itself is the deliverable (e.g. an `exceptions.py` of `class NotFoundError(Exception): pass`).
- **Canonical-structure fallback + duplicate-implementation detector.** A `structure_gate` flags two parallel implementations (root + package dup symbols); the architect fallback forces a single canonical layout when the duplication pattern is detected.

## Key Numbers

| Metric | Value |
|--------|-------|
| Codebase | 262 Python files in `belief/`, ~77,700 lines |
| Version | 3.3.0 + Synthesis Engine + Mycorrhizal Stages 1-8 + post-stress A-D |
| Benchmark | **17/20 (85%)** on 20-challenge suite |
| Cost per build | **$0.18** |
| Build time | ~5 minutes |
| ChromaDB collections | 6 — belief_tools, belief_episodes, belief_principles, belief_failures, belief_covenants, belief_biological_primitives |
| Ledger / state stores | reciprocity.db, niches.db, routing.db, signal.db, archive.db, experiments.db, snapshots/ |
| Test suite | 2032+ hermetic tests, hard gate green |

## Quick Start

```bash
pip install belief-engine

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Build something
belief build --goal "Build a URL shortener with FastAPI and SQLite"

# Build + deploy
belief build --goal "Build a REST API" --deploy docker_local --deploy-name myapi

# Run the benchmark
belief benchmark --tiers 1 2 3 4 5
```

### Local-only quick start

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
belief build --goal "Build a Python script that prints hello world"
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
weighting**: a nutrient's retention is proportional to how
often its descendants succeed in later builds. Nutrients whose
downstream uses keep working stay sharp; orphans fade. Contradicted
nutrients are soft-deleted with a `valid_until` timestamp, never
purged — `belief manifold` can show the soil as it was on any
historical date.

The v3.3 ecology organs run on top of this layer: **Predator** prunes
low-utility nutrients (with `Soil.revalidate` to bring back borderline
ones), **Sleep** drives covenant crystallization against accumulated
traces, and **Garbage Collector** removes broken tools and duplicate
sources.

You can watch this happen:

```bash
belief dashboard        # metrics: pass rate, cost, nutrients, covenants
belief manifold         # clusters by domain + coverage gaps
belief economy          # daily budget tracker (v3.3)
belief predator         # what would Predator cull right now
belief sleep            # one consolidation cycle
belief gc               # show / clean broken tools, dup sources
```

### Checking progression per vertical

The generative-chain progression tracker scores each of
eight verticals independently — `fastapi`, `cli`, `mcp`, `data`,
`async`, `library`, `script`, `general` — so you can see which
domains the engine has matured in and which it hasn't touched yet.

```bash
belief progression
```

Output lists every domain and its current stage (Seed → Cluster →
Tessellation → Basis → Connectivity → Archetypes). Domains stuck at
Seed are the ones to target with the next round of builds — or feed
to the Curiosity Gate:

```bash
belief curiosity        # propose build goals that fill soil gaps (v3.3)
```

### Adding Photosynthesis for autonomous goal generation

The Grinder daemon picks goals out of a queue and
builds them continuously. The Photosynthesis daemon
populates that queue by harvesting candidate build goals from
GitHub, PyPI, HN, Stack Overflow, RSS feeds, and ArXiv, then
filtering them through a four-stage cascade (bloom blocklist →
keyword regex → TF-IDF cosine → MiniLM embedding). Together they
turn the engine into a self-running research workshop:

```bash
# Background the grinder (drains the goal queue):
belief grinder start --max-builds 100

# Photosynthesis lives in its own package extras:
pip install "belief-engine[photosynthesis]"
```

The cascade is fail-closed against missing local model cache:
`BELIEF_OFFLINE=1` raises cleanly; `BELIEF_EMBED_ALLOW_DOWNLOAD=1`
opts back in to in-band Hugging Face fetches.

### Adding Claude for hard tasks (hybrid mode)

Hybrid mode routes mechanical agents (intake, tester, synthesizer,
validator) to the local model and keeps reasoning agents (research,
planner, architect, builder, debugger) on Claude — the same
quality ceiling as cloud mode at roughly 1/4 the cost.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BELIEF_MODEL_MODE=hybrid
belief build --goal "Build a distributed task queue with priority lanes"
```

A **confidence-probe-gated
escalation** path runs on top of this: when the probe judges the
local model unlikely to succeed on a given call (confidence < 0.4),
that single call escalates to Claude automatically. Local-first;
Claude is only paid for when needed.

## CLI Commands

| Command | Description |
|---------|-------------|
| `belief build --goal "..."` | Build software from a goal |
| `belief build --goal "..." --sidecar PATH` | Build with hydrated structural mechanism from a synthesis sidecar |
| `belief fix --repo PATH --issue "..."` | Fix an issue in existing code (brownfield) |
| `belief benchmark` | Run benchmark challenges |
| `belief benchmark-compare` | Run benchmark in cloud and local modes; print a comparison table |
| `belief sica --iterations N` | Run SICA self-improvement |
| `belief jitterbug` | Run compression-reconstruction cycle |
| `belief jitterbug --dry-run` | Expansion + compression only |
| `belief progression` | Per-domain generative-chain stage |
| `belief manifold` | Knowledge topology: clusters, cross-links, gaps |
| `belief manifold --json` | Manifold as machine-readable JSON |
| `belief optimize [agent]` | DSPy/GEPA prompt optimization |
| `belief dashboard` | Metrics dashboard |
| `belief dashboard --json` | Metrics as JSON |
| `belief library` | Named library of promoted tools |
| `belief grinder {start,status,pause,resume}` | Autonomous build loop daemon |
| `belief models` | Show active model routing table |
| `belief archive` | Inspect the DGM-style build archive |
| `belief repomap` | PageRank-ranked symbol map (v3.2) |
| `belief covenants {review,approve,reject,run-proposer}` | Review / approve auto-proposed covenants |
| `belief validator add-hallucination` | Package-validator utilities |
| `belief experiment {run,quick,report,ablate}` | Controlled A/B experiments (engine vs raw) |
| `belief probe {train,test,offline}` | Confidence probes + Stage 8 weekly offline probe |
| `belief recombine` | Cross-pollinate soil nutrients |
| `belief mine` | Run as Bittensor subnet miner |
| **v3.3 ecology** | |
| `belief economy` | Economist daily-budget tracker |
| `belief predator` | Soft-tombstone low-utility nutrients |
| `belief sleep` | Offline soil consolidation (replay + crystallize) |
| `belief gc` | Tombstone broken tools, invalid covenants, duplicate sources |
| `belief curiosity` | Suggest build goals that fill soil gaps |
| **Synthesis Engine** | |
| `belief synth words "a,b"` | Cross-domain mechanism synthesis (writes sidecar) |
| `belief synth words "..." --no-novelty-gate` | Bypass novelty filter (demo iteration) |
| `belief synth words "..." --critic-tolerance N` | Allow N of 6 LLM critic checks to fail |
| `belief synth words "..." --novelty-threshold T` | Bio-store similarity threshold |
| **Mycorrhizal** | |
| `belief reciprocity` | Per-agent reciprocity ledger (Stage 1) |
| `belief niches` | Niche-modification ledger (Stage 2) |
| `belief snapshot {take,restore,list,verify}` | Durable soil snapshots (Stage 3) |
| `belief cold-start --snapshot PATH` | Restore + print soil-health summary (Stage 3) |
| `belief signal {capacity,emit,show}` | Signal alphabet + capacity harness (Stage 4) |
| `belief topology` | Routing topology + hub set (Stage 5) |
| `belief warnings` | Active priming / covenant warnings (Stage 6) |
| `belief quarantine {review,approve,reject}` | Quarantined-build review (Stage 7) |
| `belief succession` | Current succession mode + policy (Stage 7) |

## Architecture

```
belief/
  agents/          -- LangGraph agents (intake -> validator -> brownfield)
  validators/      -- AST covenant enforcers + dynamic covenant registry
  covenants/       -- LibCST covenant proposers + precision gate (v3.2)
  memory/          -- ChromaDB soil (6 collections), FSRS, tool registry, episode recorder
  ecology/         -- v3.3 organs: economist, predator, sleep, garbage_collector, curiosity
  photosynthesis/  -- Signal acquisition daemon + synthesis/ (Synthesis Engine) + 9 source adapters
  grinder/         -- Daemon envelope pickup (hydrates synthesis sidecars at build time)
  refinement/      -- Water cycle (analyze -> fix -> revalidate)
  evolution/       -- SICA, archive, crystallizer, jitterbug, progression, tool_validator
  archive/         -- DGM-style retrieval layer (planner priors, v3.2)
  repomap/         -- tree-sitter + PageRank symbol map (v3.2)
  optimization/    -- DSPy/GEPA prompt optimization (optional)
  safety/          -- Overseer, probes, Goodhart canary, defense priming
  metrics/         -- Dashboard, growth analysis
  deploy/          -- Docker + Railway deployment
  codebase/        -- Brownfield (localizer, repo_graph, patch_sampler, change_impact)
  languages/       -- Python + TypeScript adapters
  polarity/        -- Latios/Latias incompleteness engine
  verification/    -- Schemathesis / Hypothesis API verification (optional)
  models/          -- Pydantic models (state, artifacts, skeleton, contracts)
  config/          -- Settings, model routing, local model table
  tools/           -- Composition planner, deployment generator
  prompts/         -- Agent prompts + protocol skeletons
  core/            -- Shared HTTP client (BreakerAsyncClient), small primitives
  daemons/         -- Long-running orchestration helpers
  routing/         -- Mycorrhizal Stage 5: hubs, router, sanctions, topology
  signal/          -- Mycorrhizal Stage 4: signal alphabet, capacity harness
  lifecycle/       -- Mycorrhizal Stage 8: offline probe + succession orchestration
  protocol/        -- Mycorrhizal Stage 8: PROTOCOL_VERSION + compatibility_check
  utils/           -- Cross-cutting small utilities
  cache/           -- On-disk cache scaffolding
  bittensor/       -- Top-level Bittensor integration
  experiments/     -- One-off experimental modules (not in default pipeline)
  hardening.py     -- Budget limits, rate limiter, security scanner, audit log  [IMMUTABLE]
  benchmark.py     -- Scoring logic                                              [IMMUTABLE]
  graph.py         -- LangGraph pipeline wiring (all nodes + edges)
  llm.py           -- Anthropic + Ollama clients (streaming, retry, breaker, JSON repair)
  cli.py           -- CLI entry point (subcommands listed above)
```

## Model Routing

| Agent | Model | Role |
|-------|-------|------|
| Research, Planner, Architect, Builder, Debugger | Sonnet 4.6 | Deep reasoning |
| Intake, Tester, Gap Analyst, Synthesizer, Validator, Latios, Executor | Haiku 4.5 | Mechanical tasks |
| Skeleton, Covenant Enforcer, Import Fix, Validator core | None | Deterministic (zero tokens) |
| Safety overseer | Haiku 4.5 | **Different model than agent** — prevents self-deception |

Prompt caching provides 90% savings on repeated system prompts. Combined with Haiku routing, builds cost **$0.15-0.25**.

## Tech Stack

- **Python 3.11+** (tested on 3.14)
- **LangGraph** for agent orchestration
- **Anthropic Claude** (Sonnet 4.6 + Haiku 4.5)
- **Ollama** for local-mode inference (qwen2.5-coder:14b default)
- **ChromaDB** for learning memory (6 collections with FSRS-4.5 decay)
- **SQLite** for evolutionary archive + reciprocity / niches / routing / signal / experiments ledgers
- **pybreaker** for circuit-breaker protection on every LLM call (cloud + local)
- **LibCST** for AST-level rewrite covenants
- **tree-sitter + PageRank** for the repo-map
- **APScheduler** for the photosynthesis daemon job loop
- **Docker** for deployment
- **DSPy** (optional) for prompt optimization

## License

MIT

## Author

Built by [Fio](https://github.com/metafiopy-tech) -- solo, from scratch, while making pizzas.

*"The remainder after every operation drives the next cycle."*
