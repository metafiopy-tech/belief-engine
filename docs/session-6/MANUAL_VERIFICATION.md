# Session 6 — Manual verification checklist for Joe

Assumes sessions 1-5 are merged on main. Session 6 is the first of the three architectural deltas (6 archive → 7 repomap → 8 covenant auto-extraction).

## 0. Files changed

**New files:**
```
belief/archive/__init__.py             # public API exports
belief/archive/config.py               # AgentConfiguration dataclass
belief/archive/outcome.py              # BuildOutcome dataclass
belief/archive/fitness.py              # SICA-style utility function
belief/archive/store.py                # AgentArchive (ChromaDB wrapper)
belief/archive/sampler.py              # parent_sample (Boltzmann)
belief/archive/priors.py               # format_priors_block for planner
belief/archive/persist.py              # post-build hook
tests/test_agent_archive.py            # 16 hermetic tests
docs/session-6/MANUAL_VERIFICATION.md  # this file
```

**Modified:**
```
belief/agents/planner.py   # injects top-3 priors into system prompt (append-only)
belief/cli.py              # `belief archive inspect [--goal ...] [--top N]` + post-build persist hook
```

**NOT touched:**
- `belief/evolution/archive.py` — the lineage-tracking SQLite DAG stays intact.
  Session 6's archive is a RETRIEVAL layer keyed by goal similarity; the two
  coexist without overlap.

## 1. Commit

```bash
cd ~/Desktop/belief-engine
git checkout main
git pull
git checkout -b session-6-agent-archive

git add belief/archive/ belief/agents/planner.py belief/cli.py \
        tests/test_agent_archive.py docs/session-6/

git commit -m "session-6: archive-as-first-class-citizen (DGM pattern + SICA utility)

- belief/archive/config.py AgentConfiguration: full per-agent snapshot
  (system_prompt, user_prompt_template, model, model_options,
  covenant_set, tool_schemas, code_hash). to_dict/from_dict/to_json
  round-trip.
- belief/archive/outcome.py BuildOutcome: per-build bundle. Embeds
  goal+planner_config as the document the archive indexes; stores
  full state in metadata['outcome_json'] for rehydration.
- belief/archive/fitness.py utility(): SICA-style scalar U in [0,1].
  Weights 0.5 score + 0.2 (1-cost/10) + 0.15 (1-time/600) +
  0.15 covenant_rate. Env vars U_W_* override.
- belief/archive/store.py AgentArchive: ChromaDB wrapper owning one
  collection named 'agent_archive'. Persistent at
  ~/.belief-engine/agent_archive/. DefaultEmbeddingFunction (ONNX,
  ships with chromadb). Injectable embedding_function for hermetic
  tests.
- belief/archive/sampler.py parent_sample: Boltzmann-weighted
  sampling (tau=0.2 default) over the top-3k semantic-similar
  candidates. Filters pass+fail_fixable by default; env
  BELIEF_PARENT_TAU tunes temperature.
- belief/archive/priors.py: formats top-k priors as an append-only
  'PRIOR SUCCESSFUL CONFIGURATIONS' block. Header + per-prior goal /
  verdict / score / truncated planner-prompt snippet. Preserves
  session-1 num_keep=512 prefix cache.
- belief/archive/persist.py: persist_build_outcome(final_state) —
  builds a BuildOutcome from the LangGraph final state and writes it
  to the default AgentArchive. Never raises.
- belief/agents/planner.py: calls format_priors_block before
  generate_structured and concatenates to PLANNER_SYSTEM. Append-only.
- belief/cli.py: persist_build_outcome fires after BUILD COMPLETE
  (alongside the decomposer). New 'belief archive inspect' subcommand
  lists top-N highest-utility past builds (optionally filtered by
  --goal).
- 16 hermetic tests with a hash-based embedding function — no ONNX
  download in CI."
```

## 2. Reinstall (only if pyproject changed — in session 6 it didn't)

```bash
# No new deps in session 6 — chromadb is already there.
# Verify:
python3 -c "import chromadb; print(chromadb.__version__)"
```

## 3. Full test suite

```bash
python3 -m pytest tests/ -q --timeout=60
```

Expected: **~1099 passed** (1083 after session 5 + 16 new session-6 tests). 0 failed, 7 skipped.

Session-6 isolated:
```bash
python3 -m pytest tests/test_agent_archive.py -v
```
Expected: 16 passed in <1s.

## 4. Smoke test — archive fills + planner injects

First build writes a BuildOutcome. Second build should see the prior injected.

```bash
# Fresh archive (optional — skip if you want to preserve existing data).
rm -rf ~/.belief-engine/agent_archive/

# First build — archive starts empty.
belief --mode local --goal "Build a FizzBuzz script"
```

Log should end with something like:
```
AgentArchive: persisted belief-xxxxx (verdict=pass, score=1.00, U=0.937)
```

Now a second, similar build:
```bash
belief --mode local --goal "Build a FizzBuzz clone with Click"
```

Log at planner start should show:
```
Planner: injected 1 prior(s) from agent archive
```

## 5. Inspect the archive

```bash
belief archive inspect --top 5
belief archive inspect --goal "FastAPI settings" --top 3
```

Both should print a table of prior builds with `verdict`, `weighted_score`, and `utility_score` (U).

## 6. What was NOT validated in the sandbox

- **Real ONNX embedding model** — tests use a hash-based embedder to avoid
  a HuggingFace download from inside the sandbox. Production uses
  ChromaDB's `DefaultEmbeddingFunction` (all-MiniLM-L6-v2). First build
  after install will trigger a one-time ~90MB ONNX model download to
  `~/.cache/chroma/onnx_models/`. Subsequent builds are cache-hits.
- **End-to-end retrieval quality** — the hash-based embedder passes the
  "query returns persisted docs in roughly-goal-similar order" test,
  but real all-MiniLM embeddings will give much better retrieval.
  Only a live run against 424 priors can validate that.
- **Migration from existing experiments.db** — deferred. The spec
  mentioned a `scripts/migrate_v31_to_v32_archive.py`; skip for now.
  If you want past builds in the archive, run some builds post-merge
  and they'll accumulate. Backfilling the 424 historical builds is a
  follow-up session.

## 7. Known limitations / follow-ups

- **Per-agent AgentConfiguration capture** — `persist.py` currently
  synthesises a minimal planner AgentConfiguration at persist time.
  A proper implementation would have `BaseAgent.__call__` record the
  exact system prompt (including the injected priors, for lineage),
  model, options, and code_hash on each call, then propagate them
  into state. That's a session-6-followup touching `BaseAgent`.
- **Covenant file write-protection** — the session doc mentioned
  `chmod 444` during agent execution to prevent reward-hacking.
  Deferred — it's not trivially portable (macOS vs Linux permissions,
  interaction with import_fix_node writing code files), and the
  proposer-pipeline safety in Session 8 (human-review gate) is a
  stronger defence regardless.

## 8. Merge

```bash
git checkout main
git merge --no-ff session-6-agent-archive
git push origin main
```

Session 7 next — tree-sitter + PageRank repo-map, ported from Aider.
