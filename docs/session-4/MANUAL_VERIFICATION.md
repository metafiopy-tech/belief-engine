# Session 4 — Manual verification checklist for Joe

Assumes sessions 1-3 are merged on main. Session 4's code ships independently; the synthesizer KEEP/ROUTE/DELETE decision is deferred until you run the ablation harness overnight.

## 0. Files changed

**New files:**
```
belief/synthesizer_router.py                 # should_polish(state) — 5 signals
belief/prompts/builder_strict.md             # stronger builder prompt (unused until someone wires it)
scripts/synthesizer_ablation.py              # 40×3×3 ablation harness, SQLite-backed
tests/test_synthesizer_router.py             # 18 hermetic tests
docs/SYNTHESIZER_DECISION.md                 # template — fill in after ablation
docs/session-4/MANUAL_VERIFICATION.md        # this file
```

**Modified:**
```
belief/graph_local.py                        # _route_after_executor now returns validator|synthesizer
belief/agents/synthesizer.py                 # 1.5B polish fallback via SYNTHESIZER_POLISH_MODEL
belief/cli.py                                # `belief experiment ablation-synth` + `belief validator add-hallucination`
pyproject.toml                               # radon>=6.0 added
tests/test_graph_local.py                    # updated to cover new router behaviour + old opt-out
```

## 1. Commit

```bash
cd ~/Desktop/belief-engine
git checkout main
git pull
git checkout -b session-4-synthesizer-router

git add \
  belief/synthesizer_router.py \
  belief/prompts/builder_strict.md \
  belief/graph_local.py \
  belief/agents/synthesizer.py \
  belief/cli.py \
  scripts/synthesizer_ablation.py \
  tests/test_synthesizer_router.py \
  tests/test_graph_local.py \
  pyproject.toml \
  docs/SYNTHESIZER_DECISION.md \
  docs/session-4/

git commit -m "session-4: synthesizer router + ablation harness (pre-decision)

- belief/synthesizer_router.py: should_polish(state) returns (polish?, reason)
  based on 5 signals: tests_failed, ruff_errors>3, cyclomatic>12,
  lines_added>150, wallclock_so_far<180. Env SYNTHESIZER_ROUTE_ENABLED=0
  restores pre-router behaviour (always polish).
- belief/graph_local.py _route_after_executor: on success, call
  should_polish and route to either synthesizer or validator directly.
  Failure path unchanged.
- belief/agents/synthesizer.py: when local mode, temporarily override
  router.local_model to qwen2.5-coder:1.5b (env SYNTHESIZER_POLISH_MODEL
  to tune). 14B polish is ~180s; 1.5B is ~8s. Original model restored
  in finally. Fresh LLMClient → no cached Ollama client → swap is clean.
- scripts/synthesizer_ablation.py: 3-condition × N-run harness.
  SQLite at ~/.belief-engine/ablations.db. Resumable. Prints summary
  + decision signals.
- belief/cli.py: `belief experiment ablation-synth --n N` delegates to
  scripts/synthesizer_ablation.py via subprocess. Also wires session-3
  follow-up: `belief validator add-hallucination <name>`.
- docs/SYNTHESIZER_DECISION.md: hand-over template, decision deferred
  to post-ablation.
- tests: 18 new (test_synthesizer_router.py), 1 updated
  (test_graph_local.py::test_route_after_executor_success) to cover
  both the new router behaviour and the env-disabled opt-out."
```

## 2. Reinstall

```bash
pip3 install -e ".[dev,local]"
```

Expected new install: `radon` (if not already a transitive).

## 3. Run the full test suite

```bash
python3 -m pytest tests/ -q --timeout=60
```

Expected: **~1083 passed** (1065 after session 3 + 18 new session-4 tests). Zero failed, 7 skipped.

## 4. Spot-check the router

```bash
belief --mode local --goal "Build a FizzBuzz script"
```

In the logs, look for a line like:
```
synthesizer router: skip-polish: tests_passed, ruff_err=0, max_cc=0, lines=15, wallclock=30s — nothing to polish → validator
```

That means the router fired and sent the clean build directly to validator, skipping the ~180s polish.

If you see:
```
synthesizer router: polish: ruff_errors=7 > 3 → synthesizer
```
...then the builder produced code that genuinely needs polish — the router is doing its job.

## 5. Spot-check the 1.5B polish fallback

Requires qwen2.5-coder:1.5b be pulled first:
```bash
ollama pull qwen2.5-coder:1.5b
```

Then on a build that DOES go through the synthesizer:
```bash
# Force polish by disabling the router:
SYNTHESIZER_ROUTE_ENABLED=0 belief --mode local --goal "Build a URL shortener"
```

Log should show:
```
Synthesizer: routing polish through qwen2.5-coder:1.5b (local) instead of qwen2.5-coder:14b
Synthesizer: polished N file(s)
Synthesizer completed in ~8-15s
```

(vs ~180-300s on 14B). If you want to disable the fallback and measure the 14B baseline, set `SYNTHESIZER_POLISH_MODEL=""` before the build.

## 6. Run the ablation (OVERNIGHT)

This is the decision-generator.

```bash
# 10 challenges × 3 conditions × 3 runs ≈ 90 builds × ~10min = ~15 hours.
belief experiment ablation-synth --n 3
```

Resumable — if you cancel mid-run, re-running picks up where it left off. To see current state without adding more:
```bash
belief experiment ablation-synth --report
```

When it finishes, it prints a summary table and a tentative recommendation (DELETE / ROUTE / KEEP). Fill in `docs/SYNTHESIZER_DECISION.md` by hand with:
- The summary table (copy from stdout).
- The lift percentages.
- Your decision with a 1-paragraph explanation.

If the decision is DELETE: open a follow-up session to:
- Remove the synthesizer node from `graph_local.py` and `graph.py`.
- Move deployment artifact generation (Dockerfile, docker-compose, run.sh)
  to a deterministic templater in `belief/agents/deploy_artifacts.py`.
- Delete `belief/agents/synthesizer.py` and its tests.

If the decision is ROUTE: no code changes needed; session 4 is already
shipping the router as default.

If the decision is KEEP: revert session 4's router changes (keep the
1.5B fallback — that's pure win regardless).

## 7. What was NOT validated in the sandbox

- **The 1.5B swap against a real Ollama** — the swap logic is in a
  finally block and is hermetic against test assertions of the
  router.local_model value; it hasn't been exercised against a live
  1.5B runner.
- **The ablation harness end-to-end** — the harness uses `belief.cli`
  via subprocess; the subprocess path works but the actual 90-build
  run is overnight work.
- **Integration between the new `belief experiment ablation-synth --report`
  CLI and the subprocess harness** — individually tested; the
  CLI dispatcher delegates via subprocess.

## 8. Merge

```bash
git checkout main
git merge --no-ff session-4-synthesizer-router
git push origin main
```

Session 5 next — this one's a write-up session (v3.1 consistency results → README / LinkedIn / 2-page technical note). No code, fast merge.
