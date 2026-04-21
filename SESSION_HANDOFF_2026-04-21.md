# Belief Engine — Session Handoff (2026-04-21)

Copy this file (or paste its contents) into a new chat to resume without losing context.

## Where things stand

- **Branch:** `main`, 22 commits ahead of `origin/main`, **nothing pushed yet.**
- **Tests:** `661 passed, 7 skipped, 0 failed` (python3 -m pytest tests/ -q --timeout=60).
- **Repo location:** `~/Desktop/belief-engine/` on the user's Mac. (It's inside iCloud Desktop sync — this caused real problems; see "Environment quirks" below.)
- **Working on:** sessions from `~/Desktop/belief-engine/COMPLETE_CLAUDE_CODE_SESSIONS.md` (the original 18-session plan, still uploaded in the chat).

## Completed sessions

| Session | Phase | Commit | What landed |
|--------|-------|--------|-------------|
| 1 | 1 — Fixes | `9b39571` | Skeleton CLI schema normalizers (dict→str), Nutrient `created_at` ISO-string fix |
| 2 | 1 — Fixes | `004a3bf` | `formulate_tool_goal` embeds real error examples; +8 benchmark challenges |
| 3 | 2 — Photosynthesis | `f0bff63` | Harvester core: 6 sources, 4-stage cascade filter, APScheduler, systemd, state.py (SQLite WAL) |
| 4 | 2 — Photosynthesis | `3a09acb` | Goal synthesis: novelty bands, ZPD difficulty, ACCEL heap, Sonnet generator, renderer |
| 5 | 2 — Photosynthesis | `2902b38` | Safety stack (real cost_tracker + kill_switch + anomaly + audit hash-chain + rate_limits + HITL) + bittensor module |
| 6 | 3 — Local models | `4e56a48` | Ollama backend, hybrid ModelRouter, local_cost_tracker (respects "don't modify hardening.py"), `belief models` CLI |
| 7 | 3 — Local models | `590ede3` | Per-vertical progression tracking, domain-aware recomposer re-ranking, `belief benchmark-compare`, dashboard `soil_lift` field |
| 8 | 4 — Grinder | `9b99642` | GrinderDaemon autonomous build loop, goal_queue, atomic status file, systemd, `belief grinder {start,status,pause,resume}` |
| 9 | 5 — Metacognition | `c33c4d8` | TraceCollector (async SQLite writes), `_traced` wrapper in graph.py, opt-in via `BELIEF_ENABLE_TRACE=1` |
| 10 | 5 — Metacognition | `668fb91` | ConfidenceProbe (sklearn GradientBoosting + CalibratedClassifierCV), opt-in routing via `BELIEF_ENABLE_PROBE=1`, `belief probe {train,test}` |
| 11 | 6 — Trophic PRM | `a5470bb` | Belief competition: sandboxed subprocess `compete()`, `TrophicRelation`, Haiku-backed test synthesizer (injectable client) |
| 12 | 6 — Trophic PRM | `4320f3f` | Library inductor (Haiku names apex predators), `belief library` CLI, strict Pydantic NamingResult schema with retry-once |

## Still to do

| Session | Phase | Title |
|---------|-------|-------|
| 13 | 7 — Advanced Memory | Clade-Productivity FSRS + Voyage Embeddings (line 1156 of session file) |
| 14 | 7 — Advanced Memory | Bi-Temporal Knowledge + Domain Manifold (line 1213) |
| 15 | 8 — Self-Regulating | Active Inference Jitterbug + Assembly Theory (line 1258) |
| 16 | 8 — Self-Regulating | Danger-Theory Safety Gates (line 1322) |
| 17 | 9 — Full Local | Full Local Mode + Performance Optimization (line 1374) |
| 18 | 9 — Full Local | Packaging + Distribution (line 1416) |

**Next up:** Session 13 — Clade-Productivity FSRS + Voyage Embeddings.

## Conventions the user has approved (don't re-ask)

- **One session at a time.** Commit + merge per session before moving on.
- **Branch per session.** `v3.x/session-N-slug-description`, merged `--no-ff` to main.
- **Stop and ask on failure.** Don't silently work around test failures.
- **Don't push to origin.** The user runs `git push` separately when ready.
- **User runs pytest on their Mac.** Sandbox can't load chromadb/langgraph.
- **Commit messages: heredoc format**, include "Acceptance criteria deferred" section for anything that needs live services (Anthropic, Ollama, real ChromaDB, etc.).

## Hard constraints (project CLAUDE.md)

- **DO NOT modify** `belief/benchmark.py` scoring logic or `belief/hardening.py`.
- All 412+ baseline tests must stay green (now 661).
- Python 3.14, LangGraph, Anthropic Claude, ChromaDB.

## Patterns that have worked well

- **Inject collaborators** (LLM clients, validators, probes) via `Protocol` or simple callables. Tests pass fakes; production passes real clients.
- **Lazy imports for heavy deps** (sklearn, ollama, feedparser, sentence-transformers, bittensor, chromadb). Module loads without them; runtime gracefully degrades.
- **Opt-in via env var** for anything that changes build behavior (`BELIEF_ENABLE_TRACE=1`, `BELIEF_ENABLE_PROBE=1`). Default off keeps tests unchanged.
- **Workarounds for "don't modify hardening.py":** add side-car modules (e.g. `belief/config/local_cost_tracker.py`) rather than extending BuildBudget.
- **Defer wiring** into big existing files (decomposer, jitterbug) when the spec's intent can be met by a public helper + a note in the commit message.

## Environment quirks to remember

- **iCloud Desktop sync** evicted files to stubs at the start of the work session. User granted Terminal "Files and Folders → Desktop" permission which fixed it. Repo is still on Desktop.
- **Stale `.git/index.lock`** appears after most commits from the sandbox side. Every commit block starts with `rm -f .git/index.lock`.
- **Sandbox pytest fails to collect** anything that imports `belief.memory.__init__` because soil.py imports `chromadb`. Do AST-parse + smoke-test non-memory modules in sandbox, then ask the user to run the full suite on their Mac.
- **Vim opens for merge commit messages.** User's workflow: Esc → `:wq` → Return.

## How to resume in a new chat

1. Open a new chat in the same working directory (`~/Desktop/belief-engine`).
2. Paste this handoff doc or drop it in as a file.
3. Ask Claude to read `COMPLETE_CLAUDE_CODE_SESSIONS.md` starting at the line for Session 13 (1156) and execute it, following the conventions above.
4. If the filesystem mount works immediately and `git log --oneline -1` shows `719e560`, you're good.

---

## Pushing to GitHub

Remote is already configured:

```
origin	https://github.com/metafiopy-tech/belief-engine.git (push)
```

To push the 22 pending commits:

```
cd ~/Desktop/belief-engine
git push origin main
```

If GitHub prompts for credentials, it wants a Personal Access Token, not your password. On macOS the token is usually cached in Keychain after the first push.

First-time push (if the token isn't cached):

```
# Option A: HTTPS + PAT
# Go to github.com → Settings → Developer settings → Personal access tokens
# → Generate a classic token with 'repo' scope
# Paste the token when git prompts for "Password"

# Option B: switch to SSH if you have a key already loaded
git remote set-url origin git@github.com:metafiopy-tech/belief-engine.git
git push origin main
```

**Don't push branches.** We used `v3.x/session-N-*` as local-only feature branches and merged them `--no-ff` to main. If you want to prune them:

```
git branch | grep 'v3.x/session-' | xargs git branch -d
```

(They're already merged to main, so `-d` succeeds; `-D` would force-delete.)

To verify the push landed:

```
git fetch origin
git status
# Expect: "Your branch is up to date with 'origin/main'."
```

---

Generated at session-12 completion. Next session: #13.
