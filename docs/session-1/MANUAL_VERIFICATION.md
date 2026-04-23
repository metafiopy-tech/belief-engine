# Session 1 — Manual verification checklist for Joe

Everything below must be run on your MacBook Air against a live Ollama and Python 3.14 install. The sandbox this code was built in has only Python 3.10 and no Ollama, so these steps were NOT executed as part of the session.

## 0. Files changed by this session

Copy / sync these from the Cowork workspace before doing anything else. `git` in the sandbox was broken (silent bus-error on `git log`) so **no commit was made from inside the session** — you will need to do it yourself on the Mac.

**New files:**
```
belief/llm_errors.py
belief/thermal.py
scripts/setup_ollama_env.sh
scripts/thermal_gate.py
tests/test_llm_client.py
docs/session-1/system-prompt-audit.md
docs/session-1/MANUAL_VERIFICATION.md   # this file
```

**Modified files:**
```
belief/llm.py                  # AsyncOllamaClient rewrite + ROLE_BUDGETS + breaker wiring
belief/agents/base.py          # thermal_gate preflight at start of __call__
pyproject.toml                 # tenacity>=9.0, pybreaker>=1.0 added to main deps
```

**Not touched** (verified clean):
- `belief/benchmark.py` — scoring logic untouched (project rule)
- `belief/hardening.py` — untouched (project rule)
- All 130+ other Python files

## 1. Create the branch and commit on the Mac

```bash
cd ~/Desktop/belief-engine
git checkout -b session-1-ollama-hardening

# Stage the session-1 files explicitly (not `git add -A`, which would pick
# up noise from other work).
git add \
  belief/llm.py \
  belief/llm_errors.py \
  belief/thermal.py \
  belief/agents/base.py \
  scripts/setup_ollama_env.sh \
  scripts/thermal_gate.py \
  tests/test_llm_client.py \
  pyproject.toml \
  docs/session-1/

git commit -m "session-1: bulletproof Ollama client + env config + thermal gate

- New belief/llm_errors.py: OllamaError hierarchy (transient/permanent/stall/context)
- belief/llm.py: streaming w/ per-chunk inactivity watchdog, tenacity retry,
  pybreaker per-model breaker, per-role ROLE_BUDGETS, graceful_degradation_cascade,
  4xx context-length classifier, num_keep=512 for prefix-cache hits
- New belief/thermal.py: macOS thermal pressure gate, wired into BaseAgent.__call__
- New scripts/setup_ollama_env.sh: launchctl setenv for GUI-launched Ollama.app
- New scripts/thermal_gate.py: CLI shim for belief.thermal
- New tests/test_llm_client.py: 13 hermetic tests for watchdog / retry / breaker / budgets
- pyproject.toml: tenacity + pybreaker promoted from photosynthesis extra to core deps
- No changes to belief/benchmark.py or belief/hardening.py (project rule)
- System prompts audited as byte-stable for num_keep=512 (see docs/session-1/)"
```

## Environment activation (after commit)

```bash
pip install -e ".[dev,local]"             # tenacity + pybreaker now auto-install
```

## 1. Run the full test suite

The session's hard gate. Expected: 877+ pass, only pre-existing skips.

```bash
python -m pytest tests/ -q --timeout=60
```

Pass criteria:
- Previous baseline was **877 passed, 0 failed, 7 skipped** (per `SESSION_HANDOFF_2026-04-22.md`).
- New baseline should be **890 passed, 0 failed, 7 skipped** (+13 new tests in `tests/test_llm_client.py`).
- If anything NEW fails, `git revert HEAD` and tell me which test. Don't patch forward — sessions are designed to be atomic.

## 2. Run the session-1 test suite in isolation

Faster signal on whether the hardening itself is wired correctly:

```bash
python -m pytest tests/test_llm_client.py -v
```

Expected:
```
13 passed in ~20s
```

Each test corresponds 1:1 to a success criterion in the session doc:

| Test | Validates |
|---|---|
| `test_watchdog_fires_when_stream_stalls` | Inactivity watchdog + `keep_alive=0` runner unload |
| `test_context_exceeded_does_not_retry` | Context-length error is permanent, not retried |
| `test_transient_retries_three_times_then_raises` | `stop_after_attempt(3)` |
| `test_breaker_opens_after_five_failures` | pybreaker per-model, `fail_max=5` |
| `test_executor_short_budget_enforced` | `ROLE_BUDGETS` wall-clock ceiling |
| `test_role_budgets_dictionary_matches_session_doc` | Budget values match the doc |
| `test_health_ok_returns_false_on_unroutable_host` | 5s preflight ping |
| `test_thermal_gate_unknown_is_noop_off_macos` | Thermal gate safe on CI |
| Plus 4 classifier unit tests + 1 async thermal gate test |

## 3. Set up the Ollama environment

**This is the step that unlocks the prefix-cache speedup.** Run it once. Must be done by a logged-in user — Claude Code can't run `launchctl`.

```bash
bash scripts/setup_ollama_env.sh
```

Then:
1. Quit Ollama.app completely (menu bar → Quit).
2. Re-launch from Applications.
3. Verify the env actually applied to the GUI-launched daemon:
   ```bash
   launchctl getenv OLLAMA_KEEP_ALIVE     # should print -1
   launchctl getenv OLLAMA_FLASH_ATTENTION # should print 1
   launchctl getenv OLLAMA_KV_CACHE_TYPE   # should print q8_0
   ```

**CRITICAL:** If `OLLAMA_KV_CACHE_TYPE` ever ends up as `q4_0`, stop and switch back to `q8_0`. q4_0 measurably degrades Qwen2-family code quality (llama.cpp PR#7527).

## 4. Run the smoke build

```bash
belief --mode local --goal "Build a FizzBuzz script"
```

Success criteria:
- [ ] Completes without an `httpx.ReadTimeout` crash on the architect.
- [ ] Builder-through-executor loop converges.
- [ ] Final wall clock < 300s (was 500-600s pre-session for small builds).

If you see a `circuit breaker open for qwen2.5-coder:14b` error on the FIRST build: the breaker from a prior session is persisting. Restart the Belief Engine process; breakers are in-memory per process.

## 5. Run the consistency benchmark

Three runs back-to-back. This is the one that validates the prefix-cache speedup materialised.

```bash
belief experiment quick --n 3
```

Compare against the baseline from `SESSION_HANDOFF_2026-04-22.md` (run 1):

| Challenge | Baseline | Target |
|---|---|---|
| challenge 1 | 734s | ≤550s (-25%) |
| challenge 2 | 325s | ≤245s (-25%) |
| challenge 3 | 822s | ≤615s (-25%) |
| challenge 4 | 870s | ≤650s (-25%) |
| challenge 5 | 579s | ≤435s (-25%) |

Pass criteria:
- Per-challenge wall clock drops **≥25%** on at least 3 of 5 challenges.
- `belief experiment report` shows the engine lift still holds (engine 5/5 vs raw N/5).

If the timing drop is smaller than 25%:
- Check `launchctl getenv OLLAMA_KEEP_ALIVE` — most likely the env didn't apply.
- Check that Ollama.app was actually restarted after step 3.
- Check `num_keep=512` wasn't overridden by an older `BELIEF_OLLAMA_*` env var.

## 6. Spot-check the thermal gate

On a warm machine (`notifyutil -g com.apple.system.thermalpressurelevel` returns "1" or higher), a build's agent timings should show occasional 10-30s gaps between agent starts, logged as:

```
belief.thermal INFO: thermal_gate: pressure=moderate sleeping=10s
```

Those gaps are the gate doing its job — they prevent the 90-180s throttle event that would otherwise hit later in the build. If `pressure=nominal` throughout, you won't see log entries (2s sleep is too brief to notice).

## 7. What was NOT validated in the sandbox

For honesty: Claude Code built this in a Linux sandbox with Python 3.10 and no Ollama. The following were verified via unit tests only (mocked httpx, no real network):

- Real httpx streaming behaviour against Ollama's NDJSON format → unit-tested with a fake stream, not with a live runner.
- `launchctl setenv` side-effects → script written, not executed.
- Actual prefix-cache TTFT speedup → depends on step 3 being run and Ollama restarted.
- Thermal-gate `notifyutil` invocation → no-op on Linux, tested on the macOS-unknown branch.

Step 4 (smoke build) and Step 5 (consistency benchmark) are the only real-world validations. If they pass, the session is shippable. If step 5 shows <25% wall-clock drop, the change is still net-positive (no crashes, cleaner error handling) but the prefix-cache win didn't land — investigate the Ollama env first.

## 8. Merge

Once steps 1-5 pass, merge the branch:

```bash
git checkout main
git merge --no-ff session-1-ollama-hardening
git push origin main
```

The next session (Session 2 — LibCST Pydantic v2 covenant) can start immediately after.
