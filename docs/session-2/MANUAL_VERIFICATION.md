# Session 2 — Manual verification checklist for Joe

This session assumes Session 1 is already committed and merged on your Mac. If not, finish Session 1 first (see `docs/session-1/MANUAL_VERIFICATION.md`).

Everything below must be run on your MacBook Air against Python 3.14. Sandbox was Python 3.10 and cannot reproduce the full 1012-test baseline; zero regressions were confirmed against the subset of 834 tests the sandbox can run.

## 0. Files changed by this session

**New files:**
```
belief/covenants/__init__.py            # 4-stage enforce_python_covenants pipeline
belief/covenants/pydantic_v2.py         # LibCST transformer (~700 lines)
belief/covenants/forbidden_imports.py   # requirements.txt stdlib stripping
belief/prompts/pydantic_v2_cheatsheet.md # 1-2KB contrastive cheatsheet
belief/prompts/cheatsheets.py           # cheatsheet loader + trigger detection
conftest.py                              # repo-root pytest config (langchain_core warning filter)
tests/test_pydantic_covenant.py         # 30 hermetic tests
docs/session-2/MANUAL_VERIFICATION.md   # this file
```

**Modified files:**
```
belief/graph.py                          # _covenant_enforce_node now runs LibCST pipeline FIRST, then existing AST validators
belief/agents/builder.py                 # per-file Pydantic v2 cheatsheet injection (append-only)
pyproject.toml                           # libcst>=1.8.6, bump-pydantic==0.8.0, ruff>=0.9.0 added to main deps
```

**Not touched** (verified clean):
- `belief/benchmark.py` — scoring logic untouched (project rule)
- `belief/hardening.py` — untouched (project rule)
- `belief/validators/` — EXISTING AST covenant infrastructure preserved; session-2 pipeline runs UPSTREAM of it (complement, not replace)

## 1. Create the branch and commit

```bash
cd ~/Desktop/belief-engine

# Only if you're not already on session-1's tip:
git checkout session-1-ollama-hardening
git checkout -b session-2-pydantic-covenant

# Stage the session-2 files explicitly.
git add \
  belief/covenants/ \
  belief/prompts/cheatsheets.py \
  belief/prompts/pydantic_v2_cheatsheet.md \
  belief/graph.py \
  belief/agents/builder.py \
  conftest.py \
  pyproject.toml \
  tests/test_pydantic_covenant.py \
  docs/session-2/

git commit -m "session-2: LibCST covenants kill Pydantic v1↔v2 thrash

- New belief/covenants/: 4-stage pipeline (regex prepass → LibCST →
  ruff --fix → bump-pydantic) runs upstream of the debugger so
  Qwen's v1 output is rewritten before the debugger sees it.
- belief/covenants/pydantic_v2.py: LibCST transformer for imports
  (pydantic.v1 / langchain_core.pydantic_v1 / BaseSettings),
  Config → ConfigDict with field renames, @validator → @field_validator,
  @root_validator → @model_validator, method renames (.dict / .json /
  .parse_obj / .parse_raw / .schema / .copy / .update_forward_refs),
  conint/constr → Annotated, __root__ TODO marker.
- belief/covenants/forbidden_imports.py: strips stdlib names from
  requirements.txt using sys.stdlib_module_names (3.10+ authoritative).
- belief/prompts/cheatsheets.py + pydantic_v2_cheatsheet.md: builder
  injects a 1-2KB cheatsheet into the system prompt for pydantic-
  relevant files (models.py/schemas.py/settings.py/config.py, any
  planned import of pydantic/langchain/fastapi, or goal mentioning
  fastapi/pydantic).  Append-only — preserves session-1 num_keep=512
  prefix cache.
- conftest.py: repo-root filter scoped to langchain_core's v1 warning
  only; our own v1 imports still error via pyproject filterwarnings.
- belief/graph.py _covenant_enforce_node: runs LibCST pipeline FIRST,
  then existing validators.enforce_all as a belt-and-suspenders pass.
- 30 hermetic tests in tests/test_pydantic_covenant.py.
- pyproject.toml: libcst + bump-pydantic + ruff in core deps."
```

## 2. Reinstall to pick up new deps

```bash
pip3 install -e ".[dev,local]"
```

Expected success line:
```
Successfully installed libcst-1.8.6 bump-pydantic-0.8.0 ruff-0.9.x
(plus any transitives)
```

## 3. Run the full test suite

Hard gate. Expected: **~1042 passed** (was 1012 after session 1; +30 new tests from `tests/test_pydantic_covenant.py`).

```bash
python3 -m pytest tests/ -q --timeout=60
```

Pass criteria:
- 1042 passed, 0 failed, 7 skipped (± any tests you added between sessions).
- Session-2 suite in isolation:
  ```bash
  python3 -m pytest tests/test_pydantic_covenant.py -v
  ```
  Expected: 30 passed.

If anything NEW fails, revert the commit and flag the test name.

## 4. The v1↔v2 thrash smoke test

This is the central Session 2 claim: Qwen's v1 pattern output is now rewritten deterministically BEFORE the debugger, so the debug loop doesn't oscillate.

```bash
belief --mode local --goal "Build a FastAPI server with Pydantic settings loaded from YAML"
```

Watch for in the logs:

1. **`Covenant pipeline (LibCST+ruff): N rewrites across M files — pydantic_v2.import.rewrite_v1_to_v2×K, …`**
   This line appears only if the covenant fired. If Qwen emitted clean v2 on the first try (it sometimes does, especially with priors in the archive), this line is absent — that's fine.

2. **NO `pydantic.v1` in the final generated code.** Inspect:
   ```bash
   grep -r "pydantic.v1\|langchain_core.pydantic_v1\|from pydantic import BaseSettings" output/<run-id>/
   ```
   Should return zero matches.

3. **The debugger should converge in ≤1 iteration on v1-related issues**, OR the build should pass on first executor run. (The build may still debug on OTHER issues — that's expected.)

4. **Cheatsheet injected** — in the builder log line when it starts `models.py` / `settings.py`, you should see (at DEBUG level) "Builder: injected Pydantic v2 cheatsheet for …". Upgrade your log level if you want to see it:
   ```bash
   BELIEF_LOG_LEVEL=DEBUG belief --mode local --goal "..."
   ```

## 5. Consistency benchmark — should show lift

```bash
belief experiment quick --n 3
```

Target: quality lift + wall-clock improvement vs. Session 1-only baseline. The specific benchmark wins depend on how often v1↔v2 thrashing was eating time; if the overnight 30% debug-iteration figure holds, each challenge should shave ~3-5 minutes off on average.

Per-challenge wall clock goal (vs Session 1 baseline, each challenge independently):
- Challenges that previously hit the v1↔v2 loop: ≥ 20% drop.
- Challenges that didn't: unchanged or small gain from cheatsheet preconditioning.

Also check `belief experiment report` — the engine's pass-rate vs. raw should stay at 20/20 or improve.

## 6. What was NOT validated in the sandbox

- **bump-pydantic subprocess** — ran only via path checks; no real bump-pydantic invocation against generated code. Safe because (a) it's wrapped in try/except and (b) it's Stage 4 of 4 (output is already v2-clean from LibCST).
- **ruff 0.9 on Python 3.14** — sandbox ran ruff 0.15 on Python 3.10. If you see unexpected ruff diagnostics, downgrading to pinned 0.9 may help.
- **Real FastAPI + Pydantic build** — this is step 4 above.

## 7. Known limitations (follow-up, not blockers)

- `@validator("*")` (wildcard) → no rewrite yet. Rare in LLM output; covenant emits a log line but leaves the decorator in place.
- `GenericModel` in v1 → v2 generics — bump-pydantic Stage 4 catches these if installed.
- `Field(env="FOO")` passthrough behaviour for `BaseSettings` — the import is routed to `pydantic_settings` but the `env=` kwarg usage is unchanged; pydantic-settings v2 handles this natively.

## 8. Merge when steps 3-5 pass

```bash
git checkout main
git merge --no-ff session-2-pydantic-covenant
git push origin main
```

Session 3 (layered package validator) is next. It's independent of session 2 (different failure mode — stops wasted pip installs on hallucinated names) so the order can flex if the consistency benchmark comes back weird and you want a quick win.
