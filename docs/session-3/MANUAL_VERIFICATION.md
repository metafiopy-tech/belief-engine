# Session 3 — Manual verification checklist for Joe

Assumes Sessions 1 and 2 are merged on main. This session is independent of Session 2's concerns (different failure mode), so ordering is flexible.

## 0. Files changed by this session

**New files:**
```
belief/validators/package_validator.py       # 6-layer validator (async-first)
belief/validators/import_to_package.py       # curated import→pypi table
belief/validators/known_hallucinations.txt   # seeded blocklist
scripts/refresh_pypi_corpus.sh               # weekly cron/launchd helper
tests/test_package_validator.py              # 23 hermetic tests
docs/session-3/MANUAL_VERIFICATION.md        # this file
```

**Modified files:**
```
belief/agents/executor.py                    # _install_deps uses PackageValidator;
                                             # new _post_install_guarddog and
                                             # _post_install_pip_audit methods
pyproject.toml                               # rapidfuzz, guarddog, pip-audit added
```

**Untouched per project rules:** `belief/benchmark.py`, `belief/hardening.py`.

## 1. Commit on a branch

```bash
cd ~/Desktop/belief-engine
git checkout main
git pull
git checkout -b session-3-package-validator

git add \
  belief/validators/package_validator.py \
  belief/validators/import_to_package.py \
  belief/validators/known_hallucinations.txt \
  belief/agents/executor.py \
  scripts/refresh_pypi_corpus.sh \
  tests/test_package_validator.py \
  pyproject.toml \
  docs/session-3/

git commit -m "session-3: layered package validator + slopsquatting defense

- belief/validators/package_validator.py: 6-layer validator —
  PEP 503 canonicalise → sys.stdlib_module_names reject →
  known_hallucinations.txt blocklist → top-15k PyPI corpus positive →
  PyPI Simple JSON lookup (24h positive / 1h negative cache) →
  rapidfuzz Levenshtein suggestion on reject.
- belief/validators/import_to_package.py: hand-curated import-name
  → pypi-name table (cv2→opencv-python, PIL→Pillow, sklearn→scikit-learn,
  yaml→PyYAML, bs4→beautifulsoup4, dotenv→python-dotenv, jwt→PyJWT,
  MySQLdb→mysqlclient, skimage→scikit-image, attr→attrs, docx→python-docx,
  grpc→grpcio, discord→discord.py, fitz→PyMuPDF, zmq→pyzmq,
  Crypto→pycryptodome, google.generativeai→google-generativeai).
- belief/validators/known_hallucinations.txt: seeded from overnight logs
  (settings, settings_library, fake_pkg, openai_helper, gpt4_utils).
- belief/agents/executor.py: _install_deps now drives PackageValidator
  via asyncio.run inside the existing asyncio.to_thread sandbox, so the
  sync execution path stays unchanged.  Post-install: guarddog (warn-only)
  + pip-audit --strict (block on FAIL).
- scripts/refresh_pypi_corpus.sh: weekly cron/launchd helper for top-15k
  + full PyPI name list (~13MB offline fallback).
- 23 hermetic tests in tests/test_package_validator.py — 0 live PyPI calls.
- Rejection telemetry is LOCAL ONLY (~/.belief-engine/hallucination_log.jsonl).
  Per Seth Larson: don't ship hallucinated-name logs to external telemetry —
  that's a slopsquatter's wishlist."
```

## 2. Reinstall with new deps

```bash
pip3 install -e ".[dev,local]"
```

Expected install output includes `rapidfuzz`, `guarddog`, `pip-audit` (plus transitives — guarddog pulls in semgrep, which is heavy).

**Heads up — opentelemetry conflict.** guarddog's transitives pin older opentelemetry than chromadb needs. On my sandbox, I had to explicitly:
```bash
pip3 install 'opentelemetry-sdk>=1.41' 'opentelemetry-exporter-otlp-proto-grpc>=1.41' 'opentelemetry-exporter-otlp-proto-common>=1.41'
```
If your full pytest fails with `ImportError: cannot import name 'ReadableLogRecord'`, run that command and rerun.

## 3. Run the full test suite

```bash
python3 -m pytest tests/ -q --timeout=60
```

Expected: **~1065 passed** (1042 after session 2 + 23 new session-3 tests), 0 failed, 7 skipped.

Session-3 suite in isolation:
```bash
python3 -m pytest tests/test_package_validator.py -v
```
Expected: 23 passed in <1s. They're all hermetic (no live PyPI calls).

## 4. Refresh the top-15k corpus

First-time setup — the validator works without this (just falls through to PyPI Simple lookup more often), but a warm corpus eliminates ~90% of the lookup latency.

```bash
bash scripts/refresh_pypi_corpus.sh
ls -lh ~/.belief-engine/top-pypi-packages-15k.json
```

Expected: ~0.5MB JSON file in `~/.belief-engine/`. The validator will refetch automatically after 7 days.

If you install `jq` (`brew install jq`), the script also downloads the full PyPI name list (~13MB) for offline fallback:
```bash
ls -lh ~/.belief-engine/pypi-all-names.txt
```

## 5. The two headline smoke tests

**Test A — `pydantic-settings` now accepts:**

```bash
belief --mode local --goal "Build a Pydantic settings manager that loads from YAML"
```

Watch for:
- `Executor: VERIFIED pydantic-settings via top15k` (or `via pypi`). **No `BLOCKED` line on `pydantic-settings`.**
- No wasted 150s retry on bad pip install.
- `pip-audit: no known vulnerabilities` at end of install phase.

**Test B — `timeit` rejects at stdlib layer:**

```bash
belief --mode local --goal "Build a Python profiler that times code execution"
```

Watch for:
- `Executor: BLOCKED 'timeit' at layer=stdlib — timeit is a stdlib module, not a pip package...`
- Rejection fires in <100ms (not 150s).
- The debugger receives the stdlib message in its context and drops the line rather than flailing.

## 6. Consistency benchmark

Same bar as before:
```bash
belief experiment quick --n 3
```

The wins you should see beyond sessions 1-2: faster time to failure on hallucinated packages (was ~150s per bad name, now <100ms for stdlib, <500ms for blocklist). May not show on tier 1-2 challenges because those rarely emit hallucinated deps.

## 7. What was NOT validated in the sandbox

- **Live PyPI Simple JSON lookup** — all tests mock httpx. The code uses the real URL/headers; a real call would work the same way.
- **`guarddog pypi verify` subprocess** — installed but not exercised against real packages. It's warn-only so nothing blocks if it misbehaves.
- **`pip-audit --strict` subprocess** — installed but not exercised. `pip-audit` on a known-vulnerable `requests==2.0.0` should return non-zero; test that on demand if you want.

## 8. Known limitations / follow-ups (not blockers)

- **`belief validator add-hallucination <name>`** CLI subcommand — the Python API exists (`PackageValidator.add_hallucination`), but the CLI wiring in `belief/cli.py` wasn't added this session. Add it when session 4's ablation CLI is wired, since both live in the same file.
- **Per-service `google-cloud-*` resolution** — the translation table has `google.generativeai` but not `google.cloud.storage` / `google.cloud.firestore` / etc. Low priority — those are rare in LLM-emitted code.
- **pipreqs-style fallback** — we intentionally don't default to "return the import name unchanged" when no mapping exists. An unknown import goes through the full 6-layer validator, which may fuzzy-match to the right name. If Joe wants pipreqs-style fallback, add it as a Layer 0.

## 9. Merge

```bash
git checkout main
git merge --no-ff session-3-package-validator
git push origin main
```

Session 4 next — synthesizer router + ablation harness. The harness runs 40×3×3=360 live builds overnight, so the code ships in Session 4 but the decision doc gets filled in after you run it on your hardware.
