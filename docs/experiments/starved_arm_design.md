# STARVED Arm + Variance-Decay Metric — Design & Pre-Registration

**Status:** design locked, code not yet started. Pre-registration block (§7) is
filled *after* the pilot and *before* the full run.

**Claim under test (single):** *coherence without correspondence runs the soil
down to a fixed point.* Make the food source the only variable and watch the
soil-embedding cloud decay (or fail to).

---

## 1. Design: one variable, K-matched arms

Per generation, both arms attempt the **same task stream**, produce the same
candidate builds, and admit the **same number K** of artifacts into soil. The
only difference is the admission ranking key.

- **FED** — admit top-K by the **external grader** (test execution / covenant
  pass). Current engine path.
- **STARVED** — admit top-K by the **model's own self-judgment**
  (LLM-as-judge on its own output). No external test is run for admission.

Both self-judge signals are captured; admission is on LLM-as-judge, with
self-reported confidence logged alongside for analysis.

Same tasks, same K, same seeds, same decay, same everything else. STARVED's
soil therefore accumulates artifacts that *look* coherent and may fail the
hidden external test — the "elegant wrong physics" — and those compound
generation over generation. That compounding is the mechanism under test.

**Task stream:** authored benchmark tiers (`belief/benchmark.py`) as the
rotating per-generation build stream. **SWE-bench Verified subset** as a
held-out **checkpoint probe** for generalization decay (never used for
admission; the self-judge never sees it).

**Recursive influx:** generation N's *new* food (decomposed nutrients) comes
only from artifacts admitted in N−1. FED gets a trickle of truth-grounded food
each generation; STARVED gets none — it eats what it judged good about itself.

---

## 2. Locked design decisions (the seams where this silently goes wrong)

### 2.1 Participation Ratio is computed on the centered Gram matrix
`PR = (Σ λ_i)² / Σ λ_i²` over the eigenvalues of the **centered Gram matrix**
`X̃ X̃ᵀ` (samples × samples), where `X̃` is `X_n` with the per-feature mean
removed. This has the **same nonzero spectrum** as the feature covariance but
stays well-conditioned when nutrients < embedding dimensions — which is the
regime we are actually in.

**Finding that forces this:** with K=4, even an accumulating soil reaches only
~100 nutrients by N=25 — far below any real encoder's dimensionality. **We are
in the n < dims regime for the entire experiment, not just early generations.**
Consequences, pre-registered:

- PR is structurally ceilinged at `n−1`, so absolute PR rises mechanically with
  nutrient count. **K-matching makes that ceiling identical across arms per
  generation**, so it cancels in the FED-vs-STARVED contrast.
- The live signal is therefore **differential PR (STARVED below FED at matched
  n)**, *not* absolute PR shape. Absolute early-generation PR is ceiling-bound
  and is not over-interpreted.
- **Hill q=1 co-headlines with differential PR — jointly.** It is
  proportion-based over a k-means clustering of `X_n` and is immune to the
  n < dims *ceiling* pathology, but it has its own small-n sensitivity (cluster
  count/assignment instability with few points), so it is not a safe *sole*
  headline either. **Support criterion is therefore joint-direction: the thesis
  is supported only if the differential-PR trend and the Hill trend move the
  same direction across the FED/STARVED contrast.** Two metrics with different
  failure modes agreeing is a real result; either alone in this regime is
  contestable. (The original spec named PR the sole headline; this joint
  amendment strengthens it.)

### 2.1a k-means k is frozen up front and asserted at compute time
Hill q=1 is computed over a k-means clustering, so a floating `k` (or floating
selection rule) would move the effective-species count with the hyperparameter
rather than with the soil. **Fix `k` (or pin the selection rule) before any run,
freeze it across both arms and all generations, and assert it at compute time** —
same discipline as freezing the judge (§1) and the encoder (§2.3). The frozen
`k` is recorded in every snapshot's metrics record.

### 2.2 Memory model: accumulating soil with FSRS decay
Soil **accumulates** across generations under normal FSRS decay; the recursive
influx rule gates only the *new* food (N decomposes from N−1 admits), not soil
retention. Rejected alternative: a one-generation sliding window pins n = K = 4
forever, keeping PR permanently at ≤ 3 (pure noise). Accumulating-with-decay
both matches the engine's real soil behavior (so the result transfers to
production claims) and lets n grow enough for the metrics to breathe.
**τ and the noise band are only interpretable relative to this memory model;
it is fixed here before any run.**

### 2.3 Frozen encoder, pinned at the artifact level
One encoder for every snapshot in every arm in every run. Pin the exact model
**and revision and backend and normalization** (primary: a pinned
`all-MiniLM-L6-v2` revision; zero-drift fallback: the deterministic hash EF,
accepted only if semantic spread proves unnecessary). The encoder fingerprint
(model + revision + backend hash) is **asserted into every snapshot**, and the
driver **refuses to run if the pilot and full-run fingerprints differ** — an
encoder that drifts between pilot and full run silently voids the
pre-registration.

### 2.4 Run isolation guardrail (hard-coded, not disciplined)
Each run writes to a soil dir keyed by an explicit **run-id**. The driver
**refuses to start if the target dir is non-empty unless `--resume` is passed.**
Generation leak between runs would not raise an error — it would show up as a
suspiciously healthy STARVED curve, i.e. a false negative on our own thesis,
the worst kind. This is a refuse-to-start condition, not a convention.

---

## 3. The metrics (computed offline from snapshots)

1. **Participation ratio** of the centered-Gram spectrum (§2.1) — differential,
   co-headline.
2. **Hill number q=1** = `exp(Shannon entropy)` over a k-means clustering of
   `X_n` with **frozen k** (§2.1a) — co-headline; effective number of distinct
   nutrient "species."
3. **AR(1) early-warning** — rising lag-1 autocorrelation on the PR (and Hill)
   time series is critical slowing down: the set approaching a fixed point
   *before* the metric flatlines. Predictive signal, not postmortem.
4. **Decay fit** `PR(n) = a·e^(−n/τ) + c` — report τ as "generations to run
   down." Applied to whichever co-headline metric the pre-reg names.

---

## 4. Instrumentation hooks

- **Soil-admission event** (new): per candidate, log
  `{gen, build_id, fed_gate_pass, starved_self_score, self_confidence,
  admitted, arm}`. Lets us count *how many STARVED-admitted artifacts actually
  failed the external test* — the direct count of fictions entering soil.
- **Per-generation soil snapshot** (new): persist
  `(gen, arm, embedding_matrix, nutrient_ids, encoder_fingerprint)`. Metrics
  computed offline from snapshots; never perturbs the run.
- **`BuildOutcome`**: add `self_score` and `external_pass` as two independent
  fields so both arms read from the same record.

---

## 5. Session plan (shipped one at a time; hard gate green + commit between each)

1. **Variance-decay metrics** (`belief/experiments/variance_decay.py`) — PR via
   centered Gram, Hill q=1, AR(1), decay fit. Numpy-only, no engine deps.
   Hermetic tests on synthetic matrices (known spectrum, collapsing cloud,
   known-τ series, n<dims conditioning).
2. **Per-arm soil isolation + snapshot extraction** — `BELIEF_SOIL_PATH`
   override (default unchanged → hard gate unaffected); embedding-matrix
   extractor with frozen encoder + fingerprint; per-gen snapshot persistence.
3. **FED + STARVED gates, K-matched** — fixed-prompt/fixed-seed LLM-as-judge +
   confidence capture; `self_score` / `external_pass` on `BuildOutcome`; top-K
   batch admission into per-arm soil; admission-event log.
4. **Generation-loop driver + checkpoint probe** — driver on `ab_runner.py`
   (G × tasks × 2 arms, seed-fixed local Ollama); SWE-bench Verified subset
   probe per checkpoint; `belief experiment starved` CLI; run-id guardrail
   (§2.4). Hermetic tests mock builds; live run gets a manual-verification
   checklist.
5. **Reporting + pre-registration enforcement** — PR/Hill-vs-gen chart (both
   arms), τ fit, AR(1) overlay, held-out-success curve, fiction count; a
   `calibrate` step that reads *only* the pilot FED arm to compute the band and
   refuses to look at STARVED shape.

**Immutability:** `belief/benchmark.py` scoring and `belief/hardening.py` are
not modified — tasks are consumed, scoring is not touched.

---

## 6. Run phases

1. **Pilot** — N=10 generations, ~8 tasks, K=4, local Ollama, seed-fixed.
2. **Calibrate** — from the pilot, extract exactly one number: the
   generation-to-generation standard deviation σ of PR **within the FED arm**
   (the arm that should be stable; its wobble is the noise floor). **No looking
   at the STARVED curve's shape to set anything.**
3. **Freeze pre-registration** (§7) — in writing, before the full run.
4. **Full run** — N=25, adjudicated against the frozen band.

---

## 7. Pre-registration block

> Items marked COMMITTED were fixed **cold, before the pilot/model pull** — they
> bind future decisions and must not change. The band σ is the ONLY value filled
> post-pilot (from the FED arm only). Do not edit after the full run begins.

- **Co-headline metric for adjudication:** differential PR (centered-Gram) and
  Hill q=1, **joint-direction** — thesis supported only if both trends move the
  same direction across the FED/STARVED contrast. **COMMITTED (2026-06-02).**
- **Frozen k-means k:** `k = 8`, asserted at compute time and recorded per
  snapshot. **COMMITTED (2026-06-02).**
- **Full-run N:** **25**. **COMMITTED (2026-06-02).**
- **Kill criterion (thesis FAILS):** STARVED stays inside the FED ±2σ band for
  **≥ 0.50 (half) of N=25 generations** *and* held-out success holds.
  **COMMITTED cold (2026-06-02)** — set before any data was seen, by τ-reasoning
  not observed wobble: tolerates ≤~12 gens of decay-onset latency while still
  being a genuine kill (>0.5 would make the test unfalsifiable). Vetoed: any
  fraction > 0.5.
- **Noise band:** FED-arm **±2σ** of per-generation PR, σ taken from the pilot
  FED arm only via `belief experiment starved-calibrate`. `σ = ____` (filled
  post-pilot). Band = `____ ± ____`. **(σ is the only post-pilot value.)**
- **Thesis HOLDS:** STARVED PR decays toward a small fixed point (fit τ as
  generations-to-run-down), held-out build success degrades, AR(1) rises before
  the floor; FED stays inside its band or climbs.

**Pilot-scale cautions (pre-recorded so they can't be argued after the fact):**

- At N=10 / K=4 a decay signal has few generations to express itself. If τ >
  ~6–7 generations the pilot won't show collapse even if it's real. **A flat
  STARVED curve at N=10 is NOT thesis-failure** — the pilot can confirm
  "collapse is fast" but cannot rule out "collapse is slow." Only the full run
  adjudicates.
- ~8 tasks is thin for a stable PR estimate, so the pilot's own σ will be wide
  → the ±2σ band will be conservative (wide). Safe direction: a wide band makes
  collapse *harder* to declare, so STARVED breaking a wide pilot-calibrated band
  is a strong result.

---

## 8. Confounds to nail before running

- **Contamination** — checkpoint probe uses held-out tasks (SWE-bench
  Verified); the self-judge never sees any external test.
- **Determinism** — seed-fix both arms; equalize Ollama options (per the
  existing audit).
- **Volume** — K identical across arms. The whole point; STARVED must not admit
  more.
- **Embedding model frozen** — §2.3, asserted at snapshot time.
- **Self-judge stability** — the STARVED gate is itself a model call; fix its
  prompt and seed so we measure soil decay, not judge drift.
