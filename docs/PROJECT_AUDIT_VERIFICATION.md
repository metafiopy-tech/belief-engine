# Second Audit — Verification of PROJECT_AUDIT.md (2026-06-10)

Independent verification of the 2026-06-09 audit. Method: full repo sweep
(275 source files / 82,269 lines in `belief/`, excluding iCloud dupes), test
inventory + live run of everything runnable in a Linux sandbox, artifact-level
verification of every empirical claim in §0, and a quality/risk scan.

## Bottom line

**Agree with the strategy. One number in the doc is unsourced. The doc misses
one result that strengthens its own argument and a few liabilities that matter
for Track A.** A second full audit is not needed — a short pre-pivot checklist is.

## 1. Claim-by-claim verification of §0

| Claim | Verdict | Evidence |
|---|---|---|
| Validated core thesis (selection > generation) | **Verified** | STARVED/FED contrast + agent_harness_program.md charter; admission/validation layer is where measured value lives |
| Falsifiable pre-registered result held under unfavorable verdict | **Verified** | starved_arm_design.md §7 frozen pre-reg (PR band [-0.169, 1.426], joint PR+Hill headline, N=25); outcome: 3/25 collapse gens vs 13 required, Hill moved the *opposite* direction. Verdict honestly recorded in §9 |
| Reusable infrastructure (arm isolation, soil+FSRS, snapshots, ablation instrument, injectable seams) | **Verified** | `belief/experiments/` (19 files, 5,139 lines); `ablation.py` + `starved_runner.py` have real injectable `build_fn`/`oracle` seams, not stubs; ~260 test fns cover the harness and pass |
| Retrieval firewall "measured at ~0.006" | **SOURCED, reframe** | Traced (2026-06-10) to `belief experiment ablate --id soundness` (3 arms × 8 tasks, qwen2.5-coder:14b, FED full-n25 soil): `no_soil` delta = −0.0064 vs baseline 0.9106, within the ±0.05 noise band. Two caveats for external use: (a) at n=8 inside the noise band this is an *upper bound on soil's contribution* ("soil-coupling ≤ ~0.006, indistinguishable from zero"), not a precise measurement; (b) it measures removal of *good* (FED) soil on ceiling-bound code tasks — the "bad memory doesn't transmit" claim is the separate STARVED result (28 accelerating fictions, capability gap stable ~7.5–8pp). Cite the two findings separately, not as one number |
| SWE-bench at local 14B floored | **Verified** | swebench_smoke logs v1–v6 in results/: no valid patches on any attempt |
| 28 accelerating fictions / pure self-judgment unreliable | **Verified** | STARVED arm data: 0.8→1.12 fictions/gen accelerating; FED stable |

## 2. Where I agree

- **Cut list (§2) is correct.** Soil-coupling deltas: ceiling-bound and firewall-flattened, confirmed. SWE-bench: floored, confirmed. Pure self-judgment: falsified by your own data. Recursive self-improvement as north star, not deliverable: your own oracle-bottleneck finding is the argument.
- **§3 is the correct diagnosis.** Every oracle ever wired is code-shaped (pytest, HumanEval, SWE-bench). Zero economic signal has ever entered the loop. That is the missing experiment, not a research gap.
- **§4 is honest about what's built.** The seams are real. The genuinely new code for Track A is small: a money-oracle adapter behind the existing `oracle` seam, plus ops glue. Weeks is plausible.
- **A funds B; B never gates A.** No technical finding contradicts this; the burn data (e.g. $14/10min when sessions one-shot) supports strict sequencing.

## 3. Where I amend

**3.1 The doc omits substrate-transfer — its own best evidence and its sharpest
constraint.** The 140-build run (substrate_transfer_findings.md, uncommitted):
14/20 with harness vs 3/20 raw on Python-adjacent tasks (~4x lift), **0/20 on
non-Python substrates**. Two implications the doc should absorb:
(a) the harness demonstrably adds value — supports Track A; (b) transfer is
paradigm-internal, not paradigm-general. Track A's niche must be Python-shaped
or have an oracle as crisp as pytest. "Swimsuits in summer" should be read with
that bound: pick the boring thing whose success signal is unambiguous *and*
machine-checkable, because that's where the harness is proven.

**3.2 "Reusable infrastructure" has a documentation liability.** CLAUDE.md is
~3 months stale: claims 233 files/69.6K lines (actual 275/82.3K), omits four
packages (`routing/`, `signal/`, `lifecycle/`, `protocol/`), and documents ~18
of 36 CLI commands. If Track B wraps off-the-shelf agents, they will read
CLAUDE.md — stale docs directly degrade your own agents. Truth it up before B.

**3.3 The engine violates its own covenant #2 at scale.** 137/275 files exceed
the 200-line covenant it enforces on generated code (executor.py: 1,564 lines).
Zero bare-excepts, zero silent swallowing — error discipline is good. Not
urgent, but if the engine ever runs on itself, covenant #2 fires 137 times.

## 4. Findings the doc couldn't know (current repo state)

- **Test suite: healthy.** 126 files, ~2,268 test fns. Ran in sandbox (Python
  3.10, unsupported; deps partial): **~2,164 passed / 1 failed / 26 skipped**.
  The single failure is a Python-version artifact (`compile()` raises
  ValueError for null bytes on ≤3.11, SyntaxError on ≥3.12; enforce_all catches
  only SyntaxError) — passes on 3.14. Optional hardening: catch ValueError too.
- **Skip-count drift incoming:** 6 z3-gated tests added 2026-05-27; z3 not yet
  installed on the Mac → next hard gate reports ~13 skips, not 7. Expected, not
  a regression.
- **Uncommitted work at risk:** PROJECT_AUDIT.md, substrate_transfer_findings.md,
  4 scripts (run_full_substrate_transfer + chart/brief/card generators),
  README + starved_arm_design §9 edits, deletion of the falsified
  substrate_transfer_challenges.md. One iCloud hiccup loses the substrate
  findings. Commit first.
- **Daemon stubs unchanged:** 5 photosynthesis jobs still visible-but-disabled
  (correctly instrumented). Minor: conftest docstring claims
  `filterwarnings=["error"]` which pyproject doesn't set.
- **hardening.py / benchmark.py:** untouched, immutability intact.

## 5. Pre-pivot checklist (instead of a third audit)

1. Commit the uncommitted research artifacts (one commit, today).
2. ~~Source or strike the 0.006 firewall number~~ — DONE 2026-06-10: it's the
   `no_soil` delta (−0.0064, within ±0.05 noise, n=8) from the ablation
   soundness run on FED soil. Persist that run's output to
   `docs/experiments/` or `results/` so the provenance survives; reframe in
   PROJECT_AUDIT.md as "soil-coupling upper bound" distinct from the STARVED
   bad-soil-doesn't-transmit result.
3. Fold substrate-transfer findings into PROJECT_AUDIT.md §1/§5 (it bounds
   Track A's niche selection).
4. Truth-up CLAUDE.md (counts, four packages, CLI table); tag the repo state.
5. Optional: ValueError catch in enforce_all; install z3 on the Mac.

Then ship Track A. Nothing found in this verification argues for more research
before the first money oracle.
