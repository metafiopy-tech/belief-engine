# Belief Engine — Full Project Audit & Path Forward (2026-06-09)

## 0. Where you actually are (vs. where the anxiety says you are)

Anxiety scores this project on one axis — "did my external life change yet" — and
calls everything else zero. By that measure you feel stuck. By every other
measure you are not in the same spot:

- You have a **validated core thesis**: in a self-improving system, generation is
  cheap and *selection / correspondence* is the scarce, load-bearing input.
- You have a **falsifiable, pre-registered result** (STARVED full-n25) that held
  under a verdict that didn't go your way — that's real science, rare even among
  funded labs.
- You have **reusable infrastructure**: arm isolation, soil + FSRS, snapshots, the
  differential metrics, a substrate-agnostic ablation instrument, injectable seams.
- You have a **quantified mechanism finding**: the retrieval firewall (bad memory
  doesn't transmit) measured at ~0.006.

What has NOT moved is **revenue and shipping**. That's the real gap — and it's
fixable in weeks, not years. It is not "wasted time"; it's an unbalanced
portfolio: all rigor, no output.

**One-line verdict:** you built a research lab; your stated goal needs a business.
The research has paid its main dividend. The next phase is shipping a selection
harness pointed at a *real-world (money) oracle* — the one oracle you've never used.

---

## 1. What works — KEEP (load-bearing)

- **Selection > generation.** The whole engine's value is in admission/validation,
  not in out-generating a frontier model. Stop trying to out-generate; own selection.
- **Correspondence-gated admission.** FED (external grader) keeps the system honest;
  self-judgment alone rots. This is the spine.
- **The retrieval firewall.** Dilute, similarity-gated, capped memory injection makes
  the engine robust to bad soil — measured, not assumed.
- **Differential + pre-registration methodology.** The discipline that stops you
  fooling yourself. This transfers to *any* future experiment.
- **Harness scaffolding.** Arm isolation, soil + FSRS decay, per-gen snapshots, the
  ablation instrument, the injectable `build_fn`/`oracle` seams (substrate-agnostic).
- **The two-tier oracle concept.** Cheap self-signal as pre-filter + external
  validator as backstop — validated as *necessary*, mirrors biological valence.

## 2. What's fluff or premature — CUT or PARK

- **Chasing soil-coupling deltas on code.** Ceiling-bound (baseline ~0.91), firewall
  makes it ~0; 5-hour runs for nitpicky deltas. Done giving returns.
- **SWE-bench at a local 14B.** Floored (no valid patches). Park until a stronger
  model or different niche.
- **Pure self-judgment as an oracle.** Proven unreliable (28 accelerating fictions).
  Only usable two-tier, with an external backstop.
- **Recursive self-improving meta-agent AS A NEAR-TERM BUILD.** This is the field's
  hardest unsolved problem, and your *own data* shows it's gated on the exact oracle
  bottleneck you found (self-judgment without correspondence decays). Keep it as the
  north star; do NOT let survival depend on it.
- **Perfecting one agent before shipping a system.** This is the trap you're feeling.

## 3. The real gap

**No economic oracle. Nothing ships. Nothing earns.** Every test so far was
conditioned on code output with no money/usage signal. The loop never touched the
one form of reality that matters for "sustain my life": *did someone pay for or use
this.* That's not a research gap — it's the missing experiment.

## 4. The "perfect harness," specified

You already have the hard part. The universal harness is three pieces, two of them
built:

- **Core (built, domain-agnostic):** the selection loop — propose → `oracle(candidate)`
  → admit/reflex → adapt (soil/FSRS). Substrate-agnostic via injectable seams.
- **Plug-in A — the oracle (per domain).** Code = tests. Math = a checker. **Business
  = sales / usage / did-it-make-money.** Same loop, swap the validator. *This is the
  universal-adaptation mechanism.*
- **Plug-in B — the action space + executor.** Slot an off-the-shelf agent (Hermes /
  OpenHands / Claude) as the *hands*. Your contribution is the adaptation/selection
  layer, NOT the generation. Don't rebuild the agent — wrap one.

This is the proven RL-environment/Gym interface + FunSearch's cheap-verifier insight
+ your selection layer on top. Reuse the scaffolding; innovate only where you have an
edge (selection, firewall, two-tier oracle).

## 5. The path (sequenced — and the order is the whole point)

**Track A — SHIP THE BORING THING (now, weeks).** One narrow operation with an
unambiguous money oracle. The "swimsuits in summer" instinct is correct: pick ONE
clear-demand, low-ambiguity product/service. Use off-the-shelf agents + your harness
to run the ops (sourcing, listings, pricing, customer messages). Purpose, in order:
1. **Runway** — revenue kills the existential clock that is *causing the anxiety*.
2. **Your first real oracle** — sales is the correspondence signal the research never had.
3. **Proof of the harness on a real niche**, which is the actual thesis.

**Track B — THE SELF-IMPROVING HARNESS (funded BY A, months).** Build the meta-loop
on the proven core, pointed at cheap-oracle domains first, scaled only after A pays.
**A funds B. B is never allowed to gate A.** Invert this and you spin for another
three months with no runway.

## 6. The honest risk check (constructive, not discouraging)

The "agents that research and upgrade themselves and ship me a new harness every
month" is recursive self-improvement — unsolved everywhere, and gated on the oracle
problem you already found. It is a legitimate, exciting north star. It is **not** a
near-term deliverable and must not be the thing your survival rests on. Build it with
Track A's money, on Track A's schedule.

## 7. The anti-spinning rule

You spin when you research without shipping, because research has no end condition.
The cure is a **deadline + a money oracle**: one shippable thing, one real customer,
before any further experiments. Ship first. Let reality grade it. Then iterate — with
correspondence, exactly like the engine that works.
