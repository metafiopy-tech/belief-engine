# Agent-Harness Program — the Minimal Nervous System

**Status:** opening. Follows the STARVED/FED arc (see `starved_arm_design.md`).
**Premise from what we proved:** the engine's value is in *selection*, not
generation — admission gating, the retrieval firewall, the validators. A capable
cheap agent is therefore a *good selector* (cheap reflexes + a trustworthy
oracle), not a big reasoner. This program builds that agent by **subtraction** —
isolate each candidate mechanism, prove it load-bearing, compose only the
survivors — then orchestrates a swarm over shared soil.

## What the STARVED arc established (the inputs)

- **Self-judgment alone is an unreliable oracle** — STARVED admitted accelerating
  fictions (0.8→1.12/gen). A pure self-signal feed/starve loop catches the wrong
  prey efficiently.
- **The retrieval firewall is real** — dilute, similarity-gated, capped soil→build
  injection is *why* bad soil didn't collapse capability (success gap flat ~7.5pp
  while fictions accelerated). Robustness to bad memory is an architectural
  property, not luck.
- **Differential + pre-registration is the method that adjudicates** — band on the
  STARVED−FED differential, frozen kill criterion, unspinnable verdict.

## Definition — "load-bearing"

A mechanism is load-bearing if toggling it OFF (one variable, all else fixed)
moves the outcome metric beyond the baseline arm's noise band, in a
pre-registered direction. Anything that doesn't is fluff and is dropped from the
minimal harness. Same differential discipline as STARVED.

## Substrate principle

The harness is **substrate-agnostic**: every external action is an injectable
seam (`build_fn`, `metric_fn`), so the compute substrate — cheap proxy task,
tiny real builds, or a smaller model for iteration — is chosen *per run*, not
baked in. "Whatever gets us what we need." Survivors get re-confirmed on real
14B builds before they count.

## Build order (one stage at a time; hard gate + commit between; differential adjudication each)

1. **#3 — self-ablation instrument (THIS first).** Generalize the STARVED arm /
   differential / pre-registration machinery into a reusable "toggle any
   mechanism, auto-apply the differential, attribute the effect" harness. It is
   the ruler; build it before measuring. Falsifiable utility: it must reproduce
   the STARVED result as one ablation (soil-on vs soil-off, decompose-on/off) to
   prove the instrument is sound.
2. **#1 — calibrated oracle arm.** Self-judge as a *cheap pre-filter* whose
   verdicts are continuously scored against the external validators (the engine
   already owns these: covenant AST, mutation, property tests, coverage/structure
   gates); track reliability per domain; let the crystallizer learn which signals
   to trust. **Falsifiable claim:** once calibrated, external-validator call
   frequency *falls* while quality holds. (This is the Spider's gating + step-7
   question, answered with attribution.)
3. **#2 — adaptive coupling arm.** Retrieval coupling as a live controller driven
   by soil-health (fiction rate, silhouette, differential trend). **Falsifiable
   claim:** a coupling schedule exists where clean soil accelerates builds without
   bad soil poisoning them — a sweet spot the static firewall leaves on the table.
4. **Compose survivors → minimal harness.** Only the mechanisms that passed (2),
   (3) go in. Single-agent capability test: "what can this do that a vanilla
   agent can't," with attribution from the prior stages. The minimal nervous
   system is whatever survived, nothing more.
5. **Swarm — shared soil / stigmergy.** Drones coordinate through traces in shared
   soil (the existing soil + reciprocity/niche ledgers + routing hubs), not direct
   messaging. **New requirement the swarm imposes:** a shared soil makes one
   drone's fictions every drone's input — so the calibrated oracle (#1) and the
   coupling controller (#2) become the swarm's **immune system**. Bad oracle +
   shared memory = contagion. Single-agent harness with attributed, validated
   mechanisms is a *prerequisite*, not a nicety.

## Honest boundary

Everything here wins **only in niches with a trustworthy oracle** (the FED/
verifiable domains). The swarm inherits the correspondence problem; it does not
escape it. The oracle remains the scarce asset — now shared.
