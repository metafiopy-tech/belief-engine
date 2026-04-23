# Synthesizer: keep, route, or delete?

**Status: UNDECIDED — awaiting n=3 ablation run.**

Session 4 (v3.2) shipped the router, the 1.5B polish fallback, and
the A/B harness. The keep / route / delete decision is deferred until
the harness has produced evidence. This document is the hand-over
template — Joe fills in the data and chooses.

## The question

Does the synthesizer agent earn its 180–300s wall-clock cost on
already-passing builds? Per the research report, no mainstream
agentic coder in 2025-2026 (Aider, Claude Code, Cursor, OpenHands,
SWE-agent) has a separate polish pass. So the prior is weak: absent
evidence of a quality lift, the synthesizer should be deleted.

## Experimental protocol

- **N challenges:** 10 (tier-1 and tier-2 from the benchmark set).
- **Conditions:**
  1. `builder_only` — synthesizer skipped entirely (`BELIEF_ABLATION_SKIP_SYNTHESIZER=1`).
  2. `builder_plus_synth` — pre-session-4 behaviour (`SYNTHESIZER_ROUTE_ENABLED=0`).
  3. `router` — session-4 default; router decides per-build.
- **Runs per cell:** 3 (gives 90 builds total; ~10 minutes each on
  local → ~15 hours overnight).
- **Metrics captured** per run:
  - `tests_passed`, `tests_total`, `weighted_score`
  - `ruff_errors` (count of E,F,B,UP findings)
  - `radon_mi` (maintainability index)
  - `wallclock_s`, `cost_usd`
- **Storage:** SQLite at `~/.belief-engine/ablations.db`.
- **Statistic:** mean + stddev per cell. Paired t-test
  (builder_plus_synth vs builder_only) is printed when you have
  SciPy installed; otherwise eyeball the effect size against the
  5% threshold.

## Decision rule

Apply in order:

1. **If `builder_plus_synth` shows < 5% quality lift over `builder_only`
   on every metric** → **DELETE** the synthesizer.
   Deployment artifacts (Dockerfile, docker-compose.yml, run.sh) move
   to a deterministic templater at `belief/agents/deploy_artifacts.py`
   (not yet implemented; build during the follow-up session).
2. **Else if `router` matches `builder_plus_synth` quality at lower
   wall clock** → **ROUTE** (what session 4 already ships). Keep the
   synthesizer, keep the 1.5B polish fallback. No code changes from
   the merged session 4.
3. **Else** → **KEEP** the synthesizer unconditionally. Roll back the
   router. (Unlikely based on the research prior; included for
   completeness.)

## Running the ablation

```bash
# Full run (overnight):
python3 scripts/synthesizer_ablation.py --n 3
# or via the CLI:
belief experiment ablation-synth --n 3
```

Resumable — if interrupted, rerun; already-completed
(challenge, condition, run_n) tuples are skipped.

To see the current state without running more:
```bash
belief experiment ablation-synth --report
```

## Results (fill in after run)

### Summary table

| Condition            | weighted_score | ruff_errors | radon_mi | wallclock_s | cost_usd | n |
|----------------------|----------------|-------------|----------|-------------|----------|---|
| builder_only         |                |             |          |             |          |   |
| builder_plus_synth   |                |             |          |             |          |   |
| router               |                |             |          |             |          |   |

### Lift analysis

- `builder_plus_synth` vs `builder_only`:
  - weighted_score lift: ___%
  - ruff_errors delta: ___
  - radon_mi delta: ___
  - wallclock overhead: +___s
- `router` vs `builder_only`:
  - weighted_score match: within ±___%
  - wallclock overhead: +___s
  - polish-fire rate: ___% of builds

### Decision

(Circle one and write a paragraph explaining.)

- [ ] DELETE — evidence and reasoning:
- [ ] ROUTE  — evidence and reasoning:
- [ ] KEEP   — evidence and reasoning:

### Follow-ups after the decision

- If DELETE: implement `belief/agents/deploy_artifacts.py` (deterministic
  Dockerfile / docker-compose / run.sh templates).
- If ROUTE (shipped default): tune thresholds from the ablation data.
  For example, if most polish-fires were driven by ruff > 3 but none
  of those polished files improved any metric, raise the threshold
  to 10.
- If KEEP: redo session 4 as a session-4-revert commit; start session 5
  with the current architecture intact.
