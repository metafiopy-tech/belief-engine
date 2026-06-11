# Ablation soundness run — 2026-06-10

Provenance for the "soil-coupling ~0.006" figure cited in PROJECT_AUDIT.

```
belief experiment ablate --id soundness --tasks 8 --seed-soil ~/.belief-engine/starved/full-n25/FED/soil

Ablation soundness: 3 arms x 8 tasks
Model: qwen2.5-coder:14b  |  run dir: ~/.belief-engine/ablation/soundness

Ablation — soundness  (baseline: baseline)
============================================================
  baseline                 metric=0.9106 (n=8)
  no_soil                  metric=0.9042 (n=8)  delta=-0.0064 [within noise]
  no_decompose             metric=0.9513 (n=8)  delta=+0.0407 [within noise]

  noise band: +/-0.0500 (mechanism load-bearing if |delta| exceeds)
```

## Interpretation

- `no_soil` delta −0.0064, within ±0.05 noise at n=8: an **upper bound on
  soil's contribution** to ceiling-bound code tasks — indistinguishable from
  zero. This is removal of *good* (FED) soil; the "bad memory doesn't
  transmit" finding is the separate STARVED result (28 accelerating fictions,
  capability gap stable ~7.5–8pp). Cite the two separately.
- `no_decompose` delta +0.0407: within noise but 6x the soil delta and
  positive — removing the decomposer *improved* the metric. Hint, not
  finding (n=8). Candidate cheap follow-up: single-arm rerun at n=25.
