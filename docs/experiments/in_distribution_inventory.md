# In-Distribution Challenge Inventory

**Status:** Drafted 2026-05-27 as task #2 of the substrate-transfer experiment.
**Purpose:** Identify the 15 in-distribution slots from the existing 43 challenges in `belief/benchmark.py`. Surface gaps where the existing pool can't cleanly fill a bucket.

## Validator situation (important)

In-distribution challenges already have validators **inherent to the engine pipeline**: the `validator` node runs `pytest` on the engine's own generated tests, and `verify_commands` on each Challenge provide a smoke check. There's **no per-challenge bespoke validator code to write** for these 15 — they ride on the existing infrastructure. This is the key difference from the novel-artifact bucket, which needs custom validators (task #8).

What this means: the experiment runner already knows how to score in-distribution challenges. The only work for these 15 is selecting which ones to include.

## Bucket 1 — Python Microservices (5 slots)

All FastAPI, all SQLite, all single-service. Tier 2-3 to keep build times manageable inside the 10 min/build budget. All exist with `verify_commands` already.

| Slot | Challenge ID | Tier | Why |
|------|-------------|------|-----|
| 1 | `t2-health-api` | 2 | Simplest FastAPI; floor-finder for raw_local |
| 2 | `t3-url-shortener` | 3 | Canonical CRUD + redirect; classic soil-territory |
| 3 | `t3-bookmark-api` | 3 | CRUD + tags; tests JSON nested-object handling |
| 4 | `t3-notes-api` | 3 | CRUD + search + markdown; multi-feature single-service |
| 5 | `t3-contact-api` | 3 | CRUD + CSV import/export; tests file-handling code paths |

**Coverage assessment:** Clean. Five canonical FastAPI CRUD challenges with varying secondary features. This is the engine's strongest territory — soil should help most here, which is what we expect to see in Chart 3.

## Bucket 2 — CLI / Scripts (5 slots)

Mix of Click-based CLIs and standalone scripts. Tier 1-3.

| Slot | Challenge ID | Tier | Why |
|------|-------------|------|-----|
| 1 | `t1-wordcount` | 1 | Simplest script; baseline-finder |
| 2 | `t2-todo-cli` | 2 | Canonical Click CLI with JSON persistence |
| 3 | `t2-calculator-cli` | 2 | Click CLI, tests error-handling (div/zero) |
| 4 | `t2-csv-stats` | 2 | CLI + Rich formatting; tests stdin/file handling |
| 5 | `t3-expense-tracker` | 3 | Click + SQLite + reporting; more complex |

**Coverage assessment:** Clean. Mix of tier-1/2/3 lets us see whether soil helps more at low complexity (where there's room to grow) or high (where the substrate's iteration loop has more material to work with). I dropped `t1-fizzbuzz` and `t1-fibonacci` because they're too simple — full and raw both score ~1.0 on them, no signal.

## Bucket 3 — Data Pipelines (5 slots) — partial fit

This is the weakest bucket. The existing benchmark is API-heavy; "data pipeline" isn't a category it was designed around. Three challenges are cleanly pipeline-shaped, two are borderline.

| Slot | Challenge ID | Tier | Fit | Why |
|------|-------------|------|-----|-----|
| 1 | `t3-data-pipeline` | 3 | clean | Named for the bucket; pydantic + CSV processing |
| 2 | `t6-data-pipeline` | 6 | clean | Two-service CSV ingestion + analytics — pipeline-shaped; **but** tier 6 means longer build times (1500s timeout) |
| 3 | `t5-workflow-engine` | 5 | clean | DAG of steps with retry — canonical pipeline pattern |
| 4 | `t3-task-queue` | 3 | borderline | Async work-queue concept is pipeline-adjacent; might be too API-shaped |
| 5 | `t3-schema-validator` | 3 | borderline | Validation pipeline; weaker fit but the closest remaining candidate |

**Coverage assessment:** Weaker. Honest options going forward:

(a) **Accept the weaker bucket** as-is. The experiment still works; Chart 3's data-pipeline bar will have somewhat heterogeneous content. This is OK if the bar comes out positive across the bucket; less OK if results are bimodal within the bucket.

(b) **Draft 2 new clean data-pipeline challenges.** ETL (CSV→transform→JSON) and log-aggregator are both small enough to write in an hour. This adds a small task before the experiment can run but produces a tighter bucket.

(c) **Drop the bucket to 4 challenges,** rebalancing to e.g. 6 microservices + 5 CLI + 4 pipelines + 5 novel. Loses statistical power on the data-pipeline bar but uses only clean fits.

**Recommendation: (a) for now.** The weak bucket is a real concession, but the experiment is exploratory. If the data-pipeline bar turns out interesting (positive lift, large effect), tighten it later with bespoke challenges. If the bar is flat or noisy, the bucket weakness was correctly diagnosed and you can write that up honestly. Option (b) is the kind of work that bloats scope without strengthening the headline.

## Build-time budget check

10 min/build is the planning assumption. Reality from existing challenge timeouts:

- Tier 1-2: 600s timeout (10 min) — fits
- Tier 3: 600-900s — borderline
- Tier 4-5: 900-1200s — exceeds budget
- Tier 6: 1500s — significantly exceeds
- Tier 7-8: 900-2400s — significantly exceeds

**Concern:** `t6-data-pipeline` (tier 6, 1500s timeout) is in the data-pipeline bucket. At 25 min/build worst-case, this challenge alone could blow the 25-28 hour total experiment estimate by ~30%.

**Mitigation options:**
- Tighten the timeout on `t6-data-pipeline` to 900s and accept higher failure rates on this challenge specifically (which is itself informative)
- Replace `t6-data-pipeline` with a tier-3 challenge — but the alternatives are borderline-fit
- Accept the longer wall-clock estimate

**Recommendation:** keep `t6-data-pipeline` with its native 1500s timeout. Update the experiment wall-clock estimate from "25-28 hours" to "28-34 hours" to be honest about the variance.

## Final selected 15

```python
IN_DISTRIBUTION_CHALLENGES = [
    # Microservices
    "t2-health-api",
    "t3-url-shortener",
    "t3-bookmark-api",
    "t3-notes-api",
    "t3-contact-api",
    # CLI / Scripts
    "t1-wordcount",
    "t2-todo-cli",
    "t2-calculator-cli",
    "t2-csv-stats",
    "t3-expense-tracker",
    # Data Pipelines
    "t3-data-pipeline",
    "t6-data-pipeline",
    "t5-workflow-engine",
    "t3-task-queue",
    "t3-schema-validator",
]
```

## Gaps surfaced

1. **Data-pipeline bucket is borderline.** Two of the five slots are weaker fits. See above for recommendation.
2. **No automated way to mark which 5 are in which sub-bucket.** The Challenge dataclass has `tags` but the tags don't cleanly partition into microservices/CLI/pipeline. The experiment runner will need a hardcoded mapping or a new `domain` field on Challenge. Lightweight fix; folds into task #4 (runner wiring).
3. **Tier-6 challenge in the bucket may blow wall-clock.** Real but manageable; surfaced above.

## Next-task linkage

This inventory feeds directly into task #4 (runner wiring), which needs to know:
- Which 15 challenges to load from `benchmark.CHALLENGES`
- Which domain bucket each one belongs to (for Chart 3 aggregation)
- That the engine's built-in `pytest`-based validator handles all 15 — only novel-artifact challenges need the validators from task #8.
