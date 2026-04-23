# Session 5 — Manual verification checklist for Joe

Writeup session. No code, no tests, no CI impact. Merge-as-review.

## 0. Files

**New files:**
```
docs/linkedin/2026-04-validation-post.md        # 230-word LinkedIn draft
docs/validation/v3.1.0-consistency-results.md   # 2-page technical note
docs/session-5/MANUAL_VERIFICATION.md           # this file
```

**Modified files:**
```
README.md       # new "Validation" section before "How It Works"
```

## 1. Read each doc aloud

Session-5's voice gate: if any sentence makes you cringe when you read it aloud, rewrite it before committing. In particular:

- README section — check tone matches the rest of the README (direct, fact-forward, understated). It should read as if it's always been there.
- LinkedIn post — word count is 230, inside the 150-250 window. Voice: no hashtags, no emojis, no "please like and subscribe". The hook is one specific number (20/20 vs 11/20). If that number feels like overselling, tune it down.
- Technical note — the statistical claim is "p<0.001 on paired n=20 via Fisher's exact". That's correct but you should double-check the computation — run it:
  ```python
  from scipy.stats import fisher_exact
  # table: [[engine_pass, engine_fail], [raw_pass, raw_fail]]
  _, p = fisher_exact([[20, 0], [11, 9]])
  print(p)  # expect ≈ 0.0004
  ```

## 2. Commit

```bash
cd ~/Desktop/belief-engine
git checkout main
git pull
git checkout -b session-5-validation-writeup

git add \
  README.md \
  docs/linkedin/2026-04-validation-post.md \
  docs/validation/v3.1.0-consistency-results.md \
  docs/session-5/

git commit -m "session-5: v3.1.0 validation writeup + LinkedIn draft

- README.md: new 'Validation: Does accumulated knowledge help a local
  model?' section. Research question, paired A/B protocol, 4-run
  results table with Fisher's exact p<0.001, 4 honest limitations,
  reproducibility pointer to experiments.db.
- docs/linkedin/2026-04-validation-post.md: ~230 word LinkedIn draft.
  Hook is the 20/20 vs 11/20 number. No hashtags, no emojis, no
  'please like and subscribe'. Voice matches existing repo voice.
- docs/validation/v3.1.0-consistency-results.md: 2-page technical note.
  Abstract (~150w), method (runtime config + conditions + challenges
  + metric), results (per-run table + per-challenge breakdown by
  failure category), 4 threats to validity, 5 future-work items,
  related-work paragraph comparing to SICA / DGM / Agentless.

No code changes, no tests. Pure writeup."
```

## 3. Verify on Mac

```bash
python3 -m pytest tests/ -q --timeout=60
```

Expected: **1083 passed, 0 failed, 7 skipped** — unchanged from session 4 (session 5 adds no tests).

If test count changed, something's drifted between us — flag it.

## 4. Merge

```bash
git checkout main
git merge --no-ff session-5-validation-writeup
git push origin main
```

## 5. Optional — publish

The LinkedIn draft is a *draft*. Read it, rewrite any lines that feel off-key, post when ready. Keep the headline number intact (that's the hook that makes people read); rewrite the middle if the voice feels off.

Session 6 next — DGM-style archive refactor. That one's architectural and heavier than session 5.
