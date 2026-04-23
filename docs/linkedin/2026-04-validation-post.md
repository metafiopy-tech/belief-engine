# LinkedIn draft — 2026-04-23

**Target length:** 150-250 words.
**Voice:** compressed, understated. No hashtags. No "please like and subscribe". No emojis.

---

Over four overnight runs, a local 14B model (qwen2.5-coder:14b on a MacBook Air M2) paired with my engine's accumulated knowledge solved 20 out of 20 Python challenges. The same model, same hardware, same challenges, without the engine's knowledge base, solved 11 of 20. Fisher's exact test on the paired n=20 gives p < 0.001.

The Belief Engine is an autocatalytic multi-agent build system. Every build deposits patterns, antipatterns, covenants, and skeletons into a ChromaDB soil with FSRS-based decay. The result I cared about was not "beats the frontier on SWE-bench" — it was "does this actually compound". Over 424 builds, the answer appears to be yes: the accumulated priors carry enough signal to let a local 14B match performance on problems the raw model cannot solve alone.

The caveats are real. n=20 is still underpowered for strong claims across the full benchmark set; engine wall clock is 10-15× slower per build; a factorial ablation to attribute the lift to specific subsystems is next.

What's next: n=50 across all 20 challenges, then a soil × covenants × debug × skeleton factorial to see which subsystem is actually carrying the weight.

Repo + raw data: https://github.com/metafiopy-tech/belief-engine

---

**Word count:** ~230 (inside the 150-250 window).
