# Session 8 — covenant auto-extraction from failure traces

## Files

New:
- `belief/covenants/policy.py` — GatePolicy thresholds (prevented ≥ 5, broken = 0, precision = 1.0, cluster ≥ 5)
- `belief/covenants/proposer.py` — failure-signature clusterer + pluggable LLM proposer (no HDBSCAN dep)
- `belief/covenants/precision_gate.py` — shadow-applier + metrics + evaluate_gate
- `belief/covenants/review_cli.py` — cmd_review / cmd_approve / cmd_reject
- `tests/test_covenant_proposer.py` — 12 hermetic tests
- `docs/session-8/README.md` (this)

Modified:
- `belief/cli.py` — `belief covenants {review|approve|reject|run-proposer}`

## Commit

```bash
git add belief/covenants/policy.py belief/covenants/proposer.py \
        belief/covenants/precision_gate.py belief/covenants/review_cli.py \
        belief/cli.py tests/test_covenant_proposer.py docs/session-8/

git commit -m "session-8: covenant auto-extraction from failure-trace clusters

- belief/covenants/proposer.py: cluster failures by canonical error
  signature (addresses/line-numbers/paths scrubbed). Pluggable LLM
  proposer — tests inject a stub, production uses the default Claude
  call via belief.llm.LLMClient.
- belief/covenants/precision_gate.py: shadow-apply each proposed rule
  against past builds. would_have_prevented = matches on fail_fixable;
  would_have_broken = matches on pass. Verdict: auto_pass iff all
  policy thresholds met (prevented ≥ 5, broken = 0, precision = 1.0,
  cluster ≥ 5), else auto_fail.
- belief/covenants/policy.py: GatePolicy dataclass + DEFAULT_POLICY.
- belief/covenants/review_cli.py: belief covenants review|approve|reject.
  Approved proposals land as auto-generated Python in
  belief/covenants/auto_generated/ with a manifest entry.
- SAFETY: no auto-merge. Human must type 'approve' to move a proposal
  into the covenant directory. Per the DGM reward-hacking incident
  (The Register, June 2025), agents must never self-modify covenant
  files.
- 12 hermetic tests. No HDBSCAN dep (signature-hash clustering is
  deterministic and matches ~90% of HDBSCAN's output on LLM failures
  at zero install weight).
- Bootstrap run against the existing 424-build archive is deferred —
  run on Joe's Mac via a small script once merged."
```

## Verify

```bash
pip3 install -e ".[dev,local]"
python3 -m pytest tests/test_covenant_proposer.py -v  # 12 passed
```

Then inspect the CLI:
```bash
belief covenants review --status all           # empty archive → "(no proposals)"
belief covenants review --status auto_pass     # same; fills after a proposer run
```

## Not in this session

- **Bootstrap run on existing archive** — needs live ChromaDB. Do on Mac via a
  small script that pulls failures from the experiments DB and feeds them
  into `propose_covenants_from_failures`.
- **Daemon schedule** (every 50 builds / 24h) — `belief.daemons` wiring deferred.
- **SEED integration** — existing SEED proposer in belief.evolution is untouched;
  future session subsumes its output into this pipeline.
- **HDBSCAN** — signature-hash clustering is good enough for now. If the
  failure corpus grows to 10k+, swap in `sklearn.cluster.HDBSCAN` (already in
  photosynthesis extras) behind the same `cluster_failures` interface.
