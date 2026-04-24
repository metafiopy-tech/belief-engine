# Gate 5 — Contamination Findings

## Method
Probed Qwen 2.5 Coder 14B (via Ollama, temperature=0, seed=42) with 5
HumanEval-style function signatures + docstrings. Recorded completions.

## Result: HEAVY HumanEval contamination

All 5 probes produced canonical HumanEval reference solutions or near-verbatim
variants. Variable names differ slightly but algorithms are identical.

## Implication for benchmarks

- HumanEval vanilla: NOT USABLE as main result. Raw Qwen already knows the
  answers. Engine lift will be dominated by noise.
- HumanEval+ (EvalPlus): Partially usable. Same problems but more
  rigorous test cases. Some mitigation but residual contamination likely.
- MBPP: Likely similar contamination pattern. Use MBPP+ instead.
- SWE-bench: Use SWE-bench Verified (the 500-issue human-curated subset
  from 2024). Or filter to issues with created_at > Feb 2024 (post-Qwen-cutoff).
  Vanilla SWE-bench overlaps Qwen's training data.

## Benchmark plan implied by these findings

Headline benchmark: **SWE-bench Verified** (or post-cutoff filtered)
Secondary: **MBPP+** via BigCode harness, 5 seeds, pass@1 and pass@10
Sanity check only: HumanEval vanilla, reported with "known contaminated" caveat

## Raw probe data
See humaneval_probes.json for exact completions.
