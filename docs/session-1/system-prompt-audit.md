# Session 1 — System-prompt byte-stability audit

**Date:** 2026-04-22
**Purpose:** Verify that `num_keep=512` in `belief/llm.py` will actually produce the ~18× TTFT speedup it promises. That speedup only materializes if the **first 512 tokens** of each call's system prompt are byte-identical across calls to the same model.

## Method

1. `grep` `belief/` for `datetime.now`, `time.time`, `uuid.`, `os.getpid`, and f-string interpolation into any variable named `system*` or `_SYSTEM*`.
2. Read each agent's prompt-assembly path in `belief/agents/*.py`.
3. Trace where `LLMClient.generate_text` / `generate_structured` receive `system=…` — confirm the string is a constant (or at least has a stable prefix).

## Findings

### A. No datetimes, UUIDs, or PIDs in agent system prompts

Every `datetime.now`, `time.time`, and `uuid.` hit is in telemetry, memory-subsystem records, or photosynthesis captures — none reach an agent's `system` argument.

### B. All agent system prompts start from module-level constants

| Agent | Constant | Injected dynamically? |
|---|---|---|
| intake | `INTAKE_SYSTEM` | No |
| planner | `PLANNER_SYSTEM` | No |
| architect | `ARCHITECT_SKELETON_SYSTEM` | No |
| builder | `BUILDER_SYSTEM` (with per-file `_builder_system` override) | **Appended-only** (see C) |
| tester | `TESTER_SYSTEM` | No |
| debugger | `DEBUGGER_SYSTEM` | No |
| gap_analyst | `GAP_ANALYST_SYSTEM` | No |
| synthesizer | `SYNTHESIZER_SYSTEM` | No |
| validator | — (deterministic, no LLM) | N/A |
| latios | `LATIOS_SYSTEM` | No |
| executor | small constant | No |

### C. Append-only dynamic content — safe for prefix cache

Two places add content to the system message. Both **append** (never prepend), so the leading bytes stay stable and `num_keep=512` still captures the prefix cache hit.

1. **`belief/agents/builder.py:325`**
   ```python
   system = f"{system}\n\nLANGUAGE: {lang.value.upper()}\n{lang_additions}"
   ```
   Per-build language is stable (one language per build). The insertion comes AFTER the full `BUILDER_SYSTEM` constant, so the first ~512 tokens of `BUILDER_SYSTEM` itself are cache-stable.

2. **`belief/llm.py::LLMClient.generate_structured`**
   ```python
   augmented_system = (
       f"{system}\n\n"
       f"Respond ONLY with a valid JSON object matching this schema. ..."
       f"Schema:\n{schema_json}"
   )
   ```
   Schema-per-role is stable (the planner always emits the same schema). The schema JSON is appended after the full `{system}` prefix.

### D. `TOKEN_EFFICIENCY_BLOCK` is defined but unused

`belief/agents/base.py:31` defines a token-efficiency block constant. A grep confirms it is never actually injected into any prompt. No action needed, but flagging in case a future session wants to wire it in — **if wired in, it must come AFTER the per-role system constant**, not before.

### E. Protocol skeleton injection in builder (TypeScript builds)

`belief/agents/builder.py:330+` injects protocol skeletons (x402, mcp, a2a, erc8004) into the system for TypeScript builds. This is also appended. Within one build's TS files it will be stable; across builds of different protocols it will differ. That's fine — the cross-build cache miss is expected, the intra-build hit is what we care about.

## Conclusion

**Prefix caching is safe to enable.** `num_keep=512` pins the first 512 tokens of the KV cache across requests, and no code path mutates the leading bytes of a per-role system prompt between calls.

No code changes required for this audit.

## Recommended follow-up (not required for Session 1 merge)

- Measure token length of each `_SYSTEM` constant. If any is below 512 tokens, `num_keep=512` is wastefully large for that agent; drop to roughly the 90th-percentile system-prompt length per role.
- Add a `belief prompts audit` CLI command that prints per-role system-prompt lengths + byte-stability hashes, so regressions on this property are caught automatically in CI.
