# Belief Engine — Mycorrhizal Protocol v1

**Version:** `1.0` (see `belief/protocol/__init__.py::PROTOCOL_VERSION`)
**Status:** Stable. Backward-compatible changes bump the minor; breaking
changes bump the major after a deprecation window.

This document is the canonical specification of the engine's external
mycorrhizal surface — the contracts that autonomous agents and future
Claude Code sessions build against. It is intentionally readable by both
humans and machines: an agent author should be able to implement a
compliant client from this document alone.

The protocol is the conserved-interface analog of the common-symbiosis
signaling pathway (CSSP/SYM) that lets land plants and fungi coevolve
without breaking each other (Strullu-Derrien et al. 2018). Deep coupling
is only possible against an interface that holds still.

---

## 1. Signal alphabet (Stage 4)

Agents communicate state to the engine through a **closed five-token
alphabet**. Expressivity comes from sequences and blends over time, not
per-event richness (the herbivore-induced-plant-volatile pattern; Babikova
et al. 2013).

| Token | Meaning |
|-----------|--------------------------------------------------|
| `STRESS` | Agent under load or encountering difficulty |
| `DISCOVER`| Agent found a new pattern / primitive / capability |
| `REQUEST` | Agent asking the engine for help |
| `OFFER` | Agent contributing validated output back |
| `WARN` | Agent observed a failure mode worth propagating |

**Signal fields** (`belief.signal.alphabet.Signal`, Pydantic v2):

- `agent_id: str` — non-empty, ≤128 chars.
- `token: Literal[...]` — one of the five above. The set is **closed**;
  adding a token is a protocol migration, not a free change.
- `magnitude: float` — in `[0.0, 1.0]`.
- `timestamp: datetime` — **timezone-aware required** (naive datetimes are
  rejected; ambiguous TZ corrupts decay math).
- `payload: Optional[dict]` — JSON-serializable, ≤200 bytes serialized.
  Diagnostic context only, not a channel for prompts.
- `idempotency_key: Optional[str]` — emitters SHOULD set this; absent, a
  stable SHA-256 digest of the content is derived.

**Temporal integration.** Receivers respond to the *integral* of recent
signal, not single events:

```
concentration(agent, token, window, half_life)
    = Σ over events e in window:  e.magnitude · (1/2) ** ((now − e.ts) / half_life)
```

`joint_concentration(agent, (token_a, token_b), ...)` is the product of the
two single-token concentrations — the HIPV-blend semantic for conjunction
triggers (e.g. `STRESS ∧ REQUEST`).

**Defaults:** window 5m, half-life 2m, per-(agent, token) circular buffer
of 1000 emissions.

---

## 2. Reciprocity contract (Stage 1)

Every agent is accounted for in the reciprocity ledger
(`belief.memory.reciprocity.ReciprocityLedger`).

- **`carbon_received`** — cumulative compute spent on the agent's requests
  (tokens + tool calls). Incremented via `record_request(agent_id, cost)`.
- **`nutrients_returned`** — cumulative validated contributions back to
  soil. Incremented via `record_contribution(agent_id, nutrient_value, ...)`.
- **`exchange_rate(window)`** — derived: `nutrients_returned /
  max(carbon_received, ε)` over a rolling window (default 7d, ε = 1e-3).

A never-seen agent has exchange rate `0.0` (defined, not an error). All
writes accept an `idempotency_key`; duplicate keys are dropped.

**The incentive:** allocation and privilege follow the exchange rate.
Acting in the system's interest (returning validated nutrients) is the
strategy with the highest payoff.

---

## 3. Niche-ledger schema (Stage 2)

Capability additions are logged in the niche ledger
(`belief.memory.niche_ledger.NicheLedger`).

A **niche** row:

| Field | Meaning |
|--------------------------|----------------------------------------------|
| `niche_id` | UUID. |
| `constructing_agent_id` | Who built the capability. |
| `kind` | One of `tool`, `primitive`, `pattern`, `covenant` (closed set). |
| `soil_reference` | The ChromaDB/soil id that materializes it. |
| `pre_state_description` | One line — what was not possible before. |
| `post_state_description` | One line — what is now possible. |
| `reference_count` | Downstream builds that have used this niche. |

Dedup is by `(kind, soil_reference)`. When a downstream build consumes a
niche via `record_reference(niche_id, referring_build_id)`, the **original
constructor** (not the referrer) receives a fixed `0.1` credit in the
reciprocity ledger. Widely-used niches accrue substantial credit; unused
ones accrue none. Attribution is the load-bearing rule: build things others
use.

---

## 4. Warning protocol (Stage 6)

Safety findings propagate as **primed** warnings, not immediate behavior
changes (Babikova et al. 2013 priming pattern; Heil & Karban 2009 cost
calculus). Two classes:

- **Priming-class** (`emit_priming`): gossip with TTL. Receivers raise a
  sentinel threshold on the warned pattern; they do **not** refuse the
  operation. Default half-life 24h.
- **Covenant-class** (`emit_covenant`): eager broadcast through hubs.
  Receivers **refuse** the matching operation until the warning is
  cleared. Default half-life 7d.

`check_operation(agent_id, operation_description)` returns a
`CheckResult` with `primed_patterns` (raised sentinels) and
`blocking_warnings` (covenant refusals). Matching is case-insensitive
substring. Warnings decay on read; re-observing a pattern refreshes its
expiry.

---

## 5. Onboarding contract (Stage 6)

A new agent (no reciprocity-ledger history) must pass the onboarding gate
(`belief.routing.onboarding.OnboardingGate`) before being served:

1. `submit(agent_id, self_description)` → a `DemoTask` with a known-good
   validator.
2. `complete(agent_id, output)` → on pass, the agent is admitted with an
   initial nutrient credit equal to the task value; on fail, the attempt
   counts toward a rate limit.
3. An `agent_id` matching a previously-archived (graveyard) agent requires
   explicit operator approval (`approve_manually`).

This is partner choice (West, Griffin & Gardner 2007): the engine can
refuse to take on a partner that hasn't demonstrated reciprocity.

---

## 6. Hub topology (Stage 5)

Hub status is **derived, not assigned** (`belief.routing.hubs.HubRegistry`):
an agent is a hub iff its 30-day exchange rate is in the top fraction of
all agents AND its lifetime `nutrients_returned` exceeds a floor. Promotion
is immediate; demotion is lagged (hysteresis) so the hub set doesn't thrash.

Routing defaults to **bypass** when no hubs exist (fresh ledger) — every
request flows directly to the engine, so the build path is unaltered until
enough reciprocity history accumulates to designate hubs.

---

## 7. Deprecation policy

- **Additive changes** (new optional Signal field, a new niche `kind` via
  migration, a new token) bump the **minor** version and remain
  backward-compatible.
- **Breaking changes** bump the **major** version and only after a labelled
  deprecation phase of at least one full release window, during which the
  old surface keeps working with a logged warning.
- `belief.protocol.compatibility_check(client_version)` reports
  compatibility: exact match (silent), same-major minor drift (warn,
  proceed), major mismatch (warn, proceed best-effort, recommend upgrade).
  A version mismatch **never** hard-refuses an interaction — refusing would
  be more disruptive than a degraded interaction during a deprecation
  window.

---

## 8. Coevolution discipline

The engine commits to:

1. **A stable versioned surface** (this document) so agents can specialize
   against it.
2. **Specialization permitted, not forced** — the engine rewards validated
   output (Section 2); it does not require any specific agent-internal
   behavior.
3. **Dependency tracking via the weekly offline probe**
   (`belief.lifecycle.offline_probe`) — periodically running with the
   engine's cross-domain synthesizer disabled reveals which agent
   operations have become *obligately* dependent on the engine. This is
   information for the operator, not a failure: it tells you when coupling
   has become structural.

What would foreclose coevolution and is therefore avoided: frequent
breaking protocol changes, forcing agents into a uniform shape, and hiding
the engine's capabilities so agents can't discover (and specialize against)
them.
