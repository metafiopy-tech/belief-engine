# HTTP and LLM boundaries

**Session 0.5, 2026-04-23.** Last updated when the external audit flagged
`library_inductor.py` and `package_validator.py` for bypassing the shared
HTTP stack.  This document is the source of truth for which module is
allowed to call what.

## TL;DR

Three outbound-call surfaces.  One of them has two legitimate
dispatchers, which this document justifies and pins.

| Surface            | Module                                            | Who may call it                                    |
|--------------------|---------------------------------------------------|----------------------------------------------------|
| Main pipeline LLM  | `belief/llm.py`                                   | Build-pipeline agents (intake → validator)         |
| Photosynthesis LLM | `belief/photosynthesis/safety/cost_tracker.py`    | Photosynthesis daemon (and helpers driven by it)   |
| Everything else    | `belief/core/http.py`                             | Every other outbound HTTP caller in the codebase   |

The boundary test (`tests/test_http_boundary.py`) enforces this by
forbidding raw `httpx.Client(` / `httpx.AsyncClient(` / `from anthropic
import` outside the exemption list.

## The two LLM paths

The Session 0.5 prompt asked for *one* dispatcher.  The actual code has
two and this is intentional, at least for now:

### `belief/llm.py` — main pipeline dispatcher

* Called from every agent in the build pipeline: `intake`, `planner`,
  `architect`, `builder`, `debugger`, `synthesizer`, `validator`, and
  the Session 1 Ollama-hardened local path.
* Talks directly to `https://api.anthropic.com/v1/messages` via raw
  `httpx.AsyncClient` (no `anthropic` SDK dependency) and to local
  Ollama at `127.0.0.1` via raw `httpx.AsyncClient`.  Both transports
  are owned by this module — see the exemption list below.
* Owns: per-role time budgets, retry with per-request error
  classification, Ollama inactivity watchdogs, graceful-degradation
  cascade across models.
* Does NOT own: authoritative per-call $ accounting reconciled against
  the Anthropic Admin API.  The pipeline's budget is wall-clock and
  role-scoped, not dollar-scoped.

### `belief/photosynthesis/safety/cost_tracker.py` — daemon dispatcher

* Exports `BreakerAnthropic`, structurally typed against
  `HasMessagesCreate` so tests can inject fakes without the
  `anthropic` SDK installed.
* Wraps the `anthropic.Anthropic` SDK specifically to stamp every call
  with the cost-tracker's per-call $ metering, reconciled daily
  against the Anthropic Admin API.  This is the part that enforces the
  daemon's hard daily-budget cap.
* **Instantiated only from `belief/photosynthesis/`.**  It can be
  *passed through* to helpers the daemon drives — e.g.,
  `belief.memory.library_inductor.promote_eligible` accepts any
  `HasMessagesCreate`, and the daemon passes `BreakerAnthropic`.
  That's consistent with the rule: the *dispatcher choice* is the
  photosynthesis daemon's, not library_inductor's.

### Why not merge now

Merging means folding $-per-call accounting + Admin-API reconciliation
into `belief/llm.py`.  That's the right long-term shape but a
non-trivial change with real regression risk in the cost-cap path (a
miscount silently lifts the daemon's budget cap).  Session 0.5 scoped
to *documenting* the boundary and *enforcing* no further bypass;
merger is a later session.  This doc should be deleted the day that
happens.

## `belief/core/http.py` — everything else

Shared `httpx` wrapper with tenacity retry, pybreaker circuit-breaker,
and a domain allowlist (`DEFAULT_ALLOWED_DOMAINS`).  Use it for:

* PyPI existence checks (`package_validator.py`)
* arXiv / GitHub / Hacker News / Stack Exchange fetches
  (`belief/photosynthesis/sources/`)
* Release-feed polls (`github_releases.py`)
* Telegram / other sync webhook sends (`post_form_sync`)
* Anything new that needs outbound HTTP

### Helpers

* `get_async_client(...)` — factory for `BreakerAsyncClient`, used via
  `async with`.  Pass `allowed_domains=DEFAULT_ALLOWED_DOMAINS` for
  production traffic.
* `conditional_get(client, url, ...)` — ETag / If-Modified-Since
  wrapper (caller already inside retry/breaker context).
* `head_sync(url, ...)` — one-shot status-code probe.
* `get_bytes_sync(url, ...)` — one-shot body fetch (added Session 0.5
  for the top-15k PyPI-corpus refresh path).
* `post_form_sync(url, data, ...)` — one-shot form POST (Telegram
  notifications, etc.).

### Adding a new domain

Add to `DEFAULT_ALLOWED_DOMAINS` in `belief/core/http.py` *and* leave
a comment explaining what the domain is for.  Don't silently append —
the allowlist is meant to be eyeballed.

## Exemptions from the boundary test

`tests/test_http_boundary.py` enforces:

* No `httpx.Client(` or `httpx.AsyncClient(` outside:
  * `belief/core/http.py` — the shared wrapper itself
  * `belief/llm.py` — main-pipeline transport (Anthropic + Ollama)
  * `belief/experiments/raw_runner.py` — the raw-model comparison
    baseline, deliberately bypassing the engine's stack
* No `from anthropic import` or `anthropic.Anthropic(` outside:
  * `belief/photosynthesis/safety/cost_tracker.py` — the only
    module that instantiates the `anthropic` SDK

Any grep match outside these paths is a boundary violation.  The test
prints the offending files and fails.

## Historical notes

* **Session 11** introduced `BreakerAnthropic` when photosynthesis
  needed authoritative $-metering that the pipeline dispatcher didn't
  provide.
* **Session 3** (v3.2) built `package_validator.py` against raw
  `httpx`; Session 0.5 rerouted it through `belief.core.http`.
* **Session 0.5** dropped the `arxiv` pip package because it exposed
  only a private `_session: requests.Session` attribute (no public
  hook to inject the shared transport).  The HTTP fallback — already
  using `BreakerAsyncClient` — was promoted to the sole path.
