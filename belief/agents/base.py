"""Base class for all pipeline agents.

Source: forge/agents/base.py (production-tested crash protection)

Agent contract:
  - Read from state fields relevant to the agent's role
  - Write exactly one output artifact field
  - Set state.phase to the next phase before returning
  - Never raise exceptions that reach the graph

If run() raises, __call__ catches it, logs the traceback, appends to
state.errors, and sets phase=FAILED so the pipeline reaches END cleanly.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from belief.config.models import ModelRole, ModelRouter
from belief.llm import _usage_ctx
from belief.models.artifacts import TokenUsage
from belief.models.state import Phase, UnifiedState
from belief.thermal import async_thermal_gate

logger = logging.getLogger("belief.agents")

# Injected into every agent prompt — keeps token burn down
TOKEN_EFFICIENCY_BLOCK = """
## TOKEN EFFICIENCY CONSTRAINTS (mandatory)
1. Read before writing. Never regenerate a whole file when a surgical edit suffices.
2. Surgical edits only. If a file is >100 lines, edit only the lines that need changing.
3. No redundant reads. If you already have content from this prompt, don't re-read it.
4. One tool call = one purpose.
5. Output cap: ≤800 words (≤1200 if writing code). Be dense, not exhaustive.
6. No narration of obvious steps. Just do the thing.
7. Emit your completion signal as the very last line.
""".strip()


class BaseAgent(ABC):
    """Base class for pipeline agents.

    Subclasses define:
        role  (class attr): which ModelRole this agent uses
        name  (class attr): human-readable label
        run() (async): the agent's logic
    """

    role: ModelRole
    name: str

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self.model = router.get_model(self.role)

    @abstractmethod
    async def run(self, state: UnifiedState) -> UnifiedState:
        """Execute agent logic. Read state, write output, advance phase."""
        ...

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node interface: dict in → dict out.

        Wraps run() with state hydration, timing, cost tracking,
        polarity updates, and crash protection.

        Session 1 (v3.2): a thermal-gate preflight sleeps proportionally
        to macOS thermal pressure before the agent starts.  No-op on
        non-macOS.  The sleep is NOT attributed to the agent's timing
        because the gate runs before ``_t0`` — it's system-level
        overhead, not agent work.
        """
        # Thermal gate — never allow this to fail the agent.
        try:
            await async_thermal_gate()
        except Exception as thermal_err:  # pragma: no cover
            logger.debug("thermal_gate skipped: %s", thermal_err)

        _t0 = time.monotonic()
        agent_log = logging.getLogger(f"belief.agents.{self.name.lower()}")

        # Hydrate state from dict
        try:
            forge_state = UnifiedState(
                **{k: v for k, v in state.items() if k in UnifiedState.model_fields}
            )
        except Exception as e:
            logger.error(f"{self.name}: state deserialization failed: {e}", exc_info=True)
            failed: dict[str, Any] = dict(state)
            failed["phase"] = Phase.FAILED.value
            failed.setdefault("errors", []).append(f"{self.name}: state error: {e}")
            return failed

        # Set up per-agent token tracking
        agent_usage = TokenUsage(backend=self.router.backend)
        ctx_token = _usage_ctx.set(agent_usage)

        try:
            result = await self.run(forge_state)
            elapsed = time.monotonic() - _t0
            agent_log.info(f"{self.name} completed in {elapsed:.1f}s")
            output = result.model_dump()

            # ── Polarity update after every agent action ──────────────────
            await self._update_polarity(output, elapsed)

            # Merge token usage
            existing_raw = output.get("token_usage")
            if existing_raw:
                existing = (
                    TokenUsage(**existing_raw) if isinstance(existing_raw, dict) else existing_raw
                )
                output["token_usage"] = existing.merge(agent_usage).model_dump()
            elif agent_usage.total_prompt_tokens > 0:
                output["token_usage"] = agent_usage.model_dump()

            # Record timing
            timings: dict[str, float] = dict(output.get("agent_timings") or {})
            timings[self.name] = round(elapsed, 2)
            output["agent_timings"] = timings
            return output

        except Exception as exc:
            elapsed = time.monotonic() - _t0
            agent_log.error(f"{self.name} crashed after {elapsed:.1f}s: {exc}", exc_info=True)
            forge_state.errors.append(f"{self.name} crashed: {exc}")
            forge_state.phase = Phase.FAILED
            output = forge_state.model_dump()
            timings = dict(output.get("agent_timings") or {})
            timings[self.name] = round(elapsed, 2)
            output["agent_timings"] = timings
            return output

        finally:
            _usage_ctx.reset(ctx_token)

    async def _update_polarity(self, output: dict[str, Any], elapsed: float) -> None:
        """Run polarity update after each agent action.

        Extracts remainder (what did this agent fail to account for?)
        and updates frequency coherence. Uses heuristics only — no LLM calls
        here to keep per-agent overhead at zero cost.
        """
        try:
            from belief.polarity.frequency import FrequencyLayer
            from belief.models.state import PolarityState

            polarity_raw = output.get("polarity", {})
            if isinstance(polarity_raw, dict):
                polarity = PolarityState(**polarity_raw)
            else:
                polarity = polarity_raw

            # Heuristic remainder — zero LLM cost
            errors = output.get("errors", [])
            warnings = output.get("warnings", [])

            if errors:
                # Agent produced errors — low coherence signal
                remainder = f"{self.name}: {errors[-1][:100]}"
                latios_signal = 0.2
            elif warnings:
                remainder = f"{self.name}: {warnings[-1][:100]}"
                latios_signal = 0.5
            elif elapsed > 30:
                remainder = f"{self.name}: took {elapsed:.0f}s — may indicate complexity"
                latios_signal = 0.6
            else:
                remainder = None
                latios_signal = 0.8

            # Update polarity state
            if remainder:
                remainders = polarity.accumulated_remainders or []
                remainders.append(remainder)
                if len(remainders) > 50:
                    remainders = remainders[-50:]
                polarity.accumulated_remainders = remainders
                polarity.current_remainder = remainder

            # Frequency update — latias signal based on whether output is substantive
            code_files = output.get("code_files", {})
            latias_signal = 0.7 if code_files else 0.5

            freq = FrequencyLayer()
            polarity = freq.update(polarity, latios_signal, latias_signal)

            output["polarity"] = polarity.model_dump()

        except Exception as e:
            # Polarity is never a blocker — fail silently
            logger.debug(f"Polarity update skipped: {e}")
