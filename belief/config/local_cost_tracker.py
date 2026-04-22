"""Per-model token tracker for local (Ollama) calls.

Why this module exists: the project-level rule "do not modify
belief/hardening.py" is non-negotiable, but Session 6 Task 4 still
requires per-model cost visibility. Rather than widen BuildBudget,
local-model calls route here. Dashboards aggregate `BuildBudget`
(paid-API) and `LocalCostTracker` (local, $0) separately and sum.

Local calls are always $0 USD. We still record token counts so we can
measure efficiency (tokens per build, tokens per agent role) and
detect runaway local inference.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class LocalCallRecord:
    model: str
    role: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LocalCostTracker:
    """Thread-safe per-build local-call ledger.

    Parallel to BuildBudget in hardening.py but with cost always 0 —
    the value here is in the breakdown, not the dollar total.
    """

    records: list[LocalCallRecord] = field(default_factory=list)
    fallback_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        role: str = "",
    ) -> float:
        """Append one local call. Returns 0.0 (always) for API parity."""
        with self._lock:
            self.records.append(
                LocalCallRecord(
                    model=model,
                    role=role,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                )
            )
        return 0.0

    def record_fallback(self) -> None:
        """One local->cloud fallback happened. Counter visible to dashboard."""
        with self._lock:
            self.fallback_count += 1

    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.total_tokens for r in self.records)

    def total_calls(self) -> int:
        with self._lock:
            return len(self.records)

    def by_model(self) -> dict[str, dict[str, int]]:
        """Aggregate tokens per model name."""
        out: dict[str, dict[str, int]] = {}
        with self._lock:
            for r in self.records:
                bucket = out.setdefault(
                    r.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
                )
                bucket["calls"] += 1
                bucket["prompt_tokens"] += r.prompt_tokens
                bucket["completion_tokens"] += r.completion_tokens
        return out

    def by_role(self) -> dict[str, dict[str, int]]:
        """Aggregate tokens per agent role."""
        out: dict[str, dict[str, int]] = {}
        with self._lock:
            for r in self.records:
                bucket = out.setdefault(
                    r.role or "unknown",
                    {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                )
                bucket["calls"] += 1
                bucket["prompt_tokens"] += r.prompt_tokens
                bucket["completion_tokens"] += r.completion_tokens
        return out

    def summary(self) -> str:
        calls = self.total_calls()
        toks = self.total_tokens()
        return (
            f"local: {calls} calls, {toks} tokens, "
            f"{self.fallback_count} fallbacks, $0.00"
        )


__all__ = ["LocalCallRecord", "LocalCostTracker"]
