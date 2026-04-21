"""Cost tracker stub — Session 5 replaces with SQLite-backed real impl.

Session 4 synthesis modules call these methods before every LLM
request. The stub is permissive (always under budget, always records
zero) so Session 4 tests can run without a pre-configured budget DB.

Session 5 will:
  - Swap the stub for a SQLite-backed log (one row per LLM call,
    (ts, model, input_tokens, output_tokens, cost_usd, tag)),
  - Enforce daily ($5) / weekly ($25) / monthly ($80) caps,
  - Raise BudgetExceeded when a cap would be crossed.

No synthesis caller should rely on stub behavior beyond the public
method signatures.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


class BudgetExceeded(RuntimeError):
    """Raised when a caller would cross a daily/weekly/monthly cap."""


@dataclass
class CostRecord:
    """One LLM call's cost row — Session 5 persists these."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tag: str = ""


@dataclass
class CostTracker:
    """Per-process cost tracker.

    Stub behavior: every under_budget() returns True; every record()
    appends to an in-memory list. Session 5 replaces with persistent
    SQLite + cap enforcement.
    """

    daily_cap_usd: float = 5.0
    weekly_cap_usd: float = 25.0
    monthly_cap_usd: float = 80.0

    _records: list[CostRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def under_budget(self, projected_cost_usd: float = 0.0, tag: str = "") -> bool:
        """Session 5: consult persistent log. Stub: always True."""
        return True

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        tag: str = "",
    ) -> None:
        """Append a cost record. Persisted to SQLite in Session 5."""
        with self._lock:
            self._records.append(
                CostRecord(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    tag=tag,
                )
            )

    def total(self, tag: Optional[str] = None) -> float:
        """Sum the tracked cost, optionally filtered by tag."""
        with self._lock:
            if tag is None:
                return sum(r.cost_usd for r in self._records)
            return sum(r.cost_usd for r in self._records if r.tag == tag)


__all__ = ["BudgetExceeded", "CostRecord", "CostTracker"]
