"""Error Classifier — OTP-inspired failure categorization for the agent pipeline.

Maps Erlang/OTP supervision patterns to LLM agent failure handling:

| OTP Concept          | Belief Engine Equivalent                          |
|----------------------|---------------------------------------------------|
| one_for_one restart  | Retry individual agent without affecting pipeline |
| one_for_all restart  | Rebuild from architect when shared state corrupted|
| Process isolation    | Fresh context per agent (no accumulated history)  |
| Supervisor limits    | Circuit breaker on repeated failures              |
| "Let it crash"       | Fail fast on non-retriable errors                 |

Four error categories:
  TRANSIENT  — retry with backoff (rate limits, timeouts, 5xx)
  REPAIRABLE — re-prompt with error context (test failures, type errors, import errors)
  TERMINAL   — fail fast, don't waste tokens (auth failures, content policy, missing tools)
  DEGRADED   — circuit break (repeated hallucination, same error 3x, quality collapse)

Usage:
    from belief.agents.error_classifier import classify_error, ErrorCategory, RecoveryStrategy

    category = classify_error(error_summary, error_type, iteration)
    if category.strategy == RecoveryStrategy.RETRY_BACKOFF:
        await asyncio.sleep(category.backoff_seconds)
        # retry same agent
    elif category.strategy == RecoveryStrategy.REPROMPT:
        # feed error context to debugger
    elif category.strategy == RecoveryStrategy.FAIL_FAST:
        # skip to synthesizer
    elif category.strategy == RecoveryStrategy.CIRCUIT_BREAK:
        # stop retrying, use best result so far
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("belief.agents.error_classifier")


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    REPAIRABLE = "repairable"
    TERMINAL = "terminal"
    DEGRADED = "degraded"


class RecoveryStrategy(str, Enum):
    RETRY_BACKOFF = "retry_backoff"
    REPROMPT = "reprompt"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAK = "circuit_break"


@dataclass
class ClassifiedError:
    """A classified error with recovery strategy."""
    category: ErrorCategory
    strategy: RecoveryStrategy
    reason: str
    backoff_seconds: float = 0.0
    should_change_model: bool = False
    should_rebuild: bool = False


def classify_error(
    error_summary: str,
    stderr: str = "",
    iteration: int = 0,
    max_iterations: int = 3,
    previous_errors: list[str] | None = None,
) -> ClassifiedError:
    """Classify an error and determine recovery strategy.

    Args:
        error_summary: Structured error diagnosis from _extract_error
        stderr: Raw stderr output
        iteration: Current pipeline iteration
        max_iterations: Max allowed iterations
        previous_errors: List of previous error summaries for dedup

    Returns:
        ClassifiedError with category, strategy, and recovery hints
    """
    text = (error_summary + " " + stderr).lower()
    prev = previous_errors or []

    # ── TRANSIENT: retry with backoff ────────────────────────────────
    # Rate limits, timeouts, temporary network failures
    transient_patterns = [
        (r"429|rate.?limit|too many requests", 30.0),
        (r"timeout|timed? out", 5.0),
        (r"50[0-3]|internal server error|bad gateway|service unavailable", 10.0),
        (r"connection.?(reset|refused|error)|econnreset", 5.0),
        (r"temporary|transient|retry.?after", 10.0),
    ]
    for pattern, backoff in transient_patterns:
        if re.search(pattern, text):
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                strategy=RecoveryStrategy.RETRY_BACKOFF,
                reason=f"Transient error (matched: {pattern.split('|')[0]})",
                backoff_seconds=backoff * (2 ** min(iteration, 3)),  # exponential
            )

    # ── TERMINAL: fail fast, don't waste tokens ─────────────────────
    # Auth failures, content policy, missing infrastructure
    terminal_patterns = [
        r"401|unauthorized|invalid.?api.?key|authentication",
        r"403|forbidden|access.?denied",
        r"content.?policy|safety|refused to generate",
        r"context.?length|context.?window|token.?limit exceeded",
        r"model.?not.?found|invalid.?model",
        r"npm not found|node not found|python not found",
        r"billing|quota|insufficient.?funds",
    ]
    for pattern in terminal_patterns:
        if re.search(pattern, text):
            return ClassifiedError(
                category=ErrorCategory.TERMINAL,
                strategy=RecoveryStrategy.FAIL_FAST,
                reason=f"Terminal error — cannot recover (matched: {pattern.split('|')[0]})",
            )

    # ── DEGRADED: circuit break on repeated failures ────────────────
    # Same error 3x, oscillation, quality collapse
    if len(prev) >= 2:
        # Same error appearing repeatedly
        recent_normalized = [_normalize(e) for e in prev[-2:]]
        current_normalized = _normalize(error_summary)
        if all(e == current_normalized for e in recent_normalized):
            return ClassifiedError(
                category=ErrorCategory.DEGRADED,
                strategy=RecoveryStrategy.CIRCUIT_BREAK,
                reason="Same error 3 consecutive times — circuit break",
            )

    if iteration >= max_iterations:
        return ClassifiedError(
            category=ErrorCategory.DEGRADED,
            strategy=RecoveryStrategy.CIRCUIT_BREAK,
            reason=f"Max iterations ({max_iterations}) reached",
        )

    # ── REPAIRABLE: re-prompt with error context ────────────────────
    # Import errors, test failures, type errors, syntax errors
    # These are the errors the debugger can actually fix
    repairable_patterns = {
        r"modulenotfounderror|importerror|cannot import": "import_error",
        r"syntaxerror": "syntax_error",
        r"attributeerror": "attribute_error",
        r"typeerror|type.?mismatch": "type_error",
        r"nameerror|undefined": "name_error",
        r"assertionerror|assert": "test_failure",
        r"keyerror": "key_error",
        r"error ts\d{4}": "typescript_error",
        r"failed.*test|test.*failed|\d+ failed": "test_failure",
        r"npm install failed|pip install failed": "dependency_error",
    }

    for pattern, error_type in repairable_patterns.items():
        if re.search(pattern, text):
            should_rebuild = error_type in ("syntax_error", "dependency_error") and iteration > 1
            return ClassifiedError(
                category=ErrorCategory.REPAIRABLE,
                strategy=RecoveryStrategy.REPROMPT,
                reason=f"Repairable {error_type} — feed to debugger",
                should_rebuild=should_rebuild,
                should_change_model=iteration >= 2,  # try different model after 2 failures
            )

    # Default: repairable (optimistic — let the debugger try)
    return ClassifiedError(
        category=ErrorCategory.REPAIRABLE,
        strategy=RecoveryStrategy.REPROMPT,
        reason="Unknown error — attempting repair",
    )


def _normalize(error: str) -> str:
    """Normalize error for dedup comparison."""
    s = error.lower().strip()
    s = re.sub(r'/tmp/\S+', '/tmp/X', s)
    s = re.sub(r'line \d+', 'line N', s)
    s = re.sub(r'\d+', 'N', s)
    return s
