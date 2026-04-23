"""Ollama client error hierarchy — Session 1 (v3.2).

Classifies Ollama failures into retryable vs non-retryable so the
tenacity retry + pybreaker circuit breaker only retry things that
can plausibly succeed on a second attempt.

Design:
  OllamaError                          (base)
    OllamaTransientError               (retryable)
      OllamaStreamStall                (watchdog fired: no chunks in N seconds)
    OllamaPermanentError               (non-retryable)
      OllamaContextExceeded            (input too large for num_ctx)

The separation matters because the overnight logs showed the architect
retrying a context-length-exceeded error 3 times, burning 300s of wall
clock on a failure that can never succeed. By excluding
OllamaPermanentError (and subclasses) from the retry/breaker predicate
we fail fast on these.
"""

from __future__ import annotations

from typing import Any


class OllamaError(Exception):
    """Base class for all Ollama-specific errors."""

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class OllamaTransientError(OllamaError):
    """Retryable failure — stream stall, connection reset, server restart."""


class OllamaPermanentError(OllamaError):
    """Non-retryable failure — context-length exceeded, model-not-found,
    malformed-request. Retrying will produce the same outcome."""


class OllamaStreamStall(OllamaTransientError):
    """The per-chunk inactivity watchdog fired — runner is wedged.

    When this is raised the caller should POST ``{"keep_alive": 0}`` to
    ``/api/generate`` with the same model name to force-unload the
    wedged runner before the retry layer sees it. That logic lives in
    :class:`belief.llm.AsyncOllamaClient._generate_stream`.
    """

    def __init__(
        self,
        message: str = "Ollama stream stalled: no chunks received within inactivity window",
        *,
        inactivity_s: float | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, details={"inactivity_s": inactivity_s, "model": model})
        self.inactivity_s = inactivity_s
        self.model = model


class OllamaContextExceeded(OllamaPermanentError):
    """Input tokens exceed the runner's ``num_ctx`` window.

    Ollama typically surfaces this as a 4xx response whose body contains
    phrases like ``"context length"`` or ``"exceeds"``.  :func:`classify_4xx`
    detects those substrings and raises this so tenacity's retry layer
    skips it (configured via ``retry=retry_if_exception_type(...)`` that
    excludes :class:`OllamaPermanentError`).
    """


__all__ = [
    "OllamaError",
    "OllamaTransientError",
    "OllamaPermanentError",
    "OllamaStreamStall",
    "OllamaContextExceeded",
]
